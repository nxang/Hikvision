import csv
import sys
import tempfile
import threading
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aquaculture_system"))

import config
import main
from ai.detector import AIAnalysisWorker
from hardware.camera import HikvisionCamera
from hardware.relay import ExternalRelayWorker
from utils.logger import calculate_historical_adjustment


def result(value=None, error=None):
    future = Future()
    if error:
        future.set_exception(error)
    else:
        future.set_result(value)
    return future


class SafetyTests(unittest.TestCase):
    def test_scan_waits_before_next_camera_move(self):
        camera = Mock()
        camera.go_to_preset.return_value = True
        order = []
        camera.go_to_preset.side_effect = lambda preset: order.append(f"move{preset}") or True
        def capture(**kwargs):
            future = Mock()
            def finish(**kwargs):
                order.append("captured")
                return "image.jpg", result(0)
            future.result.side_effect = finish
            return future
        camera.queue_snapshot_job.side_effect = capture
        with patch.object(main.time, "sleep"):
            self.assertEqual(main.scan_and_evaluate_perimeter(camera, Mock(), [4, 5], "unused", "test"), 0)
        self.assertEqual(order, ["move4", "captured", "move5", "captured"])

    def test_camera_move_failure_aborts_scan(self):
        camera = Mock()
        camera.go_to_preset.return_value = False
        with self.assertRaises(RuntimeError):
            main.scan_and_evaluate_perimeter(camera, Mock(), [4], "unused", "test")
        camera.queue_snapshot_job.assert_not_called()

    def test_inference_failure_aborts_scan(self):
        camera = Mock()
        camera.queue_snapshot_job.return_value = result(("image.jpg", result(error=ValueError("inference"))))
        with patch.object(main.time, "sleep"), self.assertRaises(ValueError):
            main.scan_and_evaluate_perimeter(camera, Mock(), [4], "unused", "test")

    def test_capture_timeout_aborts_scan(self):
        camera = Mock()
        camera.queue_snapshot_job.return_value = Future()
        with patch.object(main.time, "sleep"), patch.object(config, "CAPTURE_TIMEOUT_SECONDS", 0.01):
            with self.assertRaises(TimeoutError):
                main.scan_and_evaluate_perimeter(camera, Mock(), [4], "unused", "test")

    def test_capture_errors_complete_jobs(self):
        for error in (OSError("capture failed"), RuntimeError("network")):
            camera = HikvisionCamera("example.invalid", "test", "test")
            with patch("hardware.camera.requests.get", side_effect=error):
                camera.start_capture_thread(Mock())
                try:
                    future = camera.queue_snapshot_job("unused", "zone", "test")
                    with self.assertRaises(type(error)):
                        future.result(timeout=2)
                    camera.capture_queue.join()
                    self.assertEqual(camera.capture_queue.unfinished_tasks, 0)
                finally:
                    camera.stop_capture_thread()

    def test_capture_disk_failure_is_reported(self):
        camera = HikvisionCamera("example.invalid", "test", "test")
        response = Mock(content=b"image")
        with patch("hardware.camera.requests.get", return_value=response), patch("builtins.open", side_effect=OSError("disk full")):
            camera.start_capture_thread(Mock())
            try:
                with self.assertRaises(OSError):
                    camera.queue_snapshot_job(str(ROOT), "zone", "test").result(timeout=2)
            finally:
                camera.stop_capture_thread()

    def test_detector_reports_invalid_images_and_inference_errors(self):
        for image, model in ((None, Mock()), (Mock(), Mock(side_effect=RuntimeError("inference")))):
            worker = AIAnalysisWorker("unused")
            with patch("ai.detector.YOLO", return_value=model), patch("ai.detector.cv2.imread", return_value=image):
                worker.start()
                try:
                    with self.assertRaises((ValueError, RuntimeError)):
                        worker.queue_analysis_job("unused.jpg", "unused", "zone").result(timeout=2)
                    worker.queue.join()
                    self.assertEqual(worker.queue.unfinished_tasks, 0)
                finally:
                    worker.stop()

    def test_detector_accepts_successful_zero_count(self):
        worker = AIAnalysisWorker("unused")
        prediction = Mock(boxes=[])
        with tempfile.TemporaryDirectory() as folder, patch("ai.detector.YOLO", return_value=Mock(return_value=[prediction])), patch("ai.detector.cv2.imread", return_value=Mock()), patch("ai.detector.cv2.imwrite", return_value=True):
            worker.start()
            try:
                self.assertEqual(worker.queue_analysis_job("image.jpg", folder, "zone").result(timeout=2), 0)
            finally:
                worker.stop()

    def test_relay_on_failure_attempts_off_and_latches_fault(self):
        relay = ExternalRelayWorker("unused")
        relay._set_hardware_state = Mock(side_effect=lambda state: state == "off")
        relay.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "ON unconfirmed"):
                relay.trigger_feeding(1).result(timeout=2)
            with self.assertRaises(RuntimeError):
                relay.trigger_feeding(1).result(timeout=2)
            self.assertEqual([c.args[0] for c in relay._set_hardware_state.call_args_list], ["on", "off"])
        finally:
            relay.stop()

    def test_relay_off_failure_retries_and_never_confirms_delivery(self):
        relay = ExternalRelayWorker("unused")
        relay._set_hardware_state = Mock(side_effect=lambda state: state == "on")
        with patch.object(config, "SECONDS_PER_CUP", 0.05):
            relay.start()
            with self.assertRaisesRegex(RuntimeError, "OFF unconfirmed"):
                relay.trigger_feeding(1).result(timeout=2)
            self.assertEqual(relay._set_hardware_state.call_count, 1 + config.RELAY_OFF_RETRIES)
            with self.assertRaises(RuntimeError):
                relay.stop()

    def test_relay_shutdown_interrupts_active_portion(self):
        relay = ExternalRelayWorker("unused")
        on = threading.Event()
        def command(state):
            if state == "on":
                on.set()
            return True
        relay._set_hardware_state = Mock(side_effect=command)
        relay.start()
        future = relay.trigger_feeding(1)
        self.assertTrue(on.wait(2))
        relay.stop()
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            future.result(timeout=2)
        self.assertFalse(relay.thread.is_alive())
        self.assertEqual(relay._set_hardware_state.call_args.args, ("off",))

    def test_relay_success_reports_confirmed_cups(self):
        relay = ExternalRelayWorker("unused")
        relay._set_hardware_state = Mock(return_value=True)
        with patch.object(config, "SECONDS_PER_CUP", 0.05):
            relay.start()
            try:
                self.assertEqual(relay.trigger_feeding(1).result(timeout=2), 1)
            finally:
                relay.stop()

    def history(self, budget, statuses):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ledger.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["time", "budget", "mult", "actual", "status"])
                for status in statuses:
                    writer.writerow(["time", budget, 1, 1, status])
            return calculate_historical_adjustment(path, 6, max_budget=6)

    def test_history_caps_growth_and_recovers_nonfinite_budget(self):
        self.assertEqual(self.history(100, ["SUCCESS_FULL_RUN"] * 3), 1)
        self.assertEqual(self.history("nan", ["SUCCESS_FULL_RUN"] * 3), 1)

    def test_hardware_errors_do_not_change_appetite_budget(self):
        self.assertEqual(self.history(3, ["SUCCESS_FULL_RUN", "ERROR_RuntimeError", "SUCCESS_FULL_RUN"]), 0.5)

    def test_error_rows_do_not_hide_overfeeding(self):
        self.assertAlmostEqual(self.history(6, ["SUCCESS_FULL_RUN", "ABORTED_FINAL_LEFTOVERS", "ERROR_RuntimeError"]), 0.9)

    def test_errors_do_not_dilute_chronic_overfeeding_window(self):
        self.assertAlmostEqual(self.history(6, ["ABORTED_DIRTY_TANK_PRECHECK", "ERROR_RuntimeError", "ABORTED_FINAL_LEFTOVERS", "ABORTED_AT_P1_C2"]), 0.7)

    def test_oversized_history_is_clamped_before_reduction(self):
        self.assertAlmostEqual(self.history(100, ["ABORTED_FINAL_LEFTOVERS"]), 0.9)

    def run_session(self, counts, feed_error=None, capture_error=None, expected_error=None):
        with tempfile.TemporaryDirectory() as folder, patch.object(config, "MEDIA_OUTPUT_DIR", folder), patch.object(main, "run_biomass_preflight_analysis"), patch.object(main, "calculate_historical_adjustment", return_value=1), patch.object(main, "capture_pellet_count", side_effect=capture_error), patch.object(main, "scan_and_evaluate_perimeter", side_effect=counts), patch.object(main, "dispense_feed", side_effect=feed_error or (lambda camera, relay, preset, cups: cups)) as dispense, patch.object(main, "log_session_summary") as ledger, patch.object(main.time, "sleep"), patch.object(main.time, "time", side_effect=range(0, 10000, 60)):
            if expected_error:
                with self.assertRaises(expected_error):
                    main.run_dynamic_feeding_engine(Mock(), Mock(), Mock(), Mock())
            else:
                main.run_dynamic_feeding_engine(Mock(), Mock(), Mock(), Mock())
            return dispense.call_count, ledger.call_args.args

    def test_two_strikes_precede_active_velocity_branch(self):
        # The second sweep has high apparent consumption but still exceeds limits.
        with patch.object(config, "VELOCITY_ACTIVE_THRESHOLD", 0):
            calls, record = self.run_session([0, 3, 20, 20, 3, 10, 10])
        self.assertEqual(calls, 1)
        self.assertEqual(record[-1], "ABORTED_AT_P1_C2")

    def test_two_strikes_precede_stagnation_branch(self):
        calls, record = self.run_session([0, 3, 10, 10, 3, 10, 10])
        self.assertEqual(calls, 1)
        self.assertEqual(record[-1], "ABORTED_AT_P1_C2")

    def test_failed_precheck_never_dispenses(self):
        calls, record = self.run_session(RuntimeError("camera failed"), expected_error=RuntimeError)
        self.assertEqual(calls, 0)
        self.assertEqual(record[1], 0)

    def test_failed_relay_does_not_record_requested_delivery(self):
        calls, record = self.run_session([0], feed_error=RuntimeError("relay failed"), expected_error=RuntimeError)
        self.assertEqual(calls, 1)
        self.assertEqual(record[1], 0)
        self.assertTrue(record[-1].startswith("ERROR_"))

    def test_failed_verification_preserves_confirmed_delivery(self):
        calls, record = self.run_session([0], capture_error=RuntimeError("capture failed"), expected_error=RuntimeError)
        self.assertEqual(calls, 1)
        self.assertEqual(record[1], 1.2)

    def test_clean_session_never_exceeds_budget(self):
        calls, record = self.run_session([0] * 20)
        self.assertEqual(calls, 5)
        self.assertAlmostEqual(record[1], 6)

    def test_dirty_precheck_aborts_before_any_portion(self):
        calls, record = self.run_session([3, 3])
        self.assertEqual(calls, 0)
        self.assertEqual(record[1], 0)
        self.assertEqual(record[-1], "ABORTED_DIRTY_TANK_PRECHECK")

    def test_exact_leftover_threshold_allows_feeding_with_cap(self):
        calls, record = self.run_session([2] * 20)
        self.assertEqual(calls, 5)
        self.assertAlmostEqual(record[1], 6)
        self.assertEqual(record[-1], "SUCCESS_FULL_RUN")

    def test_final_leftovers_prevent_success_after_full_budget(self):
        # Precheck + four clean inter-portion checks, then two dirty final checks.
        calls, record = self.run_session([0] * 9 + [3, 3])
        self.assertEqual(calls, 5)
        self.assertAlmostEqual(record[1], 6)
        self.assertEqual(record[-1], "ABORTED_FINAL_LEFTOVERS")

    def test_final_consumption_can_clear_on_second_check(self):
        calls, record = self.run_session([0] * 9 + [3, 0])
        self.assertEqual(calls, 5)
        self.assertEqual(record[-1], "SUCCESS_FULL_RUN")

    def test_final_scan_failure_records_error(self):
        calls, record = self.run_session([0] * 9 + [RuntimeError("final scan failed")], expected_error=RuntimeError)
        self.assertEqual(calls, 5)
        self.assertAlmostEqual(record[1], 6)
        self.assertTrue(record[-1].startswith("ERROR_"))

    def test_early_budget_exhaustion_still_checks_final_consumption(self):
        with patch.object(config, "APPETITE_BOOST_MULTIPLIER", 10):
            calls, record = self.run_session([0] * 5 + [3, 3])
        self.assertEqual(calls, 2)
        self.assertAlmostEqual(record[1], 6)
        self.assertEqual(record[-1], "ABORTED_FINAL_LEFTOVERS")

    def test_dry_run_and_live_sessions_use_different_history(self):
        for dry_run, expected in ((True, config.DRY_RUN_CSV_LOG_FILE), (False, config.CSV_LOG_FILE)):
            with self.subTest(dry_run=dry_run), patch.object(config, "DRY_RUN", dry_run):
                calls, record = self.run_session([0] * 20)
                self.assertEqual(record[0], expected)
                self.assertEqual(calls, 5)

    def test_dry_run_uses_only_simulated_history(self):
        # Run the real ledger writer and history reader against disposable files.
        from utils.logger import log_session_summary
        with tempfile.TemporaryDirectory() as folder:
            live = Path(folder) / "live.csv"
            dry = Path(folder) / "dry.csv"
            log_session_summary(live, 2, 3, 0.5, "SUCCESS_FULL_RUN")
            original_live = live.read_bytes()
            with patch.object(config, "DRY_RUN", True), patch.object(config, "CSV_LOG_FILE", str(live)), patch.object(config, "DRY_RUN_CSV_LOG_FILE", str(dry)), patch.object(config, "MEDIA_OUTPUT_DIR", folder), patch.object(main, "run_biomass_preflight_analysis"), patch.object(main, "capture_pellet_count"), patch.object(main, "scan_and_evaluate_perimeter", return_value=0), patch.object(main, "dispense_feed", side_effect=lambda c, r, p, cups: cups), patch.object(main.time, "sleep"):
                main.run_dynamic_feeding_engine(Mock(), Mock(), Mock(), Mock())
            self.assertEqual(live.read_bytes(), original_live)
            with dry.open(newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(float(rows[-1][1]), 6)
            self.assertEqual(rows[-1][4], "SUCCESS_FULL_RUN")

    def test_on_acknowledgment_time_is_subtracted(self):
        relay = ExternalRelayWorker("unused")
        relay._set_hardware_state = Mock(return_value=True)
        relay.stop_event.wait = Mock(return_value=False)
        with patch("hardware.relay.time.monotonic", side_effect=[100, 102]), patch.object(config, "SECONDS_PER_CUP", 11):
            relay.start()
            try:
                self.assertEqual(relay.trigger_feeding(1).result(timeout=2), 1)
                relay.stop_event.wait.assert_called_once_with(9)
            finally:
                relay.stop()

    def test_late_on_acknowledgment_faults_and_turns_off(self):
        relay = ExternalRelayWorker("unused")
        relay._set_hardware_state = Mock(return_value=True)
        relay.stop_event.wait = Mock(return_value=False)
        with patch("hardware.relay.time.monotonic", side_effect=[100, 112]), patch.object(config, "SECONDS_PER_CUP", 11):
            relay.start()
            try:
                with self.assertRaisesRegex(RuntimeError, "deadline"):
                    relay.trigger_feeding(1).result(timeout=2)
                relay.stop_event.wait.assert_not_called()
                self.assertTrue(relay.faulted)
                self.assertEqual(relay._set_hardware_state.call_args.args, ("off",))
            finally:
                relay.stop()

    def test_recovered_off_retry_still_marks_delivery_unknown(self):
        relay = ExternalRelayWorker("unused")
        relay._set_hardware_state = Mock(side_effect=[True, False, True, True])
        relay.stop_event.wait = Mock(return_value=False)
        relay.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "OFF required retries"):
                relay.trigger_feeding(1).result(timeout=2)
            self.assertTrue(relay.faulted)
        finally:
            relay.stop()

    def test_missing_camera_password_rejected(self):
        with self.assertRaisesRegex(ValueError, "HIKVISION_CAMERA_PASSWORD"):
            HikvisionCamera("example.invalid", "test", "")

    def test_camera_uses_certificate_verification(self):
        for verification in (True, "camera-ca.pem"):
            with self.subTest(verification=verification), patch.object(config, "CAMERA_TLS_VERIFY", verification), patch("hardware.camera.requests.put", return_value=Mock()) as put:
                camera = HikvisionCamera("example.invalid", "test", "test")
                self.assertTrue(camera.go_to_preset(4))
                self.assertEqual(put.call_args.kwargs["verify"], verification)

    def test_invalid_configuration_rejected(self):
        with patch.object(config, "NUM_DISTRIBUTIONS", 0), self.assertRaises(ValueError):
            main.validate_feeding_config()


if __name__ == "__main__":
    unittest.main()
