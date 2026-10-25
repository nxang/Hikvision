"""Recoverable camera failures must skip a session without ending the scheduler."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aquaculture_system"))

import requests

import config
import feeding
import main as application
from hardware.camera import CameraUnavailableError, HikvisionCamera
from utils.logger import calculate_historical_adjustment


class CameraRecoveryTests(unittest.TestCase):
    def camera(self):
        camera = HikvisionCamera("example.invalid", "test", "test")
        camera._stop_event.wait = Mock(return_value=False)
        return camera

    def test_preset_recovers_on_third_attempt(self):
        camera = self.camera()
        with patch(
            "hardware.camera.requests.put",
            side_effect=[
                requests.Timeout("timeout"),
                requests.ConnectionError("offline"),
                Mock(),
            ],
        ) as request:
            self.assertTrue(camera.go_to_preset(2))
        self.assertEqual(request.call_count, 3)
        self.assertEqual(camera._stop_event.wait.call_count, 2)
        camera._stop_event.wait.assert_called_with(config.CAMERA_RETRY_DELAY_SECONDS)

    def test_preset_stops_after_retry_limit(self):
        camera = self.camera()
        with patch(
            "hardware.camera.requests.put", side_effect=requests.Timeout("offline")
        ) as request:
            self.assertFalse(camera.go_to_preset(2))
        self.assertEqual(request.call_count, config.CAMERA_RETRY_ATTEMPTS)
        self.assertEqual(
            camera._stop_event.wait.call_count, config.CAMERA_RETRY_ATTEMPTS - 1
        )

    def test_snapshot_recovers_without_restarting_worker(self):
        camera = self.camera()
        ai_worker = Mock()
        with tempfile.TemporaryDirectory() as folder, patch(
            "hardware.camera.requests.get",
            side_effect=[requests.Timeout("offline"), Mock(content=b"image")],
        ) as request:
            camera.start_capture_thread(ai_worker)
            try:
                path, analysis = camera.queue_snapshot_job(
                    folder, "zone", "test"
                ).result(timeout=2)
                self.assertEqual(Path(path).read_bytes(), b"image")
                self.assertIs(analysis, ai_worker.queue_analysis_job.return_value)
                self.assertEqual(request.call_count, 2)
                self.assertTrue(camera.capture_thread.is_alive())
            finally:
                camera.stop_capture_thread()

    def test_snapshot_exhaustion_completes_job_with_camera_error(self):
        camera = self.camera()
        with patch(
            "hardware.camera.requests.get",
            side_effect=requests.ConnectionError("offline"),
        ) as request:
            camera.start_capture_thread(Mock())
            try:
                with self.assertRaises(CameraUnavailableError):
                    camera.queue_snapshot_job("unused", "zone", "test").result(
                        timeout=2
                    )
                camera.capture_queue.join()
                self.assertEqual(request.call_count, config.CAMERA_RETRY_ATTEMPTS)
                self.assertTrue(camera.capture_thread.is_alive())
            finally:
                camera.stop_capture_thread()

    def test_shutdown_interrupts_retry_wait(self):
        camera = self.camera()
        camera._stop_event.wait.return_value = True
        with patch(
            "hardware.camera.requests.put", side_effect=requests.Timeout("offline")
        ) as request:
            self.assertFalse(camera.go_to_preset(2))
        self.assertEqual(request.call_count, 1)

    def test_biomass_preset_failure_skips_without_dispensing(self):
        camera, relay = Mock(), Mock()
        camera.go_to_preset.return_value = False
        # Exercise the real biomass preflight up to the failed preset movement.
        with tempfile.TemporaryDirectory() as folder, patch.object(
            config, "MEDIA_OUTPUT_DIR", folder
        ), patch(
            "biomass.os.path.exists",
            side_effect=lambda path: (
                True if path == config.BIOMASS_MATRIX_PATH else Path(path).exists()
            ),
        ), patch(
            "biomass.np.load"
        ), patch.object(
            feeding, "dispense_feed"
        ) as dispense, patch.object(
            feeding, "log_session_summary"
        ) as ledger:
            feeding.run_dynamic_feeding_engine(camera, relay, Mock(), Mock())
        dispense.assert_not_called()
        relay.trigger_feeding.assert_not_called()
        self.assertEqual(ledger.call_args.args[1], 0)
        self.assertEqual(ledger.call_args.args[-1], "SKIPPED_CAMERA_UNAVAILABLE")

    def test_camera_failure_after_portion_preserves_delivery_and_stops_session(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(
            config, "MEDIA_OUTPUT_DIR", folder
        ), patch.object(feeding, "run_biomass_preflight_analysis"), patch.object(
            feeding, "calculate_historical_adjustment", return_value=1
        ), patch.object(
            feeding, "scan_and_evaluate_perimeter", return_value=0
        ), patch.object(
            feeding, "dispense_feed", return_value=1.2
        ) as dispense, patch.object(
            feeding,
            "capture_pellet_count",
            side_effect=CameraUnavailableError("offline"),
        ), patch.object(
            feeding, "log_session_summary"
        ) as ledger:
            feeding.run_dynamic_feeding_engine(Mock(), Mock(), Mock(), Mock())
        dispense.assert_called_once()
        self.assertEqual(ledger.call_args.args[1], 1.2)
        self.assertEqual(ledger.call_args.args[-1], "STOPPED_CAMERA_UNAVAILABLE")

    def test_scheduler_runs_next_session_after_camera_skip(self):
        clock = {"minute": "09:00"}
        original_strftime = application.time.strftime

        def local_time(format_string):
            if format_string == "%H:%M":
                return clock["minute"]
            if format_string == "%Y-%m-%d %H:%M":
                return "2026-10-25 " + clock["minute"]
            return original_strftime(format_string)

        def wait(seconds):
            if seconds == 60:
                clock["minute"] = "12:00"
            if seconds == 10 and ledger.call_count == 2:
                raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as folder, patch.object(
            config, "MEDIA_OUTPUT_DIR", folder
        ), patch.object(config, "FEED_TIMES", ["09:00", "12:00"]), patch.object(
            application.time, "strftime", side_effect=local_time
        ), patch.object(
            application.time, "sleep", side_effect=wait
        ), patch.object(
            feeding,
            "run_biomass_preflight_analysis",
            side_effect=[CameraUnavailableError("offline"), None],
        ), patch.object(
            feeding, "calculate_historical_adjustment", return_value=1
        ), patch.object(
            feeding, "scan_and_evaluate_perimeter", return_value=0
        ), patch.object(
            feeding, "capture_pellet_count", return_value=0
        ), patch.object(
            feeding, "dispense_feed", side_effect=lambda c, r, p, cups: cups
        ) as dispense, patch.object(
            feeding, "log_session_summary"
        ) as ledger:
            with self.assertRaises(KeyboardInterrupt):
                application.run_schedule(Mock(), Mock(), Mock(), Mock())
        self.assertEqual(
            [call.args[-1] for call in ledger.call_args_list],
            ["SKIPPED_CAMERA_UNAVAILABLE", "SUCCESS_FULL_RUN"],
        )
        self.assertEqual(dispense.call_count, config.NUM_DISTRIBUTIONS)

    def test_camera_skips_do_not_reset_reduced_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ledger.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["time", "budget", "mult", "actual", "status"])
                writer.writerow(["time", 3, 0.5, 3, "SUCCESS_FULL_RUN"])
                writer.writerow(["time", 6, 1, 0, "SKIPPED_CAMERA_UNAVAILABLE"])
                writer.writerow(["time", 6, 1, 1.2, "STOPPED_CAMERA_UNAVAILABLE"])
            self.assertEqual(calculate_historical_adjustment(path, 6), 0.5)

    def test_invalid_retry_settings_rejected(self):
        with patch.object(config, "CAMERA_RETRY_ATTEMPTS", 0):
            with self.assertRaises(ValueError):
                feeding.validate_feeding_config()
        with patch.object(config, "CAMERA_RETRY_DELAY_SECONDS", -1):
            with self.assertRaises(ValueError):
                feeding.validate_feeding_config()


if __name__ == "__main__":
    unittest.main()
