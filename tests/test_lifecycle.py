"""Application lifecycle and logging checks for the reorganized entry point."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aquaculture_system"))

import config
import main as application
from utils.logger import tee_console


class LifecycleTests(unittest.TestCase):
    def test_scheduler_does_not_repeat_a_matching_minute(self):
        def local_time(format_string):
            return "2026-10-23 09:00" if "%Y" in format_string else "09:00"

        with (
            patch.object(config, "FEED_TIMES", ["09:00"]),
            patch.object(application.time, "strftime", side_effect=local_time),
            patch.object(
                application.time, "sleep", side_effect=[None, None, KeyboardInterrupt]
            ),
            patch.object(application, "run_dynamic_feeding_engine") as session,
        ):
            with self.assertRaises(KeyboardInterrupt):
                application.run_schedule(Mock(), Mock(), Mock(), Mock())
        session.assert_called_once()

    def test_shutdown_attempts_all_workers_in_order_after_relay_failure(self):
        camera, relay, ai_worker = Mock(), Mock(), Mock()
        order = []

        def stop_relay():
            order.append("relay")
            raise RuntimeError("OFF failed")

        relay.stop.side_effect = stop_relay
        camera.stop_capture_thread.side_effect = lambda: order.append("camera")
        ai_worker.stop.side_effect = lambda: order.append("ai")
        with self.assertRaisesRegex(RuntimeError, "OFF failed"):
            application.stop_workers(camera, relay, ai_worker)
        self.assertEqual(order, ["relay", "camera", "ai"])

    def test_worker_startup_failure_still_cleans_up(self):
        camera, relay, ai_worker = Mock(), Mock(), Mock()
        ai_worker.start.side_effect = RuntimeError("model failed")
        with (
            patch.object(application, "HikvisionCamera", return_value=camera),
            patch.object(application, "ExternalRelayWorker", return_value=relay),
            patch.object(application, "AIAnalysisWorker", return_value=ai_worker),
            patch.object(application, "YOLO"),
            patch.object(application, "run_schedule") as schedule,
        ):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                application.main()
        schedule.assert_not_called()
        relay.stop.assert_called_once()
        camera.stop_capture_thread.assert_called_once()
        ai_worker.stop.assert_called_once()

    def test_console_capture_restores_stdout_after_exception(self):
        original_stdout = sys.stdout
        with tempfile.TemporaryDirectory() as folder:
            log_path = Path(folder) / "console.txt"
            with self.assertRaisesRegex(RuntimeError, "session failed"):
                with tee_console(log_path) as logger:
                    print("session output")
                    raise RuntimeError("session failed")
            self.assertIs(sys.stdout, original_stdout)
            self.assertTrue(logger.log_file.closed)
            self.assertEqual(log_path.read_text(encoding="utf-8"), "session output\n")


if __name__ == "__main__":
    unittest.main()
