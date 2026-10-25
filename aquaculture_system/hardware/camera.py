"""Hikvision preset movement and queued snapshot capture."""

import os
import queue
import re
import threading
import warnings
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError

import requests
from requests.auth import HTTPDigestAuth
from urllib3.exceptions import InsecureRequestWarning

import config


class CameraUnavailableError(RuntimeError):
    """The camera did not recover within the configured retry window."""


def wait_for_snapshot(job):
    """Treat an exhausted capture wait as a recoverable camera failure."""
    try:
        return job.result(timeout=config.CAPTURE_TIMEOUT_SECONDS)
    except FutureTimeoutError as exc:
        raise CameraUnavailableError(
            "Camera capture exceeded its retry window"
        ) from exc


class HikvisionCamera:
    """Every capture reports success or failure explicitly."""

    def __init__(self, ip, user, password):
        if not password:
            raise ValueError("Set HIKVISION_CAMERA_PASSWORD before starting the camera")
        self.auth = HTTPDigestAuth(user, password)
        self.verify = config.CAMERA_TLS_VERIFY
        if self.verify is False:
            # Silence only the expected warning for this camera host.
            warnings.filterwarnings(
                "ignore",
                message=rf"Unverified HTTPS request is being made to host '{re.escape(ip)}'\.",
                category=InsecureRequestWarning,
                module=r"urllib3\.connectionpool",
            )
        self.base_url = f"https://{ip}/ISAPI"
        self.headers = {"X-Requested-With": "XMLHttpRequest"}
        self.capture_queue = queue.Queue()
        self.capture_thread = None
        self.is_running = False
        self._stop_event = threading.Event()

    def start_capture_thread(self, ai_worker):
        self._stop_event.clear()
        self.is_running = True
        self.capture_thread = threading.Thread(
            target=self._capture_worker_loop, args=(ai_worker,), daemon=True
        )
        self.capture_thread.start()

    def stop_capture_thread(self):
        self.is_running = False
        self._stop_event.set()
        if self.capture_thread:
            self.capture_thread.join(
                timeout=2 * config.CAMERA_SNAPSHOT_TIMEOUT_SECONDS + 2
            )

    def queue_snapshot_job(self, folder, prefix, custom_timestamp, analyze=True):
        future = Future()
        if not self.is_running:
            future.set_exception(RuntimeError("Camera worker is not running"))
            return future
        self.capture_queue.put((folder, prefix, custom_timestamp, analyze, future))
        return future

    def _capture_worker_loop(self, ai_worker):
        while self.is_running or not self.capture_queue.empty():
            try:
                folder, prefix, timestamp, analyze, future = self.capture_queue.get(
                    timeout=0.5
                )
            except queue.Empty:
                continue
            try:
                if not self.is_running:
                    raise RuntimeError("Camera worker stopped")
                filename = self._capture_snapshot(folder, prefix, timestamp)
                analysis = (
                    ai_worker.queue_analysis_job(filename, folder, prefix)
                    if analyze
                    else None
                )
                future.set_result((filename, analysis))
            except Exception as exc:
                future.set_exception(exc)
            finally:
                self.capture_queue.task_done()

    def go_to_preset(self, preset_id):
        try:
            self._request_with_retries(
                requests.put,
                f"{self.base_url}/PTZCtrl/channels/1/presets/{preset_id}/goto",
                operation=f"Move to preset {preset_id}",
                auth=self.auth,
                headers=self.headers,
                verify=self.verify,
                timeout=config.CAMERA_PRESET_TIMEOUT_SECONDS,
            )
            return True
        except CameraUnavailableError:
            return False

    def _request_with_retries(self, request, url, operation, **kwargs):
        """Retry only camera HTTP failures, with an interruptible delay."""
        for attempt in range(1, config.CAMERA_RETRY_ATTEMPTS + 1):
            if self._stop_event.is_set():
                raise CameraUnavailableError("Camera worker is stopping")
            try:
                response = request(url, **kwargs)
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                print(
                    f"[Camera] {operation}: attempt {attempt}/{config.CAMERA_RETRY_ATTEMPTS} failed: {exc}"
                )
                if attempt == config.CAMERA_RETRY_ATTEMPTS:
                    raise CameraUnavailableError(
                        f"{operation} failed after {attempt} attempts"
                    ) from exc
                if self._stop_event.wait(config.CAMERA_RETRY_DELAY_SECONDS):
                    raise CameraUnavailableError(
                        "Camera retry interrupted by shutdown"
                    ) from exc

    def _capture_snapshot(self, folder, prefix, timestamp):
        """Download and save one frame before the next camera movement."""
        response = self._request_with_retries(
            requests.get,
            f"{self.base_url}/Streaming/channels/101/picture",
            operation="Capture snapshot",
            auth=self.auth,
            verify=self.verify,
            timeout=config.CAMERA_SNAPSHOT_TIMEOUT_SECONDS,
        )
        filename = os.path.join(folder, f"{prefix}_{timestamp}_LEFTOVER.jpg")
        os.makedirs(folder, exist_ok=True)
        with open(filename, "wb") as image_file:
            image_file.write(response.content)
        return filename
