import os
import queue
import threading
from concurrent.futures import Future
import requests
from requests.auth import HTTPDigestAuth
import config


class HikvisionCamera:
    """Every capture reports success or failure explicitly."""
    def __init__(self, ip, user, password):
        if not password:
            raise ValueError("Set HIKVISION_CAMERA_PASSWORD before starting the camera")
        self.auth = HTTPDigestAuth(user, password)
        self.verify = config.CAMERA_TLS_VERIFY
        self.base_url = f'https://{ip}/ISAPI'
        self.headers = {'X-Requested-With': 'XMLHttpRequest'}
        self.capture_queue = queue.Queue()
        self.capture_thread = None
        self.is_running = False

    def start_capture_thread(self, ai_worker):
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_worker_loop, args=(ai_worker,), daemon=True)
        self.capture_thread.start()

    def stop_capture_thread(self):
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=12)

    def queue_snapshot_job(self, folder, prefix, custom_timestamp, analyze=True):
        future = Future()
        if not self.is_running:
            future.set_exception(RuntimeError('Camera worker is not running'))
            return future
        self.capture_queue.put((folder, prefix, custom_timestamp, analyze, future))
        return future

    def _capture_worker_loop(self, ai_worker):
        while self.is_running or not self.capture_queue.empty():
            try:
                folder, prefix, timestamp, analyze, future = self.capture_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if not self.is_running:
                    raise RuntimeError('Camera worker stopped')
                response = requests.get(f'{self.base_url}/Streaming/channels/101/picture', auth=self.auth, verify=self.verify, timeout=10)
                response.raise_for_status()
                filename = os.path.join(folder, f'{prefix}_{timestamp}_LEFTOVER.jpg')
                os.makedirs(folder, exist_ok=True)
                with open(filename, 'wb') as image_file:
                    image_file.write(response.content)
                analysis = ai_worker.queue_analysis_job(filename, folder, prefix) if analyze else None
                future.set_result((filename, analysis))
            except Exception as exc:
                future.set_exception(exc)
            finally:
                self.capture_queue.task_done()

    def go_to_preset(self, preset_id):
        try:
            response = requests.put(f'{self.base_url}/PTZCtrl/channels/1/presets/{preset_id}/goto', auth=self.auth, headers=self.headers, verify=self.verify, timeout=5)
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False
