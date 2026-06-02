# hardware/camera.py
import os
import queue
import threading
import requests
import urllib3
from requests.auth import HTTPDigestAuth

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class HikvisionCamera:
    """Manages hardware capture pipelines and relays completed frames to the AI thread."""
    def __init__(self, ip, user, password):
        self.ip = ip
        self.user = user
        self.password = password
        self.auth = HTTPDigestAuth(user, password)
        self.base_url = f"https://{ip}/ISAPI"
        self.headers = {"X-Requested-With": "XMLHttpRequest"}
        
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
            self.capture_thread.join(timeout=1.0)

    def queue_snapshot_job(self, folder, prefix, custom_timestamp):
        self.capture_queue.put({
            "folder": folder,
            "prefix": prefix,
            "timestamp": custom_timestamp
        })

    def _capture_worker_loop(self, ai_worker):
        while self.is_running:
            try:
                job = self.capture_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            pic_url = f"{self.base_url}/Streaming/channels/101/picture"
            filename = os.path.join(job["folder"], f"{job['prefix']}_{job['timestamp']}_LEFTOVER.jpg")
            
            try:
                r = requests.get(pic_url, auth=self.auth, verify=False, timeout=10)
                if r.status_code == 200:
                    with open(filename, 'wb') as f:
                        f.write(r.content)
                    ai_worker.queue_analysis_job(raw_image_path=filename, output_folder=job["folder"], zone_prefix=job["prefix"])
                else:
                    print(f"    ❌ [Capture Thread] Snapshot Error: {r.status_code}")
            except requests.RequestException:
                print("    ❌ [Capture Thread] Camera network timeout")
            
            self.capture_queue.task_done()

    def go_to_preset(self, preset_id):
        url = f"{self.base_url}/PTZCtrl/channels/1/presets/{preset_id}/goto"
        try:
            return requests.put(url, auth=self.auth, headers=self.headers, verify=False, timeout=5).status_code == 200
        except requests.RequestException:
            return False