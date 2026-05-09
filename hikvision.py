import cv2
import requests
from requests.auth import HTTPDigestAuth
import xml.etree.ElementTree as ET
import time
import urllib3
import os

# Silence SSL warnings for the camera's self-signed certificate
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class HikvisionMaster:
    def __init__(self, ip, user, password):
        self.ip = ip
        self.user = user
        self.password = password
        self.auth = HTTPDigestAuth(user, password)
        self.base_url = f"https://{ip}/ISAPI"
        
        # Headers captured from your browser logs to ensure access
        self.headers = {
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest"
        }
        
        # RTSP Stream (Main stream 101 for high-quality recording)
        self.rtsp_main = f"rtsp://{user}:{password}@{ip}:554/Streaming/Channels/101"

    def _get_timestamp(self):
        """Generates a sortable timestamp for filenames."""
        return time.strftime("%Y%m%d_%H%M%S")

    # --- 1. MOVEMENT & LENS CONTROL ---
    def move(self, pan, tilt, zoom=0):
        """Continuous movement (-100 to 100). Send (0,0,0) to stop."""
        url = f"{self.base_url}/PTZCtrl/channels/1/continuous"
        payload = f"<?xml version='1.0' encoding='UTF-8'?><PTZData><pan>{pan}</pan><tilt>{tilt}</tilt><zoom>{zoom}</zoom></PTZData>"
        try:
            r = requests.put(url, auth=self.auth, data=payload, headers=self.headers, verify=False, timeout=5)
            return r.status_code == 200
        except: return False

    def stop(self):
        """Emergency stop for all motors."""
        return self.move(0, 0, 0)

    def set_focus(self, focus_val):
        """Manual focus control (Positive = Far, Negative = Near)."""
        url = f"{self.base_url}/Image/channels/1/focus"
        payload = f"<?xml version='1.0' encoding='UTF-8'?><FocusData><focus>{focus_val}</focus></FocusData>"
        try:
            requests.put(url, auth=self.auth, data=payload, headers=self.headers, verify=False)
        except: pass

    # --- 2. PRESET MANAGEMENT ---
    def go_to_preset(self, preset_id):
        """Move camera to a previously saved preset."""
        url = f"{self.base_url}/PTZCtrl/channels/1/presets/{preset_id}/goto"
        try:
            r = requests.put(url, auth=self.auth, headers=self.headers, verify=False, timeout=5)
            return r.status_code == 200
        except: return False

    def save_current_as_preset(self, preset_id):
        """Saves current physical position to a Preset ID (overwrites if exists)."""
        url = f"{self.base_url}/PTZCtrl/channels/1/presets/{preset_id}/setup"
        try:
            r = requests.put(url, auth=self.auth, headers=self.headers, verify=False, timeout=5)
            if r.status_code == 200:
                print(f"✔ Position locked to Preset {preset_id}")
                return True
        except: pass
        print(f"✘ Failed to save Preset {preset_id}")
        return False

    # --- 3. STATUS & FEEDBACK ---
    def get_ptz_info(self):
        """Retrieves exact Pan/Tilt/Zoom coordinates."""
        url = f"{self.base_url}/PTZCtrl/channels/1/status"
        try:
            r = requests.get(url, auth=self.auth, headers=self.headers, verify=False, timeout=5)
            if r.status_code == 200:
                root = ET.fromstring(r.text)
                ns = {'h': 'http://www.hikvision.com/ver20/XMLSchema'}
                return {
                    "pan": int(root.find('.//h:azimuth', ns).text) / 10,
                    "tilt": int(root.find('.//h:elevation', ns).text) / 10,
                    "zoom": int(root.find('.//h:absoluteZoom', ns).text)
                }
        except: return None

    # --- 4. DATA CAPTURE (IMAGE & VIDEO) ---
    def take_snapshot(self, position_name="Manual"):
        """Saves a high-res JPG named by position and time."""
        filename = f"{position_name}_{self._get_timestamp()}.jpg"
        url = f"{self.base_url}/Streaming/channels/101/picture"
        try:
            r = requests.get(url, auth=self.auth, verify=False, timeout=10)
            if r.status_code == 200:
                with open(filename, 'wb') as f: f.write(r.content)
                print(f"📸 Saved Photo: {filename}")
                return filename
        except: print("Snapshot Failed")

    def record_video(self, position_name="Manual", duration=10):
        """Records MP4 video named by position and time."""
        filename = f"{position_name}_{self._get_timestamp()}.mp4"
        cap = cv2.VideoCapture(self.rtsp_main)
        if not cap.isOpened(): return
        
        w, h = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(5) or 25
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        print(f"🎥 Recording {position_name} for {duration}s...")
        start = time.time()
        while (time.time() - start) < duration:
            ret, frame = cap.read()
            if ret: out.write(frame)
            else: break
        
        cap.release()
        out.release()
        print(f"✔ Video Saved: {filename}")

# --- EXAMPLE THESIS WORKFLOW ---
if __name__ == "__main__":
    # Initialize
    cam = HikvisionMaster("192.168.1.64", "admin", "Hikvision")

    # A. Setup/Calibration Phase (Example: Saving a new spot)
    # cam.move(40, 0) # Move right manually
    # time.sleep(2)
    # cam.stop()
    # cam.save_current_as_preset(5) # Save this as 'Pellet Drop Zone'

    # B. Automated Data Collection Phase
    test_positions = {1: "Tank_Left", 2: "Tank_Right"}

    for pid, name in test_positions.items():
        if cam.go_to_preset(pid):
            time.sleep(4) # Allow camera to stabilize
            
            # Log coordinates for the thesis paper
            coords = cam.get_ptz_info()
            print(f"Logged {name} at: {coords}")

            # Capture evidence
            cam.take_snapshot(position_name=name)
            cam.record_video(position_name=name, duration=5)