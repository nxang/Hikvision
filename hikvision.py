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
    # MODIFIED: Added an optional 'suffix' parameter to differentiate rapid shots
    def take_snapshot(self, position_name="Manual", suffix=""):
        """Saves a high-res JPG named by position, time, and sequential suffix."""
        filename = f"{position_name}_{self._get_timestamp()}{suffix}.jpg"
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
        
    def force_color_mode(self):
        url_base = f"{self.base_url}/Image/channels/1"
        payload_base = """<?xml version="1.0" encoding="UTF-8"?>
        <ImageChannel>
            <DayNightSwitch>
                <dayNightFilterType>day</dayNightFilterType>
            </DayNightSwitch>
        </ImageChannel>"""

        url_icr = f"{self.base_url}/Image/channels/1/icr"
        payload_icr = """<?xml version="1.0" encoding="UTF-8"?>
        <IcrIrcType>
            <mode>day</mode>
        </IcrIrcType>"""

        try:
            print("Trying primary image endpoint...")
            r = requests.put(url_base, auth=self.auth, data=payload_base, headers=self.headers, verify=False, timeout=5)
            if r.status_code == 200:
                print("☀ Camera successfully locked to Day/Color Mode via primary endpoint!")
                return True
        except: pass

        try:
            print("Primary endpoint unavailable (404). Trying alternative ICR endpoint...")
            r = requests.put(url_icr, auth=self.auth, data=payload_icr, headers=self.headers, verify=False, timeout=5)
            if r.status_code == 200:
                print("☀ Camera successfully locked to Day/Color Mode via ICR endpoint!")
                return True
            else:
                print(f"❌ Both configurations rejected. Latest Status Code: {r.status_code}")
        except Exception as e:
            print(f"Network error during execution: {e}")
        return False
        
    def show_live_feed(self, window_name="Hikvision Live"):
        import queue
        import threading
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;10240000|max_delay;500000"

        cap = cv2.VideoCapture(self.rtsp_main, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print("Error: Could not open RTSP stream.")
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

        print(f"\n--- Control Mode Active: {window_name} ---")
        speed = 20
        is_moving = False
        frame_queue = queue.Queue(maxsize=2)
        stop_event = threading.Event()

        video_writer = None
        recording_end_time = 0
        is_recording = False

        def _stream_reader():
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                if frame_queue.full():
                    try: frame_queue.get_nowait()
                    except queue.Empty: pass
                frame_queue.put(frame)

        reader_thread = threading.Thread(target=_stream_reader, daemon=True)
        reader_thread.start()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                if frame_queue.empty():
                    time.sleep(0.005)
                    continue
                frame = frame_queue.get()
                if is_recording:
                    if time.time() < recording_end_time:
                        video_writer.write(frame)
                    else:
                        is_recording = False
                        video_writer.release()
                        print("✔ Video Recording Complete!")

                display_frame = frame.copy()
                if is_recording:
                    cv2.putText(display_frame, "● RECORDING", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(10) & 0xFF

                if key == ord('w'): self.move(0, speed, 0); is_moving = True
                elif key == ord('s'): self.move(0, -speed, 0); is_moving = True
                elif key == ord('a'): self.move(-speed, 0, 0); is_moving = True
                elif key == ord('d'): self.move(speed, 0, 0); is_moving = True
                elif key == ord(' '): self.stop(); is_moving = False
                elif key == ord('c'):
                    self.stop(); is_moving = False
                    self.take_snapshot(position_name="Tank_Capture")
                elif key == ord('v'):
                    if not is_recording:
                        self.stop(); is_moving = False
                        filename = f"Tank_Direct_Stream_{self._get_timestamp()}.mp4"
                        video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        recording_end_time = time.time() + 5.0
                        is_recording = True
                elif key == ord('q'):
                    self.stop()
                    break
                else:
                    if is_moving and key == 255:
                        self.stop()
                        is_moving = False
        finally:
            stop_event.set()
            reader_thread.join(timeout=1.0)
            if video_writer is not None and is_recording: video_writer.release()
            cap.release()
            cv2.destroyAllWindows()
            del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]


# --- BATCH EXECUTION RUNNER WORKFLOW ---
if __name__ == "__main__":
    cam = HikvisionMaster("192.168.1.64", "admin", "Hikvision")
    
    presets_to_capture = [2,5, 6, 7, 8, 9, 10, 11, 12] 
    stabilization_delay = 4.0
    
    # NEW CONFIGURATION FOR TEMPORAL MULTI-SHOT FILTERING
    num_shots_per_position = 1    # Total rapid pictures to take at each spot
    shot_interval_delay = 0.25    # Wait 250 milliseconds between shots to let water ripples move
    
    current_date = time.strftime("%Y-%m-%d")
    output_folder = f"Captures_{current_date}"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 Created folder: {output_folder}")
    
    for preset_id in presets_to_capture: 
        position_name = os.path.join(output_folder, f"Preset_{preset_id}") 
        print(f"\n🔄 Command: Moving to Preset {preset_id}...") 
        
        if cam.go_to_preset(preset_id): 
            print(f"⏳ Arrived. Waiting {stabilization_delay}s for physical lens stabilization...") 
            time.sleep(stabilization_delay) 
            
            coords = cam.get_ptz_info() 
            if coords:
                print(f"📊 Telemetry logged -> Pan: {coords['pan']}°, Tilt: {coords['tilt']}°, Zoom: {coords['zoom']}") 
            
            # --- MODIFIED: RAPID CONSECUTIVE CAPTURE SUB-LOOP ---
            print(f"📸 Starting rapid sequence capture ({num_shots_per_position} shots)...")
            for shot_idx in range(1, num_shots_per_position + 1):
                suffix_string = f"_shot_{shot_idx}"
                
                saved_file = cam.take_snapshot(position_name=position_name, suffix=suffix_string)
                
                if saved_file:
                    print(f"   ✔ Captured shot {shot_idx}/{num_shots_per_position}: {saved_file}")
                else:
                    print(f"   ✘ Error: Failed to write shot {shot_idx} at Preset {preset_id}")
                
                # Small pause between snapshots so water waves shift position slightly
                if shot_idx < num_shots_per_position:
                    time.sleep(shot_interval_delay)
                    
        else:
            print(f"✘ Error: Camera rejected command or failed to reach Preset {preset_id}")

    print("\n-----------------------------------------------------------------")
    print(f"🏁 Custom batch capture complete! Check the '{output_folder}' directory for images.")