import os
import cv2
import time
import queue
import threading
import urllib3
import requests
import csv
from requests.auth import HTTPDigestAuth
import xml.etree.ElementTree as ET

# Try importing YOLO from ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Please install the Ultralytics package to use YOLO: pip install ultralytics")

# Silence SSL warnings for the camera's self-signed certificate
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class ExternalRelayWorker:
    """Manages a dedicated thread for the Feeder Relay hardware using Cup portions."""
    def __init__(self, url="http://100.80.73.62:1880/api/do1"):
        self.base_url = url
        self.queue = queue.Queue()
        self.thread = None
        self.is_running = False
        self.SECONDS_PER_CUP = 11.0 

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def trigger_feeding(self, num_cups, callback_event=None):
        self.queue.put({"cups": num_cups, "event": callback_event})

    def _set_hardware_state(self, state):
        try:
            r = requests.get(f"{self.base_url}/{state}", timeout=5)
            return r.status_code == 200
        except requests.RequestException as e:
            print(f"💥 [Relay Thread] Tailscale network drop: {e}")
            return False

    def _worker_loop(self):
        while self.is_running:
            try:
                job = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            cups = job["cups"]
            runtime = cups * self.SECONDS_PER_CUP
            
            if runtime > 0:
                print(f"🐟 [Relay Thread] FEEDER ACTIVATED: Dispensing {cups} cup(s) ({runtime}s)...")
                self._set_hardware_state("on")
                time.sleep(runtime)
                self._set_hardware_state("off")
                print(f"✔ [Relay Thread] FEEDER DEACTIVATED: Portion complete.")
            
            if job["event"]:
                job["event"].set()
                
            self.queue.task_done()


class AIAnalysisWorker:
    """NEW: Dedicates a standalone thread to run YOLO inference and generate annotated audit images."""
    def __init__(self, model_path=r"runs\detect\aquaculture_fish\fish\weights\best.pt"):
        self.model_path = model_path
        self.queue = queue.Queue()
        self.thread = None
        self.is_running = False
        
        # Thread-safe variables to track pellet tallies across asynchronous runs
        self.cycle_total_leftovers = 0
        self.lock = threading.Lock()

    def start(self):
        # Load weights directly inside the worker thread initialization
        print(f"🧠 [AI Thread] Initializing YOLO Engine from: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.is_running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def reset_cycle_counter(self):
        """Clears out the cumulative tally at the start of a fresh patrol run."""
        with self.lock:
            self.cycle_total_leftovers = 0

    def queue_analysis_job(self, raw_image_path, output_folder, zone_prefix):
        """Accepts raw saved images from the capture thread for background processing."""
        self.queue.put({
            "image_path": raw_image_path,
            "output_folder": output_folder,
            "prefix": zone_prefix
        })

    def _worker_loop(self):
        while self.is_running:
            try:
                job = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            img_path = job["image_path"]
            if not os.path.exists(img_path):
                self.queue.task_done()
                continue

            try:
                # 1. Run Core Object Recognition Inference
                results = self.model(img_path, verbose=False)
                detected_pellets = len(results[0].boxes)
                
                # 2. Safely increment the global counter sharing state
                with self.lock:
                    self.cycle_total_leftovers += detected_pellets
                
                print(f"   🧠 [AI Thread] Analyzed {os.path.basename(img_path)} -> Found: {detected_pellets} pellets.")

                # 3. Build Visual Bounding Box Annotations
                annotated_frame = results[0].plot() # Automatically renders custom model bounding boxes
                
                # Construct path targeting the 'Annotated' subfolder
                annotated_dir = os.path.join(job["output_folder"], "Annotated")
                if not os.path.exists(annotated_dir):
                    os.makedirs(annotated_dir, exist_ok=True)
                    
                annotated_filename = os.path.basename(img_path).replace("_LEFTOVER.jpg", "_ANNOTATED.jpg")
                annotated_path = os.path.join(annotated_dir, annotated_filename)
                
                # Write file out to disk storage
                cv2.imwrite(annotated_path, annotated_frame)
                
            except Exception as e:
                print(f"❌ [AI Thread] Error analyzing image {img_path}: {e}")
                
            self.queue.task_done()


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

    def start_capture_thread(self, ai_worker: AIAnalysisWorker):
        self.is_running = True
        self.capture_thread = threading.Thread(
            target=self._capture_worker_loop, 
            args=(ai_worker,), 
            daemon=True
        )
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

    def _capture_worker_loop(self, ai_worker: AIAnalysisWorker):
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
                    print(f"    💾 [Capture Thread] Raw snapshot saved: {filename}")
                    
                    # LINKAGE: Pass the raw saved file directly into the AI worker thread queue!
                    ai_worker.queue_analysis_job(
                        raw_image_path=filename, 
                        output_folder=job["folder"], 
                        zone_prefix=job["prefix"]
                    )
                else:
                    print(f"    ❌ [Capture Thread] Snapshot Error: {r.status_code}")
            except requests.RequestException:
                print("    ❌ [Capture Thread] Camera network timeout")
            
            self.capture_queue.task_done()

    # =========================================================================
    # --- PHYSICAL CAMERA MOTOR OPERATIONS (KEPT LOWER DOWN) ---
    # =========================================================================
    def go_to_preset(self, preset_id):
        url = f"{self.base_url}/PTZCtrl/channels/1/presets/{preset_id}/goto"
        try: return requests.put(url, auth=self.auth, headers=self.headers, verify=False, timeout=5).status_code == 200
        except: return False

    def get_ptz_info(self):
        url = f"{self.base_url}/PTZCtrl/channels/1/status"
        try:
            r = requests.get(url, auth=self.auth, headers=self.headers, verify=False, timeout=5)
            if r.status_code == 200:
                root = ET.fromstring(r.text)
                ns = {'h': 'http://www.hikvision.com/ver20/XMLSchema'}
                return {"pan": int(root.find('.//h:azimuth', ns).text) / 10}
        except: return None


# =========================================================================
# --- ADAPTIVE LOGIC & FEEDING MANAGEMENT ---
# =========================================================================
def calculate_adjusted_feed(current_cups, total_leftovers):
    MAX_SAFE_LIMIT = 3.0  
    MIN_SAFE_LIMIT = 0.2  
    
    print(f"📊 [Analysis Engine] Global Evaluation Summary -> Cumulative Leftovers: {total_leftovers} pellets.")
    
    if total_leftovers > 40:
        new_cups = max(current_cups - 0.3, MIN_SAFE_LIMIT)
        print(f"⚠️ High Leftovers! Cutting portion from {current_cups} to {round(new_cups, 2)} cups.")
    elif total_leftovers > 10:
        new_cups = max(current_cups - 0.1, MIN_SAFE_LIMIT)
        print(f"📉 Slight overfeed. Trimming portion from {current_cups} to {round(new_cups, 2)} cups.")
    elif total_leftovers == 0:
        new_cups = min(current_cups + 0.1, MAX_SAFE_LIMIT)
        print(f"🚀 Plate completely clear! Increasing portion from {current_cups} to {round(new_cups, 2)} cups.")
    else:
        new_cups = current_cups
        print(f"💚 Optimal feeding balance maintained at {current_cups} cups.")
        
    return round(new_cups, 2)


def log_cycle_data(log_file, initial_feed, total_pellets, calculated_next_feed):
    file_exists = os.path.isfile(log_file)
    try:
        with open(log_file, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                writer.writerow(["Timestamp", "Initial_Feed_Cups", "Total_Pellets_Left", "Next_Scheduled_Feed_Cups"])
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), initial_feed, total_pellets, calculated_next_feed])
            print(f"📝 Logs appended to: {log_file}")
    except Exception as e:
        print(f"❌ Failed to commit entry to CSV log: {e}")


def run_aquaculture_cycle(camera, relay, ai_worker, patrol_presets, feeder_preset, current_feeding_cups, wait_minutes, log_filepath):
    """Executes full automated routine using synchronized standalone threads."""
    target_folder = f"Leftover_Inspection_{time.strftime('%Y-%m-%d')}"
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    print(f"\n🔔 [System] Starting Scheduled Feed & Monitoring Cycle at {time.strftime('%H:%M:%S')}")
    
    # Reset internal total variable before triggering snapshots
    ai_worker.reset_cycle_counter()
    
    # PHASE 1: ALIGN TO FEEDING ZONE & DISPENSE
    print(f"🔄 [System] Aligning camera to Feeding Station (Preset {feeder_preset})...")
    camera.go_to_preset(feeder_preset)
    time.sleep(4.0)
    
    feeder_done = threading.Event()
    relay.trigger_feeding(num_cups=current_feeding_cups, callback_event=feeder_done)
    feeder_done.wait()

    # PHASE 2: APPETITE TIMEOUT WINDOW
    print(f"⏳ [System] Feeding complete. Sleeping for {wait_minutes} minutes to allow feeding window...")
    time.sleep(wait_minutes * 60)

    # PHASE 3: LEFTOVER PATROL SCAN RUN
    print(f"🔎 [System] Timeout expired. Commencing Leftover Pellet Inspection Routine across zones...")
    cycle_timestamp = time.strftime("%Y%m%d_%H%M%S")

    for preset_id in patrol_presets:
        prefix_string = f"Zone_Preset_{preset_id}"
        print(f"  🔄 Moving to observation point: Preset {preset_id}...")
        if camera.go_to_preset(preset_id):
            time.sleep(4.0) # Settle lens mechanics
            camera.queue_snapshot_job(folder=target_folder, prefix=prefix_string, custom_timestamp=cycle_timestamp)
        else:
            print(f"  ❌ Failed to reach Inspection Zone Preset {preset_id}")
            
    # Step A: Wait for the network calls to finish writing raw files to disk
    camera.capture_queue.join()
    
    # Step B: Wait for the AI Thread to finish executing YOLO and writing annotated files
    print("⏳ [System] Camera patrol completed. Waiting for background AI thread to finish analytics...")
    ai_worker.queue.join()

    # PHASE 4: RECALCULATE DEMAND VALUES
    # Safe to read directly because .join() guarantees the AI thread is temporarily idle
    grand_total_leftovers = ai_worker.cycle_total_leftovers

    next_cycle_feed_cups = calculate_adjusted_feed(current_feeding_cups, grand_total_leftovers)
    log_cycle_data(log_filepath, current_feeding_cups, grand_total_leftovers, next_cycle_feed_cups)
    
    return next_cycle_feed_cups


# =========================================================================
# --- MAIN APPLICATION ENTRY WORKFLOW ---
# =========================================================================
if __name__ == "__main__":
    YOLO_WEIGHTS = r"runs\detect\aquaculture_pellet\pellet_run_s\weights\best.pt"
    CSV_LOG_FILE = "aquaculture_feeding_ledger.csv"
    INITIAL_CUP_PORTIONS = 1.5 

    # Initialize workers
    yolo_worker = AIAnalysisWorker(model_path=YOLO_WEIGHTS)
    relay_worker = ExternalRelayWorker("http://100.80.73.62:1880/api/do1")
    camera_system = HikvisionCamera("192.168.1.64", "admin", "Hikvision")

    # Start all 3 threads
    relay_worker.start()
    yolo_worker.start()
    camera_system.start_capture_thread(ai_worker=yolo_worker)
    
    print("🚦 3-Thread Automation Ecosystem Active.")

    FEED_TIMES = ["08:00", "16:18", "16:20"] 
    FEEDER_STATION_PRESET = 9                
    EATING_WINDOW_MINUTES = 0.1             
    INSPECTION_PRESETS = [5, 6, 7, 8, 9, 10, 11, 12]     

    print(f"🔒 Monitoring loop armed. Tracking target times: {FEED_TIMES}")
    
    try:
        while True:
            current_time_str = time.strftime("%H:%M")
            if current_time_str in FEED_TIMES:
                INITIAL_CUP_PORTIONS = run_aquaculture_cycle(
                    camera=camera_system,
                    relay=relay_worker,
                    ai_worker=yolo_worker,
                    patrol_presets=INSPECTION_PRESETS,
                    feeder_preset=FEEDER_STATION_PRESET,
                    current_feeding_cups=INITIAL_CUP_PORTIONS,
                    wait_minutes=EATING_WINDOW_MINUTES,
                    log_filepath=CSV_LOG_FILE
                )
                print(f"💾 Next feeding execution size updated to: {INITIAL_CUP_PORTIONS} cups.")
                time.sleep(60) 
            time.sleep(10)

    except KeyboardInterrupt:
        print("\nStopping ecosystem execution cleanly...")
    finally:
        camera_system.stop_capture_thread()
        yolo_worker.stop()
        relay_worker.stop()