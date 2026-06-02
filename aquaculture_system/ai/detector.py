# ai/detector.py
import os
import cv2
import queue
import threading
from ultralytics import YOLO
import config

class AIAnalysisWorker:
    """Dedicated background thread running YOLO inference tracking single-zone peak maximums."""
    def __init__(self, model_path):
        self.model_path = model_path
        self.queue = queue.Queue()
        self.thread = None
        self.is_running = False
        self.model = None
        
        # Peak-Hold Metric to conquer double-counting drifting objects
        self.cycle_max_zone_leftovers = 0
        self.lock = threading.Lock()
    
    def start(self):
        print(f"🧠 [AI Thread] Deploying YOLO Engine from: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.is_running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def reset_cycle_counter(self):
        with self.lock:
            self.cycle_max_zone_leftovers = 0

    def queue_analysis_job(self, raw_image_path, output_folder, zone_prefix):
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
                results = self.model(img_path, conf=config.YOLO_CONF_THRESHOLD, verbose=False)
                detected_pellets = len(results[0].boxes)
                
                # Dynamic Peak Hold Evaluation
                with self.lock:
                    if detected_pellets > self.cycle_max_zone_leftovers:
                        self.cycle_max_zone_leftovers = detected_pellets
                
                print(f"   🧠 [AI Engine] Zone Scan -> {os.path.basename(img_path)}: {detected_pellets} pellets.")

                # Render output validation image
                annotated_frame = results[0].plot()
                annotated_dir = os.path.join(job["output_folder"], "Annotated")
                os.makedirs(annotated_dir, exist_ok=True)
                    
                annotated_filename = os.path.basename(img_path).replace("_LEFTOVER.jpg", "_ANNOTATED.jpg")
                cv2.imwrite(os.path.join(annotated_dir, annotated_filename), annotated_frame)
                
            except Exception as e:
                print(f"❌ [AI Engine] Core processing error on image {img_path}: {e}")
                
            self.queue.task_done()