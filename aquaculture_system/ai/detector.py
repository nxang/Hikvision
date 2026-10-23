"""Queued pellet detection and annotated image output."""

import os
import queue
import threading
from concurrent.futures import Future

import cv2
from ultralytics import YOLO

import config


class AIAnalysisWorker:
    """Failed inference never means zero pellets; results belong to each image."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.queue = queue.Queue()
        self.thread = None
        self.is_running = False
        self.model = None

    def start(self):
        print(f"[AI] Loading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.is_running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=config.ANALYSIS_TIMEOUT_SECONDS)

    def queue_analysis_job(self, raw_image_path, output_folder, zone_prefix):
        future = Future()
        if not self.is_running:
            future.set_exception(RuntimeError("AI worker is not running"))
            return future
        self.queue.put((raw_image_path, output_folder, zone_prefix, future))
        return future

    def _worker_loop(self):
        while self.is_running or not self.queue.empty():
            try:
                img_path, output_folder, prefix, future = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if not self.is_running:
                    raise RuntimeError("AI worker stopped")
                count = self._analyze_image(img_path, output_folder, prefix)
                future.set_result(count)
            except Exception as exc:
                future.set_exception(exc)
            finally:
                self.queue.task_done()

    def _analyze_image(self, img_path, output_folder, prefix):
        """Validate an image, count pellets, and save the annotated result."""
        if cv2.imread(img_path) is None:
            raise ValueError(f"Missing or invalid image: {img_path}")
        results = self.model(img_path, conf=config.YOLO_CONF_THRESHOLD, verbose=False)
        if not results or results[0].boxes is None:
            raise ValueError("Detector returned no valid detection result")
        count = len(results[0].boxes)
        annotated_dir = os.path.join(output_folder, "Annotated")
        os.makedirs(annotated_dir, exist_ok=True)
        filename = os.path.basename(img_path).replace("_LEFTOVER.jpg", "_ANNOTATED.jpg")
        if not cv2.imwrite(os.path.join(annotated_dir, filename), results[0].plot()):
            raise OSError("Could not save annotated image")
        print(f"[AI] {prefix}: {count} pellets")
        return count
