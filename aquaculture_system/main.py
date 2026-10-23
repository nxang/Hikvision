"""Application startup, schedule polling, and worker shutdown.

Run from the project root:
    .\\.venv\\Scripts\\python.exe -X utf8 aquaculture_system\\main.py
"""

import time

from ultralytics import YOLO

import config
from ai.detector import AIAnalysisWorker
from feeding import run_dynamic_feeding_engine, validate_feeding_config
from hardware.camera import HikvisionCamera
from hardware.relay import ExternalRelayWorker


def run_schedule(camera, relay, ai_worker, biomass_model):
    """Start one session per matching schedule minute while the process runs."""
    last_trigger = None
    print(f"[Scheduler] Feeding times: {config.FEED_TIMES}")
    while True:
        trigger = time.strftime("%Y-%m-%d %H:%M")
        if time.strftime("%H:%M") in config.FEED_TIMES and trigger != last_trigger:
            last_trigger = trigger
            run_dynamic_feeding_engine(camera, relay, ai_worker, biomass_model)
            time.sleep(60)
        time.sleep(10)


def stop_workers(camera, relay, ai_worker):
    """Stop dispensing first and attempt all worker cleanup even after a failure."""
    try:
        relay.stop()
    finally:
        try:
            camera.stop_capture_thread()
        finally:
            ai_worker.stop()


def main():
    """Load models once, start workers, and run the feeding schedule."""
    validate_feeding_config()
    ai_worker = AIAnalysisWorker(model_path=config.PROJECT_ROOT / "best.pt")
    relay = ExternalRelayWorker(config.RELAY_API_URL)
    camera = HikvisionCamera(config.CAMERA_IP, config.CAMERA_USER, config.CAMERA_PASS)
    biomass_path = config.PROJECT_ROOT / "fish.pt"
    print(f"[Startup] Loading biomass model: {biomass_path}")
    biomass_model = YOLO(biomass_path)

    try:
        ai_worker.start()
        relay.start()
        camera.start_capture_thread(ai_worker=ai_worker)
        run_schedule(camera, relay, ai_worker, biomass_model)
    except KeyboardInterrupt:
        print("[Shutdown] Stopping workers")
    finally:
        stop_workers(camera, relay, ai_worker)


if __name__ == "__main__":
    main()
