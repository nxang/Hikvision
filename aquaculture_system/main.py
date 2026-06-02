# main.py
import os
import time
import threading
import config

from hardware.camera import HikvisionCamera
from hardware.relay import ExternalRelayWorker
from ai.detector import AIAnalysisWorker
from utils.logger import log_session_summary, calculate_historical_adjustment


def dispense_feed(camera, relay, feeder_preset, cups, ai_worker=None, target_folder=None, current_distribution=0):
    """Aligns camera tracking frame to the feeder point, discharges portion payload,
    and takes an immediate post-drop verification photo.
    """
    print(f"🔄 [Feeder Align] Target Preset {feeder_preset}...")
    camera.go_to_preset(feeder_preset)
    time.sleep(4.0)  # Settle mechanical lens vibration
    
    feeder_done = threading.Event()
    relay.trigger_feeding(num_cups=cups, callback_event=feeder_done)
    feeder_done.wait()  # Block until hardware motor completes execution run

    # Immediate Post-Drop Audit Capture at Preset 9
    if ai_worker and target_folder:
        print(f"📸 [Feeder Shield] Portion drop complete. Capturing drop verification photo...")
        verify_timestamp = f"{time.strftime('%Y%m%d_%H%M%S')}_PORTION_{current_distribution}_POST_DROP"
        
        camera.queue_snapshot_job(
            folder=target_folder, 
            prefix=f"Feeder_Zone_Preset_{feeder_preset}", 
            custom_timestamp=verify_timestamp
        )
        camera.capture_queue.join()  # Wait for file write before unlocking PTZ frame


def scan_and_evaluate_perimeter(camera, ai_worker, patrol_presets, target_folder, step_label):
    """Executes a perimeter dragnet pass and extracts the peak single-zone density count."""
    ai_worker.reset_cycle_counter()
    cycle_timestamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{step_label}"

    print(f"🔎 [Scanning Matrix: {step_label}] Panning across targets: {patrol_presets}")
    for preset_id in patrol_presets:
        prefix_string = f"Zone_Preset_{preset_id}"
        if camera.go_to_preset(preset_id):
            time.sleep(4.0)  # Mechanical settle window
            camera.queue_snapshot_job(folder=target_folder, prefix=prefix_string, custom_timestamp=cycle_timestamp)
            
    camera.capture_queue.join()
    ai_worker.queue.join()
    
    return ai_worker.cycle_max_zone_leftovers


def run_dynamic_feeding_engine(camera, relay, ai_worker):
    """Executes top-down matrix splitting using explicit, absolute wait step intervals."""
    target_folder = os.path.join(config.MEDIA_OUTPUT_DIR, f"Leftover_Inspection_{time.strftime('%Y-%m-%d')}")
    os.makedirs(target_folder, exist_ok=True)

    print(f"\n🔔 [Automation Core] Session initialized at {time.strftime('%H:%M:%S')}")
    total_cups_fed_session = 0.0

    # 1. FETCH BI-DIRECTIONAL HISTORICAL MULTIPLIER (Growth or Protection Scale)
    history_multiplier = calculate_historical_adjustment(config.CSV_LOG_FILE, lookback_sessions=3)
    
    # 2. CALCULATE FRACTIONAL ALLOCATIONS FROM TOTAL TARGET
    adjusted_total_cups = config.TOTAL_SESSION_CUPS * history_multiplier
    cups_per_distribution = adjusted_total_cups / config.NUM_DISTRIBUTIONS

    print(f"📊 Baseline Target: {config.TOTAL_SESSION_CUPS} Cups over {config.NUM_DISTRIBUTIONS} distributions.")
    print(f"📈 History Adjustment Factor: {history_multiplier}x multiplier applied.")
    print(f"⚖️ Session Budget: Total {adjusted_total_cups:.2f} Cups ({cups_per_distribution:.2f} Cups per drop).")

    # 3. MANDATORY PRE-FLIGHT SANITATION CHECK
    peak_precheck = scan_and_evaluate_perimeter(camera, ai_worker, config.PRECHECK_PRESETS, target_folder, "PRE_CHECK")
    print(f"📊 [Sanitation Check] Highest single-zone pellet count found: {peak_precheck}")
    
    if peak_precheck > config.MAX_SINGLE_ZONE_LEFTOVERS:
        print("⚠️ [Sanitation Check] Stale food discovered before feeding! Pausing for 5-minute absorption cooldown...")
        time.sleep(5 * 60)
        
        peak_precheck = scan_and_evaluate_perimeter(camera, ai_worker, config.PRECHECK_PRESETS, target_folder, "PRE_CHECK_EXTENDED")
        if peak_precheck > config.MAX_SINGLE_ZONE_LEFTOVERS:
            print("🛑 [Sanitation Check] Tank is chronically dirty. Aborting entire session to protect water chemistry!")
            log_session_summary(config.CSV_LOG_FILE, total_cups_fed_session, "ABORTED_DIRTY_TANK_PRECHECK")
            return

    # 4. EXECUTE EQUAL PORTION DISTRIBUTION LOOP
    for drop_num in range(1, config.NUM_DISTRIBUTIONS + 1):
        print(f"\n🎬 [Portion {drop_num}/{config.NUM_DISTRIBUTIONS}] Discharging feed allotment fraction...")
        
        # Fire feeder actuator mechanism
        dispense_feed(
            camera=camera, 
            relay=relay, 
            feeder_preset=config.FEEDER_STATION_PRESET, 
            cups=cups_per_distribution,
            ai_worker=ai_worker,
            target_folder=target_folder,
            current_distribution=drop_num
        )
        total_cups_fed_session += cups_per_distribution
        
        # Immediate break pattern: Skip sleeping/scanning after the absolute final portion dose is dropped
        if drop_num == config.NUM_DISTRIBUTIONS:
            break

        # 5. EXPLICIT FIXED INTERVAL TIMING LOOPS
        # Converts your static configuration parameters into absolute countdown segments
        interval_sleep_secs = config.DELAY_PER_INTERVAL_MINUTES * 60
        
        for check_idx in range(1, config.INSPECTION_COUNT_PER_DELAY + 1):
            print(f"⏳ Sleeping explicit window segment {check_idx}/{config.INSPECTION_COUNT_PER_DELAY} ({config.DELAY_PER_INTERVAL_MINUTES} mins)...")
            time.sleep(interval_sleep_secs)
            
            # Wake up and instantly loop boundary checkpoints (Skips chaotic eye-of-storm Preset 9)
            run_label = f"PORTION_{drop_num}_CHECK_{check_idx}"
            peak_peripheral_count = scan_and_evaluate_perimeter(
                camera=camera, 
                ai_worker=ai_worker, 
                patrol_presets=config.ACTIVE_MEAL_PRESETS, 
                target_folder=target_folder, 
                step_label=run_label
            )
            
            print(f"📊 [Telemetry Update] Maximum peripheral quadrant pellet density count: {peak_peripheral_count}")
            
            # Active Overfeed Shield Check
            if peak_peripheral_count > config.MAX_SINGLE_ZONE_LEFTOVERS:
                print(f"🛑 [Overfeed Shield] Boundary leakage detected ({peak_peripheral_count} > {config.MAX_SINGLE_ZONE_LEFTOVERS})!")
                print("🛑 Satiety achieved early. Terminating remaining distributions to preserve biofilter health.")
                log_session_summary(config.CSV_LOG_FILE, total_cups_fed_session, f"ABORTED_AT_PORTION_{drop_num}_CHECK_{check_idx}")
                return

    # Full Run Success Logging
    print(f"\n🏁 [Automation Core] Session complete. Total volume delivered: {total_cups_fed_session:.2f} cups.")
    log_session_summary(config.CSV_LOG_FILE, total_cups_fed_session, "SUCCESS_FULL_RUN")


if __name__ == "__main__":
    yolo_worker = AIAnalysisWorker(model_path=config.YOLO_WEIGHTS_PATH)
    relay_worker = ExternalRelayWorker(config.RELAY_API_URL)
    camera_system = HikvisionCamera(config.CAMERA_IP, config.CAMERA_USER, config.CAMERA_PASS)

    relay_worker.start()
    yolo_worker.start()
    camera_system.start_capture_thread(ai_worker=yolo_worker)
    
    print(f"🔒 Smart Loop Armed. Dynamic top-down distributions targeted for execution hour: {config.FEED_TIMES}")
    
    try:
        while True:
            current_time_str = time.strftime("%H:%M")
            if current_time_str in config.FEED_TIMES:
                run_dynamic_feeding_engine(
                    camera=camera_system,
                    relay=relay_worker,
                    ai_worker=yolo_worker
                )
                time.sleep(60)  # Clear the current matched runtime minute path
            time.sleep(10)

    except KeyboardInterrupt:
        print("\nShutting down automated aquaculture execution blocks cleanly...")
    finally:
        camera_system.stop_capture_thread()
        yolo_worker.stop()
        relay_worker.stop()