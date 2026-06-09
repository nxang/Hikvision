# main.py
import os
import time
import threading
import config
import sys  # <--- Added for stdout redirection
from hardware.camera import HikvisionCamera
from hardware.relay import ExternalRelayWorker
from ai.detector import AIAnalysisWorker
from utils.logger import ThreadSafeTeeLogger, log_session_summary, calculate_historical_adjustment


def dispense_feed(camera, relay, feeder_preset, cups, ai_worker=None, target_folder=None, current_distribution=0):
    """Aligns camera to feeder point, discharges portion payload, and saves verification capture."""
    print(f"🔄 [Feeder Align] Target Preset {feeder_preset}...")
    camera.go_to_preset(feeder_preset)
    time.sleep(4.0)  
    
    feeder_done = threading.Event()
    relay.trigger_feeding(num_cups=cups, callback_event=feeder_done)
    feeder_done.wait()  

    if ai_worker and target_folder:
        print(f"📸 [Feeder Shield] Portion drop complete. Capturing drop verification photo...")
        verify_timestamp = f"{time.strftime('%Y%m%d_%H%M%S')}_PORTION_{current_distribution}_POST_DROP"
        camera.queue_snapshot_job(folder=target_folder, prefix=f"Feeder_Zone_Preset_{feeder_preset}", custom_timestamp=verify_timestamp)
        camera.capture_queue.join()  


def scan_and_evaluate_perimeter(camera, ai_worker, patrol_presets, target_folder, step_label):
    """Executes a perimeter dragnet pass and extracts the peak single-zone density count."""
    ai_worker.reset_cycle_counter()
    cycle_timestamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{step_label}"

    print(f"🔎 [Scanning Matrix: {step_label}] Panning targets: {patrol_presets}")
    for preset_id in patrol_presets:
        prefix_string = f"Zone_Preset_{preset_id}"
        if camera.go_to_preset(preset_id):
            time.sleep(4.0)  
            camera.queue_snapshot_job(folder=target_folder, prefix=prefix_string, custom_timestamp=cycle_timestamp)
            
    camera.capture_queue.join()
    ai_worker.queue.join()
    return ai_worker.cycle_max_zone_leftovers


def run_dynamic_feeding_engine(camera, relay, ai_worker):
    """Executes top-down matrix splitting with real-time eating time calculations and portion adjustments."""
    target_folder = os.path.join(config.MEDIA_OUTPUT_DIR, f"Leftover_Inspection_{time.strftime('%Y-%m-%d')}")
    os.makedirs(target_folder, exist_ok=True)

    # Handle Tee logging setup
    log_file_path = os.path.join(target_folder, f"daily_console_stream_{time.strftime('%Y-%m-%d')}.txt")
    tee_logger = ThreadSafeTeeLogger(log_file_path)
    original_stdout = sys.stdout
    sys.stdout = tee_logger

    try:
        print(f"\n🔔 [Automation Core] Session initialized at {time.strftime('%H:%M:%S')}")
        total_cups_fed_session = 0.0
        consecutive_failures = 0  

        # 1. READ HISTORICAL PROFILE LOG DATA
        history_multiplier = calculate_historical_adjustment(config.CSV_LOG_FILE, lookback_sessions=3)
        adjusted_total_cups = config.TOTAL_SESSION_CUPS * history_multiplier
        
        base_cups_per_distribution = adjusted_total_cups / config.NUM_DISTRIBUTIONS
        next_drop_cups = base_cups_per_distribution  

        print(f"⚖️ Session Budget Strategy: Baseline {base_cups_per_distribution:.2f} Cups per drop.")

        # 2. MANDATORY PRE-FLIGHT SANITATION CHECK
        peak_precheck = scan_and_evaluate_perimeter(camera, ai_worker, config.PRECHECK_PRESETS, target_folder, "PRE_CHECK")
        if peak_precheck > config.MAX_SINGLE_ZONE_LEFTOVERS:
            print("⚠️ [Sanitation Check] Stale food discovered before feeding! Pausing for 5-minute absorption cooldown...")
            time.sleep(5 * 60)
            peak_precheck = scan_and_evaluate_perimeter(camera, ai_worker, config.PRECHECK_PRESETS, target_folder, "PRE_CHECK_EXTENDED")
            if peak_precheck > config.MAX_SINGLE_ZONE_LEFTOVERS:
                print("🛑 Tank is dirty. Aborting entire session!")
                # Updated with rich logger parameter inputs
                log_session_summary(config.CSV_LOG_FILE, total_cups_fed_session, adjusted_total_cups, history_multiplier, "ABORTED_DIRTY_TANK_PRECHECK")
                return

        # 3. EXECUTE ADAPTIVE DISTRIBUTION LOOP
        for drop_num in range(1, config.NUM_DISTRIBUTIONS + 1):
            remaining_session_budget = adjusted_total_cups - total_cups_fed_session
            if next_drop_cups > remaining_session_budget:
                print(f"🔒 [Ceiling Safeguard] Trimming portion from {next_drop_cups:.2f} to {remaining_session_budget:.2f} to avoid breaching total session limits.")
                next_drop_cups = remaining_session_budget

            if next_drop_cups <= 0:
                print("🏁 [Ceiling Safeguard] Total session budget fully exhausted early. Closing engine loops.")
                log_session_summary(config.CSV_LOG_FILE, total_cups_fed_session, adjusted_total_cups, history_multiplier, "SUCCESS_BUDGET_EXHAUSTED_EARLY")
                return # Exit the function directly so it doesn't log standard SUCCESS_FULL_RUN at the bottom
            
            print(f"\n🎬 [Portion {drop_num}/{config.NUM_DISTRIBUTIONS}] Discharging {next_drop_cups:.2f} cups...")
            dispense_feed(camera, relay, config.FEEDER_STATION_PRESET, next_drop_cups, ai_worker, target_folder, drop_num)
            total_cups_fed_session += next_drop_cups
            
            if drop_num == config.NUM_DISTRIBUTIONS:
                break 

            portion_drop_timestamp = time.time()
            pellets_cleaned_early = False

            # 4. FIXED COUNTDOWN INTERVAL LOOPS WITH EARLY EXIT
            for check_idx in range(1, config.INSPECTION_COUNT_PER_DELAY + 1):
                print(f"⏳ Waiting standard window segment {check_idx}/{config.INSPECTION_COUNT_PER_DELAY}...")
                time.sleep(config.DELAY_PER_INTERVAL_MINUTES * 60)
                
                print("⚡ Running Double-Sweep Verification Pass...")
                peak_a = scan_and_evaluate_perimeter(camera, ai_worker, config.ACTIVE_MEAL_PRESETS, target_folder, f"P{drop_num}_C{check_idx}_SWEEP_A")
                peak_b = scan_and_evaluate_perimeter(camera, ai_worker, config.ACTIVE_MEAL_PRESETS, target_folder, f"P{drop_num}_C{check_idx}_SWEEP_B")
                
                print(f"📊 Double-Sweep Telemetry -> Sweep A Peak Max: {peak_a} | Sweep B Peak Max: {peak_b}")
                
                # STAGNATION EVALUATION
                count_variance = abs(peak_a - peak_b)
                if (peak_a >= config.STAGNATION_MIN_PELLETS and peak_b >= config.STAGNATION_MIN_PELLETS and count_variance <= config.DOUBLE_SWEEP_TOLERANCE):
                    print(f"⚠️ [Stagnation Engine] Leftovers are sitting completely static!")
                    time.sleep(config.POSTPONEMENT_MINUTES * 60)
                    peak_a = scan_and_evaluate_perimeter(camera, ai_worker, config.ACTIVE_MEAL_PRESETS, target_folder, f"P{drop_num}_C{check_idx}_POSTPONED_SWEEP_A")
                    peak_b = scan_and_evaluate_perimeter(camera, ai_worker, config.ACTIVE_MEAL_PRESETS, target_folder, f"P{drop_num}_C{check_idx}_POSTPONED_SWEEP_B")

                highest_verified_leftover = max(peak_a, peak_b)
                
                # --- DYNAMIC APPETITE INTERCEPTION (Config Parameters Added) ---
                if highest_verified_leftover <= config.CLEAN_PLATE_PELLET_THRESHOLD:
                    eating_time_minutes = (time.time() - portion_drop_timestamp) / 60
                    print(f"🟢 [Dynamic Engine] Clean Plate Detected! Fish cleared feed in {eating_time_minutes:.1f} minutes.")
                    
                    if eating_time_minutes <= config.FAST_EATING_TIME_THRESHOLD_MINS:
                        print(f"🚀 [Appetite: RAVENOUS] Scaling up next portion drop size by factor of {config.APPETITE_BOOST_MULTIPLIER}x.")
                        next_drop_cups = base_cups_per_distribution * config.APPETITE_BOOST_MULTIPLIER
                    else:
                        print("⚖️ [Appetite: NORMAL] Standard clearing pacing. Keeping standard baseline portion layout.")
                        next_drop_cups = base_cups_per_distribution
                    
                    pellets_cleaned_early = True
                    break  

                # EVALUATE OVERFEED SHIELD MATRIX
                if highest_verified_leftover > config.MAX_SINGLE_ZONE_LEFTOVERS:
                    consecutive_failures += 1
                    print(f"⚠️ [Shield Warning] Strike {consecutive_failures}/2! Boundary density ({highest_verified_leftover}) broke safety limit.")
                else:
                    if consecutive_failures > 0:
                        print("🟢 [Shield Recovery] Clean area matrix verified. Resetting failure strike counter to 0.")
                    consecutive_failures = 0
                
                if consecutive_failures >= 2:
                    print("🛑 [Overfeed Shield] TERMINATION CRITERIA MET: 2 consecutive failed phases registered.")
                    # Updated with rich logger parameter inputs
                    log_session_summary(config.CSV_LOG_FILE, total_cups_fed_session, adjusted_total_cups, history_multiplier, f"ABORTED_AT_P{drop_num}_C{check_idx}")
                    return

            # Downscale Next Drop if time window expired with lingering pellets
            if not pellets_cleaned_early:
                print(f"📉 [Appetite: SLUGGISH] Pellets lingering. Tapering next drop by factor of {config.APPETITE_REDUCTION_MULTIPLIER}x.")
                next_drop_cups = base_cups_per_distribution * config.APPETITE_REDUCTION_MULTIPLIER

        print(f"\n🏁 [Automation Core] Run Successful. Disbursed: {total_cups_fed_session:.2f} cups.")
        # Updated with rich logger parameter inputs
        log_session_summary(config.CSV_LOG_FILE, total_cups_fed_session, adjusted_total_cups, history_multiplier, "SUCCESS_FULL_RUN")

    finally:
        print(f"🔒 [Automation Core] Session closed at {time.strftime('%H:%M:%S')}\n")
        sys.stdout = original_stdout
        tee_logger.close()


if __name__ == "__main__":
    yolo_worker = AIAnalysisWorker(model_path=config.YOLO_WEIGHTS_PATH)
    relay_worker = ExternalRelayWorker(config.RELAY_API_URL)
    camera_system = HikvisionCamera(config.CAMERA_IP, config.CAMERA_USER, config.CAMERA_PASS)

    relay_worker.start()
    yolo_worker.start()
    camera_system.start_capture_thread(ai_worker=yolo_worker)
    
    print(f"🔒 Master Loop Initialized. Listening for trigger schedule matching times: {config.FEED_TIMES}")
    try:
        while True:
            if time.strftime("%H:%M") in config.FEED_TIMES:
                run_dynamic_feeding_engine(camera=camera_system, relay=relay_worker, ai_worker=yolo_worker)
                time.sleep(60)
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nShutting down execution routines...")
    finally:
        camera_system.stop_capture_thread()
        yolo_worker.stop()
        relay_worker.stop()