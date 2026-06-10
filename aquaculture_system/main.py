# main.py
import os
import time
import threading
import config
import sys  # Handled for top-level stdout stream interception

from hardware.camera import HikvisionCamera
from hardware.relay import ExternalRelayWorker
from ai.detector import AIAnalysisWorker
from utils.logger import ThreadSafeTeeLogger, log_session_summary, calculate_historical_adjustment


def dispense_feed(camera, relay, feeder_preset, cups, ai_worker=None, target_folder=None, current_distribution=0):
    """Aligns camera to feeder point, discharges portion payload, and saves verification capture."""
    print(f"🔄 [Feeder Align] Target Preset {feeder_preset}...")
    camera.go_to_preset(feeder_preset)
    time.sleep(4.0)  # Mechanical settling period
    
    feeder_done = threading.Event()
    relay.trigger_feeding(num_cups=cups, callback_event=feeder_done)
    feeder_done.wait()  # Block until the requested motor runtime interval expires

    if ai_worker and target_folder:
        print(f"📸 [Feeder Shield] Portion drop complete. Capturing drop verification photo...")
        verify_timestamp = f"{time.strftime('%Y%m%d_%H%M%S')}_PORTION_{current_distribution}_POST_DROP"
        camera.queue_snapshot_job(folder=target_folder, prefix=f"Feeder_Zone_Preset_{feeder_preset}", custom_timestamp=verify_timestamp)
        camera.capture_queue.join()  


def scan_and_evaluate_perimeter(camera, ai_worker, patrol_presets, target_folder, step_label):
    """Executes a targeted perimeter pass over listed presets and extracts the peak zone density count."""
    ai_worker.reset_cycle_counter()
    cycle_timestamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{step_label}"

    print(f"🔎 [Scanning Matrix: {step_label}] Panning targets: {patrol_presets}")
    for preset_id in patrol_presets:
        prefix_string = f"Zone_Preset_{preset_id}"
        if camera.go_to_preset(preset_id):
            time.sleep(4.0)  # Standard pan-tilt mechanical delay gap
            camera.queue_snapshot_job(folder=target_folder, prefix=prefix_string, custom_timestamp=cycle_timestamp)
            
    camera.capture_queue.join()  # Synchronize download queues
    ai_worker.queue.join()        # Synchronize AI evaluation routines
    return ai_worker.cycle_max_zone_leftovers


def run_dynamic_feeding_engine(camera, relay, ai_worker):
    """Executes top-down matrix splitting utilizing real-time consumption velocity optimization parameters."""
    target_folder = os.path.join(config.MEDIA_OUTPUT_DIR, f"Leftover_Inspection_{time.strftime('%Y-%m-%d')}")
    os.makedirs(target_folder, exist_ok=True)

    # Initialize Thread-Safe Tee Logger to echo outputs into daily subdirectory text files
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

        print(f"⚖️ Session Budget Strategy: Total {adjusted_total_cups:.2f} Cups ({base_cups_per_distribution:.2f} Cups baseline per drop).")

        # 2. MANDATORY PRE-FLIGHT SANITATION CHECK
        peak_precheck = scan_and_evaluate_perimeter(camera, ai_worker, config.PRECHECK_PRESETS, target_folder, "PRE_CHECK")
        if peak_precheck > config.MAX_SINGLE_ZONE_LEFTOVERS:
            print("⚠️ [Sanitation Check] Stale food discovered before feeding! Pausing for 5-minute absorption cooldown...")
            time.sleep(5 * 60)
            peak_precheck = scan_and_evaluate_perimeter(camera, ai_worker, config.PRECHECK_PRESETS, target_folder, "PRE_CHECK_EXTENDED")
            if peak_precheck > config.MAX_SINGLE_ZONE_LEFTOVERS:
                print("🛑 Tank is dirty. Aborting entire session!")
                log_session_summary(config.CSV_LOG_FILE, total_cups_fed_session, adjusted_total_cups, history_multiplier, "ABORTED_DIRTY_TANK_PRECHECK")
                return

        # 3. EXECUTE ADAPTIVE DISTRIBUTION LOOP
        for drop_num in range(1, config.NUM_DISTRIBUTIONS + 1):
            remaining_session_budget = adjusted_total_cups - total_cups_fed_session
            
            # --- 🔒 HARD CEILING CLAMP SAFEGUARD ---
            if next_drop_cups > remaining_session_budget:
                print(f"🔒 [Ceiling Safeguard] Trimming portion from {next_drop_cups:.2f} to {remaining_session_budget:.2f} to avoid breaching total session limits.")
                next_drop_cups = remaining_session_budget

            if next_drop_cups <= 0:
                print("🏁 [Ceiling Safeguard] Total session budget fully exhausted early. Closing engine loops.")
                log_session_summary(config.CSV_LOG_FILE, total_cups_fed_session, adjusted_total_cups, history_multiplier, "SUCCESS_BUDGET_EXHAUSTED_EARLY")
                return 
            
            print(f"\n🎬 [Portion {drop_num}/{config.NUM_DISTRIBUTIONS}] Discharging {next_drop_cups:.2f} cups...")
            dispense_feed(camera, relay, config.FEEDER_STATION_PRESET, next_drop_cups, ai_worker, target_folder, drop_num)
            total_cups_fed_session += next_drop_cups
            
            if drop_num == config.NUM_DISTRIBUTIONS:
                break  # Complete run on final section release

            portion_drop_timestamp = time.time()
            last_sweep_pellet_count = None  
            last_sweep_timestamp = time.time()
            appetite_mitigated_early = False

            # 4. FIXED COUNTDOWN INTERVAL LOOPS WITH CENTER SHORT-CIRCUIT & VELOCITY
            for check_idx in range(1, config.INSPECTION_COUNT_PER_DELAY + 1):
                print(f"⏳ Waiting standard window segment {check_idx}/{config.INSPECTION_COUNT_PER_DELAY}...")
                time.sleep(config.DELAY_PER_INTERVAL_MINUTES * 60)
                
                # --- 🎯 STEP 4A: SHORT-CIRCUIT CENTER FEEDER STATION CHECK ---
                clean_plate_thresh = getattr(config, 'CLEAN_PLATE_PELLET_THRESHOLD', 2)
                print(f"🎯 [Center Check] Panning directly to Feeder Station (Preset {config.FEEDER_STATION_PRESET}) to verify consumption rate...")
                center_leftovers = scan_and_evaluate_perimeter(
                    camera, ai_worker, [config.FEEDER_STATION_PRESET], target_folder, f"P{drop_num}_C{check_idx}_CENTER"
                )
                print(f"📊 Feeder Station Direct Count: {center_leftovers} pellet(s).")

                if center_leftovers <= clean_plate_thresh:
                    print("🟢 Feeder station is clear! Doing a quick peripheral validation sweep...")
                    peak_peripheral = scan_and_evaluate_perimeter(
                        camera, ai_worker, config.ACTIVE_MEAL_PRESETS, target_folder, f"P{drop_num}_C{check_idx}_PERIMETER_VALIDATION"
                    )
                    
                    if peak_peripheral <= clean_plate_thresh:
                        print("🚀 [Appetite: RAVENOUS SHORT-CIRCUIT] Food disappeared instantly at source! Accelerating next drop layout.")
                        boost_mult = getattr(config, 'APPETITE_BOOST_MULTIPLIER', 1.20)
                        next_drop_cups = base_cups_per_distribution * boost_mult
                        appetite_mitigated_early = True
                        break # Break out of waiting intervals to dispense feed early!

                # --- 📈 STEP 4B: RUN STANDARD DOUBLE-SWEEP CONVERGENCE SECTORS ---
                print("⚡ Center has food lingering or drifting. Running full Double-Sweep Perimeter Dragnet Pass...")
                peak_a = scan_and_evaluate_perimeter(camera, ai_worker, config.ACTIVE_MEAL_PRESETS, target_folder, f"P{drop_num}_C{check_idx}_SWEEP_A")
                peak_b = scan_and_evaluate_perimeter(camera, ai_worker, config.ACTIVE_MEAL_PRESETS, target_folder, f"P{drop_num}_C{check_idx}_SWEEP_B")
                
                highest_verified_leftover = max(peak_a, peak_b)
                current_timestamp = time.time()
                print(f"📊 Double-Sweep Telemetry -> Perimeter Peak Leftovers: {highest_verified_leftover}")

                # --- 📉 HEURISTIC CONSUMPTION VELOCITY (Vc) MATH MATRIX ---
                if last_sweep_pellet_count is not None:
                    elapsed_minutes = (current_timestamp - last_sweep_timestamp) / 60.0
                    pellets_consumed = last_sweep_pellet_count - highest_verified_leftover
                    
                    if elapsed_minutes > 0:
                        consumption_velocity = pellets_consumed / elapsed_minutes
                        print(f"📉 [Velocity Engine] Consumption Rate: {consumption_velocity:.2f} pellets/min")

                        # Dynamic Trigger 1: High Velocity Trend matches heavy feeding profiles
                        v_active = getattr(config, 'VELOCITY_ACTIVE_THRESHOLD', 5.0)
                        if consumption_velocity >= v_active:
                            print("🚀 [Velocity: ACTIVE FORAGING] Perimeter clearing rate is fast. Accelerating next portion batch early.")
                            boost_mult = getattr(config, 'APPETITE_BOOST_MULTIPLIER', 1.20)
                            next_drop_cups = base_cups_per_distribution * boost_mult
                            appetite_mitigated_early = True
                            break
                        
                        # Dynamic Trigger 2: Flattened consumption velocity implies full fish
                        v_stagnant = getattr(config, 'VELOCITY_STAGNATION_THRESHOLD', 0.5)
                        if consumption_velocity <= v_stagnant and highest_verified_leftover > clean_plate_thresh:
                            print("⚠️ [Velocity: APATHY DETECTED] Slope flattened out with food left over. Throttling back next portion size.")
                            reduce_mult = getattr(config, 'APPETITE_REDUCTION_MULTIPLIER', 0.70)
                            next_drop_cups = base_cups_per_distribution * reduce_mult
                            break
                
                # Re-index history frames
                last_sweep_pellet_count = highest_verified_leftover
                last_sweep_timestamp = current_timestamp

                # EVALUATE OVERFEED SHIELD MATRIX (Failsafe Hardware Circuit Protection)
                if highest_verified_leftover > config.MAX_SINGLE_ZONE_LEFTOVERS:
                    consecutive_failures += 1
                    print(f"⚠️ [Shield Warning] Strike {consecutive_failures}/2! Boundary density broke safety limits.")
                else:
                    consecutive_failures = 0
                
                if consecutive_failures >= 2:
                    print("🛑 [Overfeed Shield] TERMINATION CRITERIA MET: Hard lockout triggered.")
                    log_session_summary(config.CSV_LOG_FILE, total_cups_fed_session, adjusted_total_cups, history_multiplier, f"ABORTED_AT_P{drop_num}_C{check_idx}")
                    return

            # Standard countdown cutoff loop timeout fallback fallback 
            if not appetite_mitigated_early and last_sweep_pellet_count is not None:
                if last_sweep_pellet_count > clean_plate_thresh:
                    print(f"📉 [Appetite: SLUGGISH TIMEOUT] Delay window closed with pellets floating. Throttling next drop size.")
                    reduce_mult = getattr(config, 'APPETITE_REDUCTION_MULTIPLIER', 0.70)
                    next_drop_cups = base_cups_per_distribution * reduce_mult

        print(f"\n🏁 [Automation Core] Run Successful. Disbursed: {total_cups_fed_session:.2f} cups.")
        log_session_summary(config.CSV_LOG_FILE, total_cups_fed_session, adjusted_total_cups, history_multiplier, "SUCCESS_FULL_RUN")

    finally:
        # Guarantee system default standard output streams are restored and files flushed cleanly
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