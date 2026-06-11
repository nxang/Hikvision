# main.py
import os
import time
import threading
import sys  # Intercepts standard stdout print streams for daily log tracking files
import csv  # Appends biomass population profiles to a continuous growth spreadsheet
import cv2
import numpy as np
from ultralytics import YOLO

import config
from hardware.camera import HikvisionCamera
from hardware.relay import ExternalRelayWorker
from ai.detector import AIAnalysisWorker
from utils.logger import ThreadSafeTeeLogger, log_session_summary, calculate_historical_adjustment


def convert_pixels_to_cm(pixel_points, homography_matrix):
    """Transforms tracking pixel coordinates into physical pond floor centimeter values."""
    pts_format = np.array(pixel_points, dtype=np.float32).reshape(-1, 1, 2)
    transformed_pts = cv2.perspectiveTransform(pts_format, homography_matrix)
    return transformed_pts.reshape(-1, 2)


def log_biomass_summary(log_file, total_count, avg_length, avg_weight):
    """Appends rich, aggregated population metrics to a continuous CSV growth ledger."""
    file_exists = os.path.isfile(log_file)
    try:
        with open(log_file, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                writer.writerow([
                    "Timestamp", 
                    "Total_Fish_Count", 
                    "Avg_Length_cm", 
                    "Avg_Weight_Grams"
                ])
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                total_count,
                round(avg_length, 2),
                round(avg_weight, 2)
            ])
            print(f"📝 [Biomass Core] Aggregated metrics saved to spreadsheet ledger: {log_file}")
    except Exception as e:
        print(f"❌ [Biomass Core] Failed to append data matrix to CSV log: {e}")


def run_biomass_preflight_analysis(camera, biomass_model, target_folder):
    """
    Swings camera to inspection Zone 2, triggers snapshot acquisition,
    plots native YOLO masks, maps geometric unwarping vectors, renders 
    the top-right OSD telemetry box, and records stats to the CSV ledger.
    """
    print("\n📏 [Biomass Core] Initializing stock preflight cohort assessment...")
    matrix_path = getattr(config, 'BIOMASS_MATRIX_PATH', "pond_preset2_homography.npy")
    biomass_csv = getattr(config, 'BIOMASS_CSV_LOG_FILE', "biomass_growth_ledger.csv")
    biomass_preset = getattr(config, 'BIOMASS_PRESET', 2)
    
    if not os.path.exists(matrix_path):
        print(f"⚠️ [Biomass Core] Matrix file missing context path: {matrix_path}. Skipping estimation loop.")
        return

    # Load perspective map coordinate matrix arrays
    H_matrix = np.load(matrix_path)
    
    # 1. PTZ MECHANIZATION TRANSIT TO SECTOR 2
    print(f"🔄 [Biomass Core] Re-aligning camera array optic zoom to Sector Preset {biomass_preset}...")
    camera.go_to_preset(biomass_preset)
    time.sleep(4.0)  # Mechanical stabilization wait padding

    # 2. TRIGGER ASYNCHRONOUS HANDSHAKE CAPTURE
    snap_time = f"{time.strftime('%Y%m%d_%H%M%S')}_BIOMASS_PRECHECK"
    camera.queue_snapshot_job(folder=target_folder, prefix="Zone_Preset_2", custom_timestamp=snap_time)
    camera.capture_queue.join()  # Wait for file download array to complete disk write bounds

    # Reconstruct filesystem image path
    captured_img_path = os.path.join(target_folder, f"Zone_Preset_2_{snap_time}_LEFTOVER.jpg")
    raw_frame = cv2.imread(captured_img_path)
    
    if raw_frame is None:
        print(f"❌ [Biomass Core] System read fault on newly created frame block: {captured_img_path}")
        return

    # 3. RUN DEEP LEARNING MODEL PREDICTION
    print("[INFO] Processing instance segmentation layers over raw water boundary...")
    results = biomass_model.predict(raw_frame, conf=0.20, verbose=False)

    valid_lengths = []
    valid_weights = []

    if results[0].masks is not None:
        # ✨ CRITICAL ARCHITECTURE FIX: Overwrite baseline frame with YOLO's native color masks & boundary plots!
        frame = results[0].plot()

        fish_contours = results[0].masks.xy
        print(f"[SUCCESS] Segmented {len(fish_contours)} targets. Extracting skeletal mappings...")
        
        # Pull parameters safely utilizing getattrs to handle fallback dynamic tuning boundaries
        len_corr = getattr(config, 'LENGTH_CORRECTION_FACTOR', 0.625)
        juve_a = getattr(config, 'JUVENILE_A_COEFF', 0.020)
        grow_a = getattr(config, 'GROWOUT_A_COEFF', 0.0198)
        
        for idx, contour in enumerate(fish_contours):
            contour_int = contour.astype(np.int32)
            rect = cv2.minAreaRect(contour_int)
            box_points = cv2.boxPoints(rect).astype(np.int32)
            
            side_d0 = np.linalg.norm(box_points[0] - box_points[1])
            side_d1 = np.linalg.norm(box_points[1] - box_points[2])
            
            if side_d0 > side_d1:
                head_pixel = (box_points[0] + box_points[3]) / 2
                tail_pixel = (box_points[1] + box_points[2]) / 2
            else:
                head_pixel = (box_points[0] + box_points[1]) / 2
                tail_pixel = (box_points[2] + box_points[3]) / 2
                
            # SKELETAL SYMMETRY TRAP MITIGATION (Centroid Balance Vector Engine)
            M = cv2.moments(contour_int)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroid = np.array([cx, cy])
                
                dist_to_head = np.linalg.norm(head_pixel - centroid)
                dist_to_tail = np.linalg.norm(tail_pixel - centroid)
                
                if dist_to_tail < dist_to_head:
                    head_pixel, tail_pixel = tail_pixel, head_pixel
            
            is_heads_up = head_pixel[1] < tail_pixel[1]
            orientation_status = "HEADS-UP" if is_heads_up else "HEADS-DOWN"
                
            pixel_endpoints = [head_pixel, tail_pixel]
            cm_endpoints = convert_pixels_to_cm(pixel_endpoints, H_matrix)
            head_cm, tail_cm = cm_endpoints[0], cm_endpoints[1]
            
            # CYLINDRICAL RIM PLANE RADIAL GEOMETRIC BOUNDARY FILTER (Diameter 245cm Bounds)
            fish_center_cm = (head_cm + tail_cm) / 2
            POND_RADIUS_CM = 122.5  
            distance_from_center = np.sqrt((fish_center_cm[0] - 122.5)**2 + (fish_center_cm[1] - 122.5)**2)
            
            if distance_from_center > POND_RADIUS_CM:
                print(f"   > [SKIPPED] Target #{idx + 1:02d} parsed out: Swimming outside pond rim layout bounds.")
                continue
            
            raw_length_cm = np.sqrt((head_cm[0] - tail_cm[0])**2 + (head_cm[1] - tail_cm[1])**2)
            fish_length_cm = raw_length_cm * len_corr
            
            # DYNAMIC COHORT MATURITY EQUATION SWITCH
            if fish_length_cm < 12.0:
                a_coeff, b_coeff, cohort_type = juve_a, 3.05, "JUVENILE"
            else:
                a_coeff, b_coeff, cohort_type = grow_a, 3.00, "GROW-OUT"
                
            fish_weight_grams = a_coeff * (fish_length_cm ** b_coeff)
            
            # Log successful non-filtered elements
            valid_lengths.append(fish_length_cm)
            valid_weights.append(fish_weight_grams)
            
            print(f"   > Stock Check #{idx + 1:02d} [{cohort_type}]: Length: {fish_length_cm:5.2f} cm | Weight: {fish_weight_grams:6.2f} g | Status: {orientation_status}")
            
            # OVERLAY CUSTOM MATHEMATICAL BIOMASS ANNOTATIONS OVER NATIVE YOLO PLOTS
            hx, hy = int(head_pixel[0]), int(head_pixel[1])
            tx, ty = int(tail_pixel[0]), int(tail_pixel[1])
            
            cv2.circle(frame, (hx, hy), 6, (0, 0, 255), -1)   # Red circle at estimated head point
            cv2.circle(frame, (tx, ty), 6, (255, 0, 0), -1)   # Blue circle at estimated tail point
            track_line_color = (0, 255, 0) if is_heads_up else (0, 165, 255)
            cv2.line(frame, (hx, hy), (tx, ty), track_line_color, 2)
            
            metric_label = f"#{idx+1} ({cohort_type[:3]}): {fish_length_cm:.1f}cm | {fish_weight_grams:.1f}g"
            cv2.putText(frame, metric_label, (min(hx, tx), min(hy, ty) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        # 4. COMPUTE SYSTEM STATISTICAL OUTCOMES & DRAW HIGHER TELEMETRY OSD BOX
        total_valid_fish = len(valid_lengths)
        avg_length_cm = np.mean(valid_lengths) if total_valid_fish > 0 else 0.0
        avg_weight_g = np.mean(valid_weights) if total_valid_fish > 0 else 0.0

        print("\n" + "="*55)
        print("                 BATCH BIOMASS SUMMARY                   ")
        print("="*55)
        print(f" Total Fish Quantified   : {total_valid_fish} count")
        print(f" Average Population Size : {avg_length_cm:.2f} cm")
        print(f" Average Population Mass : {avg_weight_g:.2f} g")
        print("="*55 + "\n")

        osd_dashboard_text = [
            "=== BATCH BIOMASS METRICS ===",
            f"TOTAL COUNT: {total_valid_fish} Pcs",
            f"AVG LENGTH : {avg_length_cm:.1f} cm",
            f"AVG WEIGHT : {avg_weight_g:.1f} g"
        ]

        h_img, w_img = frame.shape[:2]
        start_x = w_img - 300  
        start_y = 45          
        spacing_y = 30        

        for line_idx, text_string in enumerate(osd_dashboard_text):
            target_x = start_x
            target_y = start_y + (line_idx * spacing_y)
            
            # Layer Shadow (High contrast anchor boundary backdrop)
            cv2.putText(frame, text_string, (target_x + 1, target_y + 1), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            # Layer Foreground (Neon Green indicator data link)
            cv2.putText(frame, text_string, (target_x, target_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        # 5. COMMIT COMPILATIONS TO DISK STORES
        annotated_dir = os.path.join(target_folder, "Annotated")
        os.makedirs(annotated_dir, exist_ok=True)
        output_path = os.path.join(annotated_dir, f"Zone_Preset_2_{snap_time}_BIOMASS_ANNOTATED.jpg")
        cv2.imwrite(output_path, frame)
        print(f"🖼️ [Biomass Core] Geometric calculation dashboard written safely to disk archive: {output_path}")

        # Trigger permanent spreadsheet ledger append sequence
        log_biomass_summary(biomass_csv, total_valid_fish, avg_length_cm, avg_weight_g)
    else:
        print("⚠️ [Biomass Core] Scanning profile run completed: 0 active fish shapes mapped inside pond arena viewport.")
        log_biomass_summary(biomass_csv, 0, 0.0, 0.0)


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


def run_dynamic_feeding_engine(camera, relay, ai_worker, biomass_model):
    """Executes top-down matrix splitting utilizing real-time consumption velocity optimization parameters."""
    target_folder = os.path.join(config.MEDIA_OUTPUT_DIR, f"Leftover_Inspection_{time.strftime('%Y-%m-%d')}")
    os.makedirs(target_folder, exist_ok=True)

    # Initialize dynamic Tee logger loop handles
    log_file_path = os.path.join(target_folder, f"daily_console_stream_{time.strftime('%Y-%m-%d')}.txt")
    tee_logger = ThreadSafeTeeLogger(log_file_path)
    original_stdout = sys.stdout
    sys.stdout = tee_logger

    try:
        print(f"\n🔔 [Automation Core] Session initialized at {time.strftime('%H:%M:%S')}")
        total_cups_fed_session = 0.0
        consecutive_failures = 0  

        # =========================================================================
        # --- 🧬 STEP 1: RUN POPULATION BIOMASS CORE ASSESSMENT PREFLIGHT ---
        # =========================================================================
        run_biomass_preflight_analysis(camera, biomass_model, target_folder)
        # =========================================================================

        # 2. READ HISTORICAL PROFILE LEDGER LOGS
        history_multiplier = calculate_historical_adjustment(config.CSV_LOG_FILE, lookback_sessions=3)
        adjusted_total_cups = config.TOTAL_SESSION_CUPS * history_multiplier
        
        base_cups_per_distribution = adjusted_total_cups / config.NUM_DISTRIBUTIONS
        next_drop_cups = base_cups_per_distribution  

        print(f"\n⚖️ Session Budget Strategy: Baseline {base_cups_per_distribution:.2f} Cups per drop.")

        # 3. MANDATORY PRE-FLIGHT SANITATION CHECK
        peak_precheck = scan_and_evaluate_perimeter(camera, ai_worker, config.PRECHECK_PRESETS, target_folder, "PRE_CHECK")
        if peak_precheck > config.MAX_SINGLE_ZONE_LEFTOVERS:
            print("⚠️ [Sanitation Check] Stale food discovered before feeding! Pausing for 5-minute absorption cooldown...")
            time.sleep(5 * 60)
            peak_precheck = scan_and_evaluate_perimeter(camera, ai_worker, config.PRECHECK_PRESETS, target_folder, "PRE_CHECK_EXTENDED")
            if peak_precheck > config.MAX_SINGLE_ZONE_LEFTOVERS:
                print("🛑 Tank is dirty. Aborting entire session!")
                log_session_summary(config.CSV_LOG_FILE, total_cups_fed_session, adjusted_total_cups, history_multiplier, "ABORTED_DIRTY_TANK_PRECHECK")
                return

        # 4. EXECUTE EQUAL PORTION DISTRIBUTION LOOP
        for drop_num in range(1, config.NUM_DISTRIBUTIONS + 1):
            remaining_session_budget = adjusted_total_cups - total_cups_fed_session
            
            # --- 🔒 SAFETAIL CEILING CONSTRAINT CLAMP ---
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
                break 

            portion_drop_timestamp = time.time()
            last_sweep_pellet_count = None  
            last_sweep_timestamp = time.time()
            appetite_mitigated_early = False

            # 5. INTERVAL COUNTDOWNS PACKED WITH SHORT-CIRCUITS & CONSUMPTION SLOPES
            for check_idx in range(1, config.INSPECTION_COUNT_PER_DELAY + 1):
                print(f"⏳ Waiting standard window segment {check_idx}/{config.INSPECTION_COUNT_PER_DELAY}...")
                time.sleep(config.DELAY_PER_INTERVAL_MINUTES * 60)
                
                # --- STEP 5A: CENTER FEEDER STATION PRESET CHECK SHORT-CIRCUIT ---
                clean_plate_thresh = getattr(config, 'CLEAN_PLATE_PELLET_THRESHOLD', 2)
                print(f"🎯 [Center Check] Panning directly to Feeder Station (Preset {config.FEEDER_STATION_PRESET}) to verify immediate consumption...")
                center_leftovers = scan_and_evaluate_perimeter(
                    camera, ai_worker, [config.FEEDER_STATION_PRESET], target_folder, f"P{drop_num}_C{check_idx}_CENTER"
                )
                print(f"📊 Feeder Station Direct Count: {center_leftovers} pellet(s).")

                if center_leftovers <= clean_plate_thresh:
                    print("🟢 Feeder station is clean! Doing a quick peripheral validation sweep...")
                    peak_peripheral = scan_and_evaluate_perimeter(
                        camera, ai_worker, config.ACTIVE_MEAL_PRESETS, target_folder, f"P{drop_num}_C{check_idx}_PERIMETER_VALIDATION"
                    )
                    
                    if peak_peripheral <= clean_plate_thresh:
                        print("🚀 [Appetite: RAVENOUS SHORT-CIRCUIT] Food vanished immediately at source! Accelerating schedule timeline.")
                        boost_mult = getattr(config, 'APPETITE_BOOST_MULTIPLIER', 1.20)
                        next_drop_cups = base_cups_per_distribution * boost_mult
                        appetite_mitigated_early = True
                        break 

                # --- STEP 5B: RUN STANDARD DOUBLE-SWEEP DRAGNET (If food lingering at center) ---
                print("⚡ Center has food lingering or drifting. Running full Double-Sweep Perimeter Dragnet Pass...")
                peak_a = scan_and_evaluate_perimeter(camera, ai_worker, config.ACTIVE_MEAL_PRESETS, target_folder, f"P{drop_num}_C{check_idx}_SWEEP_A")
                peak_b = scan_and_evaluate_perimeter(camera, ai_worker, config.ACTIVE_MEAL_PRESETS, target_folder, f"P{drop_num}_C{check_idx}_SWEEP_B")
                
                highest_verified_leftover = max(peak_a, peak_b)
                current_timestamp = time.time()
                print(f"📊 Double-Sweep Telemetry -> Perimeter Peak Leftovers: {highest_verified_leftover}")

                # --- HEURISTIC CONSUMPTION VELOCITY (Vc) MATHEMATICAL DERIVATIVE ---
                if last_sweep_pellet_count is not None:
                    elapsed_minutes = (current_timestamp - last_sweep_timestamp) / 60.0
                    pellets_consumed = last_sweep_pellet_count - highest_verified_leftover
                    
                    if elapsed_minutes > 0:
                        consumption_velocity = pellets_consumed / elapsed_minutes
                        print(f"📉 [Velocity Engine] Consumption Rate: {consumption_velocity:.2f} pellets/min")

                        # Fast Consumption Check
                        v_active = getattr(config, 'VELOCITY_ACTIVE_THRESHOLD', 5.0)
                        if consumption_velocity >= v_active:
                            print("🚀 [Velocity: ACTIVE FORAGING] Clearing rate is fast. Advancing to next allotment portion branch early.")
                            boost_mult = getattr(config, 'APPETITE_BOOST_MULTIPLIER', 1.20)
                            next_drop_cups = base_cups_per_distribution * boost_mult
                            appetite_mitigated_early = True
                            break
                        
                        # Flat Stagnation Check
                        v_stagnant = getattr(config, 'VELOCITY_STAGNATION_THRESHOLD', 0.5)
                        if consumption_velocity <= v_stagnant and highest_verified_leftover > clean_plate_thresh:
                            print("⚠️ [Velocity: APATHY SLOPING] Slope flattened out with lingering leftovers. Throttling upcoming drop.")
                            reduce_mult = getattr(config, 'APPETITE_REDUCTION_MULTIPLIER', 0.70)
                            next_drop_cups = base_cups_per_distribution * reduce_mult
                            break
                
                last_sweep_pellet_count = highest_verified_leftover
                last_sweep_timestamp = current_timestamp

                # EVALUATE SAFETY THRESHOLD GATEWAYS
                if highest_verified_leftover > config.MAX_SINGLE_ZONE_LEFTOVERS:
                    consecutive_failures += 1
                    print(f"⚠️ [Shield Warning] Strike {consecutive_failures}/2! Boundary density broke safety limits.")
                else:
                    consecutive_failures = 0
                
                if consecutive_failures >= 2:
                    print("🛑 [Overfeed Shield] Hard feeding engine lockout triggered.")
                    log_session_summary(config.CSV_LOG_FILE, total_cups_fed_session, adjusted_total_cups, history_multiplier, f"ABORTED_AT_P{drop_num}_C{check_idx}")
                    return

            # Sluggish appetite fallback fallback if the loop finishes natively without early exits
            if not appetite_mitigated_early and last_sweep_pellet_count is not None:
                if last_sweep_pellet_count > clean_plate_thresh:
                    print(f"📉 [Appetite: SLUGGISH CONVERGENCE] Delay window elapsed with lingering feed. Tapering next drop size.")
                    reduce_mult = getattr(config, 'APPETITE_REDUCTION_MULTIPLIER', 0.70)
                    next_drop_cups = base_cups_per_distribution * reduce_mult

        print(f"\n🏁 [Automation Core] Run Successful. Disbursed: {total_cups_fed_session:.2f} cups.")
        log_session_summary(config.CSV_LOG_FILE, total_cups_fed_session, adjusted_total_cups, history_multiplier, "SUCCESS_FULL_RUN")

    finally:
        print(f"🔒 [Automation Core] Session closed at {time.strftime('%H:%M:%S')}\n")
        sys.stdout = original_stdout
        tee_logger.close()


if __name__ == "__main__":
    # Deploy foundational thread infrastructure hardware workers
    yolo_pellet_worker = AIAnalysisWorker(model_path=config.YOLO_WEIGHTS_PATH)
    relay_hardware_worker = ExternalRelayWorker(config.RELAY_API_URL)
    camera_system = HikvisionCamera(config.CAMERA_IP, config.CAMERA_USER, config.CAMERA_PASS)

    # Initialize the Biomass Instance Segmentation Model ONCE right at boot runtime 
    # to eliminate inference execution lag spikes during session startup routines
    BIOMASS_MODEL_PATH = "fish.pt"
    print(f"🧬 [Main Init] Compiling Instance Segmentation Network Layer from: {BIOMASS_MODEL_PATH}")
    yolo_biomass_model = YOLO(BIOMASS_MODEL_PATH)

    relay_hardware_worker.start()
    yolo_pellet_worker.start()
    camera_system.start_capture_thread(ai_worker=yolo_pellet_worker)
    
    print(f"🔒 Master Loop Initialized. Listening for trigger schedule matching times: {config.FEED_TIMES}")
    try:
        while True:
            if time.strftime("%H:%M") in config.FEED_TIMES:
                run_dynamic_feeding_engine(
                    camera=camera_system, 
                    relay=relay_hardware_worker, 
                    ai_worker=yolo_pellet_worker,
                    biomass_model=yolo_biomass_model
                )
                time.sleep(60)
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nShutting down execution routines...")
    finally:
        camera_system.stop_capture_thread()
        yolo_pellet_worker.stop()
        relay_hardware_worker.stop()