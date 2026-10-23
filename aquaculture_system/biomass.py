"""Fish size estimation, image annotations, and biomass reporting."""

import csv
import os
import time

import cv2
import numpy as np

import config


def convert_pixels_to_cm(pixel_points, homography_matrix):
    """Convert image points to pond-floor coordinates in centimeters."""
    pts_format = np.array(pixel_points, dtype=np.float32).reshape(-1, 1, 2)
    transformed_pts = cv2.perspectiveTransform(pts_format, homography_matrix)
    return transformed_pts.reshape(-1, 2)


def log_biomass_summary(log_file, total_count, avg_length, avg_weight):
    """Append the population count and average size to the biomass ledger."""
    file_exists = os.path.isfile(log_file)
    try:
        with open(log_file, mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                writer.writerow(
                    [
                        "Timestamp",
                        "Total_Fish_Count",
                        "Avg_Length_cm",
                        "Avg_Weight_Grams",
                    ]
                )
            writer.writerow(
                [
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    total_count,
                    round(avg_length, 2),
                    round(avg_weight, 2),
                ]
            )
            print(f"[Biomass] Summary saved: {log_file}")
    except Exception as e:
        print(f"[Biomass] Could not write summary: {e}")


def estimate_fish_endpoints(contour):
    """Estimate head and tail from the contour rectangle and centroid."""
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

    # Approximate the head as the endpoint closest to the contour centroid.
    M = cv2.moments(contour_int)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroid = np.array([cx, cy])

        dist_to_head = np.linalg.norm(head_pixel - centroid)
        dist_to_tail = np.linalg.norm(tail_pixel - centroid)

        if dist_to_tail < dist_to_head:
            head_pixel, tail_pixel = tail_pixel, head_pixel
    return head_pixel, tail_pixel


def draw_biomass_summary(frame, total_valid_fish, avg_length_cm, avg_weight_g):
    """Draw population metrics in the top-right corner of the image."""
    osd_dashboard_text = [
        "=== BATCH BIOMASS METRICS ===",
        f"TOTAL COUNT: {total_valid_fish} Pcs",
        f"AVG LENGTH : {avg_length_cm:.1f} cm",
        f"AVG WEIGHT : {avg_weight_g:.1f} g",
    ]

    w_img = frame.shape[1]
    start_x = w_img - 300
    start_y = 45
    spacing_y = 30

    for line_idx, text_string in enumerate(osd_dashboard_text):
        target_x = start_x
        target_y = start_y + (line_idx * spacing_y)

        # Draw a shadow to keep the text readable over the image.
        cv2.putText(
            frame,
            text_string,
            (target_x + 1, target_y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        # Draw the foreground text.
        cv2.putText(
            frame,
            text_string,
            (target_x, target_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def run_biomass_preflight_analysis(camera, biomass_model, target_folder):
    """Capture the biomass preset, estimate fish sizes, and save the results."""
    print("[Biomass] Starting size assessment")
    matrix_path = getattr(config, "BIOMASS_MATRIX_PATH", "pond_preset2_homography.npy")
    biomass_csv = (
        config.DRY_RUN_BIOMASS_CSV_LOG_FILE
        if config.DRY_RUN
        else config.BIOMASS_CSV_LOG_FILE
    )
    biomass_preset = getattr(config, "BIOMASS_PRESET", 2)

    if not os.path.exists(matrix_path):
        print(
            f"[Biomass] Calibration missing: {matrix_path}. Skipping size estimation."
        )
        return

    # Map image coordinates to pond-floor centimeters.
    H_matrix = np.load(matrix_path)

    # Move to the calibrated inspection preset.
    print(f"[Biomass] Moving to preset {biomass_preset}")
    if not camera.go_to_preset(biomass_preset):
        raise RuntimeError("Biomass camera preset movement failed")
    time.sleep(4.0)  # Allow the camera to settle.

    # Wait for the snapshot before reading it.
    snap_time = f"{time.strftime('%Y%m%d_%H%M%S')}_BIOMASS_PRECHECK"
    captured_img_path, _ = camera.queue_snapshot_job(
        folder=target_folder,
        prefix=f"Zone_Preset_{biomass_preset}",
        custom_timestamp=snap_time,
        analyze=False,
    ).result(timeout=config.CAPTURE_TIMEOUT_SECONDS)
    raw_frame = cv2.imread(captured_img_path)

    if raw_frame is None:
        print(f"[Biomass] Invalid capture: {captured_img_path}")
        raise RuntimeError("Biomass capture is not a valid image")

    # Segment individual fish.
    print("[Biomass] Segmenting fish")
    results = biomass_model.predict(raw_frame, conf=0.20, verbose=False)

    valid_lengths = []
    valid_weights = []

    if results[0].masks is not None:
        # Keep the segmentation masks in the annotated output.
        frame = results[0].plot()

        fish_contours = results[0].masks.xy
        print(f"[Biomass] Segmented {len(fish_contours)} fish")

        # Calibration and length-to-weight coefficients.
        len_corr = getattr(config, "LENGTH_CORRECTION_FACTOR", 0.625)
        juve_a = getattr(config, "JUVENILE_A_COEFF", 0.020)
        grow_a = getattr(config, "GROWOUT_A_COEFF", 0.0198)

        for idx, contour in enumerate(fish_contours):
            head_pixel, tail_pixel = estimate_fish_endpoints(contour)
            is_heads_up = head_pixel[1] < tail_pixel[1]
            orientation_status = "HEADS-UP" if is_heads_up else "HEADS-DOWN"

            pixel_endpoints = [head_pixel, tail_pixel]
            cm_endpoints = convert_pixels_to_cm(pixel_endpoints, H_matrix)
            head_cm, tail_cm = cm_endpoints[0], cm_endpoints[1]

            # Reject fish outside the calibrated 245 cm pond diameter.
            fish_center_cm = (head_cm + tail_cm) / 2
            POND_RADIUS_CM = 122.5
            distance_from_center = np.sqrt(
                (fish_center_cm[0] - 122.5) ** 2 + (fish_center_cm[1] - 122.5) ** 2
            )

            if distance_from_center > POND_RADIUS_CM:
                print(
                    f"[Biomass] Fish #{idx + 1:02d} is outside the pond boundary; skipped."
                )
                continue

            raw_length_cm = np.sqrt(
                (head_cm[0] - tail_cm[0]) ** 2 + (head_cm[1] - tail_cm[1]) ** 2
            )
            fish_length_cm = raw_length_cm * len_corr

            # Use separate length-to-weight curves for juveniles and grow-out fish.
            if fish_length_cm < 12.0:
                a_coeff, b_coeff, cohort_type = juve_a, 3.05, "JUVENILE"
            else:
                a_coeff, b_coeff, cohort_type = grow_a, 3.00, "GROW-OUT"

            fish_weight_grams = a_coeff * (fish_length_cm**b_coeff)

            # Aggregate measurements for fish within the pond.
            valid_lengths.append(fish_length_cm)
            valid_weights.append(fish_weight_grams)

            print(
                f"   > Stock Check #{idx + 1:02d} [{cohort_type}]: Length: {fish_length_cm:5.2f} cm | Weight: {fish_weight_grams:6.2f} g | Status: {orientation_status}"
            )

            # Add measured endpoints and size estimates to the image.
            hx, hy = int(head_pixel[0]), int(head_pixel[1])
            tx, ty = int(tail_pixel[0]), int(tail_pixel[1])

            cv2.circle(
                frame, (hx, hy), 6, (0, 0, 255), -1
            )  # Red circle at estimated head point
            cv2.circle(
                frame, (tx, ty), 6, (255, 0, 0), -1
            )  # Blue circle at estimated tail point
            track_line_color = (0, 255, 0) if is_heads_up else (0, 165, 255)
            cv2.line(frame, (hx, hy), (tx, ty), track_line_color, 2)

            metric_label = f"#{idx+1} ({cohort_type[:3]}): {fish_length_cm:.1f}cm | {fish_weight_grams:.1f}g"
            cv2.putText(
                frame,
                metric_label,
                (min(hx, tx), min(hy, ty) - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Summarize accepted measurements.
        total_valid_fish = len(valid_lengths)
        avg_length_cm = np.mean(valid_lengths) if total_valid_fish > 0 else 0.0
        avg_weight_g = np.mean(valid_weights) if total_valid_fish > 0 else 0.0

        print("\n" + "=" * 55)
        print("                 BATCH BIOMASS SUMMARY                   ")
        print("=" * 55)
        print(f" Total Fish Quantified   : {total_valid_fish} count")
        print(f" Average Population Size : {avg_length_cm:.2f} cm")
        print(f" Average Population Mass : {avg_weight_g:.2f} g")
        print("=" * 55 + "\n")

        draw_biomass_summary(frame, total_valid_fish, avg_length_cm, avg_weight_g)

        # Save the annotated image and population summary.
        annotated_dir = os.path.join(target_folder, "Annotated")
        os.makedirs(annotated_dir, exist_ok=True)
        output_path = os.path.join(
            annotated_dir, f"Zone_Preset_2_{snap_time}_BIOMASS_ANNOTATED.jpg"
        )
        cv2.imwrite(output_path, frame)
        print(f"[Biomass] Annotated image saved: {output_path}")

        log_biomass_summary(biomass_csv, total_valid_fish, avg_length_cm, avg_weight_g)
    else:
        print("[Biomass] No fish detected")
        log_biomass_summary(biomass_csv, 0, 0.0, 0.0)
