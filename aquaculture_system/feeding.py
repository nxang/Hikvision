"""Feeding sessions, leftover inspections, and portion adjustment rules."""

import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import config
from biomass import run_biomass_preflight_analysis
from utils.logger import (
    calculate_historical_adjustment,
    log_session_summary,
    tee_console,
)


@dataclass(frozen=True)
class AppetiteAssessment:
    """Decision from the inspections between two portions."""

    next_cups: float
    consecutive_failures: int
    abort_status: Optional[str] = None


def validate_feeding_config():
    """Reject invalid budgets, timing values, and empty inspection routes."""
    for name in (
        "TOTAL_SESSION_CUPS",
        "SECONDS_PER_CUP",
        "CAPTURE_TIMEOUT_SECONDS",
        "ANALYSIS_TIMEOUT_SECONDS",
    ):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in (
        "NUM_DISTRIBUTIONS",
        "INSPECTION_COUNT_PER_DELAY",
        "RELAY_OFF_RETRIES",
    ):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (
        not math.isfinite(config.MIN_SESSION_CUPS)
        or not math.isfinite(config.MAX_SESSION_CUPS)
        or not 0
        <= config.MIN_SESSION_CUPS
        <= config.TOTAL_SESSION_CUPS
        <= config.MAX_SESSION_CUPS
    ):
        raise ValueError("Session limits must contain TOTAL_SESSION_CUPS and be finite")
    if not config.PRECHECK_PRESETS or not config.ACTIVE_MEAL_PRESETS:
        raise ValueError("Inspection preset lists must not be empty")


def dispense_feed(camera, relay, feeder_preset, cups):
    """Align the camera and return the confirmed portion size from the relay."""
    print(f"[Feeder] Moving to preset {feeder_preset}; requesting {cups:.2f} cups")
    if not camera.go_to_preset(feeder_preset):
        raise RuntimeError("Feeder camera preset movement failed")
    time.sleep(4.0)
    timeout = cups * config.SECONDS_PER_CUP + 5 * (config.RELAY_OFF_RETRIES + 1) + 5
    return relay.trigger_feeding(num_cups=cups).result(timeout=timeout)


def capture_pellet_count(camera, target_folder, prefix, timestamp):
    """Wait for a capture and its analysis; propagate either failure."""
    _, analysis = camera.queue_snapshot_job(
        folder=target_folder, prefix=prefix, custom_timestamp=timestamp
    ).result(timeout=config.CAPTURE_TIMEOUT_SECONDS)
    return analysis.result(timeout=config.ANALYSIS_TIMEOUT_SECONDS)


def scan_and_evaluate_perimeter(
    camera, ai_worker, patrol_presets, target_folder, step_label
):
    """Return the largest single-zone count after inspecting every preset."""
    if not patrol_presets:
        raise ValueError("Inspection must include at least one preset")
    counts = []
    timestamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{step_label}"
    print(f"[Inspection: {step_label}] Presets: {patrol_presets}")
    for preset_id in patrol_presets:
        if not camera.go_to_preset(preset_id):
            raise RuntimeError(f"Camera could not move to preset {preset_id}")
        time.sleep(4.0)
        # Finish the capture before allowing the camera to move again.
        counts.append(
            capture_pellet_count(
                camera, target_folder, f"Zone_Preset_{preset_id}", timestamp
            )
        )
    return max(counts)


def check_tank_before_feeding(camera, ai_worker, target_folder):
    """Allow one five-minute cooldown before rejecting a dirty tank."""
    leftovers = scan_and_evaluate_perimeter(
        camera, ai_worker, config.PRECHECK_PRESETS, target_folder, "PRE_CHECK"
    )
    if leftovers <= config.MAX_SINGLE_ZONE_LEFTOVERS:
        return True
    print("[Precheck] Leftovers found. Waiting five minutes before checking again.")
    time.sleep(5 * 60)
    leftovers = scan_and_evaluate_perimeter(
        camera, ai_worker, config.PRECHECK_PRESETS, target_folder, "PRE_CHECK_EXTENDED"
    )
    return leftovers <= config.MAX_SINGLE_ZONE_LEFTOVERS


def inspect_between_portions(
    camera,
    ai_worker,
    target_folder,
    drop_num,
    base_cups,
    next_cups,
    consecutive_failures,
):
    """Check overfeeding first, then adjust the next portion for appetite."""
    last_count = None
    last_timestamp = time.time()
    accelerated = False

    for check_idx in range(1, config.INSPECTION_COUNT_PER_DELAY + 1):
        print(f"[Portion {drop_num}] Waiting for inspection {check_idx}")
        time.sleep(config.DELAY_PER_INTERVAL_MINUTES * 60)
        clean_threshold = getattr(config, "CLEAN_PLATE_PELLET_THRESHOLD", 2)
        label = f"P{drop_num}_C{check_idx}"
        center_count = scan_and_evaluate_perimeter(
            camera,
            ai_worker,
            [config.FEEDER_STATION_PRESET],
            target_folder,
            f"{label}_CENTER",
        )
        if center_count <= clean_threshold:
            perimeter_count = scan_and_evaluate_perimeter(
                camera,
                ai_worker,
                config.ACTIVE_MEAL_PRESETS,
                target_folder,
                f"{label}_PERIMETER_VALIDATION",
            )
            if perimeter_count <= clean_threshold:
                print(
                    "[Appetite] Center and perimeter clear; increasing the next portion."
                )
                boost = getattr(config, "APPETITE_BOOST_MULTIPLIER", 1.20)
                return AppetiteAssessment(base_cups * boost, 0)

        first_count = scan_and_evaluate_perimeter(
            camera,
            ai_worker,
            config.ACTIVE_MEAL_PRESETS,
            target_folder,
            f"{label}_SWEEP_A",
        )
        second_count = scan_and_evaluate_perimeter(
            camera,
            ai_worker,
            config.ACTIVE_MEAL_PRESETS,
            target_folder,
            f"{label}_SWEEP_B",
        )
        leftovers = max(first_count, second_count)
        timestamp = time.time()
        print(f"[Inspection] Peak leftovers: {leftovers}")

        # This check must precede every appetite-based early exit.
        if leftovers > config.MAX_SINGLE_ZONE_LEFTOVERS:
            consecutive_failures += 1
        else:
            consecutive_failures = 0
        if consecutive_failures >= 2:
            return AppetiteAssessment(
                next_cups,
                consecutive_failures,
                abort_status=f"ABORTED_AT_P{drop_num}_C{check_idx}",
            )

        if last_count is not None:
            elapsed_minutes = (timestamp - last_timestamp) / 60.0
            if elapsed_minutes > 0:
                velocity = (last_count - leftovers) / elapsed_minutes
                print(f"[Appetite] Estimated consumption: {velocity:.2f} pellets/min")
                active_threshold = getattr(config, "VELOCITY_ACTIVE_THRESHOLD", 5.0)
                if velocity >= active_threshold:
                    boost = getattr(config, "APPETITE_BOOST_MULTIPLIER", 1.20)
                    next_cups = base_cups * boost
                    accelerated = True
                    break
                stagnant_threshold = getattr(
                    config, "VELOCITY_STAGNATION_THRESHOLD", 0.5
                )
                if velocity <= stagnant_threshold and leftovers > clean_threshold:
                    reduction = getattr(config, "APPETITE_REDUCTION_MULTIPLIER", 0.70)
                    next_cups = base_cups * reduction
                    break

        last_count = leftovers
        last_timestamp = timestamp

    if not accelerated and last_count is not None and last_count > clean_threshold:
        reduction = getattr(config, "APPETITE_REDUCTION_MULTIPLIER", 0.70)
        next_cups = base_cups * reduction
    return AppetiteAssessment(next_cups, consecutive_failures)


def verify_final_consumption(camera, ai_worker, target_folder):
    """Require a delayed clean inspection after the final portion."""
    for check in range(1, config.INSPECTION_COUNT_PER_DELAY + 1):
        time.sleep(config.DELAY_PER_INTERVAL_MINUTES * 60)
        leftovers = scan_and_evaluate_perimeter(
            camera,
            ai_worker,
            config.ACTIVE_MEAL_PRESETS,
            target_folder,
            f"FINAL_CONSUMPTION_{check}",
        )
        if leftovers <= config.MAX_SINGLE_ZONE_LEFTOVERS:
            return True
    return False


def run_dynamic_feeding_engine(camera, relay, ai_worker, biomass_model):
    """Run one session from preflight through final consumption and accounting."""
    validate_feeding_config()
    ledger_file = config.DRY_RUN_CSV_LOG_FILE if config.DRY_RUN else config.CSV_LOG_FILE
    media_root = (
        os.path.join(config.MEDIA_OUTPUT_DIR, "dry_run")
        if config.DRY_RUN
        else config.MEDIA_OUTPUT_DIR
    )
    target_folder = os.path.join(
        media_root, f"Leftover_Inspection_{time.strftime('%Y-%m-%d')}"
    )
    os.makedirs(target_folder, exist_ok=True)
    log_file = os.path.join(
        target_folder, f"daily_console_stream_{time.strftime('%Y-%m-%d')}.txt"
    )

    total_fed = 0.0
    budget = config.TOTAL_SESSION_CUPS
    history_multiplier = 1.0

    def record_outcome(status):
        print(f"[Session] {status}; confirmed portions: {total_fed:.2f} cups")
        log_session_summary(ledger_file, total_fed, budget, history_multiplier, status)

    with tee_console(log_file):
        try:
            print(f"[Session] Started at {time.strftime('%H:%M:%S')}")
            run_biomass_preflight_analysis(camera, biomass_model, target_folder)
            history_multiplier = calculate_historical_adjustment(
                ledger_file,
                standard_baseline=config.TOTAL_SESSION_CUPS,
                lookback_sessions=3,
                min_budget=config.MIN_SESSION_CUPS,
                max_budget=config.MAX_SESSION_CUPS,
            )
            budget = min(
                config.MAX_SESSION_CUPS,
                max(
                    config.MIN_SESSION_CUPS,
                    config.TOTAL_SESSION_CUPS * history_multiplier,
                ),
            )
            history_multiplier = budget / config.TOTAL_SESSION_CUPS
            base_cups = budget / config.NUM_DISTRIBUTIONS
            next_cups = base_cups
            consecutive_failures = 0

            if not check_tank_before_feeding(camera, ai_worker, target_folder):
                record_outcome("ABORTED_DIRTY_TANK_PRECHECK")
                return

            for drop_num in range(1, config.NUM_DISTRIBUTIONS + 1):
                next_cups = min(next_cups, budget - total_fed)
                if next_cups <= 0:
                    break
                print(
                    f"[Portion {drop_num}/{config.NUM_DISTRIBUTIONS}] {next_cups:.2f} cups"
                )
                delivered = dispense_feed(
                    camera, relay, config.FEEDER_STATION_PRESET, next_cups
                )
                # Record confirmed delivery before verification capture can fail.
                total_fed += delivered
                capture_pellet_count(
                    camera,
                    target_folder,
                    f"Feeder_Zone_Preset_{config.FEEDER_STATION_PRESET}",
                    f"{time.strftime('%Y%m%d_%H%M%S')}_PORTION_{drop_num}_POST_DROP",
                )
                if drop_num == config.NUM_DISTRIBUTIONS:
                    break
                assessment = inspect_between_portions(
                    camera,
                    ai_worker,
                    target_folder,
                    drop_num,
                    base_cups,
                    next_cups,
                    consecutive_failures,
                )
                next_cups = assessment.next_cups
                consecutive_failures = assessment.consecutive_failures
                if assessment.abort_status:
                    record_outcome(assessment.abort_status)
                    return

            if total_fed == 0:
                status = "SKIPPED_ZERO_BUDGET"
            elif verify_final_consumption(camera, ai_worker, target_folder):
                status = "SUCCESS_FULL_RUN"
            else:
                status = "ABORTED_FINAL_LEFTOVERS"
            record_outcome(status)
        except BaseException as exc:
            print(f"[Session aborted] {type(exc).__name__}: {exc}")
            record_outcome(f"ERROR_{type(exc).__name__}_DELIVERY_MAY_BE_UNKNOWN")
            raise
        finally:
            print(f"[Session] Closed at {time.strftime('%H:%M:%S')}")
