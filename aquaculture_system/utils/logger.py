"""Session ledgers, historical budget adjustment, and console log capture."""

import csv
import math
import os
import sys
import threading
import time
from contextlib import contextmanager


def log_session_summary(
    log_file, total_dispensed, baseline_budget, historical_mult, final_status
):
    """Append the session budget, confirmed portions, and outcome to its ledger."""
    file_exists = os.path.isfile(log_file)
    try:
        with open(log_file, mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                writer.writerow(
                    [
                        "Timestamp",
                        "Baseline_Budget_Cups",
                        "Historical_Multiplier",
                        "Actual_Cups_Fed_This_Session",
                        "Termination_Status",
                    ]
                )
            writer.writerow(
                [
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    round(baseline_budget, 2),
                    round(historical_mult, 2),
                    round(total_dispensed, 2),
                    final_status,
                ]
            )
            print(f"📝 Session accounting saved to ledger spreadsheet: {log_file}")
    except Exception as e:
        print(f"❌ Failed to write to CSV log: {e}")


def calculate_historical_adjustment(
    log_file, standard_baseline, lookback_sessions=3, min_budget=0.0, max_budget=None
):
    """Adjust the previous budget using recent appetite observations and limits."""
    if not math.isfinite(standard_baseline) or standard_baseline <= 0:
        raise ValueError("Baseline must be finite and positive")
    if not isinstance(lookback_sessions, int) or lookback_sessions <= 0:
        raise ValueError("Lookback must be a positive integer")
    if max_budget is None:
        max_budget = standard_baseline
    if (
        not math.isfinite(min_budget)
        or not math.isfinite(max_budget)
        or not 0 <= min_budget <= standard_baseline <= max_budget
    ):
        raise ValueError("Invalid session budget limits")

    def bounded_multiplier(budget):
        return min(max_budget, max(min_budget, budget)) / standard_baseline

    if not os.path.isfile(log_file):
        print(
            "📈 [History Engine] No historical ledger found yet. Using standard 1.0x baseline setpoints."
        )
        return 1.0

    try:
        with open(log_file, mode="r", newline="") as csv_file:
            reader = list(csv.reader(csv_file))
            data_rows = reader[1:]  # Skip header row

        valid_rows = [row for row in data_rows if len(row) >= 5]

        if not valid_rows:
            return 1.0

        # Column 1 stores the previous adjusted session budget
        try:
            previous_budget = float(valid_rows[-1][1])
        except ValueError:
            previous_budget = standard_baseline
        if not math.isfinite(previous_budget) or previous_budget < 0:
            previous_budget = standard_baseline

        previous_budget = bounded_multiplier(previous_budget) * standard_baseline
        # Only completed appetite observations count toward the lookback window.
        # Errors and skipped sessions cannot mask actual overfeeding incidents.
        observations = [
            row
            for row in valid_rows
            if (
                row[4].strip() == "SUCCESS_FULL_RUN"
                or row[4].strip().startswith("ABORTED_")
            )
        ]
        recent_rows = observations[-lookback_sessions:]
        recent_statuses = [row[4].strip() for row in recent_rows]

    except Exception as e:
        print(f"⚠️ [History Engine] File read error: {e}")
        return 1.0

    overfeed_incidents = sum(1 for status in recent_statuses if "ABORTED" in status)

    if overfeed_incidents == lookback_sessions:
        next_budget = previous_budget * 0.70
        print(
            f"📉 [History Engine] Chronic overfeed alert. Reducing previous budget to {next_budget:.2f} cups."
        )

    elif overfeed_incidents >= 1:
        next_budget = previous_budget * 0.90
        print(
            f"📉 [History Engine] Recent failsafe detected. Reducing previous budget to {next_budget:.2f} cups."
        )

    else:
        perfect_streaks = sum(
            1 for status in recent_statuses if status == "SUCCESS_FULL_RUN"
        )

        if perfect_streaks == lookback_sessions:
            next_budget = previous_budget * 1.05
            print(
                f"🚀 [History Engine] Clean streak budget after limits: {bounded_multiplier(next_budget) * standard_baseline:.2f} cups."
            )
        else:
            next_budget = previous_budget
            print(
                f"⚖ [History Engine] Balanced meal data. Keeping previous budget at {next_budget:.2f} cups."
            )

    return bounded_multiplier(next_budget)


class ThreadSafeTeeLogger:
    """Write console output to the terminal and a log file under one lock."""

    def __init__(self, log_filepath):
        self.terminal = sys.stdout
        self.log_file = open(log_filepath, "a", encoding="utf-8")
        self.lock = threading.Lock()

    def write(self, message):
        with self.lock:
            self.terminal.write(message)
            if not self.log_file.closed:
                self.log_file.write(message)
                self.log_file.flush()

    def flush(self):
        with self.lock:
            self.terminal.flush()
            if not self.log_file.closed:
                self.log_file.flush()

    def close(self):
        with self.lock:
            if not self.log_file.closed:
                self.log_file.close()


@contextmanager
def tee_console(log_filepath):
    """Capture a session's console output and always restore the original stream."""
    original_stdout = sys.stdout
    logger = ThreadSafeTeeLogger(log_filepath)
    sys.stdout = logger
    try:
        yield logger
    finally:
        sys.stdout = original_stdout
        logger.close()
