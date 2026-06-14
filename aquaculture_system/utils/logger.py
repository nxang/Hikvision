# utils/logger.py
import os
import csv
import sys
import threading
import time

def log_session_summary(log_file, total_dispensed, baseline_budget, historical_mult, final_status):
    """Writes rich session outcomes to a CSV audit spreadsheet."""
    file_exists = os.path.isfile(log_file)
    try:
        with open(log_file, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                writer.writerow([
                    "Timestamp", 
                    "Baseline_Budget_Cups", 
                    "Historical_Multiplier", 
                    "Actual_Cups_Fed_This_Session", 
                    "Termination_Status"
                ])
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"), 
                round(baseline_budget, 2),
                round(historical_mult, 2),
                round(total_dispensed, 2), 
                final_status
            ])
            print(f"📝 Session accounting saved to ledger spreadsheet: {log_file}")
    except Exception as e:
        print(f"❌ Failed to write to CSV log: {e}")
        
        
def calculate_historical_adjustment(log_file, standard_baseline, lookback_sessions=3):
    """
    Reads the last N valid rows of the ledger.
    Uses the previous adjusted budget as the new base, so increases compound:
    4.00 -> 4.20 -> 4.41 -> 4.63
    """
    if not os.path.isfile(log_file):
        print("📈 [History Engine] No historical ledger found yet. Using standard 1.0x baseline setpoints.")
        return 1.0

    try:
        with open(log_file, mode='r', newline='') as csv_file:
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

        if len(valid_rows) < lookback_sessions:
            print("⚖ [History Engine] Not enough valid historical records. Keeping previous budget.")
            return previous_budget / standard_baseline

        recent_rows = valid_rows[-lookback_sessions:]
        recent_statuses = [row[4].strip() for row in recent_rows]

    except Exception as e:
        print(f"⚠️ [History Engine] File read error: {e}")
        return 1.0

    overfeed_incidents = sum(1 for status in recent_statuses if "ABORTED" in status)

    if overfeed_incidents == lookback_sessions:
        next_budget = previous_budget * 0.70
        print(f"📉 [History Engine] Chronic overfeed alert. Reducing previous budget to {next_budget:.2f} cups.")

    elif overfeed_incidents >= 1:
        next_budget = previous_budget * 0.90
        print(f"📉 [History Engine] Recent failsafe detected. Reducing previous budget to {next_budget:.2f} cups.")

    else:
        perfect_streaks = sum(1 for status in recent_statuses if status == "SUCCESS_FULL_RUN")

        if perfect_streaks == lookback_sessions:
            next_budget = previous_budget * 1.05
            print(f"🚀 [History Engine] Clean streak detected. Increasing previous budget to {next_budget:.2f} cups.")
        else:
            next_budget = previous_budget
            print(f"⚖ [History Engine] Balanced meal data. Keeping previous budget at {next_budget:.2f} cups.")

    return next_budget / standard_baseline

class ThreadSafeTeeLogger:
    """
    Interceptors sys.stdout, writing console prints to both the 
    standard terminal screen and a daily log file simultaneously.
    """
    def __init__(self, log_filepath):
        self.terminal = sys.stdout
        self.log_file = open(log_filepath, "a", encoding="utf-8")
        self.lock = threading.Lock()

    def write(self, message):
        with self.lock:
            self.terminal.write(message)
            self.log_file.write(message)
            self.log_file.flush()  # Force write immediate telemetry to disk

    def flush(self):
        with self.lock:
            self.terminal.flush()
            self.log_file.flush()

    def close(self):
        with self.lock:
            if not self.log_file.closed:
                self.log_file.close()