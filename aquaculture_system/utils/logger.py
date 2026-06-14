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
        
        
def calculate_historical_adjustment(log_file, lookback_sessions=3):
    """
    Reads the last N valid rows of the ledger.
    Scales down for overfeed protection, or increases feed by 5%
    if a perfect success streak is detected.
    """
    if not os.path.isfile(log_file):
        print("📈 [History Engine] No historical ledger found yet. Using standard 1.0x baseline setpoints.")
        return 1.0

    recent_statuses = []

    try:
        with open(log_file, mode='r', newline='') as csv_file:
            reader = list(csv.reader(csv_file))
            data_rows = reader[1:]  # Skip header row

            valid_rows = [row for row in data_rows if len(row) >= 5]

            for row in valid_rows[-lookback_sessions:]:
                recent_statuses.append(row[4].strip())

    except Exception as e:
        print(f"⚠️ [History Engine] File read error: {e}")
        return 1.0

    if len(recent_statuses) < lookback_sessions:
        print("⚖ [History Engine] Not enough valid historical records. Keeping feed at 100% standard baseline.")
        return 1.0

    overfeed_incidents = sum(1 for status in recent_statuses if "ABORTED" in status)

    if overfeed_incidents == lookback_sessions:
        print(f"📉 [History Engine] Chronic overfeed alert over last {lookback_sessions} meals! Safety throttle: reducing feed by 30%.")
        return 0.7

    elif overfeed_incidents >= 1:
        print("📉 [History Engine] Failsafes triggered recently. Dialing down baseline feed by 10%.")
        return 0.9

    perfect_streaks = sum(1 for status in recent_statuses if status == "SUCCESS_FULL_RUN")

    if perfect_streaks == lookback_sessions:
        print(f"🚀 [History Engine] Perfect Clean-Plate Streak! Fish are growing. Up-scaling baseline feed by 5% to match biomass.")
        return 1.05

    print("⚖ [History Engine] Balanced meal data. Keeping feed values at 100% standard baseline.")
    return 1.0

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