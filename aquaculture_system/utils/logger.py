# utils/logger.py
import os
import csv
import time

def log_session_summary(log_file, total_dispensed, final_status):
    """Writes session outcomes to a CSV audit spreadsheet."""
    file_exists = os.path.isfile(log_file)
    try:
        with open(log_file, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                writer.writerow(["Timestamp", "Total_Cups_Fed_This_Session", "Termination_Status"])
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), total_dispensed, final_status])
            print(f"📝 Session accounting saved to ledger spreadsheet: {log_file}")
    except Exception as e:
        print(f"❌ Failed to write to CSV log: {e}")
        
        
def calculate_historical_adjustment(log_file, lookback_sessions=3):
    """
    Reads the last N rows of the ledger.
    Scales down for overfeed protection, or automatically increments 
    feed volumes upward by 5% if a perfect consumption streak is detected.
    """
    if not os.path.isfile(log_file):
        print("📈 [History Engine] No historical ledger found yet. Using standard 1.0x baseline setpoints.")
        return 1.0  
        
    recent_statuses = []
    try:
        with open(log_file, mode='r') as csv_file:
            reader = list(csv.reader(csv_file))
            data_rows = reader[1:]  # Omit header row
            
            # Extract the string completion codes from the last entries
            for row in data_rows[-lookback_sessions:]:
                if len(row) >= 3:
                    recent_statuses.append(row[2])
    except Exception as e:
        print(f"⚠️ [History Engine] File read error: {e}")
        return 1.0

    # Ensure we actually have enough history to make an assessment
    if len(recent_statuses) < lookback_sessions:
        return 1.0

    # 1. EVALUATE DOWNWARD PROTECTION FACTORS
    overfeed_incidents = sum(1 for status in recent_statuses if "ABORTED" in status)
    
    if overfeed_incidents == lookback_sessions:
        print(f"📉 [History Engine] Chronic overfeed alert over last {lookback_sessions} meals! Safety throttle: reducing feed by 30%.")
        return 0.7
    elif overfeed_incidents >= 1:
        print("📉 [History Engine] Failsafes triggered recently. Dialing down baseline feed by 10%.")
        return 0.9

    # 2. EVALUATE UPWARD GROWTH FACTORS
    perfect_streaks = sum(1 for status in recent_statuses if status == "SUCCESS_FULL_RUN")
    
    if perfect_streaks == lookback_sessions:
        print(f"🚀 [History Engine] Perfect Clean-Plate Streak! Fish are growing. Up-scaling baseline feed by 5% to match biomass.")
        return 1.05

    # Fallback to normal profile values
    print("⚖ [History Engine] Balanced meal data. Keeping feed values at 100% standard baseline.")
    return 1.0