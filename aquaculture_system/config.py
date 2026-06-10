# config.py
import os

# DRY_RUN = True
DRY_RUN =False

# --- Hardware Endpoints & Specs ---
CAMERA_IP = "192.168.1.64"
CAMERA_USER = "admin"
CAMERA_PASS = "Hikvision"

RELAY_API_URL = "http://100.80.73.62:1880/api/do1"
SECONDS_PER_CUP = 11.0

# --- AI Machine Learning Settings ---
YOLO_WEIGHTS_PATH = os.path.join("best.pt")
YOLO_CONF_THRESHOLD = 0.40  # Only count pellets if the model is > 40% confident
# --- RADIAL SCATTER ANTI-DOUBLE COUNTING THRESHOLDS ---
# Since pellets blast outward in all directions, if ANY single peripheral preset 
# catches more than this limit, it means pellets are escaping the school.
MAX_SINGLE_ZONE_LEFTOVERS = 8

CLEAN_PLATE_PELLET_THRESHOLD = 2          

# Time limit (minutes). If a portion is cleared faster than this, fish are "ravenous"
FAST_EATING_TIME_THRESHOLD_MINS = 2  

# Tuning bounds for Consumption Velocity (Vc)
VELOCITY_ACTIVE_THRESHOLD = 5.0      # Min pellets eaten per minute to count as ravenous
VELOCITY_STAGNATION_THRESHOLD = 0.5  # Below this rate, fish have lost interest

# Scale factors applied to the NEXT fractional drop based on clearing velocity
APPETITE_BOOST_MULTIPLIER = 1.20          # Increase next drop by 20% if eating fast
APPETITE_REDUCTION_MULTIPLIER = 0.70      # Decrease next drop by 30% if pellets linger

# --- PTZ Presets Layout ---
FEEDER_STATION_PRESET = 4

# Pre-Check scans everything including the feeder zone to guarantee absolute cleanliness
PRECHECK_PRESETS = [4,5, 6, 7, 8, 9, 10, 11, 12]

# Active Meal checks EXCLUDE Preset 9 to completely avoid surface foam/splashing noise
ACTIVE_MEAL_PRESETS = [4,5, 6, 7, 8, 10, 11, 12]

# --- Storage Settings ---
CSV_LOG_FILE = "aquaculture_feeding_ledger.csv"
MEDIA_OUTPUT_DIR = "media"

# --- Feeding Schedule Times ---
FEED_TIMES = ["16:45","09:00", "12:00","15:00","18:00"]  # 24-hour format times to trigger feeding sessions

# =========================================================================
# --- TAPERED/BEHAVIORAL FEEDING ENGINE PROFILE ---
# =========================================================================
# FEEDING_SEQUENCE = [
#     {
#         "step_label": "STAGE_1_PRIMER",
#         "feed_cups": 0.5,           # Small test dose to verify active appetite
#         # "delay_minutes": 3.0,       # Wait 3 mins for them to clean the plate
#         "delay_minutes": 0.1,       # Wait 3 mins for them to clean the plate
#         "inspection_count": 1,      # 1 full sweep of the perimeter at the end
#         "is_active_meal": True
#     },
#     {
#         "step_label": "STAGE_2_MAIN_COURSE",
#         "feed_cups": 1.5,           # Bulk nutritional delivery
#         # "delay_minutes": 6.0,       # Longer consumption window
#         "delay_minutes": 0.1,       # Wait 3 mins for them to clean the plate

#         "inspection_count": 3,      # Scans perimeter every 2 mins to catch escapees early
#         "is_active_meal": True
#     },
#     {
#         "step_label": "STAGE_3_TAPER_CHECK",
#         "feed_cups": 0.5,           # Micro final top-up
#         # "delay_minutes": 4.0,
#         "delay_minutes": 0.1,       # Wait 3 mins for them to clean the plate

#         "inspection_count": 1,      # Final clean check
#         "is_active_meal": True
#     }
# ]

TOTAL_SESSION_CUPS = 4.0             # Total ideal amount of food for this session
NUM_DISTRIBUTIONS = 5                # How many fractional portions to split the total into

DELAY_PER_INTERVAL_MINUTES = 1.0     # EXACT time to wait before EACH camera check pass
INSPECTION_COUNT_PER_DELAY = 2       # How many perimeter photo sweeps to execute during that wait window

DOUBLE_SWEEP_TOLERANCE = 4           # Allowed difference to count as "almost the same" (e.g. 4 vs 4, or 4 vs 3)
STAGNATION_MIN_PELLETS = 4           # Minimum pellets required to trigger rule (ignores clean 0 vs 0 frames)
POSTPONEMENT_MINUTES = 3.0           # Extra delay wait time to give fish if pellets are found sitting static