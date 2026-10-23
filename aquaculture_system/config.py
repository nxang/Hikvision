# config.py
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DRY_RUN = True
# DRY_RUN =False

# --- Hardware Endpoints & Specs ---
CAMERA_IP = os.environ.get("HIKVISION_CAMERA_IP", "192.168.1.64")
CAMERA_USER = os.environ.get("HIKVISION_CAMERA_USER", "admin")
CAMERA_PASS = os.environ.get("HIKVISION_CAMERA_PASSWORD", "")
# For self-signed cameras, supply a trusted CA/certificate bundle path.
CAMERA_TLS_VERIFY = os.environ.get("HIKVISION_CAMERA_CA_BUNDLE") or True

RELAY_API_URL = "http://100.80.73.62:1880/api/do1"
SECONDS_PER_CUP = 11.0
RELAY_OFF_RETRIES = 3
CAPTURE_TIMEOUT_SECONDS = 15
ANALYSIS_TIMEOUT_SECONDS = 60

# --- AI Machine Learning Settings ---
YOLO_CONF_THRESHOLD = 0.40  # Only count pellets if the model is > 40% confident
# --- RADIAL SCATTER ANTI-DOUBLE COUNTING THRESHOLDS ---
# Since pellets blast outward in all directions, if ANY single peripheral preset 
# catches more than this limit, it means pellets are escaping the school.
MAX_SINGLE_ZONE_LEFTOVERS = 2

CLEAN_PLATE_PELLET_THRESHOLD = 2          

# Time limit (minutes). If a portion is cleared faster than this, fish are "ravenous"
FAST_EATING_TIME_THRESHOLD_MINS = 2  

# Tuning bounds for Consumption Velocity (Vc)
VELOCITY_ACTIVE_THRESHOLD = 5.0      # Min pellets eaten per minute to count as ravenous
VELOCITY_STAGNATION_THRESHOLD = 0.5  # Below this rate, fish have lost interest

# Scale factors applied to the NEXT fractional drop based on clearing velocity
APPETITE_BOOST_MULTIPLIER = 1.20          # Increase next drop by 20% if eating fast
APPETITE_REDUCTION_MULTIPLIER = 0.70      # Decrease next drop by 30% if pellets linger

BIOMASS_PRESET = 2
# --- PTZ Presets Layout ---
FEEDER_STATION_PRESET = 4

# Pre-Check scans everything including the feeder zone to guarantee absolute cleanliness
PRECHECK_PRESETS = [4, 5, 6, 7, 8, 9, 10, 11, 12,13]
# PRECHECK_PRESETS = [4]

# Active Meal checks EXCLUDE Preset 9 to completely avoid surface foam/splashing noise
ACTIVE_MEAL_PRESETS = [4,5, 6, 7, 8, 10, 11, 12,13]
# ACTIVE_MEAL_PRESETS = [5]

# --- Storage Settings ---
CSV_LOG_FILE = str(PROJECT_ROOT / "aquaculture_feeding_ledger.csv")
DRY_RUN_CSV_LOG_FILE = str(PROJECT_ROOT / "aquaculture_feeding_ledger_dry_run.csv")
MEDIA_OUTPUT_DIR = str(PROJECT_ROOT / "media")
BIOMASS_CSV_LOG_FILE = str(PROJECT_ROOT / "biomass_growth_ledger.csv")
DRY_RUN_BIOMASS_CSV_LOG_FILE = str(PROJECT_ROOT / "biomass_growth_ledger_dry_run.csv")


# =========================================================================
# --- BIOMASS ESTIMATION SCHEME TRACKING ---
# =========================================================================
BIOMASS_MATRIX_PATH = str(PROJECT_ROOT / "pond_preset2_homography.npy")
LENGTH_CORRECTION_FACTOR  = 0.625
JUVENILE_A_COEFF  = 0.020
GROWOUT_A_COEFF  = 0.0198

# --- Feeding Schedule Times ---
FEED_TIMES = ["09:00", "12:00","14:30","18:00"]  # 24-hour format times to trigger feeding sessions

TOTAL_SESSION_CUPS = 6.0             # Total ideal amount of food for this session
# Keep historical adaptation within an explicit operator-controlled range.
MIN_SESSION_CUPS = 0.0
MAX_SESSION_CUPS = 6.0
NUM_DISTRIBUTIONS = 5                # How many fractional portions to split the total into

DELAY_PER_INTERVAL_MINUTES = 2.0     # EXACT time to wait before EACH camera check pass
INSPECTION_COUNT_PER_DELAY = 2       # How many perimeter photo sweeps to execute during that wait window

DOUBLE_SWEEP_TOLERANCE = 4           # Allowed difference to count as "almost the same" (e.g. 4 vs 4, or 4 vs 3)
STAGNATION_MIN_PELLETS = 4           # Minimum pellets required to trigger rule (ignores clean 0 vs 0 frames)
POSTPONEMENT_MINUTES = 3.0           # Extra delay wait time to give fish if pellets are found sitting static
