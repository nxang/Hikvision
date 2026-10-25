"""Hardware, feeding policy, calibration, and output settings. Times use the local clock."""

import os
from pathlib import Path

# Paths and operating mode
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DRY_RUN = True

# Camera connection and TLS
CAMERA_IP = os.environ.get("HIKVISION_CAMERA_IP", "192.168.1.64")
CAMERA_USER = os.environ.get("HIKVISION_CAMERA_USER", "admin")
CAMERA_PASS = os.environ.get("HIKVISION_CAMERA_PASSWORD", "Hikvision")
# Disabled for this camera's self-signed certificate. Set True or a trusted PEM
# bundle path to restore certificate verification.
CAMERA_TLS_VERIFY = False
CAMERA_RETRY_ATTEMPTS = 3
CAMERA_RETRY_DELAY_SECONDS = 10.0
CAMERA_PRESET_TIMEOUT_SECONDS = 5
CAMERA_SNAPSHOT_TIMEOUT_SECONDS = 10

# Feeder relay and worker timeouts
RELAY_API_URL = "http://100.80.73.62:1880/api/do1"
SECONDS_PER_CUP = 11.0
RELAY_OFF_RETRIES = 3
# Allow each request's connect/read timeouts plus the waits between retries.
CAPTURE_TIMEOUT_SECONDS = (
    2 * CAMERA_SNAPSHOT_TIMEOUT_SECONDS * CAMERA_RETRY_ATTEMPTS
    + CAMERA_RETRY_DELAY_SECONDS * (CAMERA_RETRY_ATTEMPTS - 1)
    + 5
)
ANALYSIS_TIMEOUT_SECONDS = 60

# Feeding schedule and session budget
FEED_TIMES = ["09:00", "12:00", "15:00", "20:20"]
TOTAL_SESSION_CUPS = 6.0
MIN_SESSION_CUPS = 0.0
MAX_SESSION_CUPS = 6.0
NUM_DISTRIBUTIONS = 5
DELAY_PER_INTERVAL_MINUTES = 2.0
INSPECTION_COUNT_PER_DELAY = 2

# Pellet detection and appetite adjustment
YOLO_CONF_THRESHOLD = 0.40
MAX_SINGLE_ZONE_LEFTOVERS = 2
CLEAN_PLATE_PELLET_THRESHOLD = 2
VELOCITY_ACTIVE_THRESHOLD = 5.0
VELOCITY_STAGNATION_THRESHOLD = 0.5
APPETITE_BOOST_MULTIPLIER = 1.20
APPETITE_REDUCTION_MULTIPLIER = 0.70

# Camera presets
BIOMASS_PRESET = 2
FEEDER_STATION_PRESET = 4
PRECHECK_PRESETS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# Preset 9 is excluded during meals because of foam and splashing.
ACTIVE_MEAL_PRESETS = [4, 5, 6, 7, 8, 10, 11, 12, 13]

# Live and dry-run output paths
CSV_LOG_FILE = str(PROJECT_ROOT / "aquaculture_feeding_ledger.csv")
DRY_RUN_CSV_LOG_FILE = str(PROJECT_ROOT / "aquaculture_feeding_ledger_dry_run.csv")
MEDIA_OUTPUT_DIR = str(PROJECT_ROOT / "media")
BIOMASS_CSV_LOG_FILE = str(PROJECT_ROOT / "biomass_growth_ledger.csv")
DRY_RUN_BIOMASS_CSV_LOG_FILE = str(PROJECT_ROOT / "biomass_growth_ledger_dry_run.csv")

# Biomass calibration
BIOMASS_MATRIX_PATH = str(PROJECT_ROOT / "pond_preset2_homography.npy")
LENGTH_CORRECTION_FACTOR = 0.625
JUVENILE_A_COEFF = 0.020
GROWOUT_A_COEFF = 0.0198

# Reserved settings (currently unused)
FAST_EATING_TIME_THRESHOLD_MINS = 2
DOUBLE_SWEEP_TOLERANCE = 4
STAGNATION_MIN_PELLETS = 4
POSTPONEMENT_MINUTES = 3.0
