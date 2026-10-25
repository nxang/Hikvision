# Aquaculture feeding controller

Python 3.12 application using a Hikvision PTZ camera, YOLO pellet detection,
fish segmentation, and an HTTP feeder relay.

## Setup and run (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r aquaculture_system\requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
$cameraCredential = Get-Credential -UserName admin -Message 'Hikvision camera credentials'
$env:HIKVISION_CAMERA_USER = $cameraCredential.UserName
$env:HIKVISION_CAMERA_PASSWORD = $cameraCredential.GetNetworkCredential().Password
.\.venv\Scripts\python.exe -X utf8 aquaculture_system\main.py
```

Settings are in `aquaculture_system/config.py`. `DRY_RUN = True` suppresses relay
commands; it still captures images and moves the camera during a session.
Schedule times use the computer's local clock. Data paths resolve relative to
the project root, regardless of the launch directory.

Camera credentials can be overridden with `HIKVISION_CAMERA_USER` and
`HIKVISION_CAMERA_PASSWORD`. Prefer those environment variables for credentials.
The camera needs credentials even in dry-run mode because it still accesses hardware.
`CAMERA_TLS_VERIFY = False` allows this camera's self-signed certificate on both
the laptop and Pi. Traffic still uses HTTPS, but the camera's identity is not
verified. To restore verification, set `CAMERA_TLS_VERIFY` to `True` or an absolute
path to a trusted PEM certificate bundle in `config.py`.

Dry runs read and write `aquaculture_feeding_ledger_dry_run.csv`, use
`biomass_growth_ledger_dry_run.csv`, and save media under `media/dry_run/`.
Physical feeding uses the original ledger and media paths. Existing ledger rows
have no mode marker, so older mixed test/live records cannot be separated
automatically; existing files are preserved.

`MIN_SESSION_CUPS` and `MAX_SESSION_CUPS` bound historical adaptation. The maximum
defaults to the existing six-cup baseline. Increasing it requires explicitly
changing the configuration. Biomass estimates are recorded separately and do
not currently determine feeding budgets.

## Failure behavior

- Camera preset moves and snapshot HTTP requests make up to
  `CAMERA_RETRY_ATTEMPTS` attempts (default 3), waiting
  `CAMERA_RETRY_DELAY_SECONDS` between failures (default 10 seconds). The retry
  message includes the underlying network, HTTP, or certificate error. Request
  timeouts add to the total recovery window; capture waits include that window.
- If the camera remains unavailable, the controller records
  `SKIPPED_CAMERA_UNAVAILABLE` and waits for the next scheduled session. If some
  portions were already dispensed, it records `STOPPED_CAMERA_UNAVAILABLE` with
  the confirmed amount and does not dispense further portions. A retry repeats
  only the failed camera operation, never a previously dispensed portion.
  Camera skips do not change appetite history or reset a reduced budget.
- Each capture finishes before the next preset move. Invalid images, inference,
  disk writes, and relay failures retain their existing error handling.
- The two-strike leftover check runs before appetite shortcuts.
- After the last portion, the controller waits for a consumption interval and
  inspects the active meal presets. It allows the configured number of checks;
  persistent leftovers produce `ABORTED_FINAL_LEFTOVERS`, never a clean success.
  This also applies when the budget is exhausted before the last planned drop.
- Relay ON is attempted once. An ambiguous response aborts the portion; OFF is
  still attempted with bounded retries. A relay fault blocks additional portions.
- Portion timing uses a monotonic deadline starting before the ON request, so
  acknowledgment latency is deducted from the remaining wait. A late ON reply
  or an OFF command needing retries marks delivery unknown and faults the relay,
  even when a later OFF retry succeeds.
- Shutdown interrupts active dispensing and attempts OFF before waiting for
  camera and inference workers. Camera unavailability ends only the current
  session; other session errors terminate the process for operator review.
- The ledger counts only portions whose timed ON/OFF sequence was confirmed.
  Error rows flag that additional physical delivery may be unknown. Hardware
  errors and skipped sessions do not count as appetite observations for historical
  adaptation. They cannot mask real overfeeding aborts in the lookback window.

Relay acknowledgments and timing cannot measure actual feed mass or guarantee
shutdown when the network or controller loses power. Configure a maximum ON
timer at the relay/controller itself for independent protection.
The repository contains no relay/controller implementation to provision that
timer; it must be configured on the external controller. The six-cup software
cap bounds requested portions and is not a feed-mass measurement.

## Tests

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -v -b
```

Tests mock camera, inference, and relay operations. They cover scan failures,
capture ordering, relay faults and interruption, delivery accounting, budget
limits, and both appetite paths that previously bypassed the overfeeding check.

## Code organization

- `aquaculture_system/main.py`: model loading, worker startup/shutdown, and schedule polling.
- `aquaculture_system/feeding.py`: session orchestration, prechecks, appetite decisions, and budget enforcement.
- `aquaculture_system/biomass.py`: fish geometry, size estimation, annotations, and biomass summaries.
- `aquaculture_system/config.py`: settings grouped by hardware, feeding policy, calibration, and storage.
- `aquaculture_system/hardware/`: camera capture and feeder relay workers.
- `aquaculture_system/ai/detector.py`: pellet detection and annotated images.
- `aquaculture_system/utils/logger.py`: feeding history, CSV summaries, and console capture.
- `tests/`: mocked hardware regression tests and application lifecycle checks.

To format the code consistently:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m black aquaculture_system tests
```

Runtime dependencies remain in `aquaculture_system/requirements.txt`; the formatter
is an optional development dependency. Reserved settings in `config.py` are kept
for compatibility and are explicitly marked as unused.
