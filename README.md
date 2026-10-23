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

Camera credentials come from `HIKVISION_CAMERA_USER` and
`HIKVISION_CAMERA_PASSWORD`; no password is stored in source. The camera password
is required even in dry-run mode because that mode still uses the camera.
Certificate verification is enabled. For a camera using a private CA or a
self-signed certificate, set `HIKVISION_CAMERA_CA_BUNDLE` to an absolute PEM bundle
path containing the trusted certificate. The certificate must match the host/IP
configured with `HIKVISION_CAMERA_IP`.

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

- Failed preset moves, captures, invalid images, inference, or annotation writes
  abort the session. Each capture finishes before the next preset move.
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
  camera and inference workers. Session errors terminate the process, requiring
  operator review and restart.
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
