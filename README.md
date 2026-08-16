# YellowLine AI v8.0 — Platform Safety System

Real-time crowd safety monitoring for metro station platforms.
YOLOv8 detection · ByteTrack tracking · Fuzzy logic risk scoring · Telegram alerts · Multi-camera mesh

**It doesn't just count people near the yellow line — it predicts who's about to cross it, detects when someone falls, and raises an alarm when a tracked person vanishes at the platform edge.**

---

## Detection capabilities

### Predictive edge crossing — alerts *before* the line is crossed
`predict_trajectory()` extrapolates each person's motion vector forward over `PREDICT_FRAMES` and tests whether the projected path intersects the yellow-line zone. If it will, and the person is already within `APPROACH_THRESH` of the edge, the track is flagged **PREDICTED** and rendered with a live dashed projection path. This buys seconds of warning instead of reacting after the fact.

### Multi-signal fall detection — six independent cues, weighted
`check_fall()` scores every tracked person 0–100 by combining:

| Signal | Weight | What it measures |
|---|---|---|
| `was_standing` | 18 | Person was upright before the event (rules out people already sitting) |
| `horizontal_now` | 22 | Bounding-box aspect ratio flipped to horizontal |
| `sudden_drop` | 18 | Centroid dropped ≥18% of the person's own standing height |
| `height_collapse` | 14 | Box height collapsed to ≤68% of the tracked standing height |
| `immobile` | 12 | Movement span stayed under `FALL_IMMOBILE_PX` after the drop |
| `down_confirm` | 10 | Stayed horizontal across consecutive frames |
| zone bonus | +8 / +4 | In the edge zone / buffer zone |
| pose score | +30 max | Skeletal confirmation (below) |

Crossing the threshold isn't enough — a fall is only *confirmed* through a multi-condition gate requiring prior upright posture, sustained horizontal state, **and** a drop or collapse. That's what keeps someone crouching to tie a shoelace from paging the station controller. The threshold auto-tightens near the tracks (`FALL_SCORE_EDGE=62` vs `FALL_SCORE_ALERT=70`), so the system is deliberately more sensitive where a fall is most dangerous.

### Skeletal pose confirmation
`pose_fall_score()` reads COCO keypoints and computes the true torso angle from the shoulder–hip vector, then checks body compaction, head-below-torso, and leg flatness. It's a second, independent opinion on the bounding-box verdict — so a person lying down is distinguished from a person merely detected in a wide box.

### Track-loss at the edge — the highest-stakes check
`check_edge_loss()` watches for a track that was inside the edge zone and then **disappears entirely** within `EDGE_LOSS_FRAMES`. On a platform, a person vanishing at the edge has one likely explanation: they've gone over it. Per-track cooldowns prevent alert storms from ordinary tracker dropouts.

### Crowd surge detection
`check_surge()` looks at aggregate motion, not individuals. It collects velocity vectors moving toward the platform edge, then checks angular alignment — when `SURGE_MIN_IDS` people move edge-ward within `SURGE_ANGLE_DEG` of each other, that's a coordinated push, not ordinary milling about.

### Fuzzy risk inference
A scikit-fuzzy Mamdani system (`fuzzy_logic_controller.py`) maps three antecedents — `in_zone`, `dwell`, `in_buf` — through **15 rules** to a continuous 0–100 risk score. Fall, surge, and edge-loss events feed in as additional inputs. Fuzzy logic means risk degrades smoothly instead of flickering at hard thresholds.

### ML crowd forecasting
Three `GradientBoostingRegressor` models (120 estimators, depth 4) predict density at **5, 15, and 30-minute** horizons from 14 engineered features — cyclical time encoding (`sin_hour`/`cos_hour`), peak/weekend flags, minutes-to-next-train, and rolling 5-frame windows of edge count and risk. Models retrain online as the platform accumulates data.

Measured performance (`crowd_features.json`):

| Horizon | MAE | R² |
|---|---|---|
| 5 min | 1.291 | 0.561 |
| 15 min | 1.320 | 0.546 |
| 30 min | 1.329 | 0.540 |

### Live operations dashboard
WebSocket push (`flask_socketio`) streams state to the browser — no polling. Includes a cumulative crowd-density **heatmap**, per-track velocity arrows, fading motion trails colour-coded by zone, and a live MJPEG feed.

---

## What's new in v8 (vs v7)

| Area | v7 | v8 |
|---|---|---|
| TTS / announcements | Windows-only PowerShell/SAPI | pyttsx3 + espeak fallback (works on RPi, Ubuntu, macOS, Windows) |
| Buzzer | winsound (Windows only) | sounddevice → simpleaudio → aplay chain |
| Metro boarding suppression | Schedule AND HSV colour gate (almost never fired) | Schedule-primary gate — always fires when train due |
| Risk score | FIS ignores fall/surge events | FIS receives fall/surge/edge_loss as inputs |
| Fuzzy rules | 4 rules, 2 variables | 15 rules, 3 variables (in_zone, dwell, in_buf) |
| Privacy | None | Silhouette mode — `YL_PRIVACY_MODE=1` |
| Snapshot retention | Permanent | Auto-purge after `YL_SNAP_RETENTION_HOURS` (default 24h) |
| Database | Events + sessions tables | + `metrics` time-series table |
| API | 6 endpoints | + `/api/history`, `/api/audit`, `/api/cameras`, `/api/crowd`, `/api/privacy` |
| Auth | Single operator token | Operator token + read-only audit token |
| Demo mode | Manual source edit | `YL_DEMO_LOOP=1` loops any video file |
| ToF sensor | Not supported | Optional VL53L1X depth sensor (RPi I2C) |
| Multi-camera | Not supported | `station_coordinator.py` — polls nodes, stitches track IDs |
| Crowd prediction | Single horizon, Windows only | 5/15/30-min GradientBoosting, online retraining |

---

## Quick start

```bash
git clone https://github.com/g0vindoh/yellowline-ai.git
cd yellowline-ai
pip install -r requirements.txt

# Configure
cp .env.example .env
nano .env   # fill in tokens, DB credentials, camera source

# Run (single camera)
python server.py

# Run (multi-camera coordinator, separate terminal)
YL_NODES=http://192.168.1.10:5000,http://192.168.1.11:5000 python station_coordinator.py
```

Open `http://localhost:5000` for the dashboard.

---

## Demo mode (no live camera needed)

Record ~3 minutes of metro platform footage on your phone, then:

```bash
YL_SOURCE=/path/to/platform_demo.mp4 YL_DEMO_LOOP=1 python server.py
```

The engine loops the file indefinitely. Ideal for ELP demo day.

---

## File structure

```
yellowline_v8/
├── core.py                  # Detection engine (main loop)
├── server.py                # Flask web server + all API endpoints
├── fuzzy_logic_controller.py # FIS risk scorer (15 rules, 3 variables)
├── metro_sync_monitor.py    # Train arrival / boarding window logic
├── crowd_prediction_ai.py   # ML crowd forecasting (5/15/30 min)
├── station_coordinator.py   # Multi-camera coordinator (NEW)
├── audio_backend.py         # Cross-platform TTS + buzzer chain
├── storage.py               # DB layer / metrics persistence
├── tof_sensor.py            # Optional VL53L1X depth sensor (RPi I2C)
├── requirements.txt
├── .env.example             # Config template — copy to .env
└── snapshots/               # Incident snapshots (auto-purged, gitignored)
```

---

## API reference

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/` | GET | — | Dashboard HTML |
| `/healthz` | GET | — | Health check (200/503) |
| `/video` | GET | operator | MJPEG live stream |
| `/api/stats` | GET | any | Full system state JSON |
| `/api/control` | POST | operator | Change zones, pause, heatmap, privacy |
| `/api/report` | GET | any | Session summary report |
| `/api/history` | GET | any | Time-series metrics (`?minutes=60`) |
| `/api/audit` | GET | any | Event log by date (`?date=YYYY-MM-DD`) |
| `/api/cameras` | GET | any | Multi-camera node list + states |
| `/api/cameras/register` | POST | operator | Register a secondary camera node |
| `/api/crowd` | GET | any | Crowd density forecast |
| `/api/privacy` | POST | operator | Toggle silhouette mode |
| `/snapshots/<file>` | GET | any | Serve incident snapshot image |

---

## Hardware upgrade path

### Minimum (laptop demo)
- Any webcam or phone IP camera
- Python 3.10+, ~4GB RAM

### Recommended (RPi deployment)
- Raspberry Pi 5 (8GB) per camera zone
- PoE IP camera (RTSP stream)
- VL53L1X ToF sensor on I2C for exact depth confirmation
- PA speaker for TTS announcements

### Multi-camera station
- One RPi per camera (or one GPU server for all)
- All nodes on same LAN
- Station coordinator runs on a central server or the strongest RPi
- `station_coordinator.py` polls all nodes, stitches IDs, sends merged state to ops room

---

## Privacy & compliance notes

- **No face recognition** — YOLOv8 detects `person` bounding boxes only; no face encoding.
- **Silhouette mode** (`YL_PRIVACY_MODE=1`) — OpenCV stylization removes facial detail from stored frames. Detection runs on raw feed; storage is clean.
- **Snapshot auto-purge** — `YL_SNAP_RETENTION_HOURS=24` deletes incident images after 24 hours.
- **Audit log** — `/api/audit?date=YYYY-MM-DD` returns the full event log for any date. Designed for post-incident review.
- **Role-based access** — `YL_API_TOKEN` (operator, full control) + `YL_AUDIT_TOKEN` (read-only, audit endpoints only).

---

## ELP presentation talking points

1. **The v7 bug we found and fixed**: Metro boarding suppression was gated on HSV colour detection of the train body — a camera pointing along the platform would never see the train. Now schedule-based primary, visual as additive confidence. False positive rate during boarding events drops to near-zero.

2. **Cross-platform deployment**: Removed all Windows-only dependencies (PowerShell TTS, winsound).

3. **Privacy-first**: Silhouette mode + auto-purge means the system never stores faces and auto-deletes after 24 hours — directly addresses BMRCL's surveillance concerns.

4. **Real historical data**: TimescaleDB-compatible metrics table means you can show a 6-hour risk graph, not just live numbers.

5. **Scales to a real station**: The coordinator architecture means adding a second camera is one config line, not a rewrite.

6. **We predict, we don't just react**: Trajectory extrapolation flags a person heading for the yellow line before they reach it. A system that alarms only *after* the breach has already lost the seconds that matter.

7. **Engineered against false alarms**: A station controller who gets three false alarms stops trusting the system entirely, so fall detection requires six weighted signals plus a multi-condition confirmation gate — not a single aspect-ratio flip. Detection thresholds tighten automatically inside the edge zone, so sensitivity is highest exactly where a fall is most dangerous.

8. **The check nobody else makes**: Track-loss detection. If a tracked person disappears at the platform edge, that is treated as a potential fall onto the track — catching the one event that matters most, precisely when the camera can no longer see it.
