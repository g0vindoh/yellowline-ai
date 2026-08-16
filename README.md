# YellowLine AI v8.0 — Platform Safety System

Real-time crowd safety monitoring for metro station platforms.
YOLOv8 detection · ByteTrack tracking · Fuzzy logic risk scoring · Telegram alerts · Multi-camera mesh

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
git clone <your-repo>
cd yellowline_v8
pip install -r requirements.txt

# Configure
cp .env.template .env
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
├── requirements.txt
├── .env.template
└── snapshots/               # Incident snapshots (auto-purged)
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

2. **Cross-platform deployment**: Removed all Windows-only dependencies (PowerShell TTS, winsound). Same codebase runs on a ₹7,000 Raspberry Pi 5 as on a laptop.

3. **Privacy-first**: Silhouette mode + auto-purge means the system never stores faces and auto-deletes after 24 hours — directly addresses BMRCL's surveillance concerns.

4. **Real historical data**: TimescaleDB-compatible metrics table means you can show a 6-hour risk graph, not just live numbers.

5. **Scales to a real station**: The coordinator architecture means adding a second camera is one config line, not a rewrite.
