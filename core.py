"""
YellowLine AI  —  Core Engine  v8.0
────────────────────────────────────────────────────────────────────────────────
Changes from v7:

1. CROSS-PLATFORM TTS  — removes Windows-only PowerShell/SAPI dependency.
   Uses pyttsx3 (offline, cross-platform) with pre-rendered WAV fallback for
   fixed-phrase PA announcements. Works on Raspberry Pi, Ubuntu, macOS, Windows.

2. FIXED METRO BOARDING SUPPRESSION  — v7 required both schedule AND HSV colour
   detection to suppress alerts (almost never triggered). v8 uses schedule alone
   as the primary gate, with HSV and crowd density surge as additive signals via
   metro_sync_monitor.check_boarding().

3. CROSS-PLATFORM BUZZER  — replaces winsound with sounddevice/simpleaudio,
   falling back to beep via subprocess on any OS.

4. ToF DEPTH SENSOR SUPPORT  — optional VL53L5CX/VL53L1X I2C sensor integration.
   When present, adds exact depth reading to risk computation (hard confirmation).
   Runs in a background thread; degrades gracefully when hardware absent.

5. SILHOUETTE PRIVACY MODE  — YL_PRIVACY_MODE=1 applies OpenCV stylization to
   frames before JPEG encoding/storing. Detection runs on raw; storage is clean.

6. SNAPSHOT AUTO-PURGE  — background thread deletes snapshots older than
   YL_SNAP_RETENTION_HOURS (default 24) to comply with CCTV retention policies.

7. TIMESERIES METRICS TABLE  — in addition to events, logs a metrics row every
   YL_METRICS_INTERVAL seconds for historical dashboard queries.

8. CROWD AI INTEGRATION  — calls crowd_prediction_ai.record() every frame and
   writes 5/15/30-min predictions into shared state for dashboard display.

9. RISK FUNCTION  — now passes fall/surge/edge_loss booleans into fuzzy
   controller so they influence the output score, not just the status bar.

10. DEMO / SIMULATION MODE  — YL_SOURCE can be a local video file path.
    The engine loops the file automatically for unattended demos.
"""
from dotenv import load_dotenv
load_dotenv()
import cv2
import sys
import json
import time
import wave
import math
import os
import threading
import subprocess
import collections
import glob
import numpy as np
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import fuzzy_logic_controller as flc
import metro_sync_monitor as msm
import crowd_prediction_ai as cpa
import storage
import tof_sensor
import audio_backend

# ── env helpers ───────────────────────────────────────────────────────────────
def env_float(name, default, lo=None, hi=None):
    try:    v = float(os.environ.get(name, default))
    except: v = float(default)
    if lo is not None: v = max(lo, v)
    if hi is not None: v = min(hi, v)
    return v

def env_int(name, default, lo=None, hi=None):
    try:    v = int(os.environ.get(name, default))
    except: v = int(default)
    if lo is not None: v = max(lo, v)
    if hi is not None: v = min(hi, v)
    return v

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    import requests as req_lib
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

try:
    import mysql.connector
    MYSQL_OK = True
except ImportError:
    MYSQL_OK = False
# Audio deps now handled by audio_backend module

# ── CONFIG ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN    = os.environ.get("YL_TG_TOKEN",  "")
TELEGRAM_CHAT_ID  = os.environ.get("YL_TG_CHAT",   "")
SNAPSHOT_DIR      = os.environ.get("YL_SNAP_DIR",  "snapshots")
SNAP_RETENTION_H  = env_int("YL_SNAP_RETENTION_HOURS", 24, 1, 720)
DB_HOST           = os.environ.get("YL_DB_HOST",   "localhost")
DB_USER           = os.environ.get("YL_DB_USER",   "root")
DB_PASS           = os.environ.get("YL_DB_PASS",   "")
DB_NAME           = os.environ.get("YL_DB_NAME",   "yellowline")
METRICS_INTERVAL  = env_int("YL_METRICS_INTERVAL", 10, 5, 300)

MODEL_PATH        = os.environ.get("YL_MODEL",      "yolov8n.pt")
POSE_MODEL_PATH   = os.environ.get("YL_POSE_MODEL", "").strip()
CAMERA_SOURCE     = os.environ.get("YL_SOURCE",     "0")
CONF_THRESH       = env_float("YL_CONF", 0.4, 0.05, 0.95)
CAM_W             = env_int("YL_CAM_W", 1280, 320, 3840)
CAM_H             = env_int("YL_CAM_H", 720,  240, 2160)

ZONE_EDGE_RATIO   = env_float("YL_ZONE_EDGE",   0.20, 0.05, 0.45)
ZONE_BUFFER_RATIO = env_float("YL_ZONE_BUFFER", 0.38, 0.10, 0.70)
DWELL_ALERT_SEC   = env_float("YL_DWELL_SEC",   5.0,  1.0,  60.0)
TRAIL_LENGTH      = 30
VELOCITY_FRAMES   = 8
PREDICT_FRAMES    = 20
APPROACH_THRESH   = 60
HEATMAP_ALPHA     = 0.30
HEATMAP_DECAY     = 0.97
SKIP_FRAME_FPS    = 15.0

FALL_ASPECT_NORMAL  = 0.9
FALL_ASPECT_FALLEN  = 0.6
FALL_CONFIRM_FRAMES = 3
FALL_HISTORY_FRAMES = env_int("YL_FALL_HISTORY_FRAMES", 30, 10, 120)
FALL_DOWN_FRAMES    = env_int("YL_FALL_DOWN_FRAMES",     6,  3,  60)
FALL_STILL_FRAMES   = env_int("YL_FALL_STILL_FRAMES",    8,  3,  60)
FALL_SCORE_ALERT    = env_int("YL_FALL_SCORE_ALERT",    70, 40, 100)
FALL_SCORE_EDGE     = env_int("YL_FALL_SCORE_EDGE",     62, 40, 100)
FALL_DROP_PX        = env_float("YL_FALL_DROP_PX",   35.0, 5.0, 250.0)
FALL_IMMOBILE_PX    = env_float("YL_FALL_IMMOBILE_PX",18.0, 3.0, 100.0)

SURGE_MIN_IDS    = 4
SURGE_ANGLE_DEG  = 40
EDGE_LOSS_FRAMES = 8

SURGE_ALERT_COOLDOWN = env_float("YL_SURGE_COOLDOWN",      20.0, 3.0, 300.0)
EDGE_LOSS_COOLDOWN   = env_float("YL_EDGE_LOSS_COOLDOWN",   30.0, 3.0, 300.0)
TRACKER_TTL_FRAMES   = env_int("YL_TRACKER_TTL_FRAMES",    300,  30,  5000)
CAMERA_RETRY_SEC     = env_float("YL_CAMERA_RETRY_SEC",     2.0, 0.5,  30.0)
AUTO_PAUSE_ENABLED   = os.environ.get("YL_AUTO_PAUSE_ENABLED","1").lower() not in ("0","false","no","off")
AUTO_PAUSE_INTERVAL  = env_float("YL_AUTO_PAUSE_INTERVAL_SEC", 300.0, 30.0, 3600.0)
AUTO_PAUSE_DURATION  = env_float("YL_AUTO_PAUSE_DURATION_SEC",  10.0,  3.0,  300.0)

# NEW v8 flags
PRIVACY_MODE      = os.environ.get("YL_PRIVACY_MODE", "0").lower() in ("1","true","yes","on")
TOF_ENABLED       = os.environ.get("YL_TOF_ENABLED",  "0").lower() in ("1","true","yes","on")
TOF_DANGER_MM     = env_int("YL_TOF_DANGER_MM", 500, 100, 2000)   # <500mm from edge = danger
DEMO_LOOP         = os.environ.get("YL_DEMO_LOOP", "0").lower() in ("1","true","yes","on")

# Multilingual announcement phrases
_ANNOUNCE_PHRASES = {
    "online":      {"en": "YellowLine system online.", "kn": "ಯೆಲ್ಲೋಲೈನ್ ಸಿಸ್ಟಮ್ ಆನ್‌ಲೈನ್."},
    "edge_enter":  {"en": "Attention. Passenger near platform edge. Step back immediately.",
                    "kn": "ಗಮನಿಸಿ. ಪ್ರಯಾಣಿಕರು ಪ್ಲಾಟ್‌ಫಾರ್ಮ್ ಅಂಚಿನ ಬಳಿ ಇದ್ದಾರೆ. ದಯವಿಟ್ಟು ಹಿಂದೆ ಸರಿಯಿರಿ."},
    "dwell_crit":  {"en": "Critical alert. Passenger in danger zone. Immediate action required.",
                    "kn": "ತುರ್ತು ಎಚ್ಚರಿಕೆ. ಪ್ರಯಾಣಿಕರು ಅಪಾಯದ ವಲಯದಲ್ಲಿದ್ದಾರೆ."},
    "approach":    {"en": "Warning. Passenger approaching platform edge.",
                    "kn": "ಎಚ್ಚರಿಕೆ. ಪ್ರಯಾಣಿಕರು ಅಂಚಿನತ್ತ ಹೊರಟಿದ್ದಾರೆ."},
    "fall":        {"en": "Emergency. Passenger has fallen. Immediate response required.",
                    "kn": "ತುರ್ತು ಪರಿಸ್ಥಿತಿ. ಪ್ರಯಾಣಿಕರು ಬಿದ್ದಿದ್ದಾರೆ."},
    "edge_loss":   {"en": "Emergency. Passenger may have fallen from platform.",
                    "kn": "ತುರ್ತು. ಪ್ರಯಾಣಿಕರು ಪ್ಲಾಟ್‌ಫಾರ್ಮ್‌ನಿಂದ ಬಿದ್ದಿರಬಹುದು."},
    "surge":       {"en": "Warning. Crowd surge detected. All passengers stand clear of platform edge.",
                    "kn": "ಎಚ್ಚರಿಕೆ. ಜನಸಂದಣಿ ಉಲ್ಬಣ. ಎಲ್ಲರೂ ಅಂಚಿನಿಂದ ದೂರ ನಿಲ್ಲಿ."},
}
ANNOUNCE_LANG = os.environ.get("YL_LANG", "en")   # "en" or "kn"

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ── SHARED STATE ──────────────────────────────────────────────────────────────
state_lock = threading.Lock()

state = {
    "total": 0, "in_edge": 0, "in_buffer": 0,
    "risk": 0, "max_dwell": 0.0, "fps": 0.0, "uptime": 0.0,
    "status": "SAFE", "buzzer": False, "pred_warn": False,
    "fall_alert": False, "surge_alert": False, "edge_loss": False,
    "peak_risk": 0, "incidents": 0, "alerts": 0, "falls": 0,
    "surges": 0, "snapshots": 0,
    "count_history": collections.deque([0]*600, maxlen=600),
    "risk_history":  collections.deque([0]*600, maxlen=600),
    "event_log":     collections.deque(maxlen=100),
    "all_events":    [],
    "frame_bytes":   None,
    "incidents_feed": collections.deque(maxlen=20),
    "show_heatmap":  True,
    "paused": False, "manual_paused": False,
    "pause_reason": "",
    "auto_pause_enabled": AUTO_PAUSE_ENABLED,
    "auto_pause_interval": AUTO_PAUSE_INTERVAL,
    "auto_pause_duration": AUTO_PAUSE_DURATION,
    "auto_pause_remaining": 0.0,
    "next_auto_pause_in": AUTO_PAUSE_INTERVAL,
    "zone_edge":   ZONE_EDGE_RATIO,
    "zone_buffer": ZONE_BUFFER_RATIO,
    "session_start": time.time(),
    "online": False, "degraded": False,
    "engine_error": "", "last_frame_ts": None,
    # v8 additions
    "metro_boarding": False,
    "metro_confidence": 0,
    "metro_schedule_due": False,
    "metro_visual_seen": False,
    "tof_in_danger": False,
    "tof_dist_mm": -1,
    "privacy_mode": PRIVACY_MODE,
    "crowd_pred": {},          # {5min, 15min, 30min, surge_predicted}
    "fuzzy_detail": {},        # full breakdown from fuzzy controller
    "train_present_gpio": False,
    "camera_id": 0,
}

# ── Internal per-engine tracking ──────────────────────────────────────────────
_dwell_tracker    = {}
_trail_tracker    = {}
_alerted_ids      = set()
_predicted_ids    = set()
_person_aspect    = {}
_fall_history     = {}
_fall_scores      = {}
_fall_ids         = set()
_edge_presence    = {}
_surge_frames     = 0
_last_surge_alert = 0
_edge_loss_reported = {}
_last_seen_track  = {}
_heatmap_acc      = None
_last_results     = None
_last_pose_results = None
_frame_count      = 0

# ── DATABASE ──────────────────────────────────────────────────────────────────
_db_conn   = None
_db_cursor = None

def init_db():
    global _db_conn, _db_cursor
    if not MYSQL_OK:
        return
    try:
        _db_conn   = mysql.connector.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME)
        _db_cursor = _db_conn.cursor()
        _db_cursor.execute("""CREATE TABLE IF NOT EXISTS sessions (
            id INT AUTO_INCREMENT PRIMARY KEY, started_at DATETIME,
            duration_s FLOAT, peak_risk INT, incidents INT, alerts INT,
            falls INT, surges INT)""")
        _db_cursor.execute("""CREATE TABLE IF NOT EXISTS events (
            id INT AUTO_INCREMENT PRIMARY KEY, session_id INT,
            ts DATETIME, message TEXT)""")
        # v8: time-series metrics table for historical dashboard queries
        _db_cursor.execute("""CREATE TABLE IF NOT EXISTS metrics (
            id INT AUTO_INCREMENT PRIMARY KEY,
            ts DATETIME NOT NULL,
            in_edge INT, in_buffer INT, total INT,
            risk INT, max_dwell FLOAT,
            metro_boarding TINYINT, INDEX(ts))""")
        _db_conn.commit()
        print("[DB] MySQL connected (v8 schema).")
    except Exception as e:
        print(f"[DB] Not connected: {e}")
        _db_conn = None

_last_metrics_write = 0.0

def write_metrics_row(in_edge, in_buffer, total, risk, max_dwell, metro_boarding):
    global _last_metrics_write
    now = time.time()
    if now - _last_metrics_write < METRICS_INTERVAL:
        return
    _last_metrics_write = now
    with state_lock:
        cam_id = state.get("camera_id", "cam-1")
        fps    = state.get("fps", 0)
        tof    = dict(state.get("tof", {}))
    # TimescaleDB / PostgreSQL via storage module
    storage.write_metrics(cam_id, {
        "total": total, "in_edge": in_edge, "in_buffer": in_buffer,
        "risk": risk, "fps": fps, "max_dwell": round(max_dwell, 2), "tof": tof,
    })
    # MySQL fallback
    if _db_conn and _db_cursor:
        try:
            _db_cursor.execute(
                """INSERT INTO metrics (ts, in_edge, in_buffer, total, risk, max_dwell, metro_boarding)
                   VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                (datetime.now(), in_edge, in_buffer, total, risk, round(max_dwell,2), int(metro_boarding))
            )
            _db_conn.commit()
        except Exception:
            pass   # non-fatal


# ── SNAPSHOT AUTO-PURGE ───────────────────────────────────────────────────────
def _purge_old_snapshots():
    """Background thread: delete snapshots older than retention window."""
    while True:
        time.sleep(3600)   # run hourly
        cutoff = time.time() - SNAP_RETENTION_H * 3600
        count = 0
        for f in glob.glob(os.path.join(SNAPSHOT_DIR, "incident_*.jpg")):
            try:
                if os.path.getmtime(f) < cutoff:
                    os.remove(f)
                    count += 1
            except Exception:
                pass
        if count:
            print(f"[Purge] Removed {count} snapshot(s) older than {SNAP_RETENTION_H}h")

threading.Thread(target=_purge_old_snapshots, daemon=True).start()


# ── ToF DEPTH SENSOR — delegated to tof_sensor module ────────────────────────
# tof_sensor.start() is called from engine_loop after init.
# Use tof_sensor.snapshot() everywhere to read current state.
# Supports mock mode: YL_TOF_MOCK=1 for demos without hardware.


# ── TELEGRAM ──────────────────────────────────────────────────────────────────
_tg_last = 0
TG_COOLDOWN = 15.0

def telegram_send(text, photo_path=None):
    global _tg_last
    if not REQUESTS_OK or not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    now = time.time()
    if now - _tg_last < TG_COOLDOWN:
        return
    _tg_last = now
    def _send():
        base = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
        try:
            if photo_path and os.path.exists(photo_path):
                with open(photo_path,"rb") as f:
                    req_lib.post(f"{base}/sendPhoto", timeout=8,
                                 data={"chat_id": TELEGRAM_CHAT_ID, "caption": text},
                                 files={"photo": f})
            else:
                req_lib.post(f"{base}/sendMessage", timeout=8,
                             json={"chat_id": TELEGRAM_CHAT_ID, "text": text,
                                   "parse_mode": "HTML"})
        except Exception as e:
            print(f"[TG] Send error: {e}")
    threading.Thread(target=_send, daemon=True).start()


# ── INCIDENT SNAPSHOTS ────────────────────────────────────────────────────────
_snap_lock = threading.Lock()
_last_snap = 0
SNAP_COOLDOWN = 8.0

def _apply_privacy(frame):
    """Return a silhouette-only version of the frame (no face detail)."""
    if not PRIVACY_MODE:
        return frame
    try:
        # Stylization removes facial detail while keeping body silhouettes
        return cv2.stylization(frame, sigma_s=60, sigma_r=0.45)
    except Exception:
        # Fallback: heavy bilateral blur
        return cv2.bilateralFilter(frame, 15, 75, 75)

def save_snapshot(frame, label: str, risk: int):
    global _last_snap
    with _snap_lock:
        now = time.time()
        if now - _last_snap < SNAP_COOLDOWN:
            return None
        _last_snap = now

    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(SNAPSHOT_DIR, f"incident_{ts}.jpg")

    out = _apply_privacy(frame.copy())
    cv2.rectangle(out, (0,0),(out.shape[1],48),(20,10,80),-1)
    cv2.putText(out, f"YellowLine INCIDENT  |  {label}  |  Risk {risk}/100",
                (12,32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(out, datetime.now().strftime("%d %b %Y  %H:%M:%S"),
                (out.shape[1]-260,32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,200), 1, cv2.LINE_AA)
    if PRIVACY_MODE:
        cv2.putText(out, "SILHOUETTE MODE — No face data stored",
                    (12,out.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,200,100), 1, cv2.LINE_AA)
    cv2.imwrite(fname, out, [cv2.IMWRITE_JPEG_QUALITY, 88])

    with state_lock:
        state["snapshots"] += 1
        state["incidents_feed"].appendleft({
            "time":  datetime.now().strftime("%H:%M:%S"),
            "label": label, "risk": risk,
            "file":  os.path.basename(fname),
        })
    telegram_send(
        f"🚨 <b>YellowLine INCIDENT</b>\n<b>Type:</b> {label}\n"
        f"<b>Risk:</b> {risk}/100\n<b>Time:</b> {datetime.now().strftime('%H:%M:%S')}",
        photo_path=fname)
    return fname


# ── VOICE / TTS  — delegated to audio_backend ────────────────────────────────
_last_voice = 0
VOICE_CD    = 6.0

def _phrase(key: str) -> str:
    phrases = _ANNOUNCE_PHRASES.get(key, {})
    return phrases.get(ANNOUNCE_LANG, phrases.get("en", key))

def announce(msg_or_key: str, cooldown: float = None):
    global _last_voice
    cd  = cooldown if cooldown is not None else VOICE_CD
    now = time.time()
    if now - _last_voice < cd:
        return
    _last_voice = now
    text = _phrase(msg_or_key) if msg_or_key in _ANNOUNCE_PHRASES else msg_or_key
    # Map phrase key for audio_backend WAV lookup
    phrase_key = msg_or_key if msg_or_key in _ANNOUNCE_PHRASES else None
    audio_backend.speak_text(text, phrase_key=phrase_key)


# ── BUZZER — delegated to audio_backend ──────────────────────────────────────
ALARM_FILE = "alarm.wav"
audio_backend.make_alarm(ALARM_FILE)   # generate if missing
_last_buzz  = 0
_buzz_lock  = threading.Lock()
BUZZ_CD     = 2.0

def trigger_buzzer():
    global _last_buzz
    with _buzz_lock:
        now = time.time()
        if now - _last_buzz < BUZZ_CD:
            return
        _last_buzz = now
    audio_backend.play_buzzer(ALARM_FILE)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def log_event(msg):
    ts = time.strftime("%H:%M:%S")
    e  = f"{ts}  {msg}"
    with state_lock:
        state["event_log"].appendleft(e)
        state["all_events"].append(e)
        cam_id = state.get("camera_id", "cam-1")
    storage.write_event(cam_id, msg)
    if _db_conn and _db_cursor:
        try:
            _db_cursor.execute(
                "INSERT INTO events (session_id, ts, message) VALUES (%s,%s,%s)",
                (None, datetime.now(), msg))
            _db_conn.commit()
        except Exception:
            pass

def status_label(score):
    if score < 30:   return "SAFE"
    elif score < 65: return "CAUTION"
    else:            return "CRITICAL"

def status_color_bgr(score):
    if score < 30:   return (70, 200,  90)
    elif score < 65: return (30, 165, 235)
    else:            return (55,  55, 225)

def compute_platform_risk(in_edge, in_buffer, max_dwell, fall, surge, edge_loss):
    """
    v8: passes fall/surge/edge_loss into fuzzy controller for proper scoring.
    Stores full breakdown in state["fuzzy_detail"] for dashboard display.
    Also hardens score if ToF sensor confirms danger proximity.
    """
    detail = flc.get_risk_detail(in_edge, in_buffer, max_dwell, fall, surge, edge_loss)
    score  = detail["score"]

    tof = tof_sensor.snapshot()
    tof_adjusted = False
    if tof.get("enabled") and tof.get("available"):
        if tof.get("tof_in_danger"):
            score = max(score, 70)
            tof_adjusted = True
        elif score >= 65 and not (fall or surge or edge_loss):
            score = 64
            tof_adjusted = True

    detail["score"]        = score
    detail["tof_adjusted"] = tof_adjusted

    with state_lock:
        state["fuzzy_detail"] = detail

    return score

def format_uptime(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── DRAWING PRIMITIVES ────────────────────────────────────────────────────────
def blend_rect(frame, x1, y1, x2, y2, color, alpha):
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(frame.shape[1],x2), min(frame.shape[0],y2)
    if x2<=x1 or y2<=y1: return
    roi = frame[y1:y2, x1:x2]
    ov  = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(ov, alpha, roi, 1-alpha, 0, roi)
    frame[y1:y2, x1:x2] = roi

def draw_filled_rounded(frame, x1, y1, x2, y2, color, r=8):
    cv2.rectangle(frame,(x1+r,y1),(x2-r,y2),color,-1)
    cv2.rectangle(frame,(x1,y1+r),(x2,y2-r),color,-1)
    for cx_,cy_,a in [(x1+r,y1+r,180),(x2-r,y1+r,270),(x1+r,y2-r,90),(x2-r,y2-r,0)]:
        cv2.ellipse(frame,(cx_,cy_),(r,r),a,0,90,color,-1)

def draw_rounded_rect(frame, x1, y1, x2, y2, color, r=8, t=1):
    cv2.line(frame,(x1+r,y1),(x2-r,y1),color,t)
    cv2.line(frame,(x1+r,y2),(x2-r,y2),color,t)
    cv2.line(frame,(x1,y1+r),(x1,y2-r),color,t)
    cv2.line(frame,(x2,y1+r),(x2,y2-r),color,t)
    for cx_,cy_,a in [(x1+r,y1+r,180),(x2-r,y1+r,270),(x1+r,y2-r,90),(x2-r,y2-r,0)]:
        cv2.ellipse(frame,(cx_,cy_),(r,r),a,0,90,color,t)

def lbl(frame, text, x, y, size=0.38, color=(180,180,188), bold=False):
    cv2.putText(frame, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                size, color, 2 if bold else 1, cv2.LINE_AA)

def micro(frame, text, x, y, color=(110,112,122)):
    cv2.putText(frame, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                0.30, color, 1, cv2.LINE_AA)


# ── HEATMAP ───────────────────────────────────────────────────────────────────
def update_heatmap(acc, detections):
    if not detections: return
    dots = np.zeros_like(acc)
    for cx, cy in detections:
        if 0 <= cy < dots.shape[0] and 0 <= cx < dots.shape[1]:
            dots[cy, cx] += 6.0
    acc += cv2.GaussianBlur(dots, (71,71), 25)

def render_heatmap(frame, acc, dw, h):
    region = acc[:h, :dw]
    if region.max() < 1: return
    normed  = np.clip(region / region.max(), 0, 1)
    colored = cv2.applyColorMap((normed*255).astype(np.uint8), cv2.COLORMAP_JET)
    mask    = (normed > 0.08).astype(np.float32)
    for c in range(3):
        frame[:h,:dw,c] = np.clip(
            frame[:h,:dw,c]*(1-mask*HEATMAP_ALPHA) +
            colored[:,:,c]*mask*HEATMAP_ALPHA, 0, 255).astype(np.uint8)


# ── TRAIL + VELOCITY ──────────────────────────────────────────────────────────
def draw_trail(frame, trail, in_edge, in_buffer):
    pts = list(trail)
    col = (55,55,225) if in_edge else ((30,210,230) if in_buffer else (70,200,90))
    for i in range(1, len(pts)):
        a = i/len(pts)
        cv2.line(frame, pts[i-1], pts[i], tuple(int(c*a*0.8) for c in col),
                 max(1,int(a*2)), cv2.LINE_AA)

def get_velocity(trail):
    pts = list(trail)
    if len(pts) < VELOCITY_FRAMES: return 0.0,0.0,0.0,False
    recent = pts[-VELOCITY_FRAMES:]
    dx = (recent[-1][0]-recent[0][0])/VELOCITY_FRAMES
    dy = (recent[-1][1]-recent[0][1])/VELOCITY_FRAMES
    sp = math.sqrt(dx*dx+dy*dy)
    return dx, dy, sp, dx < -0.5

def draw_velocity_arrow(frame, trail, zone_x, cx, cy):
    dx, dy, sp, toward = get_velocity(trail)
    if sp < 3: return
    scale = min(sp*1.8, 50)
    nx, ny = dx/sp, dy/sp
    ex, ey = int(cx+nx*scale), int(cy+ny*scale)
    col = (55,55,225) if (toward and cx < zone_x+50) else (70,200,90)
    cv2.arrowedLine(frame,(cx,cy),(ex,ey),col,2,tipLength=0.3,line_type=cv2.LINE_AA)


# ── TRAJECTORY PREDICTION ─────────────────────────────────────────────────────
def predict_trajectory(frame, trail, zone_x, cx, cy):
    pts = list(trail)
    if len(pts) < VELOCITY_FRAMES: return False
    recent = pts[-VELOCITY_FRAMES:]
    dx = (recent[-1][0]-recent[0][0])/VELOCITY_FRAMES
    dy = (recent[-1][1]-recent[0][1])/VELOCITY_FRAMES
    if dx >= 0: return False
    pred = [(int(cx+dx*f), int(cy+dy*f)) for f in range(1, PREDICT_FRAMES+1)]
    for i in range(1, len(pred)):
        if i%2==0:
            cv2.line(frame, pred[i-1], pred[i], (30,130,255), 1, cv2.LINE_AA)
    if pred:
        cv2.circle(frame, pred[-1], 4, (30,130,255), -1, cv2.LINE_AA)
    will_enter   = any(p[0] < zone_x for p in pred)
    close_enough = cx < zone_x + APPROACH_THRESH
    if will_enter and close_enough:
        pad = 44
        draw_rounded_rect(frame, cx-pad, cy-pad, cx+pad, cy+pad, (30,130,255), r=6)
        lbl(frame, "PREDICTED", cx-32, cy-pad-6, size=0.3, color=(30,130,255))
        return True
    return False


# ── FALL DETECTION ────────────────────────────────────────────────────────────
def _median(values, default=0.0):
    vals = sorted(values)
    if not vals: return default
    mid = len(vals)//2
    return vals[mid] if len(vals)%2 else (vals[mid-1]+vals[mid])/2

def box_iou(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    if inter <= 0: return 0.0
    area_a = max(1,(ax2-ax1)*(ay2-ay1))
    area_b = max(1,(bx2-bx1)*(by2-by1))
    return inter/(area_a+area_b-inter)

def pose_fall_score(keypoints):
    if keypoints is None: return 0
    try:
        pts  = keypoints.xy[0].cpu().numpy()
        conf = keypoints.conf[0].cpu().numpy() if keypoints.conf is not None else np.ones(len(pts))
    except Exception:
        return 0
    def pt(idx, min_conf=0.25):
        if idx >= len(pts) or conf[idx] < min_conf: return None
        x,y = pts[idx]
        if x<=0 and y<=0: return None
        return float(x), float(y)
    nose = pt(0)
    shoulders = [p for p in (pt(5),pt(6)) if p]
    hips      = [p for p in (pt(11),pt(12)) if p]
    knees     = [p for p in (pt(13),pt(14)) if p]
    ankles    = [p for p in (pt(15),pt(16)) if p]
    if len(shoulders)<1 or len(hips)<1: return 0
    sx = sum(p[0] for p in shoulders)/len(shoulders)
    sy = sum(p[1] for p in shoulders)/len(shoulders)
    hx = sum(p[0] for p in hips)/len(hips)
    hy = sum(p[1] for p in hips)/len(hips)
    torso_angle     = abs(math.degrees(math.atan2(hy-sy, hx-sx)))
    horizontal_torso= torso_angle<35 or torso_angle>145
    compact_body    = abs(hy-sy) < max(28, abs(hx-sx)*0.65)
    head_low        = bool(nose and nose[1] > min(sy,hy)-10)
    leg_flat = False
    if knees and ankles:
        ky = sum(p[1] for p in knees)/len(knees)
        ay = sum(p[1] for p in ankles)/len(ankles)
        leg_flat = abs(ay-ky) < 35
    score = 0
    if horizontal_torso: score += 14
    if compact_body:     score += 8
    if head_low:         score += 5
    if leg_flat:         score += 3
    return min(score, 30)

def match_pose_score(pose_results, bbox):
    if pose_results is None or not hasattr(pose_results,"boxes") or pose_results.boxes is None:
        return 0
    best_iou, best_score = 0.0, 0
    keypoints = getattr(pose_results, "keypoints", None)
    for idx, pbox in enumerate(pose_results.boxes):
        try:
            pb = tuple(map(int, pbox.xyxy[0]))
        except Exception:
            continue
        iou = box_iou(bbox, pb)
        if iou > best_iou:
            kp = keypoints[idx] if keypoints is not None else None
            best_iou, best_score = iou, pose_fall_score(kp)
    return best_score if best_iou >= 0.25 else 0

def check_fall(tid, bw, bh, frame, cx, cy, in_edge=False, in_buffer=False, pose_score=0):
    if tid not in _person_aspect:
        _person_aspect[tid] = collections.deque(maxlen=FALL_HISTORY_FRAMES)
    if tid not in _fall_history:
        _fall_history[tid] = collections.deque(maxlen=FALL_HISTORY_FRAMES)
    ratio = bh / max(bw,1)
    _person_aspect[tid].append(ratio)
    _fall_history[tid].append({
        "frame": _frame_count, "ratio": ratio,
        "cx": cx, "cy": cy, "bw": bw, "bh": bh,
        "fallen_like": ratio < FALL_ASPECT_FALLEN,
    })
    hist = list(_fall_history[tid])
    if len(hist) < max(FALL_CONFIRM_FRAMES, 4):
        _fall_scores[tid] = 0
        return False
    prior  = hist[:-FALL_CONFIRM_FRAMES] or hist[:-1]
    recent = hist[-FALL_CONFIRM_FRAMES:]
    upright_ratios  = [o["ratio"] for o in prior if o["ratio"] > FALL_ASPECT_NORMAL]
    upright_heights = [o["bh"]    for o in prior if o["ratio"] > FALL_ASPECT_NORMAL]
    standing_height = _median(upright_heights, default=max(o["bh"] for o in hist))
    was_standing    = len(upright_ratios)>=2 or _median([o["ratio"] for o in prior],0)>FALL_ASPECT_NORMAL
    horizontal_now  = sum(1 for o in recent if o["fallen_like"]) >= max(2, FALL_CONFIRM_FRAMES-1)
    lookback        = hist[max(0, len(hist)-max(10, FALL_DOWN_FRAMES*2))]
    downward_drop   = cy - lookback["cy"]
    sudden_drop     = downward_drop >= max(FALL_DROP_PX, standing_height*0.18)
    height_collapse = bh <= standing_height*0.68
    still_hist      = hist[-min(len(hist), FALL_STILL_FRAMES):]
    immobile = False
    if len(still_hist) >= max(3, min(FALL_STILL_FRAMES,5)):
        xs = [o["cx"] for o in still_hist]
        ys = [o["cy"] for o in still_hist]
        move_span = math.sqrt((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2)
        immobile  = move_span <= FALL_IMMOBILE_PX
    down_confirm = (sum(1 for o in hist[-min(len(hist),FALL_DOWN_FRAMES):]
                        if o["fallen_like"]) >= max(3, FALL_DOWN_FRAMES-1))
    zone_bonus = 8 if in_edge else (4 if in_buffer else 0)
    score = 0
    if was_standing:      score += 18
    if horizontal_now:    score += 22
    if sudden_drop:       score += 18
    if height_collapse:   score += 14
    if immobile:          score += 12
    if down_confirm:      score += 10
    score += zone_bonus + min(30, int(pose_score))
    score  = min(score, 100)
    _fall_scores[tid] = score
    threshold = FALL_SCORE_EDGE if in_edge else FALL_SCORE_ALERT
    confirmed = (
        score >= threshold and
        (was_standing or pose_score>=20) and
        (down_confirm or (pose_score>=20 and horizontal_now)) and
        (sudden_drop or height_collapse or pose_score>=24)
    )
    if confirmed and tid not in _fall_ids:
        _fall_ids.add(tid)
        cv2.rectangle(frame,(cx-58,cy-58),(cx+58,cy+58),(0,0,255),3,cv2.LINE_AA)
        lbl(frame,f"FALL DETECTED {score}",cx-72,cy-64,size=0.45,color=(0,0,255),bold=True)
        return True
    if score >= max(45, threshold-15):
        cv2.rectangle(frame,(cx-44,cy-44),(cx+44,cy+44),(0,130,255),2,cv2.LINE_AA)
        lbl(frame,f"FALL RISK {score}",cx-54,cy-50,size=0.34,color=(0,130,255),bold=True)
    return False


# ── CROWD SURGE ───────────────────────────────────────────────────────────────
def check_surge(velocity_vectors):
    if len(velocity_vectors) < SURGE_MIN_IDS: return False
    toward_edge = [(dx,dy) for dx,dy in velocity_vectors
                   if dx<-1.5 and math.sqrt(dx*dx+dy*dy)>2]
    if len(toward_edge) < SURGE_MIN_IDS: return False
    angles = [math.atan2(dy,dx) for dx,dy in toward_edge]
    ref    = angles[0]
    thresh = math.radians(SURGE_ANGLE_DEG)
    aligned = sum(1 for a in angles if abs(a-ref) < thresh)
    return aligned >= SURGE_MIN_IDS


# ── TRACK-LOSS AT EDGE ────────────────────────────────────────────────────────
def update_edge_presence(tid, in_edge, current_frame):
    if in_edge: _edge_presence[tid] = current_frame
    elif tid in _edge_presence: del _edge_presence[tid]

def check_edge_loss(current_frame, active_ids):
    lost = set()
    for tid, last_frame in list(_edge_presence.items()):
        if tid not in active_ids:
            frames_since = current_frame - last_frame
            if 1 <= frames_since <= EDGE_LOSS_FRAMES:
                now = time.time()
                if now - _edge_loss_reported.get(tid,0) >= EDGE_LOSS_COOLDOWN:
                    lost.add(tid)
                    _edge_loss_reported[tid] = now
                del _edge_presence[tid]
            elif frames_since > EDGE_LOSS_FRAMES:
                del _edge_presence[tid]
    return lost

def cleanup_trackers(active_ids, current_frame):
    stale_ids = [tid for tid,seen in list(_last_seen_track.items())
                 if tid not in active_ids and current_frame-seen>TRACKER_TTL_FRAMES]
    for tid in stale_ids:
        for d in (_last_seen_track,_dwell_tracker,_trail_tracker,_person_aspect,
                  _fall_history,_fall_scores,_edge_presence,_edge_loss_reported):
            d.pop(tid, None)
        for s in (_alerted_ids,_predicted_ids,_fall_ids):
            s.discard(tid)


# ── MODEL LOADING ─────────────────────────────────────────────────────────────
model       = None
pose_model  = None
_model_lock = threading.Lock()
_pose_lock  = threading.Lock()

def get_model():
    global model
    with _model_lock:
        if model is None:
            print(f"[YellowLine] Loading model: {MODEL_PATH}")
            model = YOLO(MODEL_PATH)
            print("[YellowLine] Model ready.")
    return model

def get_pose_model():
    global pose_model
    if not POSE_MODEL_PATH: return None
    with _pose_lock:
        if pose_model is None:
            print(f"[YellowLine] Loading pose model: {POSE_MODEL_PATH}")
            pose_model = YOLO(POSE_MODEL_PATH)
    return pose_model

def build_socket_payload():
    with state_lock:
        payload = {k:v for k,v in state.items()
                   if k not in ("frame_bytes","count_history","risk_history",
                                "event_log","all_events","incidents_feed")}
        payload["event_log"]     = list(state["event_log"])[:8]
        payload["risk_history"]  = list(state["risk_history"])[-120:]
        payload["count_history"] = list(state["count_history"])[-120:]
        payload["persons"] = [
            {"id":tid,"cx":int(cx_),"cy":int(cy_),"bw":int(bw_),"bh":int(bh_),
             "zone":z_,"dwell":round(d_,2)}
            for tid,cx_,cy_,bw_,bh_,z_,d_ in state.get("person_list",[])
        ]
    return payload


# ── MAIN DETECTION ENGINE ─────────────────────────────────────────────────────
PANEL_W = 0

def engine_loop(headless=True, socketio=None):
    global _heatmap_acc, _last_results, _last_pose_results, _frame_count, _last_surge_alert

    init_db()
    storage.init()
    tof_sensor.start()
    try:
        detector      = get_model()
        pose_detector = get_pose_model()
    except Exception as exc:
        msg = f"Model load failed: {exc}"
        print(f"[Engine] ERROR: {msg}")
        with state_lock:
            state.update({"online":False,"degraded":True,"engine_error":msg})
        log_event(msg)
        return

    source = int(CAMERA_SOURCE) if CAMERA_SOURCE.isdigit() else CAMERA_SOURCE
    cap    = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    if not cap.isOpened():
        err = f"Cannot open source '{source}'"
        print(f"[Engine] ERROR: {err}")
        with state_lock:
            state.update({"online":False,"degraded":True,"engine_error":err})
        log_event(f"Camera offline: {source}")
        return

    prev_time     = time.time()
    flash_alpha   = 0.0
    session_start = time.time()
    next_auto_pause  = session_start + AUTO_PAUSE_INTERVAL
    auto_pause_until = 0.0
    auto_pause_was_active = False
    last_auto_enabled  = AUTO_PAUSE_ENABLED
    last_auto_interval = AUTO_PAUSE_INTERVAL
    last_auto_duration = AUTO_PAUSE_DURATION

    with state_lock:
        state["online"] = True
        state["session_start"] = session_start
        state["next_auto_pause_in"] = AUTO_PAUSE_INTERVAL

    log_event("System online v8.0")
    announce("online", cooldown=0)
    camera_failures = 0

    BG_PANEL = (18,18,22); C_WHITE = (240,240,245); C_GRAY2 = (110,112,122)
    S_GREEN  = (70,200,90); S_AMBER = (30,165,235)
    S_RED    = (55,55,225); S_YELLOW = (30,210,230)

    while True:
        now = time.time()
        with state_lock:
            manual_paused = state["manual_paused"]
            auto_enabled  = state["auto_pause_enabled"]
            auto_interval = state["auto_pause_interval"]
            auto_duration = state["auto_pause_duration"]
            sh_heatmap    = state["show_heatmap"]
            ze, zb        = state["zone_edge"], state["zone_buffer"]

        if (auto_enabled != last_auto_enabled or
                auto_interval != last_auto_interval or
                auto_duration != last_auto_duration):
            next_auto_pause = now + auto_interval
            if not auto_enabled: auto_pause_until = 0.0
            last_auto_enabled  = auto_enabled
            last_auto_interval = auto_interval
            last_auto_duration = auto_duration

        if auto_interval>0 and next_auto_pause < now-auto_interval:
            next_auto_pause = now + auto_interval
        if auto_enabled and now>=next_auto_pause and now>=auto_pause_until:
            auto_pause_until  = now + auto_duration
            next_auto_pause   = now + auto_interval
            auto_pause_was_active = True
            log_event(f"BOARDING WINDOW {auto_duration:.0f}s")

        auto_remaining = max(0.0, auto_pause_until - now)
        auto_active    = auto_enabled and auto_remaining > 0
        effective_paused = manual_paused or auto_active
        pause_reason   = "MANUAL" if manual_paused else ("BOARDING" if auto_active else "")

        if auto_pause_was_active and not auto_active:
            auto_pause_was_active = False
            log_event("Boarding window ended")

        with state_lock:
            state["paused"]             = effective_paused
            state["pause_reason"]       = pause_reason
            state["auto_pause_remaining"] = round(auto_remaining, 1)
            state["next_auto_pause_in"] = round(max(0.0, next_auto_pause-now), 1)

        if effective_paused:
            if socketio: socketio.emit("stats", build_socket_payload())
            time.sleep(0.05)
            continue

        ret, frame = cap.read()
        if not ret:
            camera_failures += 1
            with state_lock:
                state["degraded"]     = True
                state["engine_error"] = f"Camera read failed ({camera_failures})"
            if camera_failures == 1:
                log_event("Camera read failed; reconnecting")
            cap.release()
            time.sleep(CAMERA_RETRY_SEC)
            # Demo loop: restart video file from beginning
            cap = cv2.VideoCapture(source)
            if DEMO_LOOP and isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
            continue

        if camera_failures:
            log_event("Camera stream recovered")
        camera_failures = 0
        with state_lock:
            state["degraded"] = False
            state["engine_error"] = ""

        _frame_count += 1
        h, w   = frame.shape[:2]
        now    = time.time()
        uptime = now - session_start

        zone_edge_x   = int(w * ze)
        zone_buffer_x = int(w * zb)

        if _heatmap_acc is None:
            _heatmap_acc = np.zeros((h, w), dtype=np.float32)
        _heatmap_acc *= HEATMAP_DECAY

        # Metro boarding check — v8: schedule-based primary gate
        with state_lock:
            s_copy = dict(state)
        metro_boarding = msm.check_boarding(s_copy, frame if _frame_count%30==0 else None,
                                            in_edge=s_copy.get("in_edge", 0))
        with state_lock:
            state["metro_boarding"]    = s_copy.get("metro_boarding", False)
            state["metro_confidence"]  = s_copy.get("metro_confidence", 0)
            state["metro_schedule_due"]= s_copy.get("metro_schedule_due", False)
            state["metro_visual_seen"] = s_copy.get("metro_visual_seen", False)

        # Update ToF state into shared state
        _tof = tof_sensor.snapshot()
        with state_lock:
            state["tof_in_danger"] = _tof.get("tof_in_danger", False)
            state["tof_dist_mm"]   = _tof.get("min_dist_mm", -1)
            state["tof"]           = _tof

        # ── Zone rendering ────────────────────────────────────────────
        for xi in range(min(30, zone_edge_x)):
            blend_rect(frame, xi, 0, xi+1, h, [0,0,180], 0.20*(1-xi/30))
        blend_rect(frame, 0, 0, zone_edge_x, h, [0,0,150], 0.08)
        cv2.line(frame,(zone_edge_x,0),(zone_edge_x,h),(40,40,200),2,cv2.LINE_AA)
        blend_rect(frame, zone_edge_x, 0, zone_buffer_x, h, [0,100,200], 0.05)
        cv2.line(frame,(zone_buffer_x,0),(zone_buffer_x,h),(40,160,200),1,cv2.LINE_AA)

        blend_rect(frame,6,6,120,52,list(BG_PANEL),0.75)
        draw_rounded_rect(frame,6,6,120,52,(40,40,180),r=6)
        lbl(frame,"EDGE ZONE",14,24,size=0.34,color=(120,120,220),bold=True)
        micro(frame,"High risk",14,42,color=(80,80,160))
        blend_rect(frame,zone_edge_x+4,6,zone_edge_x+105,52,list(BG_PANEL),0.75)
        lbl(frame,"BUFFER",zone_edge_x+10,24,size=0.34,color=S_YELLOW)
        micro(frame,"Caution",zone_edge_x+10,42,color=(80,160,80))

        # ── Metro boarding overlay ───────────────────────────────────
        if metro_boarding:
            conf = s_copy.get("metro_confidence", 0)
            blend_rect(frame,0,h-90,w,h-44,list(BG_PANEL),0.85)
            lbl(frame,f"  METRO BOARDING WINDOW ACTIVE — Alert suppression ON  (conf {conf}%)",
                10,h-58,size=0.42,color=(30,210,230))

        # ── Privacy mode indicator ───────────────────────────────────
        if PRIVACY_MODE:
            micro(frame,"PRIVACY MODE — silhouette only",10,h-4,color=(100,200,100))

        # ── FPS-adaptive inference ────────────────────────────────────
        fps_now = 1.0/(now-prev_time+1e-6)
        skip    = (fps_now < SKIP_FRAME_FPS) and (_frame_count%2==0)
        if not skip:
            _last_results = detector.track(frame, classes=[0], conf=CONF_THRESH,
                                           persist=True, verbose=False)[0]
            if pose_detector is not None:
                _last_pose_results = pose_detector.predict(frame, classes=[0],
                                                           conf=CONF_THRESH, verbose=False)[0]
        results      = _last_results
        pose_results = _last_pose_results

        heatmap_pts   = []
        velocity_vecs = []
        active_ids    = set()

        total_count = in_edge_cnt = in_buf_cnt = 0
        max_dwell   = 0.0
        buzzer_on   = pred_warn = fall_alert = False
        surge_alert = edge_loss_alert = False
        person_list = []

        if results is not None:
            for box in results.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                bw,bh = x2-x1, y2-y1
                cx,cy = (x1+x2)//2, (y1+y2)//2
                total_count += 1
                tid = int(box.id[0]) if box.id is not None else -1

                in_edge   = cx < zone_edge_x
                in_buffer = zone_edge_x <= cx < zone_buffer_x
                zone_str  = "edge" if in_edge else ("buffer" if in_buffer else "safe")
                dwell_now = (now-_dwell_tracker[tid]) if (in_edge and tid in _dwell_tracker) else 0.0
                person_list.append((tid,cx,cy,bw,bh,zone_str,dwell_now))

                if tid >= 0:
                    active_ids.add(tid)
                    _last_seen_track[tid] = _frame_count
                    if tid not in _trail_tracker:
                        _trail_tracker[tid] = collections.deque(maxlen=TRAIL_LENGTH)
                    _trail_tracker[tid].append((cx,cy))
                    draw_trail(frame, _trail_tracker[tid], in_edge, in_buffer)
                    update_edge_presence(tid, in_edge, _frame_count)

                heatmap_pts.append((cx, y2))

                if tid>=0 and tid in _trail_tracker:
                    dx,dy,sp,toward = get_velocity(_trail_tracker[tid])
                    if sp>1: velocity_vecs.append((dx,dy))
                    draw_velocity_arrow(frame, _trail_tracker[tid], zone_edge_x, cx, cy)

                if tid>=0 and tid in _trail_tracker:
                    will = predict_trajectory(frame, _trail_tracker[tid], zone_edge_x, cx, cy)
                    if will:
                        pred_warn = True
                        if tid not in _predicted_ids:
                            _predicted_ids.add(tid)
                            log_event(f"PREDICT ID{tid}")
                            announce("approach")
                    else:
                        _predicted_ids.discard(tid)

                if tid>=0:
                    pose_score = match_pose_score(pose_results, (x1,y1,x2,y2))
                    fell = check_fall(tid,bw,bh,frame,cx,cy,
                                      in_edge=in_edge,in_buffer=in_buffer,
                                      pose_score=pose_score)
                    if fell:
                        fall_alert = True
                        log_event(f"FALL DETECTED ID{tid}")
                        if not metro_boarding:
                            announce("fall", cooldown=10)
                            snap = save_snapshot(frame, f"FALL ID{tid}", 100)
                            telegram_send(
                                f"🚨 <b>FALL DETECTED</b>\nID {tid} — immediate response needed!\n"
                                f"Time: {datetime.now().strftime('%H:%M:%S')}",
                                photo_path=snap)
                        with state_lock:
                            state["falls"] += 1

                dwell = 0.0
                if in_edge and tid>=0:
                    if tid not in _dwell_tracker:
                        _dwell_tracker[tid] = now
                        log_event(f"ID{tid} entered EDGE zone")
                        with state_lock:
                            state["incidents"] += 1
                        if not metro_boarding:
                            announce("edge_enter")
                    dwell     = now-_dwell_tracker[tid]
                    max_dwell = max(max_dwell, dwell)
                    in_edge_cnt += 1

                    if dwell>=DWELL_ALERT_SEC and not metro_boarding:
                        buzzer_on = True
                        if tid not in _alerted_ids:
                            _alerted_ids.add(tid)
                            log_event(f"DWELL ALERT ID{tid} {dwell:.0f}s")
                            with state_lock:
                                state["alerts"] += 1
                            flash_alpha = 0.45
                            announce("dwell_crit", cooldown=8)
                            snap = save_snapshot(frame, f"DWELL {dwell:.0f}s ID{tid}", 85)
                        trigger_buzzer()

                elif in_buffer and tid>=0:
                    in_buf_cnt += 1
                    if tid in _dwell_tracker:
                        del _dwell_tracker[tid]
                        _alerted_ids.discard(tid)
                elif tid>=0 and tid in _dwell_tracker:
                    log_event(f"ID{tid} left zone")
                    del _dwell_tracker[tid]
                    _alerted_ids.discard(tid)
                    _fall_ids.discard(tid)

                col = S_RED if in_edge else (S_YELLOW if in_buffer else S_GREEN)
                cv2.rectangle(frame,(x1-1,y1-1),(x2+1,y2+1),tuple(c//3 for c in col),1)
                cv2.rectangle(frame,(x1,y1),(x2,y2),col,2,cv2.LINE_AA)

                tag  = "[E]" if in_edge else ("[B]" if in_buffer else "")
                text = f"ID{tid} {tag}"
                if in_edge: text += f" {dwell:.1f}s"
                if in_edge and dwell>=DWELL_ALERT_SEC: text += " !"
                tw = len(text)*7+12
                draw_filled_rounded(frame,x1,y1-22,x1+tw,y1,col,r=4)
                lbl(frame,text,x1+6,y1-7,size=0.33,color=C_WHITE,bold=True)

                if in_edge and dwell>=DWELL_ALERT_SEC:
                    r_ = int(bw*0.7)
                    cv2.circle(frame,(cx,cy),r_,S_RED,1,cv2.LINE_AA)
                    prog = min(dwell/(DWELL_ALERT_SEC*2),1.0)
                    cv2.ellipse(frame,(cx,cy),(r_,r_),-90,0,int(prog*360),S_AMBER,2,cv2.LINE_AA)

        # ── Surge ────────────────────────────────────────────────────
        if check_surge(velocity_vecs):
            surge_alert = True
            if now-_last_surge_alert>=SURGE_ALERT_COOLDOWN:
                _last_surge_alert = now
                with state_lock: state["surges"] += 1
                log_event("CROWD SURGE DETECTED")
                if not metro_boarding:
                    announce("surge", cooldown=12)
                    snap = save_snapshot(frame, "CROWD SURGE", 90)
            blend_rect(frame,0,h//2-30,w,h//2+30,[0,120,200],0.75)
            lbl(frame,"CROWD SURGE DETECTED — STAND CLEAR",
                w//2-220,h//2+10,size=0.7,color=C_WHITE,bold=True)

        # ── Track-loss ───────────────────────────────────────────────
        lost = check_edge_loss(_frame_count, active_ids)
        if lost:
            edge_loss_alert = True
            for tid in lost:
                log_event(f"EDGE LOSS ID{tid} — POSSIBLE FALL")
                if not metro_boarding:
                    announce("edge_loss", cooldown=10)
                    snap = save_snapshot(frame, f"EDGE LOSS ID{tid}", 100)
                    telegram_send(
                        f"🚨 <b>EDGE LOSS ALERT</b>\nID {tid} disappeared at platform edge.\n"
                        f"Possible platform fall — check immediately!\n"
                        f"Time: {datetime.now().strftime('%H:%M:%S')}",
                        photo_path=snap)
                with state_lock: state["incidents"] += 1

        # ── Heatmap ──────────────────────────────────────────────────
        cleanup_trackers(active_ids, _frame_count)
        update_heatmap(_heatmap_acc, heatmap_pts)
        if sh_heatmap:
            render_heatmap(frame, _heatmap_acc, w, h)

        # ── Risk score ───────────────────────────────────────────────
        risk = compute_platform_risk(in_edge_cnt, in_buf_cnt, max_dwell,
                                     fall_alert, surge_alert, edge_loss_alert)
        if metro_boarding:
            risk = min(risk, 25)   # cap risk display during boarding window

        # ── Crowd AI ─────────────────────────────────────────────────
        # Record every 5th frame (avoid overloading the buffer)
        if _frame_count % 5 == 0:
            cpa.record(in_edge_cnt, in_buf_cnt, risk, total=total_count)
            cpa.log_to_csv(in_edge_cnt, in_buf_cnt, total_count, risk)
        crowd_pred = {}
        if _frame_count % 150 == 0:   # predict every ~5 seconds
            crowd_pred = cpa.predict_crowd(in_edge_cnt, in_buf_cnt, risk, total=total_count)
            with state_lock:
                state["crowd_pred"] = crowd_pred

        # ── Flash ────────────────────────────────────────────────────
        if flash_alpha > 0.01:
            blend_rect(frame,0,0,w,h,[0,0,180],flash_alpha)
            flash_alpha *= 0.75

        # ── Status bar ───────────────────────────────────────────────
        bar_h = 44
        if fall_alert or edge_loss_alert:
            blend_rect(frame,0,h-bar_h,w,h,[0,0,100],0.92)
            cv2.line(frame,(0,h-bar_h),(w,h-bar_h),S_RED,2)
            lbl(frame,"  EMERGENCY — PERSON DOWN / EDGE LOSS — IMMEDIATE RESPONSE",
                10,h-14,size=0.5,color=C_WHITE,bold=True)
        elif risk>=65:
            blend_rect(frame,0,h-bar_h,w,h,[20,10,120],0.88)
            cv2.line(frame,(0,h-bar_h),(w,h-bar_h),S_RED,1)
            lbl(frame,f"  CRITICAL  —  {in_edge_cnt} in EDGE  Risk {risk}/100  BUZZER ACTIVE",
                10,h-14,size=0.46,color=C_WHITE,bold=True)
        elif risk>=30:
            blend_rect(frame,0,h-bar_h,w,h,[0,60,100],0.85)
            cv2.line(frame,(0,h-bar_h),(w,h-bar_h),S_AMBER,1)
            lbl(frame,f"  CAUTION  —  {in_edge_cnt} EDGE  {in_buf_cnt} BUFFER  Risk {risk}/100",
                10,h-14,size=0.44,color=C_WHITE)

        micro(frame, datetime.now().strftime("%H:%M:%S"), w-70, 20, color=C_GRAY2)

        # Write metrics row (non-blocking)
        write_metrics_row(in_edge_cnt, in_buf_cnt, total_count, risk, max_dwell, metro_boarding)

        # ── Update shared state ──────────────────────────────────────
        fps_now   = 1.0/(now-prev_time+1e-6)
        prev_time = now

        with state_lock:
            state.update({
                "total": total_count, "in_edge": in_edge_cnt,
                "in_buffer": in_buf_cnt, "risk": risk,
                "max_dwell": round(max_dwell,1), "fps": round(fps_now,1),
                "uptime": round(uptime,0), "status": status_label(risk),
                "buzzer": buzzer_on, "pred_warn": pred_warn,
                "fall_alert": fall_alert, "surge_alert": surge_alert,
                "edge_loss": edge_loss_alert,
                "last_frame_ts": datetime.now().isoformat(timespec="seconds"),
            })
            state["count_history"].append(in_edge_cnt+in_buf_cnt)
            state["risk_history"].append(risk)
            state["person_list"] = person_list
            if risk > state["peak_risk"]: state["peak_risk"] = risk

        # Encode — apply silhouette if privacy mode
        out_frame = _apply_privacy(frame) if PRIVACY_MODE else frame
        _, jpg = cv2.imencode(".jpg", out_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with state_lock:
            state["frame_bytes"] = jpg.tobytes()

        if socketio:
            socketio.emit("stats", build_socket_payload())

        if not headless:
            cv2.imshow("YellowLine AI v8", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    with state_lock:
        state["online"] = False
    print("[Engine] Stopped.")
