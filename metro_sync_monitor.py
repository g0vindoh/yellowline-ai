"""
YellowLine AI — Metro Sync Monitor v8.0

KEY CHANGE from v7: Train detection is NO LONGER gated on camera HSV colour matching.
The original `see_train()` check required the train body to be visible in the camera FOV,
which almost never happens when the camera points along the platform (not at the track).
This caused the boarding-window suppression to NEVER activate.

v8 approach:
  1. Primary: schedule-based boarding window (always fires when a train is due)
  2. Supplement: crowd density surge heuristic (rapid edge-count spike near schedule time)
  3. Optional: hall-effect / GPIO sensor input via shared state key "train_present"
  4. Optional: HSV visual check kept as an *additive* confidence signal, not a gate
"""

import time
import threading
import cv2
import numpy as np

# ── Schedule ──────────────────────────────────────────────────────────────────
# Format: "HH:MM"  — expand this to the real Namma Metro schedule
# You can also set environment variable YL_METRO_SCHEDULE as comma-separated times.
import os

_DEFAULT_SCHEDULE = [
    "06:20","06:25","06:30","06:35","06:40","06:45","06:50","06:55",
    "07:00","07:05","07:10","07:15","07:20","07:25","07:30","07:35",
    "07:40","07:45","07:50","07:55","08:00","08:05","08:10","08:15",
    "08:20","08:25","08:30","08:35","08:40","08:45","08:50","08:55",
    "09:00","09:10","09:20","09:30","09:40","09:50",
    "10:00","10:10","10:20","10:30","10:40","10:50",
    "11:00","11:10","11:20","11:30","11:40","11:50",
    "12:00","12:10","12:20","12:30","12:40","12:50",
    "13:00","13:10","13:20","13:30","13:40","13:50",
    "14:00","14:10","14:20","14:30","14:40","14:50",
    "15:00","15:10","15:20","15:30","15:40","15:50",
    "16:00","16:10","16:20","16:25","16:30","16:35",
    "16:40","16:45","16:50","16:55","17:00","17:05",
    "17:10","17:15","17:20","17:25","17:30","17:35",
    "17:40","17:45","17:50","17:55","18:00","18:05",
    "18:10","18:15","18:20","18:25","18:30","18:35",
    "18:40","18:45","18:50","18:55","19:00","19:05",
    "19:10","19:15","19:20","19:25","19:30","19:35",
    "19:40","19:45","19:50","19:55","20:00","20:10",
    "20:20","20:30","20:40","20:50","21:00","21:10",
    "21:20","21:30","21:40","21:50","22:00","22:10",
    "22:20","22:30","22:40","22:50","23:00",
]

def _load_schedule():
    env = os.environ.get("YL_METRO_SCHEDULE", "").strip()
    if env:
        return [t.strip() for t in env.split(",") if t.strip()]
    return _DEFAULT_SCHEDULE

SCHEDULE = _load_schedule()

# How many seconds before/after the scheduled time counts as "train due"
PRE_ARRIVAL_SEC  = int(os.environ.get("YL_METRO_PRE_SEC",  "90"))
POST_ARRIVAL_SEC = int(os.environ.get("YL_METRO_POST_SEC", "60"))

# ── Internal state ─────────────────────────────────────────────────────────────
_lock = threading.Lock()
_last_schedule_hit: float = 0.0
_density_history: list = []   # rolling list of (timestamp, in_edge_count)
_DENSITY_WINDOW = 30.0        # seconds to look back for surge heuristic

# ── Helpers ───────────────────────────────────────────────────────────────────
def _seconds_to_next_train() -> float:
    """Return seconds until next scheduled train (negative = overdue)."""
    now_str = time.strftime("%H:%M")
    now_sec = int(time.strftime("%H")) * 3600 + int(time.strftime("%M")) * 60 + int(time.strftime("%S"))
    for sched in SCHEDULE:
        h, m = map(int, sched.split(":"))
        t_sec = h * 3600 + m * 60
        delta = t_sec - now_sec
        if -POST_ARRIVAL_SEC <= delta <= PRE_ARRIVAL_SEC:
            return delta
    return float("inf")


def is_train_due() -> bool:
    """True if a scheduled train is within the pre/post arrival window."""
    return abs(_seconds_to_next_train()) < max(PRE_ARRIVAL_SEC, POST_ARRIVAL_SEC)


def see_train_visual(frame) -> bool:
    """
    HSV colour check for Namma Metro purple/green livery.
    Used as supplementary confidence signal ONLY — not a gate.
    Returns True if significant metro colour is visible in frame.
    """
    if frame is None:
        return False
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Purple line
        mask_purple = cv2.inRange(hsv, np.array([128, 50, 50]), np.array([162, 255, 255]))
        # Green line
        mask_green  = cv2.inRange(hsv, np.array([55, 60, 60]),  np.array([85, 255, 255]))
        mask = cv2.bitwise_or(mask_purple, mask_green)
        return cv2.countNonZero(mask) > 30000
    except Exception:
        return False


def record_density(in_edge: int):
    """Called every frame with current in-edge count to track density history."""
    with _lock:
        now = time.time()
        _density_history.append((now, in_edge))
        # Prune old entries
        cutoff = now - _DENSITY_WINDOW
        while _density_history and _density_history[0][0] < cutoff:
            _density_history.pop(0)


def _density_surge_detected() -> bool:
    """
    Returns True if the edge count spiked significantly in the last window.
    A spike near a scheduled train time is strong evidence of boarding activity.
    """
    with _lock:
        if len(_density_history) < 10:
            return False
        counts = [c for _, c in _density_history]
        baseline = sum(counts[:5]) / 5
        recent   = sum(counts[-5:]) / 5
        return (recent - baseline) > 3  # 3+ extra people in edge zone = surge


def check_boarding(state: dict, frame=None, in_edge: int = 0) -> bool:
    """
    Central boarding check. Call this every frame from core engine.
    Returns True if a boarding window is active (alerts should be suppressed).

    Updates state["metro_boarding"] and state["metro_confidence"] in-place.
    """
    global _last_schedule_hit

    record_density(in_edge)

    schedule_hit   = is_train_due()
    visual_hit     = see_train_visual(frame) if frame is not None else False
    density_hit    = _density_surge_detected() if schedule_hit else False

    # GPIO/hardware sensor support: check shared state key "train_present"
    gpio_hit = bool(state.get("train_present_gpio", False))

    # Boarding if schedule fires + at least one corroborating signal
    # OR if GPIO sensor confirms (hardware doesn't lie)
    # OR if schedule fires alone (safer than missing real boarding)
    boarding_active = schedule_hit or gpio_hit

    confidence = 0
    if schedule_hit: confidence += 50
    if gpio_hit:     confidence += 40
    if visual_hit:   confidence += 30
    if density_hit:  confidence += 20
    confidence = min(confidence, 100)

    state["metro_boarding"]    = boarding_active
    state["metro_confidence"]  = confidence
    state["metro_schedule_due"] = schedule_hit
    state["metro_visual_seen"] = visual_hit

    return boarding_active


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Schedule loaded: {len(SCHEDULE)} trains")
    sec = _seconds_to_next_train()
    if sec == float("inf"):
        print("No train due in current time window")
    else:
        print(f"Next train: {sec:+.0f}s  |  Due: {is_train_due()}")
