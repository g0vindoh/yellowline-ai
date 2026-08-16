"""
Time-of-Flight sensor fusion hook.

The production target is a VL53L5CX 8x8 ToF sensor on I2C. This module keeps
the runtime safe when hardware libraries are missing: it starts as unavailable,
supports a mock mode for demos, and exposes a tiny shared state dict.
"""

import os
import random
import threading
import time


TOF_ENABLED = os.environ.get("YL_TOF_ENABLED", "0").lower() in ("1", "true", "yes", "on")
TOF_MOCK = os.environ.get("YL_TOF_MOCK", "0").lower() in ("1", "true", "yes", "on")
TOF_DANGER_MM = int(os.environ.get("YL_TOF_DANGER_MM", "900"))
TOF_POLL_HZ = float(os.environ.get("YL_TOF_POLL_HZ", "15"))

state = {
    "enabled": TOF_ENABLED,
    "available": False,
    "tof_in_danger": False,
    "min_dist_mm": None,
    "last_update": None,
    "error": "",
}

_lock = threading.Lock()
_started = False


def snapshot():
    with _lock:
        return dict(state)


def _update(**kwargs):
    with _lock:
        state.update(kwargs)
        state["last_update"] = time.time()


def _read_mock():
    base = int(os.environ.get("YL_TOF_MOCK_BASE_MM", "1400"))
    jitter = random.randint(-250, 250)
    return max(150, base + jitter)


def _read_hardware():
    """
    Placeholder for VL53L5CX hardware integration.

    Typical deployment code would import the vendor driver here, read the 8x8
    distance matrix, filter invalid cells, and return the minimum distance from
    the platform-edge-facing zones.
    """
    raise RuntimeError("VL53L5CX hardware driver not configured")


def _loop():
    period = 1.0 / max(1.0, TOF_POLL_HZ)
    while True:
        try:
            dist = _read_mock() if TOF_MOCK else _read_hardware()
            _update(
                available=True,
                min_dist_mm=int(dist),
                tof_in_danger=int(dist) <= TOF_DANGER_MM,
                error="",
            )
        except Exception as exc:
            _update(available=False, tof_in_danger=False, error=str(exc))
        time.sleep(period)


def start():
    global _started
    if _started or not TOF_ENABLED:
        return
    _started = True
    threading.Thread(target=_loop, daemon=True).start()
