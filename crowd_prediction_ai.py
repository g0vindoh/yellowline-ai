"""
YellowLine AI — Crowd Prediction AI v8.1
─────────────────────────────────────────
Trained on 186,000 rows of real metro platform data (30 days, April–May 2026).
Three separate GradientBoosting models: 5-min, 15-min, 30-min horizons.
14 features including metro schedule proximity, rolling context, peak-hour flags.

Model accuracy (held-out test set):
  5-min:  MAE = 1.29 people,  R² = 0.56
  15-min: MAE = 1.32 people,  R² = 0.55
  30-min: MAE = 1.33 people,  R² = 0.54

Dominant predictors:
  5-min:  total_count (0.70), min_to_train (0.14)
  15-min: min_to_train (0.48), cos_hour (0.13)
  30-min: hour/sin_hour/cos_hour (time of day dominates at 30min)
"""

import os
import time
import threading
import collections
import json
from datetime import datetime

import numpy as np

try:
    import pandas as pd
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    import joblib
    _ML_OK = True
except ImportError:
    _ML_OK = False
    print("[CrowdAI] scikit-learn/pandas not available — prediction disabled.")

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH      = os.environ.get("YL_FEATURE_LOG",  "feature_log.csv")
MODEL_DIR     = os.environ.get("YL_CROWD_DIR",    ".")
RETRAIN_EVERY = int(os.environ.get("YL_RETRAIN_ROWS", "500"))
MIN_ROWS      = 30

# Model file paths (suffix convention: none=15min, _5=5min, _30=30min)
_MODEL_PATHS = {
    "5min":  (f"{MODEL_DIR}/crowd_model_5.joblib",  f"{MODEL_DIR}/crowd_scaler_5.joblib"),
    "15min": (f"{MODEL_DIR}/crowd_model.joblib",    f"{MODEL_DIR}/crowd_scaler.joblib"),
    "30min": (f"{MODEL_DIR}/crowd_model_30.joblib", f"{MODEL_DIR}/crowd_scaler_30.joblib"),
}

# Feature list — must match training exactly
FEATURES = [
    'hour', 'minute', 'day_of_week', 'is_peak', 'is_weekend',
    'sin_hour', 'cos_hour', 'min_to_train',
    'in_edge', 'in_buffer', 'total_count', 'risk',
    'rolling_edge_5', 'rolling_risk_5',
]

# Namma Metro schedule (HH, MM)
_METRO_TIMES = [
    (6,20),(6,25),(6,30),(6,35),(6,40),(6,45),(6,50),(6,55),
    (7,0),(7,5),(7,10),(7,15),(7,20),(7,25),(7,30),(7,35),(7,40),(7,45),(7,50),(7,55),
    (8,0),(8,5),(8,10),(8,15),(8,20),(8,25),(8,30),(8,35),(8,40),(8,45),(8,50),(8,55),
    (9,0),(9,10),(9,20),(9,30),(9,40),(9,50),
    (17,0),(17,5),(17,10),(17,15),(17,20),(17,25),(17,30),(17,35),(17,40),(17,45),(17,50),(17,55),
    (18,0),(18,5),(18,10),(18,15),(18,20),(18,25),(18,30),(18,35),(18,40),(18,45),(18,50),(18,55),
    (19,0),(19,10),(19,20),(19,30),(19,40),(19,50),
]
_METRO_MINS = [h*60+m for h,m in _METRO_TIMES]

def _min_to_train(h: int, m: int) -> float:
    now = h*60 + m
    gaps = [t - now for t in _METRO_MINS if 0 <= (t-now) <= 60]
    return float(min(gaps)) if gaps else 60.0

# ── Model registry ────────────────────────────────────────────────────────────
_models  = {}   # horizon -> GradientBoostingRegressor
_scalers = {}   # horizon -> StandardScaler
_model_lock = threading.Lock()

def load_saved_model():
    """Load all three pre-trained models from disk. Called at startup."""
    if not _ML_OK:
        return
    loaded = []
    with _model_lock:
        for horizon, (mpath, spath) in _MODEL_PATHS.items():
            try:
                _models[horizon]  = joblib.load(mpath)
                _scalers[horizon] = joblib.load(spath)
                loaded.append(horizon)
            except Exception:
                pass
    if loaded:
        print(f"[CrowdAI] Loaded models: {', '.join(loaded)}")
    else:
        print("[CrowdAI] No saved models found — will train from data as it accumulates.")

# ── Rolling buffer for context features ──────────────────────────────────────
_buffer: collections.deque = collections.deque(maxlen=10)   # last 10 readings
_buf_lock = threading.Lock()
_rows_since_retrain = 0

def record(in_edge: int, in_buffer: int, risk: int, total: int = 0):
    """Call every N frames to update rolling context and accumulate training data."""
    global _rows_since_retrain
    if not _ML_OK:
        return
    now = datetime.now()
    with _buf_lock:
        _buffer.append({
            'in_edge': in_edge, 'in_buffer': in_buffer,
            'risk': risk, 'total_count': total,
            'ts': now,
        })
    _rows_since_retrain += 1
    if _rows_since_retrain >= RETRAIN_EVERY:
        _rows_since_retrain = 0
        threading.Thread(target=_retrain_background, daemon=True).start()


def _build_features(ts: datetime, in_edge: int, in_buffer: int,
                    risk: int, total: int,
                    rolling_edge: float, rolling_risk: float) -> list:
    """Build one feature vector matching the FEATURES list."""
    h, m, dow = ts.hour, ts.minute, ts.weekday()
    return [
        h, m, dow,
        int(h in (7,8,9,17,18,19)),        # is_peak
        int(dow >= 5),                       # is_weekend
        float(np.sin(2*np.pi*h/24)),         # sin_hour
        float(np.cos(2*np.pi*h/24)),         # cos_hour
        _min_to_train(h, m),                 # min_to_train
        in_edge, in_buffer, total, risk,
        rolling_edge, rolling_risk,
    ]

# ── Background retrainer ──────────────────────────────────────────────────────
def _retrain_background():
    """Retrain 15-min model on accumulated CSV data. Non-blocking."""
    if not _ML_OK or not os.path.exists(CSV_PATH):
        return
    try:
        import pandas as pd
        df = pd.read_csv(CSV_PATH)
        if len(df) < MIN_ROWS + 90:
            return
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['hour']           = df['timestamp'].dt.hour
        df['minute']         = df['timestamp'].dt.minute
        df['day_of_week']    = df['timestamp'].dt.dayofweek
        df['is_peak']        = df['hour'].isin([7,8,9,17,18,19]).astype(int)
        df['is_weekend']     = (df['day_of_week'] >= 5).astype(int)
        df['sin_hour']       = np.sin(2*np.pi*df['hour']/24)
        df['cos_hour']       = np.cos(2*np.pi*df['hour']/24)
        df['min_to_train']   = df.apply(lambda r: _min_to_train(r['hour'], r['minute']), axis=1)
        df['rolling_edge_5'] = df['in_edge'].rolling(5, min_periods=1).mean()
        df['rolling_risk_5'] = df['risk'].rolling(5, min_periods=1).mean()

        horizon = 90  # 15-min
        X = df[FEATURES].values[:-horizon]
        y = df['in_edge'].values[horizon:]
        if len(X) < MIN_ROWS:
            return

        sc = StandardScaler()
        m  = GradientBoostingRegressor(n_estimators=120, max_depth=4,
                                        learning_rate=0.1, subsample=0.8, random_state=42)
        m.fit(sc.fit_transform(X), y)
        mpath, spath = _MODEL_PATHS["15min"]
        joblib.dump(m,  mpath)
        joblib.dump(sc, spath)
        with _model_lock:
            _models["15min"]  = m
            _scalers["15min"] = sc
        print(f"[CrowdAI] 15-min model retrained on {len(X)} rows.")
    except Exception as e:
        print(f"[CrowdAI] Retrain error: {e}")


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_crowd(in_edge: int, in_buffer: int, risk: int, total: int = 0) -> dict:
    """
    Returns predictions for 5, 15, 30-min horizons using all three models.
    Also returns surge_predicted flag and model confidence info.
    Returns empty dict if no models loaded yet.
    """
    if not _ML_OK:
        return {}

    with _buf_lock:
        buf = list(_buffer)
    rolling_edge = float(np.mean([b['in_edge'] for b in buf])) if buf else float(in_edge)
    rolling_risk = float(np.mean([b['risk']    for b in buf])) if buf else float(risk)

    results = {}
    with _model_lock:
        for horizon, (m, sc) in [
            ("5min",  (_models.get("5min"),  _scalers.get("5min"))),
            ("15min", (_models.get("15min"), _scalers.get("15min"))),
            ("30min", (_models.get("30min"), _scalers.get("30min"))),
        ]:
            if m is None or sc is None:
                results[horizon] = None
                continue
            delta_min = int(horizon.replace("min",""))
            now = datetime.now()
            future = datetime(now.year, now.month, now.day,
                              now.hour, (now.minute + delta_min) % 60)
            feat = _build_features(future, in_edge, in_buffer, risk, total,
                                   rolling_edge, rolling_risk)
            try:
                pred = float(m.predict(sc.transform([feat]))[0])
                results[horizon] = max(0.0, round(pred, 1))
            except Exception:
                results[horizon] = None

    # Surge flag: 15-min prediction more than 2x current or >5 in edge
    pred_15 = results.get("15min") or 0
    results["surge_predicted"] = pred_15 > max(in_edge * 2, 5)

    # Which models are loaded
    with _model_lock:
        results["models_loaded"] = [h for h in ("5min","15min","30min") if _models.get(h)]

    return results


# ── CSV logger ────────────────────────────────────────────────────────────────
_csv_lock = threading.Lock()

def log_to_csv(in_edge: int, in_buffer: int, total: int, risk: int):
    with _csv_lock:
        write_header = not os.path.exists(CSV_PATH)
        try:
            with open(CSV_PATH, "a") as f:
                if write_header:
                    f.write("timestamp,in_edge,in_buffer,total_count,risk\n")
                ts = datetime.now().isoformat(timespec="seconds")
                f.write(f"{ts},{in_edge},{in_buffer},{total},{risk}\n")
        except Exception as e:
            print(f"[CrowdAI] CSV log error: {e}")


# ── Init ──────────────────────────────────────────────────────────────────────
if _ML_OK:
    load_saved_model()
