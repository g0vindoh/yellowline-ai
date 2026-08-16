"""
YellowLine AI — Web Server v8.0

New in v8:
  - /api/history  — time-series query (last N minutes) for dashboard graphs
  - /api/audit    — date-filtered audit log endpoint for incident review
  - /api/cameras  — multi-camera coordinator: list registered nodes + merged state
  - /api/crowd    — crowd prediction forecast (5/15/30 min)
  - /api/privacy  — toggle privacy/silhouette mode at runtime
  - Demo mode setup prompt (video file or live camera)
  - Role-based token system (operator vs. read-only auditor) via YL_AUDIT_TOKEN
"""

import os
import threading
import time
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_socketio import SocketIO

import core
import storage
from station_coordinator import StationCoordinator

station = StationCoordinator.from_env()

APP_DIR    = Path(__file__).resolve().parent
API_TOKEN  = os.environ.get("YL_API_TOKEN",   "").strip()
AUDIT_TOKEN= os.environ.get("YL_AUDIT_TOKEN", "").strip()  # read-only auditor token
HOST       = os.environ.get("YL_HOST",  "0.0.0.0")
PORT       = int(os.environ.get("YL_PORT", "5000"))
CORS_ORIGINS = os.environ.get("YL_CORS_ORIGINS", "*")

app      = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins=CORS_ORIGINS, async_mode="threading")

# ── Auth ──────────────────────────────────────────────────────────────────────
def _request_token():
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "): return auth.split(" ",1)[1].strip()
    return request.headers.get("X-YL-Token","") or request.args.get("token","")

def authorized(operator_only=False):
    tok = _request_token()
    if not API_TOKEN: return True                  # open (no token configured)
    if tok == API_TOKEN: return True               # operator token always works
    if not operator_only and AUDIT_TOKEN and tok == AUDIT_TOKEN:
        return True                                # auditor read-only access
    return False

def require_auth(operator_only=False):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not authorized(operator_only=operator_only):
                return jsonify({"ok": False, "error": "unauthorized"}), 401
            return fn(*args, **kwargs)
        return wrapper
    return decorator


# ── Control helpers ───────────────────────────────────────────────────────────
def apply_control(data):
    data = data or {}
    with core.state_lock:
        if "show_heatmap"       in data: core.state["show_heatmap"]       = bool(data["show_heatmap"])
        if "paused"             in data:
            core.state["manual_paused"] = bool(data["paused"])
            core.state["paused"] = bool(data["paused"]) or core.state.get("auto_pause_remaining",0)>0
        if "auto_pause_enabled" in data: core.state["auto_pause_enabled"] = bool(data["auto_pause_enabled"])
        if "auto_pause_interval"in data: core.state["auto_pause_interval"]= max(30.0,min(3600.0,float(data["auto_pause_interval"])))
        if "auto_pause_duration"in data: core.state["auto_pause_duration"]= max(3.0, min(300.0, float(data["auto_pause_duration"])))
        if "zone_edge"          in data:
            edge = max(0.05, min(0.45, float(data["zone_edge"])))
            core.state["zone_edge"] = edge
            if core.state["zone_buffer"] <= edge:
                core.state["zone_buffer"] = min(0.70, edge+0.05)
        if "zone_buffer"        in data:
            buffer = max(0.10, min(0.70, float(data["zone_buffer"])))
            core.state["zone_buffer"] = max(buffer, core.state["zone_edge"]+0.05)
        if "privacy_mode"       in data: core.state["privacy_mode"] = bool(data["privacy_mode"])
        if "train_present_gpio" in data: core.state["train_present_gpio"] = bool(data["train_present_gpio"])
    return {"ok": True}


def public_state(include_report_fields=False):
    with core.state_lock:
        s = dict(core.state)
        s.pop("frame_bytes",  None)
        s.pop("person_list",  None)
        s["event_log"]      = list(s["event_log"])[:10]
        events              = list(s["all_events"])
        s["all_events"]     = events if include_report_fields else events[-50:]
        s["incidents_feed"] = list(s["incidents_feed"])
        s["count_history"]  = list(s["count_history"])[-120:]
        s["risk_history"]   = list(s["risk_history"])[-120:]
    return s


def gen_frames():
    while True:
        with core.state_lock:
            data = core.state.get("frame_bytes")
        if data:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Cache-Control: no-store\r\n\r\n" + data + b"\r\n")
        time.sleep(0.033)


# ── Multi-camera coordinator ──────────────────────────────────────────────────
# Each additional camera node can POST its state here.
# The coordinator merges: peak risk, total counts, incident feeds.
_camera_nodes = {}   # camera_id -> {last_seen, state_snapshot}
_camera_lock  = threading.Lock()

def _prune_stale_cameras():
    """Remove camera nodes not seen in 10 seconds."""
    while True:
        time.sleep(5)
        now = time.time()
        with _camera_lock:
            stale = [cid for cid, info in _camera_nodes.items()
                     if now - info["last_seen"] > 10]
            for cid in stale:
                del _camera_nodes[cid]

threading.Thread(target=_prune_stale_cameras, daemon=True).start()


# ── Security headers ──────────────────────────────────────────────────────────
@app.after_request
def security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"]        = "no-referrer"
    resp.headers["Cache-Control"]          = "no-store"
    return resp


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    dashboard = (APP_DIR / "dashboard.html").read_text(encoding="utf-8")
    return dashboard

@app.route("/healthz")
def healthz():
    with core.state_lock:
        payload = {
            "ok":            bool(core.state.get("online")) and not bool(core.state.get("degraded")),
            "online":        core.state.get("online"),
            "degraded":      core.state.get("degraded"),
            "engine_error":  core.state.get("engine_error"),
            "last_frame_ts": core.state.get("last_frame_ts"),
            "version":       "8.0",
        }
    return jsonify(payload), 200 if payload["ok"] else 503

@app.route("/video")
@require_auth()
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/stats")
@require_auth()
def api_stats():
    return jsonify(public_state())

@app.route("/api/control", methods=["POST"])
@require_auth(operator_only=True)
def api_control():
    try:
        return jsonify(apply_control(request.get_json(silent=True) or {}))
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

@app.route("/api/report")
@require_auth()
def api_report():
    with core.state_lock:
        s = core.state
        report = {
            "version":    "8.0",
            "timestamp":  datetime.now().isoformat(),
            "duration_s": round(s["uptime"], 2),
            "peak_risk":  s["peak_risk"],
            "incidents":  s["incidents"],
            "alerts":     s["alerts"],
            "falls":      s["falls"],
            "surges":     s["surges"],
            "snapshots":  s["snapshots"],
            "degraded":   s.get("degraded", False),
            "engine_error": s.get("engine_error", ""),
            "events":     list(s["all_events"]),
            "privacy_mode": s.get("privacy_mode", False),
            "metro_boarding": s.get("metro_boarding", False),
        }
    return jsonify(report)


@app.route("/api/crowd")
@require_auth()
def api_crowd():
    """Crowd prediction: 5/15/30-minute forecasts."""
    with core.state_lock:
        pred = dict(core.state.get("crowd_pred", {}))
        in_edge  = core.state.get("in_edge", 0)
        in_buf   = core.state.get("in_buffer", 0)
        risk     = core.state.get("risk", 0)
    if not pred:
        # Try to generate on-demand
        import crowd_prediction_ai as cpa
        pred = cpa.predict_crowd(in_edge, in_buf, risk)
    return jsonify({"ok": True, "predictions": pred, "current_in_edge": in_edge})


@app.route("/api/history")
@require_auth()
def api_history():
    """
    Time-series metrics. Tries TimescaleDB (storage module) first,
    then MySQL, then in-memory ring buffer.
    Query params: ?window=6h  or  ?minutes=60  (default 6h)
    """
    window  = request.args.get("window")
    minutes = request.args.get("minutes")
    camera_id = request.args.get("camera_id")

    if not window and minutes:
        window = f"{minutes}m"
    window = window or "6h"

    rows = storage.history(window=window, camera_id=camera_id)
    if rows:
        return jsonify({"ok": True, "source": "timescaledb", "window": window, "rows": rows})

    # MySQL fallback
    if core._db_conn and core._db_cursor:
        try:
            mins  = int(minutes or 360)
            since = datetime.now() - timedelta(minutes=mins)
            core._db_cursor.execute(
                """SELECT ts, in_edge, in_buffer, total, risk, max_dwell, metro_boarding
                   FROM metrics WHERE ts >= %s ORDER BY ts ASC""", (since,))
            db_rows = core._db_cursor.fetchall()
            data = [{"ts": r[0].isoformat(), "in_edge": r[1], "in_buffer": r[2],
                     "total": r[3], "risk": r[4], "max_dwell": float(r[5]),
                     "metro_boarding": bool(r[6])} for r in db_rows]
            return jsonify({"ok": True, "source": "mysql", "rows": data})
        except Exception:
            pass

    # In-memory fallback
    with core.state_lock:
        ch = list(core.state["count_history"])[-600:]
        rh = list(core.state["risk_history"])[-600:]
    data = [{"frame": i, "count": ch[i], "risk": rh[i]} for i in range(len(ch))]
    return jsonify({"ok": True, "source": "memory", "rows": data})


@app.route("/api/audit")
@require_auth()
def api_audit():
    """
    Event log by date for post-incident review.
    Tries TimescaleDB first, then in-memory.
    Query param: ?date=YYYY-MM-DD
    """
    date_str = request.args.get("date")
    rows = storage.audit(date=date_str)
    if rows:
        return jsonify({"ok": True, "source": "timescaledb",
                        "date": date_str, "count": len(rows), "rows": rows})

    # In-memory fallback
    with core.state_lock:
        all_events = list(core.state["all_events"])
    return jsonify({"ok": True, "source": "memory", "date": date_str,
                    "count": len(all_events), "events": all_events[-500:]})


@app.route("/api/cameras", methods=["GET"])
@require_auth()
def api_cameras():
    """Multi-camera coordinator: list all registered camera nodes."""
    with _camera_lock:
        nodes = {cid: {"last_seen_ago": round(time.time()-info["last_seen"],1),
                        "state": info["state_snapshot"]}
                 for cid, info in _camera_nodes.items()}
    # Include self (camera_id=0) from main engine
    nodes["0"] = {"last_seen_ago": 0, "state": public_state()}
    return jsonify({"ok": True, "cameras": nodes})


@app.route("/api/cameras/register", methods=["POST"])
@require_auth(operator_only=True)
def api_cameras_register():
    """
    Secondary camera nodes POST their state snapshot here periodically.
    Body: {"camera_id": "1", "state": {...}}
    """
    data = request.get_json(silent=True) or {}
    cid  = str(data.get("camera_id", "unknown"))
    snap = data.get("state", {})
    with _camera_lock:
        _camera_nodes[cid] = {"last_seen": time.time(), "state_snapshot": snap}
    # Merge: if any node reports a fall/edge_loss, propagate to primary state
    if snap.get("fall_alert") or snap.get("edge_loss"):
        with core.state_lock:
            if snap.get("fall_alert"):  core.state["falls"]     += 0   # already counted by node
            if snap.get("edge_loss"):   core.state["incidents"] += 0
    return jsonify({"ok": True, "registered": cid})


@app.route("/api/privacy", methods=["POST"])
@require_auth(operator_only=True)
def api_privacy():
    """Toggle silhouette/privacy mode at runtime."""
    data = request.get_json(silent=True) or {}
    mode = bool(data.get("enabled", False))
    with core.state_lock:
        core.state["privacy_mode"] = mode
    return jsonify({"ok": True, "privacy_mode": mode})


@app.route("/api/station")
@require_auth()
def api_station():
    """Station-level coordinator snapshot (multi-camera risk map + stitched tracks)."""
    return jsonify(station.snapshot())


@app.route("/station")
def station_dashboard():
    """Simple station coordinator HTML dashboard."""
    token_js = f'"{AUDIT_TOKEN or API_TOKEN}"' if (AUDIT_TOKEN or API_TOKEN) else '""'
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>YellowLine Station</title>
<style>body{{margin:0;background:#080d12;color:#e5e7eb;font-family:Arial,sans-serif}}
main{{max-width:1100px;margin:32px auto;padding:0 20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}
.card{{border:1px solid #26313d;background:#111923;border-radius:8px;padding:14px}}
.bar{{height:8px;background:#243244;border-radius:20px;overflow:hidden;margin:6px 0}}
.fill{{height:100%;background:#22c55e;transition:width .5s}}
.crit .fill{{background:#ef4444}}.warn .fill{{background:#f59e0b}}
.muted{{color:#94a3b8;font-size:12px}}.risk{{font-size:42px;font-weight:900;margin:8px 0}}</style>
</head><body><main>
<h1>YellowLine Station Coordinator</h1>
<div class="muted" id="meta">Loading…</div>
<div class="risk" id="risk">0</div>
<div class="grid" id="zones"></div>
<h2 style="margin-top:24px">Stitched Tracks</h2><div id="tracks"></div>
</main><script>
const token={token_js};
const headers=token?{{'Authorization':'Bearer '+token}}:{{}};
async function tick(){{
  const r=await fetch('/api/station',{{headers}}); const s=await r.json();
  document.getElementById('meta').textContent=s.enabled?'Coordinator online — '+s.node_count+' nodes':'No camera nodes configured';
  document.getElementById('risk').textContent=s.station_risk||0;
  document.getElementById('zones').innerHTML=(s.risk_map||[]).map(z=>{{
    const cls=z.risk>=65?'crit':z.risk>=30?'warn':'';
    return `<div class="card ${{cls}}"><b>${{z.camera_id}}</b>
      <div class="muted">${{z.range_m[0]}}m – ${{z.range_m[1]}}m</div>
      <div class="bar"><div class="fill" style="width:${{Math.min(100,z.risk||0)}}%"></div></div>
      <div>Risk ${{z.risk||0}} | People ${{z.total||0}} | Edge ${{z.in_edge||0}}</div>
      <div class="muted">${{z.online?'online':'offline'}} ${{z.degraded?'⚠ degraded':''}}</div></div>`;
  }}).join('');
  document.getElementById('tracks').innerHTML=(s.tracks||[]).map(t=>
    `<div style="font-size:13px;color:#cbd5e1;margin:4px 0">${{t.global_id}}: ${{t.camera_id}} / ID${{t.local_id}} at ${{t.platform_m}}m — ${{t.zone}} — dwell ${{t.dwell_s}}s</div>`
  ).join('')||'<div class="muted">No stitched tracks yet</div>';
}}
setInterval(tick,1000); tick();
</script></body></html>"""


@app.route("/api/demo/reset", methods=["POST"])
@require_auth(operator_only=True)
def api_demo_reset():
    """Reset all session counters — useful between demo runs."""
    with core.state_lock:
        core.state["peak_risk"]  = 0
        core.state["incidents"]  = 0
        core.state["alerts"]     = 0
        core.state["falls"]      = 0
        core.state["surges"]     = 0
        core.state["snapshots"]  = 0
        core.state["event_log"].clear()
        core.state["all_events"].clear()
        core.state["count_history"].clear()
        core.state["risk_history"].clear()
        core.state["incidents_feed"].clear()
    return jsonify({"ok": True, "message": "Session counters reset"})


@app.route("/snapshots/<path:filename>")
@require_auth()
def serve_snapshot(filename):
    return send_from_directory(core.SNAPSHOT_DIR, filename)


# ── SocketIO ──────────────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect(auth=None):
    token = ""
    if isinstance(auth, dict): token = auth.get("token","")
    token = token or request.args.get("token","")
    if API_TOKEN and token not in (API_TOKEN, AUDIT_TOKEN):
        return False
    print("[WS] Client connected")

@socketio.on("control")
def on_control(data):
    tok = request.args.get("token","")
    if API_TOKEN and tok not in (API_TOKEN,):
        return {"ok": False, "error": "unauthorized"}
    try:
        return apply_control(data)
    except (TypeError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}


# ── Engine startup ────────────────────────────────────────────────────────────
def start_engine():
    print("[Server] Starting engine thread...")
    t = threading.Thread(
        target=core.engine_loop,
        kwargs={"headless": True, "socketio": socketio},
        daemon=True)
    t.start()
    station.start()   # start multi-camera coordinator if nodes configured
    return t


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  YellowLine AI v8 — Camera Setup")
    print("="*50)
    print("Options:")
    print("  1. Press ENTER → built-in webcam (camera 0)")
    print("  2. Type an IP camera URL (rtsp:// or http://)")
    print("  3. Type a video FILE PATH for demo/simulation mode")
    print("-"*50)
    user_input = input("Your choice (ENTER for webcam): ").strip()

    if user_input == "":
        os.environ["YL_SOURCE"] = "0"
        print("[Setup] Using built-in webcam.")
    elif os.path.isfile(user_input):
        os.environ["YL_SOURCE"]    = user_input
        os.environ["YL_DEMO_LOOP"] = "1"
        print(f"[Setup] Demo mode — looping file: {user_input}")
    else:
        os.environ["YL_SOURCE"] = user_input
        print(f"[Setup] Connecting to: {user_input}")

    # Privacy mode prompt
    priv = input("Enable privacy/silhouette mode? (y/N): ").strip().lower()
    if priv in ("y", "yes"):
        os.environ["YL_PRIVACY_MODE"] = "1"
        print("[Setup] Privacy mode ENABLED — no face data will be stored.")

    print("="*50 + "\n")

    if not API_TOKEN:
        print("[Server] WARNING: YL_API_TOKEN not set. Dashboard is open.")
    if not AUDIT_TOKEN:
        print("[Server] INFO: YL_AUDIT_TOKEN not set. Audit endpoint uses operator token.")

    start_engine()
    print(f"[Server] Dashboard:   http://{HOST}:{PORT}")
    print(f"[Server] Video feed:  http://{HOST}:{PORT}/video")
    print(f"[Server] History API: http://{HOST}:{PORT}/api/history?minutes=60")
    print(f"[Server] Audit API:   http://{HOST}:{PORT}/api/audit?date=YYYY-MM-DD")
    print(f"[Server] Cameras:     http://{HOST}:{PORT}/api/cameras")
    socketio.run(app, host=HOST, port=PORT, debug=False, use_reloader=False)
