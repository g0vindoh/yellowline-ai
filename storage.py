"""
TimescaleDB/PostgreSQL storage adapter for YellowLine.

Set YL_TSDB_DSN to enable persistence, for example:
postgresql://yellowline:password@localhost:5432/yellowline
"""

import os
from datetime import datetime, timedelta


TSDB_DSN = os.environ.get("YL_TSDB_DSN", "").strip()
RETENTION_HOURS = int(os.environ.get("YL_AUDIT_RETENTION_HOURS", "168"))

_conn = None
_ok = False


def enabled():
    return bool(TSDB_DSN)


def init():
    global _conn, _ok
    if not TSDB_DSN:
        return False
    try:
        import psycopg2
        _conn = psycopg2.connect(TSDB_DSN)
        _conn.autocommit = True
        cur = _conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS yl_metrics (
                ts TIMESTAMPTZ NOT NULL,
                camera_id TEXT NOT NULL,
                total INT,
                in_edge INT,
                in_buffer INT,
                risk INT,
                fps REAL,
                max_dwell REAL,
                tof_in_danger BOOLEAN,
                min_dist_mm INT
            )
        """)
        cur.execute("""
            SELECT create_hypertable('yl_metrics', 'ts',
                                     if_not_exists => TRUE,
                                     migrate_data => TRUE)
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS yl_audit (
                ts TIMESTAMPTZ NOT NULL,
                camera_id TEXT NOT NULL,
                message TEXT NOT NULL
            )
        """)
        cur.execute("""
            SELECT create_hypertable('yl_audit', 'ts',
                                     if_not_exists => TRUE,
                                     migrate_data => TRUE)
        """)
        _ok = True
        print("[TSDB] Connected.")
        return True
    except Exception as exc:
        _ok = False
        print(f"[TSDB] Disabled: {exc}")
        return False


def write_event(camera_id, message, ts=None):
    if not _ok:
        return
    try:
        cur = _conn.cursor()
        cur.execute(
            "INSERT INTO yl_audit (ts, camera_id, message) VALUES (%s, %s, %s)",
            (ts or datetime.utcnow(), camera_id, message),
        )
    except Exception as exc:
        print(f"[TSDB] Event write failed: {exc}")


def write_metrics(camera_id, metrics, ts=None):
    if not _ok:
        return
    tof = metrics.get("tof") or {}
    try:
        cur = _conn.cursor()
        cur.execute(
            """
            INSERT INTO yl_metrics
            (ts, camera_id, total, in_edge, in_buffer, risk, fps, max_dwell,
             tof_in_danger, min_dist_mm)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                ts or datetime.utcnow(),
                camera_id,
                metrics.get("total", 0),
                metrics.get("in_edge", 0),
                metrics.get("in_buffer", 0),
                metrics.get("risk", 0),
                metrics.get("fps", 0),
                metrics.get("max_dwell", 0),
                tof.get("tof_in_danger"),
                tof.get("min_dist_mm"),
            ),
        )
    except Exception as exc:
        print(f"[TSDB] Metric write failed: {exc}")


def parse_window(window):
    unit = window[-1:].lower()
    try:
        value = int(window[:-1])
    except Exception:
        value = 1
    if unit == "m":
        return timedelta(minutes=value)
    if unit == "d":
        return timedelta(days=value)
    return timedelta(hours=value)


def history(window="6h", camera_id=None):
    if not _ok:
        return []
    since = datetime.utcnow() - parse_window(window)
    try:
        cur = _conn.cursor()
        if camera_id:
            cur.execute(
                """
                SELECT ts, camera_id, total, in_edge, in_buffer, risk, fps,
                       max_dwell, tof_in_danger, min_dist_mm
                FROM yl_metrics
                WHERE ts >= %s AND camera_id = %s
                ORDER BY ts ASC
                """,
                (since, camera_id),
            )
        else:
            cur.execute(
                """
                SELECT ts, camera_id, total, in_edge, in_buffer, risk, fps,
                       max_dwell, tof_in_danger, min_dist_mm
                FROM yl_metrics
                WHERE ts >= %s
                ORDER BY ts ASC
                """,
                (since,),
            )
        return [
            {
                "ts": row[0].isoformat(),
                "camera_id": row[1],
                "total": row[2],
                "in_edge": row[3],
                "in_buffer": row[4],
                "risk": row[5],
                "fps": row[6],
                "max_dwell": row[7],
                "tof_in_danger": row[8],
                "min_dist_mm": row[9],
            }
            for row in cur.fetchall()
        ]
    except Exception as exc:
        print(f"[TSDB] History query failed: {exc}")
        return []


def audit(date=None):
    if not _ok:
        return []
    try:
        cur = _conn.cursor()
        if date:
            cur.execute(
                """
                SELECT ts, camera_id, message FROM yl_audit
                WHERE ts::date = %s::date
                ORDER BY ts ASC
                """,
                (date,),
            )
        else:
            since = datetime.utcnow() - timedelta(hours=RETENTION_HOURS)
            cur.execute(
                """
                SELECT ts, camera_id, message FROM yl_audit
                WHERE ts >= %s
                ORDER BY ts DESC
                LIMIT 500
                """,
                (since,),
            )
        return [
            {"ts": row[0].isoformat(), "camera_id": row[1], "message": row[2]}
            for row in cur.fetchall()
        ]
    except Exception as exc:
        print(f"[TSDB] Audit query failed: {exc}")
        return []
