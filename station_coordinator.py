"""
Station-level multi-camera coordinator.

Each camera node exposes /api/stats. This coordinator polls nodes, maps each
local camera to a platform metre range, stitches IDs across adjacent cameras,
and exposes a station-level risk map.
"""

import os
import threading
import time
from dataclasses import dataclass, field

try:
    import requests
    REQUESTS_OK = True
except Exception:
    REQUESTS_OK = False


@dataclass
class CameraNode:
    camera_id: str
    url: str
    start_m: float
    end_m: float
    token: str = ""


@dataclass
class TrackMemory:
    global_id: str
    camera_id: str
    local_id: int
    platform_m: float
    zone: str
    last_seen: float
    dwell_started: float | None = None


class StationCoordinator:
    def __init__(self, nodes=None, poll_sec=0.5, stitch_window_sec=2.0):
        self.nodes = nodes or []
        self.poll_sec = poll_sec
        self.stitch_window_sec = stitch_window_sec
        self._lock = threading.Lock()
        self._running = False
        self._tracks: dict[tuple[str, int], TrackMemory] = {}
        self._recent_exits: list[TrackMemory] = []
        self._node_stats = {}
        self._next_gid = 1

    @classmethod
    def from_env(cls):
        spec = os.environ.get("YL_CAMERA_NODES", "").strip()
        nodes = []
        if spec:
            # Format: cam1=http://host:5000,0,40;cam2=http://host2:5000,40,80
            for part in spec.split(";"):
                if not part.strip():
                    continue
                name, rest = part.split("=", 1)
                url, start_m, end_m, *token = [x.strip() for x in rest.split(",")]
                nodes.append(CameraNode(name.strip(), url.rstrip("/"), float(start_m), float(end_m), token[0] if token else ""))
        return cls(nodes=nodes)

    def start(self):
        if self._running or not self.nodes or not REQUESTS_OK:
            return
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"[Station] Coordinator started for {len(self.nodes)} camera nodes.")

    def snapshot(self):
        with self._lock:
            risk_map = []
            for node in self.nodes:
                stats = self._node_stats.get(node.camera_id, {})
                risk_map.append({
                    "camera_id": node.camera_id,
                    "range_m": [node.start_m, node.end_m],
                    "risk": stats.get("risk", 0),
                    "total": stats.get("total", 0),
                    "in_edge": stats.get("in_edge", 0),
                    "online": stats.get("online", False),
                    "degraded": stats.get("degraded", False),
                })
            return {
                "enabled": bool(self.nodes),
                "station_risk": max([z["risk"] for z in risk_map], default=0),
                "total_people": sum(z["total"] for z in risk_map),
                "risk_map": risk_map,
                "tracks": [
                    {
                        "global_id": t.global_id,
                        "camera_id": t.camera_id,
                        "local_id": t.local_id,
                        "platform_m": round(t.platform_m, 2),
                        "zone": t.zone,
                        "dwell_s": round(time.time() - t.dwell_started, 1) if t.dwell_started else 0,
                    }
                    for t in self._tracks.values()
                    if time.time() - t.last_seen < 5
                ],
            }

    def _loop(self):
        while self._running:
            for node in self.nodes:
                self._poll_node(node)
            self._expire()
            time.sleep(self.poll_sec)

    def _poll_node(self, node):
        headers = {"Authorization": f"Bearer {node.token}"} if node.token else {}
        try:
            data = requests.get(f"{node.url}/api/stats", headers=headers, timeout=1.5).json()
        except Exception:
            data = {"online": False, "degraded": True, "persons": []}
        persons = data.get("persons", [])
        now = time.time()
        with self._lock:
            self._node_stats[node.camera_id] = data
            active_keys = set()
            for p in persons:
                local_id = int(p.get("id", -1))
                if local_id < 0:
                    continue
                key = (node.camera_id, local_id)
                active_keys.add(key)
                platform_m = self._platform_position(node, p, data)
                existing = self._tracks.get(key)
                if existing:
                    existing.platform_m = platform_m
                    existing.zone = p.get("zone", "safe")
                    existing.last_seen = now
                    if existing.zone == "edge" and not existing.dwell_started:
                        existing.dwell_started = now
                    continue
                gid, dwell_started = self._match_recent_exit(node, platform_m, now)
                self._tracks[key] = TrackMemory(
                    global_id=gid,
                    camera_id=node.camera_id,
                    local_id=local_id,
                    platform_m=platform_m,
                    zone=p.get("zone", "safe"),
                    last_seen=now,
                    dwell_started=dwell_started,
                )
            self._remember_exits(node, active_keys, now)

    def _platform_position(self, node, person, stats):
        width = float(stats.get("frame_width") or stats.get("cam_w") or 1280)
        cx = float(person.get("cx", 0))
        frac = max(0.0, min(1.0, cx / max(width, 1.0)))
        return node.start_m + frac * (node.end_m - node.start_m)

    def _new_gid(self):
        gid = f"P{self._next_gid:06d}"
        self._next_gid += 1
        return gid

    def _match_recent_exit(self, node, platform_m, now):
        for old in list(self._recent_exits):
            if now - old.last_seen > self.stitch_window_sec:
                continue
            at_boundary = abs(old.platform_m - node.start_m) < 4 or abs(old.platform_m - node.end_m) < 4
            close = abs(old.platform_m - platform_m) < 8
            if old.camera_id != node.camera_id and at_boundary and close:
                self._recent_exits.remove(old)
                return old.global_id, old.dwell_started
        return self._new_gid(), None

    def _remember_exits(self, node, active_keys, now):
        for key, track in list(self._tracks.items()):
            if key[0] != node.camera_id or key in active_keys:
                continue
            if now - track.last_seen <= self.stitch_window_sec:
                near_edge = (
                    abs(track.platform_m - node.start_m) < 4 or
                    abs(track.platform_m - node.end_m) < 4
                )
                if near_edge:
                    self._recent_exits.append(track)

    def _expire(self):
        now = time.time()
        with self._lock:
            self._tracks = {k: v for k, v in self._tracks.items() if now - v.last_seen < 10}
            self._recent_exits = [t for t in self._recent_exits if now - t.last_seen <= self.stitch_window_sec]
