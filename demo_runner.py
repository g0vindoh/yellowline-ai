r"""
Clean demo launcher for recorded station footage.

Usage:
    python demo_runner.py C:\path\to\demo_footage.mp4

It sets the video source, enables privacy mode by default, uses short boarding
pauses for demonstration, and starts server.py.
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python demo_runner.py <video-file>")
        return 2
    video = Path(sys.argv[1]).resolve()
    if not video.exists():
        print(f"Video not found: {video}")
        return 2

    env = os.environ.copy()
    env["YL_SOURCE"] = str(video)
    env.setdefault("YL_PRIVACY_MODE", "1")
    env.setdefault("YL_AUTO_PAUSE_INTERVAL_SEC", "60")
    env.setdefault("YL_AUTO_PAUSE_DURATION_SEC", "8")
    env.setdefault("YL_CAMERA_ID", "demo-cam")
    env.setdefault("YL_SNAP_DIR", "demo_snapshots")

    print("[Demo] Starting YellowLine with recorded footage")
    print(f"[Demo] Source: {video}")
    return subprocess.call([sys.executable, "server.py"], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
