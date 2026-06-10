from __future__ import annotations

import argparse
import json
import socket
from typing import Any


def probe_camera() -> dict[str, Any]:
    try:
        import cv2  # type: ignore
    except ImportError:
        return {"ok": False, "message": "opencv-python is not installed"}
    capture = cv2.VideoCapture(0)
    try:
        ok, _frame = capture.read()
        return {"ok": bool(ok), "message": "camera frame captured" if ok else "no frame"}
    finally:
        capture.release()


def probe_microphone() -> dict[str, Any]:
    try:
        import sounddevice  # type: ignore
    except ImportError:
        return {"ok": False, "message": "sounddevice is not installed"}
    return {"ok": True, "devices": str(sounddevice.query_devices())}


def probe_speaker() -> dict[str, Any]:
    try:
        import sounddevice  # type: ignore
    except ImportError:
        return {"ok": False, "message": "sounddevice is not installed"}
    return {"ok": True, "devices": str(sounddevice.query_devices())}


def probe_server(host: str = "10.88.129.127", port: int = 8000) -> dict[str, Any]:
    try:
        with socket.create_connection((host, port), timeout=5):
            return {"ok": True, "message": f"connected to {host}:{port}"}
    except OSError as exc:
        return {"ok": False, "message": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an edge hardware probe")
    parser.add_argument("probe", choices=["camera", "microphone", "speaker", "server"])
    args = parser.parse_args()
    probes = {
        "camera": probe_camera,
        "microphone": probe_microphone,
        "speaker": probe_speaker,
        "server": probe_server,
    }
    print(json.dumps(probes[args.probe](), ensure_ascii=False))


if __name__ == "__main__":
    main()
