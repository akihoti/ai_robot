#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import json
import time
from dataclasses import replace
from pathlib import Path
import threading

import cv2

from ai_robot_edge.config import load_config
from ai_robot_edge.devices.gimbal import PanTiltGimbal
from ai_robot_edge.vision.tracking import PanTiltTracker, TrackingTarget
from ai_robot_edge.vision.yolov5_face import YoloV5FaceOmDetector, draw_faces


class DetectionState:
    """Thread-safe container for the latest detection results."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._faces: list = []
        self._width: int = 0
        self._height: int = 0
        self._frame_at: float = 0.0
        self._last_nonempty_faces: list = []
        self._last_nonempty_width: int = 0
        self._last_nonempty_height: int = 0
        self._last_nonempty_frame_at: float = 0.0

    def update(self, faces: list, width: int, height: int, frame_at: float) -> None:
        with self._lock:
            self._faces = faces
            self._width = width
            self._height = height
            self._frame_at = frame_at
            if faces:
                self._last_nonempty_faces = list(faces)
                self._last_nonempty_width = width
                self._last_nonempty_height = height
                self._last_nonempty_frame_at = frame_at

    def snapshot(self) -> tuple[list, int, int, float]:
        with self._lock:
            return list(self._faces), self._width, self._height, self._frame_at

    def snapshot_with_hold(
        self,
        now: float,
        hold_seconds: float,
    ) -> tuple[list, int, int, float]:
        with self._lock:
            if self._faces:
                return list(self._faces), self._width, self._height, self._frame_at
            if (
                self._last_nonempty_faces
                and hold_seconds > 0
                and self._last_nonempty_frame_at > 0
                and now - self._last_nonempty_frame_at <= hold_seconds
            ):
                return (
                    list(self._last_nonempty_faces),
                    self._last_nonempty_width,
                    self._last_nonempty_height,
                    self._last_nonempty_frame_at,
                )
            return [], self._width, self._height, self._frame_at


class LatestFrameCapture:
    def __init__(self, source: int | str, width: int, height: int, fps: int) -> None:
        capture_source = _normalize_capture_source(source)
        backend = cv2.CAP_V4L2 if _is_v4l2_source(capture_source) else 0
        self._capture = (
            cv2.VideoCapture(capture_source, backend)
            if backend
            else cv2.VideoCapture(capture_source)
        )
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open camera source: {source}")
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture.set(cv2.CAP_PROP_FPS, fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_frame_at = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while self._running:
            ok, frame = self._capture.read()
            if not ok:
                time.sleep(0.001)
                continue
            with self._lock:
                self._latest_frame = frame
                self._latest_frame_at = time.monotonic()

    def read(self) -> tuple[object | None, float]:
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0
            return self._latest_frame.copy(), self._latest_frame_at

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._capture.release()


def _is_v4l2_source(source: int | str) -> bool:
    if isinstance(source, int):
        return True
    source_path = str(source)
    return source_path.startswith("/dev/video") or source_path.startswith("/dev/v4l/")


def _normalize_capture_source(source: int | str) -> int | str:
    if isinstance(source, int):
        return source
    source_path = Path(str(source))
    if source_path.exists() and source_path.is_symlink():
        try:
            resolved = source_path.resolve(strict=True)
        except OSError:
            return str(source)
        if str(resolved).startswith("/dev/video"):
            return str(resolved)
    return str(source)


async def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    servo_config = replace(config.servo, enabled=True, dry_run=not args.live)
    tracking_config = replace(
        config.tracking,
        tilt_enabled=(
            config.tracking.tilt_enabled
            if args.tilt_enabled is None
            else args.tilt_enabled
        ),
        pan_direction=args.pan_direction or config.tracking.pan_direction,
        tilt_direction=args.tilt_direction or config.tracking.tilt_direction,
    )
    gimbal = PanTiltGimbal(servo_config)
    tracker = PanTiltTracker(gimbal, tracking_config)
    detector = YoloV5FaceOmDetector(
        config.vision.face_model_path,
        input_size=config.vision.face_input_size,
        confidence_threshold=config.vision.person_threshold,
        iou_threshold=config.vision.face_iou_threshold,
        device_id=config.vision.face_device_id,
    )
    raw_source = args.source if args.source is not None else config.camera.source
    source = int(raw_source) if str(raw_source).isdigit() else raw_source
    capture = LatestFrameCapture(source, config.camera.width, config.camera.height, config.camera.fps)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_enabled = bool(args.preview and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")))
    if args.preview and not preview_enabled:
        print(json.dumps({"warning": "preview disabled: no DISPLAY found"}))

    started = time.monotonic()
    frames = moves = 0
    detections = 0
    latest = None
    detect_state = DetectionState()
    next_detect_at = started
    next_control_at = started
    next_preview_at = started
    detect_interval_s = max(0.02, args.detect_interval_ms / 1000.0)
    control_interval_s = max(0.01, args.control_interval_ms / 1000.0)
    preview_interval_s = max(0.05, args.preview_interval_ms / 1000.0)
    target_hold_s = max(0.0, args.target_hold_ms / 1000.0)
    try:
        if args.live:
            await gimbal.center(800)
            await asyncio.sleep(1)
        while args.duration <= 0 or time.monotonic() - started < args.duration:
            now = time.monotonic()
            image, frame_at = capture.read()
            if image is None:
                await asyncio.sleep(0.005)
                continue

            if now >= next_detect_at:
                # ACL contexts are thread-affine, so inference must stay on the
                # same thread where the model was initialized.
                faces = detector.detect_faces(image)
                detect_state.update(faces, image.shape[1], image.shape[0], frame_at)
                detections += 1
                next_detect_at = now + detect_interval_s

            decision = None
            if now >= next_control_at:
                latest_faces, latest_width, latest_height, latest_frame_at = (
                    detect_state.snapshot_with_hold(now, target_hold_s)
                )
                if latest_faces:
                    decision = await tracker.update(
                        [
                            TrackingTarget(
                                face.x1,
                                face.y1,
                                face.width,
                                face.height,
                                face.confidence,
                            )
                            for face in latest_faces
                        ],
                        latest_width,
                        latest_height,
                        target_age_ms=max(0.0, (now - latest_frame_at) * 1000.0),
                    )
                else:
                    await tracker.target_lost()
                next_control_at = now + control_interval_s

                if decision is not None and decision.position is not None:
                    moves += 1
                    print(
                        json.dumps(
                            {
                                "target_center": [
                                    round(decision.target.center_x),
                                    round(decision.target.center_y),
                                ],
                                "error": [
                                    round(decision.error_x, 3),
                                    round(decision.error_y, 3),
                                ],
                                "delta": [
                                    round(decision.pan_delta, 2),
                                    round(decision.tilt_delta, 2),
                                ],
                                "position": [
                                    round(decision.position.pan, 2),
                                    round(decision.position.tilt, 2),
                                ],
                                "frame_age_ms": round((now - latest_frame_at) * 1000, 1),
                                "detect_hz": round(1 / detect_interval_s, 1),
                                "control_hz": round(1 / control_interval_s, 1),
                            }
                        )
                    )

            latest = draw_faces(image, latest_faces)
            if preview_enabled and now >= next_preview_at:
                elapsed = max(time.monotonic() - started, 1e-6)
                fps = frames / elapsed
                overlay = latest.copy()
                height, width = overlay.shape[:2]
                center = (width // 2, height // 2)
                cv2.drawMarker(
                    overlay,
                    center,
                    (255, 255, 0),
                    cv2.MARKER_CROSS,
                    32,
                    2,
                )
                cv2.putText(
                    overlay,
                    f"frames={frames} moves={moves} fps={fps:.1f} live={args.live}",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                if decision is not None:
                    target_center = (
                        round(decision.target.center_x),
                        round(decision.target.center_y),
                    )
                    cv2.line(overlay, center, target_center, (0, 200, 255), 2)
                    cv2.circle(overlay, target_center, 6, (0, 200, 255), -1)
                    cv2.putText(
                        overlay,
                    (
                        f"err=({decision.error_x:+.2f},{decision.error_y:+.2f}) "
                        f"cmd=({decision.pan_delta:+.2f},{decision.tilt_delta:+.2f}) "
                        f"pos=({gimbal.position.pan:.1f},{gimbal.position.tilt:.1f})"
                    ),
                        (16, 64),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("face-gimbal-tracking", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                next_preview_at = now + preview_interval_s
            frames += 1
            await asyncio.sleep(0.001)
    finally:
        capture.release()
        detector.close()
        if args.live and args.center_at_end:
            await gimbal.center(800)
            await asyncio.sleep(1)
        await gimbal.close()
        if preview_enabled:
            cv2.destroyAllWindows()

    if latest is not None:
        cv2.imwrite(str(output_path), latest)
    print(
        json.dumps(
            {
                "live": args.live,
                "frames": frames,
                "detections": detections,
                "moves": moves,
                "final_position": [gimbal.position.pan, gimbal.position.tilt],
                "output": str(output_path),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Test NPU face-to-gimbal tracking")
    parser.add_argument("--config", default="config/edge.yaml")
    parser.add_argument("--source", default=None)
    parser.add_argument(
        "--duration",
        type=float,
        default=0,
        help="Test duration in seconds; use 0 or a negative value to run until you quit the window",
    )
    parser.add_argument("--output", default="artifacts/face-gimbal-tracking.jpg")
    parser.add_argument("--live", action="store_true", help="Enable real servo movement")
    parser.add_argument("--preview", action="store_true", help="Show a live OpenCV preview window")
    parser.add_argument("--detect-interval-ms", type=int, default=50, help="Detection period in milliseconds")
    parser.add_argument("--control-interval-ms", type=int, default=45, help="Servo control period in milliseconds")
    parser.add_argument("--preview-interval-ms", type=int, default=100, help="Preview refresh period in milliseconds")
    parser.add_argument(
        "--target-hold-ms",
        type=int,
        default=180,
        help="Keep the last non-empty detection for this long to ride through brief detector misses",
    )
    parser.add_argument("--pan-direction", type=float, choices=(-1, 1))
    parser.add_argument("--tilt-direction", type=float, choices=(-1, 1))
    parser.add_argument(
        "--tilt-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override vertical correction; defaults to the config value",
    )
    parser.add_argument(
        "--no-center-at-end",
        dest="center_at_end",
        action="store_false",
        help="Do not return the live gimbal to center after the test",
    )
    parser.set_defaults(center_at_end=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
