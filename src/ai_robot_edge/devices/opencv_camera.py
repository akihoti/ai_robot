from __future__ import annotations

import asyncio

from ..events import CameraFrame
from .base import Camera


class OpenCvCamera(Camera):
    """USB camera adapter that keeps blocking OpenCV calls off the event loop."""

    def __init__(
        self,
        source: int | str,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = max(1, fps)

    async def frames(self):
        import cv2

        capture = await asyncio.to_thread(cv2.VideoCapture, self.source)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"could not open camera source: {self.source}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        sequence = 0
        try:
            while True:
                ok, image = await asyncio.to_thread(capture.read)
                if not ok:
                    raise RuntimeError(f"could not read camera source: {self.source}")
                yield CameraFrame(data=image, sequence=sequence)
                sequence += 1
        finally:
            await asyncio.to_thread(capture.release)
