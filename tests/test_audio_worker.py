from __future__ import annotations

import asyncio
import unittest

import numpy as np

from ai_robot_edge.audio.worker import AudioWorker
from ai_robot_edge.events import AudioFrame, Utterance


class AudioWorkerPresenceListeningTests(unittest.IsolatedAsyncioTestCase):
    async def test_presence_listening_does_not_expire_like_window_listening(self) -> None:
        microphone = QueueMicrophone()
        vad = OneShotVad()
        utterance_queue: asyncio.Queue[Utterance] = asyncio.Queue()
        worker = AudioWorker(
            microphone=microphone,
            vad=vad,
            utterance_queue=utterance_queue,
            listen_timeout_ms=1,
        )
        worker.arm_presence_listening()
        task = asyncio.create_task(worker.run())
        try:
            await asyncio.sleep(0.02)
            await microphone.put(_frame(1))

            utterance = await asyncio.wait_for(utterance_queue.get(), timeout=1)

            self.assertEqual(utterance.reason, "test")
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_suppression_disarms_presence_listening(self) -> None:
        microphone = QueueMicrophone()
        vad = OneShotVad()
        suppress_event = asyncio.Event()
        utterance_queue: asyncio.Queue[Utterance] = asyncio.Queue()
        worker = AudioWorker(
            microphone=microphone,
            vad=vad,
            utterance_queue=utterance_queue,
            listen_timeout_ms=1000,
            suppress_event=suppress_event,
        )
        worker.arm_presence_listening()
        suppress_event.set()
        task = asyncio.create_task(worker.run())
        try:
            await microphone.put(_frame(1))
            await asyncio.sleep(0.05)
            suppress_event.clear()
            await microphone.put(_frame(2))
            await asyncio.sleep(0.05)

            self.assertTrue(utterance_queue.empty())
            self.assertEqual(vad.accept_count, 0)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task


class QueueMicrophone:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[AudioFrame] = asyncio.Queue()

    async def put(self, frame: AudioFrame) -> None:
        await self.queue.put(frame)

    async def frames(self):
        while True:
            yield await self.queue.get()


class OneShotVad:
    def __init__(self) -> None:
        self.accept_count = 0

    def accept(self, frame: AudioFrame) -> Utterance | None:
        self.accept_count += 1
        return Utterance(frames=[frame], reason="test")

    def reset(self) -> None:
        pass


def _frame(sequence: int) -> AudioFrame:
    samples = np.full(480, 1000, dtype=np.int16)
    return AudioFrame(
        data=samples.tobytes(),
        sample_rate=16000,
        channels=1,
        sequence=sequence,
    )


if __name__ == "__main__":
    unittest.main()
