from __future__ import annotations

import asyncio
import io
import wave
from collections.abc import AsyncIterator

import numpy as np

from ..events import AudioFrame
from .base import Microphone, Speaker


class SoundDeviceMicrophone(Microphone):
    def __init__(
        self,
        *,
        sample_rate: int,
        channels: int,
        frame_ms: int,
        device: str | None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_ms = frame_ms
        self.device = device

    async def frames(self) -> AsyncIterator[AudioFrame]:
        import sounddevice as sd  # type: ignore

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=32)
        blocksize = max(1, int(self.sample_rate * self.frame_ms / 1000))

        def callback(indata, _frames, _time, status) -> None:
            if status:
                return
            data = bytes(indata)
            try:
                loop.call_soon_threadsafe(queue.put_nowait, data)
            except asyncio.QueueFull:
                pass

        stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=blocksize,
            device=self.device,
            callback=callback,
        )
        stream.start()
        sequence = 0
        try:
            while True:
                data = await queue.get()
                yield AudioFrame(
                    data=data,
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                    sequence=sequence,
                )
                sequence += 1
        finally:
            stream.stop()
            stream.close()


class SoundDeviceSpeaker(Speaker):
    async def play(
        self,
        audio: bytes,
        sample_rate: int,
        channels: int = 1,
        media_type: str = "audio/pcm",
    ) -> None:
        import sounddevice as sd  # type: ignore

        pcm, actual_rate, actual_channels = _decode_audio(
            audio,
            sample_rate=sample_rate,
            channels=channels,
            media_type=media_type,
        )
        await asyncio.to_thread(
            _play_blocking,
            sd,
            pcm,
            actual_rate,
            actual_channels,
        )


def _decode_audio(
    audio: bytes,
    *,
    sample_rate: int,
    channels: int,
    media_type: str,
) -> tuple[np.ndarray, int, int]:
    if "wav" not in media_type.lower():
        samples = np.frombuffer(audio, dtype=np.int16)
        if channels > 1:
            samples = samples.reshape(-1, channels)
        return samples, sample_rate, channels

    with wave.open(io.BytesIO(audio), "rb") as wav_file:
        wav_channels = wav_file.getnchannels()
        wav_sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16)
    if wav_channels > 1:
        samples = samples.reshape(-1, wav_channels)
    return samples, wav_sample_rate, wav_channels


def _play_blocking(sd_module, samples: np.ndarray, sample_rate: int, channels: int) -> None:
    sd_module.play(samples, samplerate=sample_rate, channels=channels, blocking=True)
