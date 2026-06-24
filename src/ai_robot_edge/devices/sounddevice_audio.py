from __future__ import annotations

import asyncio
import logging
import math
import struct
from collections.abc import AsyncIterator

import numpy as np

from ..events import AudioFrame
from .base import Microphone, Speaker


LOGGER = logging.getLogger(__name__)


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
    def __init__(
        self,
        *,
        device: str | int | None = None,
        normalize_loudness: bool = True,
        target_rms_dbfs: float = -18.0,
        peak_limit: float = 0.95,
        max_output_channels: int = 2,
    ) -> None:
        self.device = device
        self.normalize_loudness = normalize_loudness
        self.target_rms_dbfs = target_rms_dbfs
        self.peak_limit = peak_limit
        self.max_output_channels = max_output_channels

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
            self.device,
            self.normalize_loudness,
            self.target_rms_dbfs,
            self.peak_limit,
            self.max_output_channels,
        )

    async def stop(self) -> None:
        import sounddevice as sd  # type: ignore

        await asyncio.to_thread(sd.stop)


def _decode_audio(
    audio: bytes,
    *,
    sample_rate: int,
    channels: int,
    media_type: str,
) -> tuple[np.ndarray, int, int]:
    audio_format = _detect_audio_format(audio, media_type)
    if audio_format == "pcm_s16le":
        samples = np.frombuffer(audio, dtype=np.int16)
        if channels > 1:
            samples = samples.reshape(-1, channels)
        return samples, sample_rate, channels
    if audio_format in {"mp3", "ogg"}:
        raise ValueError(
            f"unsupported compressed audio format '{audio_format}'; "
            "server should return WAV or PCM16"
        )
    return _decode_wav(audio)


def _detect_audio_format(audio: bytes, media_type: str) -> str:
    media = media_type.split(";", 1)[0].strip().lower()
    if len(audio) >= 12 and audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
        return "wav"
    if audio[:3] == b"ID3":
        return "mp3"
    if len(audio) >= 2 and audio[0] == 0xFF and (audio[1] & 0xE0) == 0xE0:
        return "mp3"
    if audio[:4] == b"OggS":
        return "ogg"
    if "wav" in media:
        return "wav"
    if "mpeg" in media or "mp3" in media:
        return "mp3"
    if "ogg" in media or "opus" in media:
        return "ogg"
    return "pcm_s16le"


def _decode_wav(audio: bytes) -> tuple[np.ndarray, int, int]:
    fmt_chunk: bytes | None = None
    data_chunk: bytes | None = None
    if len(audio) < 12 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        raise ValueError("invalid WAV header")

    offset = 12
    while offset + 8 <= len(audio):
        chunk_id = audio[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", audio, offset + 4)[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(audio):
            break
        chunk_data = audio[chunk_start:chunk_end]
        if chunk_id == b"fmt ":
            fmt_chunk = chunk_data
        elif chunk_id == b"data":
            data_chunk = chunk_data
        offset = chunk_end + (chunk_size % 2)

    if fmt_chunk is None or data_chunk is None or len(fmt_chunk) < 16:
        raise ValueError("incomplete WAV data")

    (
        format_tag,
        wav_channels,
        wav_sample_rate,
        _byte_rate,
        _block_align,
        bits_per_sample,
    ) = struct.unpack_from("<HHIIHH", fmt_chunk, 0)

    if format_tag == 1:
        if bits_per_sample != 16:
            raise ValueError(f"unsupported PCM WAV bit depth: {bits_per_sample}")
        samples = np.frombuffer(data_chunk, dtype=np.int16)
    elif format_tag == 3:
        if bits_per_sample != 32:
            raise ValueError(f"unsupported float WAV bit depth: {bits_per_sample}")
        samples = np.frombuffer(data_chunk, dtype=np.float32)
    else:
        raise ValueError(f"unsupported WAV format tag: {format_tag}")

    if wav_channels > 1:
        samples = samples.reshape(-1, wav_channels)
    return samples, wav_sample_rate, wav_channels


def _play_blocking(
    sd_module,
    samples: np.ndarray,
    sample_rate: int,
    channels: int,
    device: str | int | None,
    normalize_loudness: bool = True,
    target_rms_dbfs: float = -18.0,
    peak_limit: float = 0.95,
    max_output_channels: int = 2,
) -> None:
    samples = _normalize_output_shape(samples, channels)
    samples, stats = _normalize_loudness(
        samples,
        enabled=normalize_loudness,
        target_rms_dbfs=target_rms_dbfs,
        peak_limit=peak_limit,
    )
    if stats["enabled"]:
        LOGGER.info(
            "playback loudness normalized rms_dbfs=%.1f target_dbfs=%.1f gain=%.3f peak_limited=%s",
            stats["rms_dbfs"],
            target_rms_dbfs,
            stats["gain"],
            stats["peak_limited"],
        )
    device_channels, output_sample_rate = _resolve_output_device_params(
        sd_module,
        device,
        fallback_channels=channels,
        fallback_sample_rate=sample_rate,
        max_output_channels=max_output_channels,
    )
    samples = _resample_samples(samples, input_rate=sample_rate, output_rate=output_sample_rate)
    samples = _adapt_samples_for_output_channels(samples, device_channels)
    sd_module.play(
        samples,
        samplerate=output_sample_rate,
        device=device,
        blocking=True,
    )


def _normalize_output_shape(samples: np.ndarray, channels: int) -> np.ndarray:
    if channels > 1 and samples.ndim == 1:
        return samples.reshape(-1, channels)
    return samples


def _resolve_output_device_params(
    sd_module,
    device: str | int | None,
    *,
    fallback_channels: int,
    fallback_sample_rate: int,
    max_output_channels: int = 2,
) -> tuple[int, int]:
    try:
        device_info = sd_module.query_devices(device, kind="output")
        device_channels = int(device_info.get("max_output_channels", fallback_channels))
        default_sample_rate = int(
            round(float(device_info.get("default_samplerate", fallback_sample_rate)))
        )
        return _limit_output_channels(device_channels, max_output_channels), max(
            1, default_sample_rate
        )
    except Exception:
        return _limit_output_channels(fallback_channels, max_output_channels), max(
            1, fallback_sample_rate
        )


def _limit_output_channels(device_channels: int, max_output_channels: int) -> int:
    effective_max = max(1, int(max_output_channels or 1))
    return max(1, min(int(device_channels), effective_max))


def _adapt_samples_for_output_channels(
    samples: np.ndarray,
    device_channels: int,
) -> np.ndarray:
    audio_channels = samples.shape[1] if samples.ndim == 2 else 1
    if device_channels <= audio_channels:
        return samples
    if samples.ndim == 1:
        return np.column_stack([samples] * device_channels)

    repeats = (device_channels + audio_channels - 1) // audio_channels
    return np.tile(samples, (1, repeats))[:, :device_channels]


def _normalize_loudness(
    samples: np.ndarray,
    *,
    enabled: bool,
    target_rms_dbfs: float,
    peak_limit: float,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    if not enabled:
        return samples, {
            "enabled": False,
            "rms_dbfs": -120.0,
            "gain": 1.0,
            "peak_limited": False,
        }

    float_samples = _samples_to_float32(samples)
    if float_samples.size == 0:
        return float_samples, {
            "enabled": True,
            "rms_dbfs": -120.0,
            "gain": 1.0,
            "peak_limited": False,
        }

    float64_samples = float_samples.astype(np.float64, copy=False)
    rms = float(np.sqrt(np.mean(float64_samples * float64_samples)))
    if rms <= 1e-6:
        return float_samples, {
            "enabled": True,
            "rms_dbfs": -120.0,
            "gain": 1.0,
            "peak_limited": False,
        }

    target_rms = 10 ** (target_rms_dbfs / 20)
    gain = target_rms / rms
    peak = float(np.max(np.abs(float_samples)))
    effective_peak_limit = min(max(float(peak_limit), 0.05), 1.0)
    peak_limited = False
    if peak > 0 and peak * gain > effective_peak_limit:
        gain = effective_peak_limit / peak
        peak_limited = True

    adjusted = np.clip(
        float_samples * gain,
        -effective_peak_limit,
        effective_peak_limit,
    ).astype(np.float32)
    rms_dbfs = 20 * math.log10(rms) if rms > 0 else -120.0
    return adjusted, {
        "enabled": True,
        "rms_dbfs": rms_dbfs,
        "gain": float(gain),
        "peak_limited": peak_limited,
    }


def _samples_to_float32(samples: np.ndarray) -> np.ndarray:
    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        scale = float(max(abs(info.min), info.max))
        return (samples.astype(np.float32) / scale).astype(np.float32)
    return samples.astype(np.float32, copy=False)


def _resample_samples(
    samples: np.ndarray,
    *,
    input_rate: int,
    output_rate: int,
) -> np.ndarray:
    if input_rate == output_rate:
        return samples

    if samples.ndim == 1:
        return _resample_channel(samples, input_rate=input_rate, output_rate=output_rate)

    channels = [
        _resample_channel(samples[:, index], input_rate=input_rate, output_rate=output_rate)
        for index in range(samples.shape[1])
    ]
    return np.column_stack(channels)


def _resample_channel(
    channel: np.ndarray,
    *,
    input_rate: int,
    output_rate: int,
) -> np.ndarray:
    if channel.size == 0:
        return channel

    source = channel.astype(np.float32, copy=False)
    output_length = max(1, int(round(channel.shape[0] * output_rate / input_rate)))
    if output_length == channel.shape[0]:
        return source

    src_positions = np.linspace(0.0, channel.shape[0] - 1, num=channel.shape[0], dtype=np.float32)
    dst_positions = np.linspace(0.0, channel.shape[0] - 1, num=output_length, dtype=np.float32)
    resampled = np.interp(dst_positions, src_positions, source).astype(np.float32)
    if np.issubdtype(channel.dtype, np.integer):
        return np.clip(resampled, -32768, 32767).astype(channel.dtype)
    return resampled
