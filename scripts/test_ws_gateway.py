#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path

import websockets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the AI robot WebSocket gateway")
    parser.add_argument(
        "--url",
        required=True,
        help="Full websocket URL, for example ws://host:18010/api/v1/edge/sessions/device-001",
    )
    parser.add_argument(
        "--token",
        required=True,
        help="Bearer token from server edge.bearer_tokens[device_id]",
    )
    parser.add_argument(
        "--mode",
        choices=("welcome", "audio"),
        required=True,
        help="Test welcome TTS only or the full ASR/RAG/TTS pipeline",
    )
    parser.add_argument(
        "--wav",
        help="16kHz mono PCM16 WAV file used in audio mode",
    )
    parser.add_argument(
        "--device-id",
        default="test-device",
        help="Device id to send in session.start payload",
    )
    parser.add_argument(
        "--request-id",
        default="",
        help="Optional fixed request id",
    )
    parser.add_argument(
        "--welcome-text",
        default="你好，我在这里。有什么可以帮你的吗？",
        help="Welcome text used in welcome mode",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Receive timeout in seconds",
    )
    parser.add_argument(
        "--save-audio",
        default="",
        help="Optional path to save returned TTS audio bytes",
    )
    parser.add_argument(
        "--play-audio",
        action="store_true",
        help="Play each returned TTS chunk immediately while continuing to receive later chunks",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    request_id = args.request_id or str(uuid.uuid4())
    headers = {"Authorization": f"Bearer {args.token}"}

    try:
        async with websockets.connect(
            args.url,
            additional_headers=headers,
            open_timeout=args.timeout,
            ping_interval=20,
        ) as websocket:
            if args.mode == "welcome":
                await send_welcome(websocket, request_id, args.welcome_text)
            else:
                if not args.wav:
                    raise ValueError("--wav is required in audio mode")
                pcm_bytes, sample_rate, channels, duration_ms = load_pcm16_wav(args.wav)
                await send_audio_session(
                    websocket=websocket,
                    request_id=request_id,
                    device_id=args.device_id,
                    pcm_bytes=pcm_bytes,
                    sample_rate=sample_rate,
                    channels=channels,
                    duration_ms=duration_ms,
                )

            await receive_frames(
                websocket=websocket,
                request_id=request_id,
                timeout=args.timeout,
                save_audio=args.save_audio,
                play_audio=args.play_audio,
            )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


async def send_welcome(websocket, request_id: str, welcome_text: str) -> None:
    await websocket.send(
        frame(
            "event.vision",
            request_id,
            {
                "event": "welcome_triggered",
                "confidence": 1.0,
                "source": "manual_test",
                "cooldown_active": False,
                "welcome_text": welcome_text,
            },
        )
    )
    print("[send] event.vision")


async def send_audio_session(
    *,
    websocket,
    request_id: str,
    device_id: str,
    pcm_bytes: bytes,
    sample_rate: int,
    channels: int,
    duration_ms: int,
) -> None:
    await websocket.send(
        frame(
            "session.start",
            request_id,
            {
                "device_id": device_id,
                "audio": {
                    "encoding": "pcm_s16le",
                    "sample_rate": sample_rate,
                    "channels": channels,
                },
                "wake_word_id": "",
                "context": {},
            },
        )
    )
    print("[send] session.start")

    await websocket.send(
        frame(
            "audio.chunk",
            request_id,
            {
                "sequence": 1,
                "duration_ms": duration_ms,
                "encoding": "pcm_s16le",
            },
        )
    )
    print(f"[send] audio.chunk metadata duration_ms={duration_ms}")

    await websocket.send(pcm_bytes)
    print(f"[send] audio bytes={len(pcm_bytes)}")

    await websocket.send(
        frame(
            "audio.end",
            request_id,
            {
                "total_chunks": 1,
                "reason": "manual_test",
            },
        )
    )
    print("[send] audio.end")


async def receive_frames(
    *,
    websocket,
    request_id: str,
    timeout: float,
    save_audio: str,
    play_audio: bool,
) -> None:
    play_queue: asyncio.Queue[_AudioChunk | None] = asyncio.Queue()
    playback_task: asyncio.Task | None = None
    if play_audio:
        playback_task = asyncio.create_task(_play_audio_queue(play_queue))

    tts_meta: dict | None = None
    received_chunks: list[_AudioChunk] = []
    try:
        while True:
            message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
            if isinstance(message, bytes):
                print(f"[recv] binary audio bytes={len(message)}")
                if tts_meta is None:
                    continue
                chunk = _AudioChunk(
                    audio=message,
                    sample_rate=int(tts_meta.get("sample_rate", 16000)),
                    channels=int(tts_meta.get("channels", 1)),
                    media_type=str(tts_meta.get("media_type", "audio/pcm")),
                    encoding=str(tts_meta.get("encoding", "pcm_s16le")),
                    is_final=bool(tts_meta.get("is_final", False)),
                    sequence=int(tts_meta.get("sequence", len(received_chunks) + 1)),
                )
                received_chunks.append(chunk)
                if play_audio and playback_task is not None:
                    await play_queue.put(chunk)
                if chunk.is_final:
                    break
                tts_meta = None
                continue

            envelope = json.loads(message)
            if envelope.get("request_id") not in {"", request_id}:
                continue

            frame_type = envelope.get("type", "")
            payload = envelope.get("payload", {})
            print(f"[recv] {frame_type}: {json.dumps(payload, ensure_ascii=False)}")

            if frame_type == "tts.chunk":
                tts_meta = payload
            elif frame_type == "error":
                raise RuntimeError(json.dumps(payload, ensure_ascii=False))
    finally:
        if playback_task is not None:
            await play_queue.put(None)
            await playback_task

    if save_audio and received_chunks:
        saved_paths = _save_audio_chunks(save_audio, received_chunks)
        for path in saved_paths:
            print(f"[save] wrote returned audio to {path}")


def load_pcm16_wav(path: str) -> tuple[bytes, int, int, int]:
    with wave.open(path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        pcm_bytes = wav_file.readframes(frame_count)

    if sample_rate != 16000:
        raise ValueError(f"WAV sample rate must be 16000, got {sample_rate}")
    if channels != 1:
        raise ValueError(f"WAV channels must be 1, got {channels}")
    if sample_width != 2:
        raise ValueError(f"WAV sample width must be 2 bytes, got {sample_width}")

    duration_ms = int(frame_count / sample_rate * 1000)
    return pcm_bytes, sample_rate, channels, duration_ms


def frame(frame_type: str, request_id: str, payload: dict) -> str:
    return json.dumps(
        {
            "type": frame_type,
            "request_id": request_id,
            "timestamp_ms": int(time.time() * 1000),
            "payload": payload,
        },
        ensure_ascii=False,
    )


@dataclass(frozen=True)
class _AudioChunk:
    audio: bytes
    sample_rate: int
    channels: int
    media_type: str
    encoding: str
    is_final: bool
    sequence: int


async def _play_audio_queue(queue: asyncio.Queue[_AudioChunk | None]) -> None:
    while True:
        chunk = await queue.get()
        if chunk is None:
            return
        await _play_chunk(chunk)


async def _play_chunk(chunk: _AudioChunk) -> None:
    suffix = ".wav" if _is_wav_chunk(chunk) else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
    try:
        if _is_wav_chunk(chunk):
            tmp_path.write_bytes(chunk.audio)
        else:
            _write_pcm_as_wav(
                path=tmp_path,
                audio=chunk.audio,
                sample_rate=chunk.sample_rate,
                channels=chunk.channels,
            )
        await _play_file(tmp_path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


async def _play_file(path: Path) -> None:
    if sys.platform == "darwin":
        process = await asyncio.create_subprocess_exec("afplay", str(path))
        await process.wait()
        return
    if sys.platform.startswith("linux"):
        process = await asyncio.create_subprocess_exec("aplay", str(path))
        await process.wait()
        return
    if sys.platform == "win32":
        await asyncio.to_thread(_play_file_windows, path)
        return
    raise RuntimeError(f"audio playback is not supported on platform {sys.platform}")


def _play_file_windows(path: Path) -> None:
    import winsound

    winsound.PlaySound(str(path), winsound.SND_FILENAME)


def _save_audio_chunks(save_audio: str, chunks: list[_AudioChunk]) -> list[Path]:
    target = Path(save_audio)
    if len(chunks) == 1:
        chunk = chunks[0]
        if _is_wav_chunk(chunk):
            target.write_bytes(chunk.audio)
            return [target]
        wav_target = _ensure_wav_suffix(target)
        _write_pcm_as_wav(
            path=wav_target,
            audio=chunk.audio,
            sample_rate=chunk.sample_rate,
            channels=chunk.channels,
        )
        return [wav_target]

    if all(not _is_wav_chunk(chunk) for chunk in chunks):
        wav_target = _ensure_wav_suffix(target)
        merged = b"".join(chunk.audio for chunk in chunks)
        _write_pcm_as_wav(
            path=wav_target,
            audio=merged,
            sample_rate=chunks[0].sample_rate,
            channels=chunks[0].channels,
        )
        return [wav_target]

    saved_paths: list[Path] = []
    for chunk in chunks:
        part_path = target.with_name(
            f"{target.stem}.part{chunk.sequence:03d}{_chunk_suffix(chunk, target)}"
        )
        if _is_wav_chunk(chunk):
            part_path.write_bytes(chunk.audio)
        else:
            _write_pcm_as_wav(
                path=part_path.with_suffix(".wav"),
                audio=chunk.audio,
                sample_rate=chunk.sample_rate,
                channels=chunk.channels,
            )
            part_path = part_path.with_suffix(".wav")
        saved_paths.append(part_path)
    return saved_paths


def _ensure_wav_suffix(path: Path) -> Path:
    return path if path.suffix.lower() == ".wav" else path.with_suffix(".wav")


def _chunk_suffix(chunk: _AudioChunk, target: Path) -> str:
    if _is_wav_chunk(chunk):
        return target.suffix or ".wav"
    return ".wav"


def _is_wav_chunk(chunk: _AudioChunk) -> bool:
    media = chunk.media_type.lower()
    encoding = chunk.encoding.lower()
    return "wav" in media or encoding == "wav"


def _write_pcm_as_wav(
    *,
    path: Path,
    audio: bytes,
    sample_rate: int,
    channels: int,
) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
