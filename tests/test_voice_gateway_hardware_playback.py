from __future__ import annotations

import asyncio
import json
import os
import socket
import unittest
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

import websockets

from ai_robot_edge.config import EdgeConfig, _parse_config, load_config
from ai_robot_edge.devices.sounddevice_audio import SoundDeviceSpeaker
from ai_robot_edge.events import AudioFrame, Utterance
from ai_robot_edge.server.client import ConversationClient


@unittest.skipUnless(
    os.environ.get("AI_ROBOT_ENABLE_HARDWARE_PLAYBACK") == "1",
    "set AI_ROBOT_ENABLE_HARDWARE_PLAYBACK=1 on the edge device to play audio",
)
class VoiceGatewayHardwarePlaybackTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        asyncio.get_running_loop().slow_callback_duration = 10.0

    async def test_configured_server_is_reachable_and_returned_tts_is_played(self) -> None:
        config = _load_edge_config()
        _assert_configured_server_reachable(config)
        audio_path = _playback_wav_path()
        audio_bytes = audio_path.read_bytes()

        gateway = _AudibleLoopbackVoiceGateway(audio_bytes)
        server = await websockets.serve(gateway.handle, "127.0.0.1", 0)
        try:
            port = server.sockets[0].getsockname()[1]
            loopback_config = replace(
                config,
                server=replace(
                    config.server,
                    websocket_url=(
                        f"ws://127.0.0.1:{port}/api/v1/edge/sessions/{{device_id}}"
                    ),
                    connect_timeout_seconds=1,
                ),
            )
            speaker = SoundDeviceSpeaker(device=config.speaker.device)
            playback_count = 0

            async def play_audio(
                data: bytes,
                sample_rate: int,
                channels: int,
                media_type: str,
            ) -> None:
                nonlocal playback_count
                playback_count += 1
                await speaker.play(data, sample_rate, channels, media_type)

            client = ConversationClient(
                loopback_config,
                on_tts=play_audio,
                on_action=lambda action: _noop(),
            )
            utterance = Utterance(
                request_id="hardware-loopback-request",
                frames=[
                    AudioFrame(
                        data=b"\0\0" * 320,
                        sample_rate=config.microphone.sample_rate,
                        channels=config.microphone.channels,
                        sequence=1,
                    )
                ],
                reason="hardware_loopback",
            )

            result = await asyncio.wait_for(
                client.send_utterance(
                    utterance,
                    context={"session_id": "hardware-loopback", "turn_index": 1},
                ),
                timeout=10.0,
            )

            self.assertTrue(result.success, result.error_message)
            self.assertTrue(result.had_audio)
            self.assertEqual(playback_count, 1)
            self.assertEqual(gateway.received_audio, utterance.audio_bytes)
        finally:
            server.close()
            await server.wait_closed()


class _AudibleLoopbackVoiceGateway:
    def __init__(self, audio_bytes: bytes) -> None:
        self.audio_bytes = audio_bytes
        self.received_audio = bytearray()

    async def handle(self, websocket) -> None:
        request_id = ""
        while True:
            message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
            if isinstance(message, bytes):
                self.received_audio.extend(message)
                continue

            envelope = json.loads(message)
            frame_type = str(envelope.get("type", ""))
            request_id = str(envelope.get("request_id", request_id))
            if frame_type != "audio.end":
                continue

            await websocket.send(
                _frame(
                    "asr.final",
                    request_id,
                    {"text": "硬件播放测试", "reason": "hardware_loopback"},
                )
            )
            await websocket.send(
                _frame("llm.final", request_id, {"text": "正在播放测试语音。"})
            )
            await websocket.send(
                _frame(
                    "tts.chunk",
                    request_id,
                    {
                        "sequence": 1,
                        "encoding": "wav",
                        "sample_rate": 16000,
                        "channels": 1,
                        "is_final": True,
                        "media_type": "audio/wav",
                        "segment_text": "正在播放测试语音。",
                    },
                )
            )
            await websocket.send(self.audio_bytes)
            return


def _load_edge_config() -> EdgeConfig:
    config_path = os.environ.get(
        "AI_ROBOT_EDGE_CONFIG",
        "/root/Desktop/ai_robot_github/config/edge.yaml",
    )
    path = Path(config_path)
    if path.exists():
        return load_config(path)
    return _parse_config(
        {
            "device_id": "hardware-loopback-edge",
            "server": {
                "websocket_url": "ws://127.0.0.1:8010/api/v1/edge/sessions/{device_id}",
                "bearer_token": "loopback-token",
            },
            "admin": {"auth_token": "admin-token"},
        }
    )


def _playback_wav_path() -> Path:
    path = Path(
        os.environ.get(
            "AI_ROBOT_PLAYBACK_WAV_PATH",
            "/root/Desktop/ai_robot_github/assets/audio/welcome.wav",
        )
    )
    if not path.exists():
        raise FileNotFoundError(
            "playback WAV not found; set AI_ROBOT_PLAYBACK_WAV_PATH to a local WAV file"
        )
    return path


def _assert_configured_server_reachable(config: EdgeConfig) -> None:
    url = config.server.websocket_url.format(device_id=config.device_id)
    parsed = urlparse(url)
    if not parsed.hostname:
        raise AssertionError(f"server websocket URL has no host: {url}")
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    with socket.create_connection(
        (parsed.hostname, port),
        timeout=config.server.connect_timeout_seconds,
    ):
        return


async def _noop() -> None:
    return None


def _frame(frame_type: str, request_id: str, payload: dict[str, object]) -> str:
    return json.dumps(
        {
            "type": frame_type,
            "request_id": request_id,
            "timestamp_ms": 0,
            "payload": payload,
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    unittest.main()
