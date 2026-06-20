from __future__ import annotations

import asyncio
import io
import json
import time
import wave
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .config import ServerAppConfig
from .connectors import RagflowClient, XinferenceClient
from .orchestrator import ConversationOrchestrator
from .registry import EdgeDeviceRegistry
from .remote import RemoteCommandService


def create_app(config: ServerAppConfig) -> FastAPI:
    app = FastAPI(title="AI Robot Server", version="0.1.0")
    registry = EdgeDeviceRegistry()
    xinference = XinferenceClient(config.xinference)
    ragflow = RagflowClient(config.ragflow)
    commands = RemoteCommandService(registry)
    orchestrator = ConversationOrchestrator(ragflow, xinference, config.voice_gateway)

    async def require_admin(
        authorization: Optional[str] = Header(default=None),
        x_admin_token: Optional[str] = Header(default=None),
    ) -> None:
        token = _extract_token(authorization) or x_admin_token
        if config.http.admin_token == "change-me":
            return
        if token != config.http.admin_token:
            raise HTTPException(status_code=401, detail="invalid admin token")

    @app.get("/admin", response_class=HTMLResponse)
    async def admin() -> str:
        return SERVER_ADMIN_HTML

    @app.get("/api/v1/health")
    async def health() -> dict[str, Any]:
        statuses = await asyncio.gather(ragflow.health(), xinference.health())
        return {
            "ok": True,
            "service": "ai-robot-server",
            "connectors": [status.__dict__ for status in statuses],
        }

    async def require_gateway(
        authorization: Optional[str] = Header(default=None),
    ) -> None:
        expected = config.voice_gateway.auth_token
        if not expected:
            return
        if _extract_token(authorization) != expected:
            raise HTTPException(status_code=401, detail="invalid gateway token")

    @app.get("/api/v1/devices", dependencies=[Depends(require_admin)])
    async def devices() -> list[dict[str, Any]]:
        return registry.list_devices()

    @app.get("/api/v1/models", dependencies=[Depends(require_admin)])
    async def models() -> dict[str, Any]:
        try:
            return {"items": await xinference.list_models()}
        except Exception as exc:  # pragma: no cover - defensive web boundary
            return {"items": [], "error": str(exc)}

    @app.post("/api/v1/models/actions", dependencies=[Depends(require_admin)])
    async def model_action(payload: dict[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action", ""))
        return await xinference.model_action(action, payload)

    @app.get("/api/v1/knowledge-bases", dependencies=[Depends(require_admin)])
    async def knowledge_bases() -> dict[str, Any]:
        try:
            return {"items": await ragflow.list_knowledge_bases()}
        except Exception as exc:  # pragma: no cover - defensive web boundary
            return {"items": [], "error": str(exc)}

    @app.post("/api/v1/knowledge-bases/sync", dependencies=[Depends(require_admin)])
    async def sync_knowledge_base(payload: dict[str, Any]) -> dict[str, Any]:
        return await ragflow.sync_knowledge_base(payload)

    @app.post("/api/v1/chat/query", dependencies=[Depends(require_admin)])
    async def chat_query(payload: dict[str, Any]) -> dict[str, Any]:
        question = str(payload.get("question", ""))
        if not question:
            raise HTTPException(status_code=400, detail="question is required")
        return await orchestrator.query(question, payload)

    @app.get("/api/v1/voice-gateway/health", dependencies=[Depends(require_gateway)])
    async def voice_gateway_health() -> dict[str, Any]:
        statuses = await asyncio.gather(ragflow.health(), xinference.health())
        return {
            "ok": all(status.reachable for status in statuses),
            "service": "voice-gateway",
            "connectors": [status.__dict__ for status in statuses],
            "configured": {
                "asr_model": config.voice_gateway.asr_model,
                "tts_model": config.voice_gateway.tts_model,
                "tts_voice": config.voice_gateway.tts_voice,
                "ragflow_chat_id": config.voice_gateway.ragflow_chat_id,
            },
        }

    @app.post("/api/v1/voice-gateway/text-chat", dependencies=[Depends(require_gateway)])
    async def voice_gateway_text_chat(payload: dict[str, Any]) -> JSONResponse:
        question = str(payload.get("question", "")).strip()
        if not question:
            raise HTTPException(status_code=400, detail="question is required")
        response = await orchestrator.text_chat(question, payload)
        return JSONResponse(response)

    @app.post("/api/v1/voice-gateway/voice-chat", dependencies=[Depends(require_gateway)])
    async def voice_gateway_voice_chat(file: UploadFile = File(...)) -> Response:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="uploaded audio is empty")
        result = await orchestrator.voice_chat(
            filename=file.filename or "audio.wav",
            audio_bytes=audio_bytes,
            content_type=file.content_type or "application/octet-stream",
            context={},
        )
        return Response(
            content=result.audio_bytes,
            media_type=result.media_type or config.voice_gateway.tts_media_type,
            headers={
                "X-ASR-Text": result.question.encode(
                    "utf-8", errors="ignore"
                ).hex(),
                "X-Answer-Text": result.answer.encode(
                    "utf-8", errors="ignore"
                ).hex(),
            },
        )

    @app.get("/health", dependencies=[Depends(require_gateway)])
    async def compatibility_health() -> dict[str, Any]:
        return await voice_gateway_health()

    @app.post("/text-chat", dependencies=[Depends(require_gateway)])
    async def compatibility_text_chat(payload: dict[str, Any]) -> JSONResponse:
        return await voice_gateway_text_chat(payload)

    @app.post("/voice-chat", dependencies=[Depends(require_gateway)])
    async def compatibility_voice_chat(file: UploadFile = File(...)) -> Response:
        return await voice_gateway_voice_chat(file)

    @app.post(
        "/api/v1/devices/{device_id}/commands",
        dependencies=[Depends(require_admin)],
    )
    async def request_command(
        device_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        command = str(payload.get("command", ""))
        parameters = payload.get("parameters", {})
        if not isinstance(parameters, dict):
            raise HTTPException(status_code=400, detail="parameters must be an object")
        return await commands.request(device_id, command, parameters)

    @app.websocket("/api/v1/edge/sessions/{device_id}")
    async def edge_session(websocket: WebSocket, device_id: str) -> None:
        if not _edge_token_is_valid(config, device_id, websocket):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        registry.mark_online(device_id)
        command_task: asyncio.Task | None = None
        conversation = _EdgeConversationSession(
            device_id=device_id,
            orchestrator=orchestrator,
        )
        try:
            while True:
                message = await websocket.receive()
                if "text" in message and message["text"] is not None:
                    frame_type = await _handle_edge_text(
                        registry,
                        device_id,
                        message["text"],
                        websocket,
                        conversation,
                    )
                    if frame_type == "device.status" and command_task is None:
                        command_task = asyncio.create_task(
                            _send_commands(websocket, registry, device_id)
                        )
                elif "bytes" in message and message["bytes"] is not None:
                    conversation.handle_audio_bytes(message["bytes"])
                else:
                    break
        finally:
            if command_task is not None:
                command_task.cancel()
            registry.mark_offline(device_id)

    return app


async def _send_commands(
    websocket: WebSocket, registry: EdgeDeviceRegistry, device_id: str
) -> None:
    while True:
        command = await registry.next_command(device_id)
        await websocket.send_text(json.dumps(command, ensure_ascii=False))


async def _handle_edge_text(
    registry: EdgeDeviceRegistry,
    device_id: str,
    raw_message: str,
    websocket: WebSocket,
    conversation: "_EdgeConversationSession",
) -> str | None:
    try:
        envelope = json.loads(raw_message)
    except json.JSONDecodeError:
        registry.add_log(device_id, f"invalid json frame: {raw_message[:80]}")
        return None
    frame_type = envelope.get("type")
    payload = envelope.get("payload", {})
    if frame_type == "device.status" and isinstance(payload, dict):
        registry.update_status(device_id, payload)
    elif frame_type in {"command.progress", "command.result"}:
        registry.add_log(device_id, json.dumps(envelope, ensure_ascii=False))
    elif frame_type == "session.start":
        registry.add_log(device_id, "conversation session started")
        conversation.start(envelope)
    elif frame_type == "audio.chunk":
        conversation.note_chunk(payload)
    elif frame_type == "audio.end":
        registry.add_log(device_id, "conversation audio ended")
        await conversation.finish_audio(websocket, payload)
    elif frame_type == "event.vision":
        await conversation.handle_vision_event(websocket, envelope)
    return frame_type


def _edge_token_is_valid(
    config: ServerAppConfig, device_id: str, websocket: WebSocket
) -> bool:
    expected = config.edge.bearer_tokens.get(device_id)
    if not expected:
        return True
    token = _extract_token(websocket.headers.get("authorization"))
    return token == expected


def _extract_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    prefix = "Bearer "
    if authorization.startswith(prefix):
        return authorization[len(prefix) :]
    return authorization


class _EdgeConversationSession:
    def __init__(
        self,
        *,
        device_id: str,
        orchestrator: ConversationOrchestrator,
    ) -> None:
        self.device_id = device_id
        self.orchestrator = orchestrator
        self.request_id = ""
        self.sample_rate = 16000
        self.channels = 1
        self.audio_buffer = bytearray()

    def start(self, envelope: dict[str, Any]) -> None:
        payload = envelope.get("payload", {})
        audio = payload.get("audio", {})
        self.request_id = str(envelope.get("request_id", "")).strip()
        self.sample_rate = int(audio.get("sample_rate", 16000))
        self.channels = int(audio.get("channels", 1))
        self.audio_buffer.clear()

    def note_chunk(self, _payload: dict[str, Any]) -> None:
        return None

    def handle_audio_bytes(self, data: bytes) -> None:
        self.audio_buffer.extend(data)

    async def finish_audio(
        self,
        websocket: WebSocket,
        payload: dict[str, Any],
    ) -> None:
        if not self.audio_buffer:
            await _send_error_frame(
                websocket,
                request_id=self.request_id,
                code="asr_failed",
                message="audio payload is empty",
                retryable=True,
            )
            return
        try:
            wav_bytes = _pcm16_to_wav(
                bytes(self.audio_buffer),
                sample_rate=self.sample_rate,
                channels=self.channels,
            )
            question = await self.orchestrator.transcribe_audio(
                filename="utterance.wav",
                audio_bytes=wav_bytes,
                content_type="audio/wav",
            )
            if not question.strip():
                raise ValueError("ASR returned empty text")
            await websocket.send_text(
                _frame(
                    "asr.final",
                    self.request_id,
                    {"text": question, "reason": payload.get("reason", "vad_silence")},
                )
            )
            answer, rag_response = await self.orchestrator.answer_question(question, {})
            await websocket.send_text(
                _frame(
                    "llm.final",
                    self.request_id,
                    {
                        "text": answer,
                        "rag_sources": rag_response.get("references", []),
                    },
                )
            )
            audio_out, media_type = await self.orchestrator.synthesize_text(answer)
            resolved_media_type = media_type or "audio/wav"
            await websocket.send_text(
                _frame(
                    "tts.chunk",
                    self.request_id,
                    {
                        "sequence": 1,
                        "encoding": (
                            "wav"
                            if "wav" in resolved_media_type.lower()
                            else "pcm_s16le"
                        ),
                        "sample_rate": self.sample_rate,
                        "channels": self.channels,
                        "is_final": True,
                        "media_type": resolved_media_type,
                    },
                )
            )
            await websocket.send_bytes(audio_out)
        except Exception as exc:
            await _send_error_frame(
                websocket,
                request_id=self.request_id,
                code="conversation_failed",
                message=str(exc),
                retryable=True,
            )
        finally:
            self.audio_buffer.clear()

    async def handle_vision_event(
        self,
        websocket: WebSocket,
        envelope: dict[str, Any],
    ) -> None:
        payload = envelope.get("payload", {})
        request_id = str(envelope.get("request_id", ""))
        if payload.get("event") != "welcome_triggered":
            return
        try:
            welcome_text, audio_out, media_type = await self.orchestrator.build_welcome_audio(
                str(payload.get("welcome_text", "")).strip() or None
            )
            resolved_media_type = media_type or "audio/wav"
            await websocket.send_text(
                _frame("llm.final", request_id, {"text": welcome_text, "rag_sources": []})
            )
            await websocket.send_text(
                _frame(
                    "tts.chunk",
                    request_id,
                    {
                        "sequence": 1,
                        "encoding": (
                            "wav"
                            if "wav" in resolved_media_type.lower()
                            else "pcm_s16le"
                        ),
                        "sample_rate": self.sample_rate,
                        "channels": self.channels,
                        "is_final": True,
                        "media_type": resolved_media_type,
                    },
                )
            )
            await websocket.send_bytes(audio_out)
        except Exception as exc:
            await _send_error_frame(
                websocket,
                request_id=request_id,
                code="welcome_failed",
                message=str(exc),
                retryable=True,
            )


def _pcm16_to_wav(audio_bytes: bytes, *, sample_rate: int, channels: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)
    return buffer.getvalue()


async def _send_error_frame(
    websocket: WebSocket,
    *,
    request_id: str,
    code: str,
    message: str,
    retryable: bool,
) -> None:
    await websocket.send_text(
        _frame(
            "error",
            request_id,
            {
                "code": code,
                "message": message,
                "retryable": retryable,
                "speak_text": "我刚刚没有处理好，请再说一次。",
            },
        )
    )


def _frame(frame_type: str, request_id: str, payload: dict[str, Any]) -> str:
    return json.dumps(
        {
            "type": frame_type,
            "request_id": request_id,
            "timestamp_ms": int(time.time() * 1000),
            "payload": payload,
        },
        ensure_ascii=False,
    )


SERVER_ADMIN_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Robot Server Console</title>
  <style>
    :root { color-scheme: light; --ink:#172126; --muted:#647178; --line:#d7dee2; --accent:#0b7a75; --warn:#b45f06; --bg:#f4f7f6; --panel:#ffffff; }
    body { margin:0; font-family: ui-sans-serif, "Avenir Next", "Segoe UI", sans-serif; background:linear-gradient(180deg,#eef5f3,#f9fbfa); color:var(--ink); }
    header { padding:24px 32px 16px; border-bottom:1px solid var(--line); background:#ffffffcc; backdrop-filter: blur(10px); }
    h1 { margin:0; font-size:28px; letter-spacing:0; }
    main { padding:24px 32px 40px; display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:16px; }
    section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px; box-shadow:0 8px 24px #1232; }
    h2 { margin:0 0 12px; font-size:17px; }
    button { border:1px solid var(--accent); background:var(--accent); color:white; border-radius:6px; padding:8px 12px; cursor:pointer; }
    input, select { width:100%; box-sizing:border-box; border:1px solid var(--line); border-radius:6px; padding:8px; margin:6px 0; }
    pre { white-space:pre-wrap; background:#f2f5f4; border-radius:6px; padding:10px; min-height:70px; overflow:auto; }
    .muted { color:var(--muted); font-size:13px; }
    .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  </style>
</head>
<body>
  <header>
    <h1>AI Robot Server Console</h1>
    <p class="muted">Models, knowledge bases, and connected edge devices.</p>
  </header>
  <main>
    <section><h2>Health</h2><button onclick="loadHealth()">Refresh</button><pre id="health"></pre></section>
    <section><h2>Devices</h2><button onclick="loadDevices()">Refresh</button><pre id="devices"></pre></section>
    <section><h2>Xinference Models</h2><button onclick="loadModels()">Refresh</button><pre id="models"></pre></section>
    <section><h2>Ragflow Knowledge Bases</h2><button onclick="loadKbs()">Refresh</button><pre id="kbs"></pre></section>
    <section>
      <h2>Remote Command</h2>
      <input id="deviceId" placeholder="device id" value="atlas-200i-dk-a2-001">
      <select id="command">
        <option>logs</option><option>restart_edge_service</option><option>pull_update</option><option>run_install</option>
        <option>test_camera</option><option>test_microphone</option><option>test_speaker</option><option>test_server_connection</option>
      </select>
      <button onclick="sendCommand()">Queue</button>
      <pre id="commandResult"></pre>
    </section>
    <section>
      <h2>Knowledge Query</h2>
      <input id="question" placeholder="Ask a question">
      <button onclick="ask()">Query</button>
      <pre id="answer"></pre>
    </section>
  </main>
  <script>
    const adminToken = localStorage.getItem("aiRobotAdminToken") || "";
    const headers = adminToken ? {"Authorization":"Bearer " + adminToken} : {};
    async function api(path, options={}) {
      const res = await fetch(path, {...options, headers:{...headers, ...(options.headers||{})}});
      const text = await res.text();
      try { return JSON.parse(text); } catch { return text; }
    }
    function show(id, data) { document.getElementById(id).textContent = JSON.stringify(data, null, 2); }
    async function loadHealth(){ show("health", await api("/api/v1/health")); }
    async function loadDevices(){ show("devices", await api("/api/v1/devices")); }
    async function loadModels(){ show("models", await api("/api/v1/models")); }
    async function loadKbs(){ show("kbs", await api("/api/v1/knowledge-bases")); }
    async function sendCommand(){
      const id = document.getElementById("deviceId").value;
      const command = document.getElementById("command").value;
      show("commandResult", await api(`/api/v1/devices/${id}/commands`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({command})}));
    }
    async function ask(){
      const question = document.getElementById("question").value;
      show("answer", await api("/api/v1/chat/query", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({question})}));
    }
    loadHealth(); loadDevices(); loadModels(); loadKbs();
  </script>
</body>
</html>"""
