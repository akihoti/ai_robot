from __future__ import annotations

import asyncio
import io
import json
import logging
import time
import wave
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .config import ConnectorConfig, ServerAppConfig
from .connectors import MeloTtsClient, RagflowClient, XinferenceClient
from .history import ConversationHistoryStore
from .orchestrator import ConversationOrchestrator
from .registry import EdgeDeviceRegistry
from .remote import RemoteCommandService


LOGGER = logging.getLogger("uvicorn.error")


def create_app(config: ServerAppConfig) -> FastAPI:
    app = FastAPI(title="AI Robot Server", version="0.1.0")
    registry = EdgeDeviceRegistry()
    xinference = XinferenceClient(config.xinference)
    ragflow = RagflowClient(config.ragflow)
    melotts = MeloTtsClient(
        ConnectorConfig(
            base_url=config.tts.base_url,
            api_key=config.tts.api_key,
            timeout_seconds=config.tts.timeout_seconds,
        )
    )
    commands = RemoteCommandService(registry)
    orchestrator = ConversationOrchestrator(
        ragflow,
        xinference,
        config.voice_gateway,
        tts_client=melotts,
        tts_config=config.tts,
    )
    history_store = ConversationHistoryStore()

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

    @app.get("/monitor", response_class=HTMLResponse)
    async def monitor() -> str:
        return SERVER_MONITOR_HTML

    @app.get("/api/v1/health")
    async def health() -> dict[str, Any]:
        checks = [ragflow.health(), xinference.health()]
        if _tts_provider(config) == "melotts_sidecar":
            checks.append(melotts.health())
        statuses = await asyncio.gather(*checks)
        return {
            "ok": True,
            "service": "ai-robot-server",
            "connectors": [status.__dict__ for status in statuses],
        }

    @app.get("/api/v1/monitor/summary", dependencies=[Depends(require_admin)])
    async def monitor_summary() -> dict[str, Any]:
        server = await _monitor_server_summary(ragflow, xinference, melotts, config)
        models_payload = await _monitor_models_summary(xinference)
        devices_payload = registry.list_devices()
        configured = _monitor_config_summary(config, models_payload)
        return {
            "timestamp_ms": int(time.time() * 1000),
            "server": server,
            "models": models_payload,
            "devices": devices_payload,
            "configured": configured,
            "workflow": _monitor_workflow(
                devices_payload,
                server=server,
                models=models_payload,
                config=config,
            ),
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
        checks = [ragflow.health(), xinference.health()]
        if _tts_provider(config) == "melotts_sidecar":
            checks.append(melotts.health())
        statuses = await asyncio.gather(*checks)
        return {
            "ok": all(status.reachable for status in statuses),
            "service": "voice-gateway",
            "connectors": [status.__dict__ for status in statuses],
            "configured": {
                "asr_model": config.voice_gateway.asr_model,
                "tts_model": config.voice_gateway.tts_model,
                "tts_voice": config.voice_gateway.tts_voice,
                "tts_provider": config.tts.provider,
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
            history_store=history_store,
            registry=registry,
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


async def _monitor_server_summary(
    ragflow: RagflowClient,
    xinference: XinferenceClient,
    melotts: MeloTtsClient | None = None,
    config: ServerAppConfig | None = None,
) -> dict[str, Any]:
    checks = [
        ("ragflow", ragflow, ragflow.health()),
        ("xinference", xinference, xinference.health()),
    ]
    if config is not None and _tts_provider(config) == "melotts_sidecar" and melotts is not None:
        checks.append(("melotts_sidecar", melotts, melotts.health()))
    results = await asyncio.gather(
        *(check for _, _, check in checks),
        return_exceptions=True,
    )
    connectors: list[dict[str, Any]] = []
    for (name, client, _), result in zip(checks, results):
        if isinstance(result, Exception):
            connectors.append(
                {
                    "name": name,
                    "configured": bool(client.config.base_url),
                    "reachable": False,
                    "message": str(result),
                }
            )
        else:
            connectors.append(result.__dict__)
    return {
        "ok": all(bool(connector.get("reachable")) for connector in connectors),
        "service": "ai-robot-server",
        "connectors": connectors,
    }


async def _monitor_models_summary(
    xinference: XinferenceClient,
) -> dict[str, Any]:
    try:
        return {"items": await xinference.list_models()}
    except Exception as exc:  # pragma: no cover - defensive web boundary
        return {"items": [], "error": str(exc)}


def _monitor_config_summary(
    config: ServerAppConfig,
    models: dict[str, Any],
) -> dict[str, Any]:
    loaded_models = _monitor_model_names(models.get("items", []))
    tts_provider = _tts_provider(config)
    return {
        "asr_model": config.voice_gateway.asr_model,
        "tts_model": config.voice_gateway.tts_model,
        "tts_voice": config.voice_gateway.tts_voice,
        "tts_provider": tts_provider,
        "tts_fallback_provider": config.tts.fallback_provider,
        "ragflow_chat_id": config.voice_gateway.ragflow_chat_id,
        "models_loaded": {
            "asr": _monitor_model_is_loaded(
                config.voice_gateway.asr_model,
                loaded_models,
            ),
            "tts": True
            if tts_provider == "melotts_sidecar"
            else _monitor_model_is_loaded(
                config.voice_gateway.tts_model,
                loaded_models,
            ),
        },
    }


def _tts_provider(config: ServerAppConfig) -> str:
    return config.tts.provider.strip().lower() or "xinference"


MONITOR_STAGE_TIMEOUT_MS = {
    "microphone": 8_000,
    "asr": 10_000,
    "llm_ragflow": 30_000,
    "tts": 12_000,
    "speaker": 20_000,
}


def _monitor_workflow(
    devices: list[dict[str, Any]],
    *,
    server: dict[str, Any],
    models: dict[str, Any],
    config: ServerAppConfig,
) -> dict[str, Any]:
    latest_device = _monitor_latest_device(devices)
    device_online = bool(latest_device and latest_device.get("online"))
    status = _as_dict(latest_device.get("status")) if latest_device else {}
    monitoring = _as_dict(status.get("monitoring"))
    edge_devices = _as_dict(status.get("devices"))
    components = _as_dict(monitoring.get("components"))
    last_events = _as_dict(monitoring.get("last_events"))
    counters = _as_dict(monitoring.get("counters"))
    queues = _as_dict(monitoring.get("queues"))
    session = _as_dict(monitoring.get("session"))
    session_state = _session_state_value(session.get("state"))
    listening = _as_dict(monitoring.get("listening"))
    playback = _as_dict(monitoring.get("playback"))
    runtime = _as_dict(status.get("runtime"))
    server_activity = _as_dict(latest_device.get("server_activity")) if latest_device else {}
    now_ms = int(time.time() * 1000)
    active_stage = _monitor_active_stage(
        server_activity,
        session=session,
        listening=listening,
        playback=playback,
        now_ms=now_ms,
    )
    connectors = {
        str(connector.get("name")): connector
        for connector in server.get("connectors", [])
        if isinstance(connector, dict)
    }
    model_names = _monitor_model_names(models.get("items", []))
    tts_provider = _tts_provider(config)
    if tts_provider == "melotts_sidecar":
        tts_health = _connector_status(connectors.get("melotts_sidecar"))
        tts_detail = "provider=melotts_sidecar"
        tts_metrics = {
            "playback_chunks": counters.get("playback_chunks", 0),
            "provider": tts_provider,
            "fallback_provider": config.tts.fallback_provider,
        }
    else:
        tts_health = _model_status(
            device_online,
            config.voice_gateway.tts_model,
            model_names,
            models.get("error"),
        )
        tts_detail = config.voice_gateway.tts_model or "not configured"
        tts_metrics = {"playback_chunks": counters.get("playback_chunks", 0)}

    groups = [
        {
            "id": "vision",
            "label": "视觉链路",
            "nodes": [
                _monitor_node(
                    "camera",
                    "摄像头",
                    _enabled_node_status(device_online, edge_devices, "camera"),
                    "active" if device_online else "offline",
                    _device_detail(edge_devices.get("camera")),
                    last_events.get("vision"),
                    {"vision_present": counters.get("vision_present", 0)},
                ),
                _monitor_node(
                    "target_detection",
                    "目标检测",
                    _vision_detection_status(device_online, last_events),
                    "active"
                    if device_online
                    and "present"
                    in str(_as_dict(last_events.get("vision")).get("event_type", ""))
                    else "idle",
                    _event_text(last_events.get("vision")),
                    last_events.get("vision"),
                    {
                        "present": counters.get("vision_present", 0),
                        "absent": counters.get("vision_absent", 0),
                    },
                ),
                _monitor_node(
                    "tracking",
                    "跟踪",
                    _degraded_status(
                        device_online,
                        bool(components.get("tracker_degraded")),
                        active=bool(session_state)
                        and session_state not in {"idle", "disengaged"},
                    ),
                    _activity_from_active(
                        device_online,
                        bool(session_state)
                        and session_state not in {"idle", "disengaged"},
                    ),
                    f"session={session.get('state', 'unknown')}, reason={session.get('reason', '-')}",
                    last_events.get("vision"),
                    {"vision_queue": queues.get("vision")},
                ),
                _monitor_node(
                    "servo",
                    "舵机",
                    _degraded_status(
                        device_online,
                        bool(components.get("servo_degraded")),
                        active=True,
                    ),
                    _activity_from_active(device_online, True),
                    _event_text(last_events.get("action")),
                    last_events.get("action"),
                    {"action_interrupts": counters.get("action_interrupts", 0)},
                ),
            ],
        },
        {
            "id": "voice",
            "label": "语音链路",
            "nodes": [
                _monitor_node(
                    "microphone",
                    "麦克风",
                    _enabled_node_status(device_online, edge_devices, "microphone"),
                    _activity_from_active(
                        device_online,
                        bool(listening.get("armed"))
                        or active_stage.get("node_id") == "microphone",
                    ),
                    _device_detail(edge_devices.get("microphone")),
                    last_events.get("utterance"),
                    {"utterances": counters.get("utterances", 0)},
                ),
                _monitor_node(
                    "listening",
                    "监听窗口",
                    _binary_activity_status(device_online, bool(listening.get("armed"))),
                    _activity_from_active(device_online, bool(listening.get("armed"))),
                    f"timeout_ms={listening.get('timeout_ms', '-')}",
                    listening,
                    {"updated_at_ms": listening.get("updated_at_ms")},
                ),
                _monitor_node(
                    "vad",
                    "VAD",
                    "ok" if device_online and edge_devices.get("vad") else "offline",
                    _activity_from_active(device_online, bool(listening.get("armed"))),
                    _device_detail(edge_devices.get("vad")),
                    last_events.get("utterance"),
                    {"utterances": counters.get("utterances", 0)},
                ),
                _monitor_node(
                    "asr",
                    "ASR",
                    _model_status(
                        device_online,
                        config.voice_gateway.asr_model,
                        model_names,
                        models.get("error"),
                    ),
                    "idle",
                    config.voice_gateway.asr_model or "not configured",
                    last_events.get("utterance"),
                    {"model": config.voice_gateway.asr_model},
                ),
                _monitor_node(
                    "llm_ragflow",
                    "LLM/RAGFlow",
                    _connector_status(connectors.get("ragflow")),
                    "idle",
                    f"chat_id={config.voice_gateway.ragflow_chat_id or '-'}",
                    last_events.get("server_turn"),
                    {
                        "turns_ok": counters.get("turns_ok", 0),
                        "turns_failed": counters.get("turns_failed", 0),
                    },
                ),
                _monitor_node(
                    "tts",
                    "TTS",
                    tts_health,
                    "idle",
                    tts_detail,
                    last_events.get("playback_chunk"),
                    tts_metrics,
                ),
                _monitor_node(
                    "speaker",
                    "扬声器",
                    _enabled_node_status(device_online, edge_devices, "speaker"),
                    _activity_from_active(device_online, bool(playback.get("active"))),
                    _device_detail(edge_devices.get("speaker")),
                    last_events.get("playback_chunk"),
                    {"playback_active": bool(playback.get("active"))},
                ),
            ],
        },
        {
            "id": "service",
            "label": "服务链路",
            "nodes": [
                _monitor_node(
                    "edge_online",
                    "边缘设备在线",
                    "ok" if device_online else "offline",
                    "active" if device_online else "offline",
                    str(latest_device.get("device_id")) if latest_device else "等待边缘设备连接",
                    latest_device,
                    {"connection_count": latest_device.get("connection_count") if latest_device else 0},
                ),
                _monitor_node(
                    "server_health",
                    "服务端健康",
                    "ok" if server.get("ok") else "warn",
                    "idle",
                    server.get("service", "ai-robot-server"),
                    server,
                    {"runtime_mode": runtime.get("mode")},
                ),
                _monitor_node(
                    "xinference",
                    "Xinference 模型",
                    _connector_status(connectors.get("xinference")),
                    "idle",
                    _models_detail(models),
                    models,
                    {"model_count": len(models.get("items", []))},
                ),
                _monitor_node(
                    "ragflow",
                    "RAGFlow 可达性",
                    _connector_status(connectors.get("ragflow")),
                    "idle",
                    _connector_detail(connectors.get("ragflow")),
                    connectors.get("ragflow"),
                    {"chat_id": config.voice_gateway.ragflow_chat_id},
                ),
            ],
        },
    ]
    if device_online:
        _apply_active_stage(groups, active_stage)
    alerts = _monitor_alerts(groups, device_online)
    overall_status = _overall_monitor_status(groups, device_online, alerts)
    return {
        "overall": {
            "status": overall_status,
            "message": _overall_monitor_message(overall_status, latest_device),
            "active_device_id": latest_device.get("device_id") if latest_device else None,
            "updated_at_ms": now_ms,
        },
        "active_stage": active_stage,
        "alerts": alerts,
        "groups": groups,
        "latest_device": latest_device,
        "session": session,
        "queues": queues,
        "counters": counters,
        "recent_events": last_events,
        "server_activity": server_activity,
        "recent_logs": list((latest_device.get("logs") if latest_device else []) or [])[-50:],
    }


def _monitor_node(
    node_id: str,
    label: str,
    status: str,
    activity: str,
    detail: str,
    event: Any,
    counters: dict[str, Any],
) -> dict[str, Any]:
    health = _health_from_status(status)
    return {
        "id": node_id,
        "label": label,
        "status": _display_status(health, activity),
        "health": health,
        "activity": activity,
        "detail": detail,
        "last_event": event,
        "event_text": _event_text(event),
        "counters": counters,
    }


def _monitor_active_stage(
    server_activity: dict[str, Any],
    *,
    session: dict[str, Any],
    listening: dict[str, Any],
    playback: dict[str, Any],
    now_ms: int,
) -> dict[str, Any]:
    node_id = str(server_activity.get("node_id", "")).strip()
    status = str(server_activity.get("status", "")).strip().lower()
    if node_id and status and status != "idle":
        started_at_ms = _safe_int(server_activity.get("started_at_ms"), default=0)
        elapsed_ms = max(0, now_ms - started_at_ms) if started_at_ms else 0
        activity = "active" if status == "active" else status
        health = "ok"
        if activity in {"error", "failed"}:
            activity = "error"
            health = "error"
        elif activity == "active" and elapsed_ms > MONITOR_STAGE_TIMEOUT_MS.get(
            node_id, 15_000
        ):
            activity = "stuck"
            health = "warn"
        elif activity not in {"active", "idle"}:
            activity = "idle"
        return {
            "node_id": node_id,
            "phase": str(server_activity.get("phase", node_id)),
            "activity": activity,
            "health": health,
            "message": str(server_activity.get("message", "")),
            "request_id": str(server_activity.get("request_id", "")),
            "started_at_ms": started_at_ms or None,
            "updated_at_ms": _safe_int(server_activity.get("updated_at_ms"), default=0)
            or None,
            "elapsed_ms": elapsed_ms,
        }
    if playback.get("active"):
        return {
            "node_id": "speaker",
            "phase": "playback",
            "activity": "active",
            "health": "ok",
            "message": "playing audio",
            "elapsed_ms": 0,
        }
    if listening.get("armed"):
        return {
            "node_id": "listening",
            "phase": "listening",
            "activity": "active",
            "health": "ok",
            "message": "listening window armed",
            "elapsed_ms": 0,
        }
    state = _session_state_value(session.get("state"))
    if state and state not in {"idle", "disengaged"}:
        return {
            "node_id": "tracking",
            "phase": state,
            "activity": "active",
            "health": "ok",
            "message": str(session.get("reason", "")),
            "elapsed_ms": 0,
        }
    return {
        "node_id": "",
        "phase": "idle",
        "activity": "idle",
        "health": "ok",
        "message": "idle",
        "elapsed_ms": 0,
    }


def _session_state_value(value: Any) -> str:
    state = str(value or "").strip()
    if "." in state:
        state = state.rsplit(".", 1)[-1]
    return state.lower()


def _apply_active_stage(
    groups: list[dict[str, Any]],
    active_stage: dict[str, Any],
) -> None:
    node_id = active_stage.get("node_id")
    if not node_id:
        return
    for group in groups:
        for node in group.get("nodes", []):
            if not isinstance(node, dict) or node.get("id") != node_id:
                continue
            activity = str(active_stage.get("activity", "idle"))
            health = str(active_stage.get("health", "ok"))
            node["activity"] = activity
            node["health"] = _worst_health(str(node.get("health", "ok")), health)
            node["status"] = _display_status(str(node["health"]), activity)
            node["elapsed_ms"] = active_stage.get("elapsed_ms", 0)
            message = str(active_stage.get("message", ""))
            if message:
                elapsed_ms = int(active_stage.get("elapsed_ms") or 0)
                suffix = f"elapsed_ms={elapsed_ms}" if elapsed_ms else ""
                node["detail"] = ", ".join(part for part in [message, suffix] if part)


def _monitor_alerts(
    groups: list[dict[str, Any]],
    device_online: bool,
) -> list[dict[str, str]]:
    if not device_online:
        return []
    alerts: list[dict[str, str]] = []
    for group in groups:
        for node in group.get("nodes", []):
            if not isinstance(node, dict):
                continue
            health = str(node.get("health", "ok"))
            activity = str(node.get("activity", "idle"))
            if health not in {"warn", "error", "offline"} and activity not in {
                "stuck",
                "error",
            }:
                continue
            level = "error" if health == "error" or activity == "error" else "warn"
            if health == "offline":
                level = "offline"
            alerts.append(
                {
                    "node_id": str(node.get("id", "")),
                    "label": str(node.get("label", "")),
                    "level": level,
                    "message": str(node.get("detail") or node.get("event_text") or ""),
                }
            )
    return alerts


def _health_from_status(status: str) -> str:
    if status in {"active", "idle", "ok"}:
        return "ok"
    if status in {"warn", "warning"}:
        return "warn"
    if status in {"error", "failed"}:
        return "error"
    if status == "offline":
        return "offline"
    return "warn"


def _activity_from_active(device_online: bool, active: bool) -> str:
    if not device_online:
        return "offline"
    return "active" if active else "idle"


def _display_status(health: str, activity: str) -> str:
    if activity in {"stuck", "error"}:
        return activity
    if health in {"warn", "error", "offline"}:
        return health
    return activity if activity == "active" else "ok"


def _worst_health(left: str, right: str) -> str:
    order = {"ok": 0, "warn": 1, "offline": 2, "error": 3}
    return left if order.get(left, 1) >= order.get(right, 1) else right


def _monitor_latest_device(
    devices: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not devices:
        return None
    candidates = [device for device in devices if device.get("online")] or devices
    return max(
        candidates,
        key=lambda device: _safe_int(device.get("last_seen_ms"), default=0),
    )


def _monitor_model_names(items: Any) -> set[str]:
    names: set[str] = set()
    if not isinstance(items, list):
        return names
    keys = ("id", "model", "model_name", "model_uid", "name")
    for item in items:
        if isinstance(item, str):
            names.add(item.lower())
            continue
        if not isinstance(item, dict):
            continue
        for key in keys:
            value = item.get(key)
            if value:
                names.add(str(value).lower())
    return names


def _monitor_model_is_loaded(model_name: str, loaded_models: set[str]) -> bool:
    if not model_name:
        return False
    normalized = model_name.lower()
    return any(normalized == loaded or normalized in loaded for loaded in loaded_models)


def _model_status(
    device_online: bool,
    model_name: str,
    loaded_models: set[str],
    model_error: Any,
) -> str:
    if not device_online:
        return "offline"
    if model_error:
        return "warn"
    if not model_name:
        return "warn"
    return "ok" if _monitor_model_is_loaded(model_name, loaded_models) else "warn"


def _enabled_node_status(
    device_online: bool,
    edge_devices: dict[str, Any],
    name: str,
) -> str:
    if not device_online:
        return "offline"
    details = _as_dict(edge_devices.get(name))
    if not details:
        return "warn"
    return "ok" if bool(details.get("enabled", True)) else "warn"


def _binary_activity_status(device_online: bool, active: bool) -> str:
    if not device_online:
        return "offline"
    return "active" if active else "idle"


def _degraded_status(device_online: bool, degraded: bool, *, active: bool) -> str:
    if not device_online:
        return "offline"
    if degraded:
        return "warn"
    return "active" if active else "idle"


def _vision_detection_status(
    device_online: bool,
    last_events: dict[str, Any],
) -> str:
    if not device_online:
        return "offline"
    vision = _as_dict(last_events.get("vision"))
    event_type = str(vision.get("event_type", ""))
    if "present" in event_type:
        return "active"
    if vision:
        return "idle"
    return "warn"


def _connector_status(connector: dict[str, Any] | None) -> str:
    if not connector:
        return "warn"
    if connector.get("reachable"):
        return "ok"
    if connector.get("configured"):
        return "warn"
    return "offline"


def _connector_detail(connector: dict[str, Any] | None) -> str:
    if not connector:
        return "no status"
    message = str(connector.get("message", ""))
    configured = "configured" if connector.get("configured") else "not configured"
    return f"{configured}; {message or '-'}"


def _models_detail(models: dict[str, Any]) -> str:
    if models.get("error"):
        return str(models["error"])
    return f"{len(models.get('items', []))} loaded"


def _device_detail(value: Any) -> str:
    details = _as_dict(value)
    if not details:
        return "not reported"
    parts: list[str] = []
    for key in ("source", "device", "width", "height", "fps", "sample_rate", "engine"):
        if key in details and details[key] not in {None, ""}:
            parts.append(f"{key}={details[key]}")
    return ", ".join(parts) if parts else json.dumps(details, ensure_ascii=False)


def _event_text(event: Any) -> str:
    if not event:
        return "无最近事件"
    if not isinstance(event, dict):
        return str(event)
    parts: list[str] = []
    for key in (
        "event_type",
        "phase",
        "reason",
        "name",
        "success",
        "duration_ms",
        "bytes",
        "error_message",
    ):
        if key in event and event[key] not in {None, ""}:
            parts.append(f"{key}={event[key]}")
    return ", ".join(parts) if parts else json.dumps(event, ensure_ascii=False)


def _overall_monitor_status(
    groups: list[dict[str, Any]],
    device_online: bool,
    alerts: list[dict[str, str]],
) -> str:
    if not device_online:
        return "offline"
    if alerts:
        return "warning"
    return "healthy"


def _overall_monitor_message(
    status: str,
    latest_device: dict[str, Any] | None,
) -> str:
    if status == "offline":
        return "等待边缘设备连接"
    if status == "warning":
        return "存在需要关注的链路节点"
    device_id = latest_device.get("device_id") if latest_device else "-"
    return f"{device_id} 工作流正常"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _single_line_log_text(text: str, *, max_chars: int = 240) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "...(truncated)"


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
        history_store: ConversationHistoryStore | None = None,
        registry: EdgeDeviceRegistry | None = None,
    ) -> None:
        self.device_id = device_id
        self.orchestrator = orchestrator
        self.history_store = history_store
        self.registry = registry
        self.request_id = ""
        self.sample_rate = 16000
        self.channels = 1
        self.context: dict[str, Any] = {}
        self.audio_buffer = bytearray()
        self._send_lock = asyncio.Lock()
        self._active_monitor_node = "microphone"

    def start(self, envelope: dict[str, Any]) -> None:
        payload = envelope.get("payload", {})
        audio = payload.get("audio", {})
        self.request_id = str(envelope.get("request_id", "")).strip()
        self.sample_rate = int(audio.get("sample_rate", 16000))
        self.channels = int(audio.get("channels", 1))
        self.context = dict(payload.get("context", {}))
        self.audio_buffer.clear()
        self._record_server_activity(
            node_id="microphone",
            phase="receiving_audio",
            status="active",
            message="receiving utterance audio",
        )

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
            request_started = time.perf_counter()
            self._record_server_activity(
                node_id="asr",
                phase="asr",
                status="active",
                message="transcribing audio",
            )
            asr_started = time.perf_counter()
            question = await self.orchestrator.transcribe_audio(
                filename="utterance.wav",
                audio_bytes=wav_bytes,
                content_type="audio/wav",
            )
            asr_text_for_log = _single_line_log_text(question)
            LOGGER.info(
                "ASR耗时 request_id=%s elapsed_ms=%.1f text_chars=%s",
                self.request_id,
                (time.perf_counter() - asr_started) * 1000,
                len(question),
            )
            LOGGER.info(
                "ASR文本 request_id=%s device_id=%s session_id=%s text_chars=%s text=%r",
                self.request_id,
                self.device_id,
                self._session_id() or "-",
                len(question),
                asr_text_for_log,
            )
            if not question.strip():
                raise ValueError("ASR returned empty text")
            await self._send_frame(
                websocket,
                "asr.final",
                self.request_id,
                {"text": question, "reason": payload.get("reason", "vad_silence")},
            )
            self._record_server_activity(
                node_id="llm_ragflow",
                phase="llm_ragflow",
                status="active",
                message="waiting for RAGFlow answer",
            )
            answer_context = self._context_with_history()
            full_answer = await self._stream_answer_with_tts(
                websocket,
                question,
                context=answer_context,
            )
            LOGGER.info(
                "语音问答总耗时 request_id=%s elapsed_ms=%.1f answer_chars=%s",
                self.request_id,
                (time.perf_counter() - request_started) * 1000,
                len(full_answer),
            )
            self._record_history_turn(question, full_answer)
            await self._send_frame(
                websocket,
                "llm.final",
                self.request_id,
                {
                    "text": full_answer,
                    "rag_sources": [],
                },
            )
            self._record_server_activity(
                node_id="speaker",
                phase="completed",
                status="idle",
                message="turn completed",
            )
        except Exception as exc:
            self._record_server_activity(
                node_id=self._active_monitor_node,
                phase=self._active_monitor_node,
                status="error",
                message=str(exc),
            )
            await _send_error_frame(
                websocket,
                request_id=self.request_id,
                code="conversation_failed",
                message=str(exc),
                retryable=True,
            )
        finally:
            self.audio_buffer.clear()

    def _context_with_history(self) -> dict[str, Any]:
        context = dict(self.context)
        if self.history_store is None:
            return context
        history = self.history_store.for_context(
            device_id=self.device_id,
            session_id=self._session_id(),
        )
        if history:
            context["conversation_history"] = history
        return context

    def _record_history_turn(self, question: str, answer: str) -> None:
        if self.history_store is None:
            return
        self.history_store.record_turn(
            device_id=self.device_id,
            session_id=self._session_id(),
            user_text=question,
            assistant_text=answer,
        )

    def _session_id(self) -> str:
        return str(self.context.get("session_id", "")).strip()

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
            await self._send_frame(
                websocket,
                "llm.final",
                request_id,
                {"text": welcome_text, "rag_sources": []},
            )
            await self._send_tts_audio(
                websocket,
                request_id=request_id,
                sequence=1,
                audio_out=audio_out,
                media_type=media_type or "audio/wav",
                is_final=True,
                segment_text=welcome_text,
            )
        except Exception as exc:
            await _send_error_frame(
                websocket,
                request_id=request_id,
                code="welcome_failed",
                message=str(exc),
                retryable=True,
            )

    async def _stream_answer_with_tts(
        self,
        websocket: WebSocket,
        question: str,
        context: dict[str, Any],
    ) -> str:
        queue: asyncio.Queue[_QueuedTtsSegment | None] = asyncio.Queue()
        tts_task = asyncio.create_task(self._run_tts_queue(websocket, queue))
        full_answer_parts: list[str] = []
        segment_buffer = ""
        answer_started = time.perf_counter()
        first_delta_seen = False
        real_segment_queued = False
        streaming_segments_queued = False
        try:
            async for delta in self.orchestrator.stream_answer(question, context):
                if not delta:
                    continue
                if not first_delta_seen:
                    first_delta_seen = True
                    LOGGER.info(
                        "RAGFlow首字耗时 request_id=%s elapsed_ms=%.1f",
                        self.request_id,
                        (time.perf_counter() - answer_started) * 1000,
                    )
                full_answer_parts.append(delta)
                segment_buffer += delta
                await self._send_frame(
                    websocket,
                    "llm.partial",
                    self.request_id,
                    {"text": delta},
                )
                ready_segments, segment_buffer = _split_ready_tts_segments(segment_buffer)
                for segment in ready_segments:
                    await queue.put(_QueuedTtsSegment(text=segment, is_final=False))
                    real_segment_queued = True
                    streaming_segments_queued = True

            full_answer = "".join(full_answer_parts).strip()
            if not full_answer:
                raise ValueError("RAGFlow returned empty streamed answer")
            LOGGER.info(
                "RAGFlow总耗时 request_id=%s elapsed_ms=%.1f answer_chars=%s",
                self.request_id,
                (time.perf_counter() - answer_started) * 1000,
                len(full_answer),
            )

            final_segment = segment_buffer.strip()
            if final_segment:
                await queue.put(
                    _QueuedTtsSegment(
                        text=final_segment,
                        is_final=not streaming_segments_queued,
                    )
                )
                real_segment_queued = True
            elif not real_segment_queued:
                await queue.put(_QueuedTtsSegment(text=full_answer, is_final=True))
                real_segment_queued = True

            await queue.put(None)
            next_sequence = await tts_task
            if streaming_segments_queued:
                await self._send_silent_final_audio(websocket, sequence=next_sequence)
            return full_answer
        except Exception:
            tts_task.cancel()
            await asyncio.gather(tts_task, return_exceptions=True)
            raise

    async def _run_tts_queue(
        self,
        websocket: WebSocket,
        queue: asyncio.Queue[_QueuedTtsSegment | None],
    ) -> int:
        sequence = 1
        while True:
            item = await queue.get()
            if item is None:
                return sequence
            self._record_server_activity(
                node_id="tts",
                phase="tts",
                status="active",
                message="synthesizing speech",
            )
            tts_started = time.perf_counter()
            audio_out, media_type = await self.orchestrator.synthesize_text(item.text)
            tts_elapsed_ms = (time.perf_counter() - tts_started) * 1000
            send_started = time.perf_counter()
            await self._send_tts_audio(
                websocket,
                request_id=self.request_id,
                sequence=sequence,
                audio_out=audio_out,
                media_type=media_type or "audio/wav",
                is_final=item.is_final,
                segment_text=item.text,
            )
            LOGGER.info(
                "每段TTS耗时 request_id=%s sequence=%s chars=%s final=%s tts_ms=%.1f 发送音频耗时_ms=%.1f",
                self.request_id,
                sequence,
                len(item.text),
                item.is_final,
                tts_elapsed_ms,
                (time.perf_counter() - send_started) * 1000,
            )
            sequence += 1

    def _record_server_activity(
        self,
        *,
        node_id: str,
        phase: str,
        status: str,
        message: str,
    ) -> None:
        self._active_monitor_node = node_id
        if self.registry is None:
            return
        now_ms = int(time.time() * 1000)
        self.registry.update_server_activity(
            self.device_id,
            {
                "node_id": node_id,
                "phase": phase,
                "status": status,
                "message": message,
                "request_id": self.request_id,
                "started_at_ms": now_ms if status == "active" else None,
                "updated_at_ms": now_ms,
            },
        )

    async def _send_silent_final_audio(
        self,
        websocket: WebSocket,
        *,
        sequence: int,
    ) -> None:
        frame_count = max(1, int(self.sample_rate * 0.02))
        silence = b"\0" * frame_count * max(1, self.channels) * 2
        audio_out = _pcm16_to_wav(
            silence,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )
        send_started = time.perf_counter()
        await self._send_tts_audio(
            websocket,
            request_id=self.request_id,
            sequence=sequence,
            audio_out=audio_out,
            media_type="audio/wav",
            is_final=True,
            segment_text="",
        )
        LOGGER.info(
            "发送音频耗时 request_id=%s sequence=%s final_marker=true elapsed_ms=%.1f",
            self.request_id,
            sequence,
            (time.perf_counter() - send_started) * 1000,
        )

    async def _send_frame(
        self,
        websocket: WebSocket,
        frame_type: str,
        request_id: str,
        payload: dict[str, Any],
    ) -> None:
        async with self._send_lock:
            await websocket.send_text(_frame(frame_type, request_id, payload))

    async def _send_tts_audio(
        self,
        websocket: WebSocket,
        *,
        request_id: str,
        sequence: int,
        audio_out: bytes,
        media_type: str,
        is_final: bool,
        segment_text: str,
    ) -> None:
        async with self._send_lock:
            await websocket.send_text(
                _frame(
                    "tts.chunk",
                    request_id,
                    {
                        "sequence": sequence,
                        "encoding": (
                            "wav"
                            if "wav" in media_type.lower()
                            else "pcm_s16le"
                        ),
                        "sample_rate": self.sample_rate,
                        "channels": self.channels,
                        "is_final": is_final,
                        "media_type": media_type,
                        "segment_text": segment_text,
                    },
                )
            )
            await websocket.send_bytes(audio_out)


@dataclass(frozen=True)
class _QueuedTtsSegment:
    text: str
    is_final: bool


def _split_ready_tts_segments(text: str) -> tuple[list[str], str]:
    ready: list[str] = []
    start = 0
    last_emit = 0
    punctuation = "。！？；!?\n"
    soft_punctuation = "，,：:"
    for index, char in enumerate(text):
        length = index - start + 1
        should_emit = char in punctuation
        if not should_emit and length >= 48 and char in soft_punctuation:
            should_emit = True
        if not should_emit and length >= 72:
            should_emit = True
        if should_emit:
            segment = text[start : index + 1].strip()
            if segment:
                ready.append(segment)
            last_emit = index + 1
            start = index + 1
    return ready, text[last_emit:]


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


SERVER_MONITOR_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Robot Workflow Monitor</title>
  <style>
    :root {
      color-scheme: light;
      --bg:#f6f8fa;
      --panel:#ffffff;
      --ink:#172126;
      --muted:#66737b;
      --line:#d9e0e4;
      --soft:#eef3f5;
      --ok:#0b7a75;
      --active:#1d65c1;
      --warn:#b45f06;
      --offline:#8a1f2d;
      --shadow:0 12px 26px rgba(24, 42, 54, .08);
    }
    * { box-sizing:border-box; }
    body {
      margin:0;
      min-width:320px;
      font-family: ui-sans-serif, "Avenir Next", "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background:var(--bg);
      color:var(--ink);
      letter-spacing:0;
    }
    header {
      display:flex;
      align-items:flex-start;
      justify-content:space-between;
      gap:18px;
      padding:22px 28px 16px;
      border-bottom:1px solid var(--line);
      background:#ffffffee;
      position:sticky;
      top:0;
      z-index:10;
      backdrop-filter: blur(10px);
    }
    h1 { margin:0; font-size:24px; line-height:1.2; font-weight:760; }
    h2 { margin:0; font-size:16px; line-height:1.3; }
    h3 { margin:0; font-size:14px; line-height:1.35; }
    p { margin:4px 0 0; color:var(--muted); font-size:13px; line-height:1.45; }
    button {
      border:1px solid var(--ink);
      background:var(--ink);
      color:#fff;
      border-radius:6px;
      padding:8px 11px;
      font-size:13px;
      cursor:pointer;
    }
    button.secondary { background:#fff; color:var(--ink); border-color:var(--line); }
    input {
      border:1px solid var(--line);
      border-radius:6px;
      padding:9px 10px;
      font-size:13px;
      min-width:250px;
      background:#fff;
      color:var(--ink);
    }
    main { padding:18px 28px 34px; display:grid; gap:16px; }
    .toolbar { display:flex; align-items:center; justify-content:flex-end; gap:8px; flex-wrap:wrap; }
    .status-pill {
      display:inline-flex;
      align-items:center;
      gap:7px;
      min-height:32px;
      border:1px solid var(--line);
      border-radius:999px;
      padding:6px 10px;
      background:#fff;
      color:var(--ink);
      font-size:13px;
      white-space:nowrap;
    }
    .dot { width:9px; height:9px; border-radius:99px; background:var(--muted); display:inline-block; }
    .dot.healthy, .dot.ok { background:var(--ok); }
    .dot.active { background:var(--active); }
    .dot.warning, .dot.warn, .dot.idle, .dot.stuck { background:var(--warn); }
    .dot.offline, .dot.error { background:var(--offline); }
    .panel {
      background:var(--panel);
      border:1px solid var(--line);
      border-radius:8px;
      box-shadow:var(--shadow);
      padding:16px;
    }
    .section-head { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; margin-bottom:14px; }
    .muted { color:var(--muted); font-size:12px; line-height:1.4; }
    .token-panel {
      display:none;
      align-items:center;
      gap:8px;
      flex-wrap:wrap;
      border:1px solid #e6c06a;
      background:#fff8e8;
      border-radius:8px;
      padding:12px;
    }
    .token-panel.visible { display:flex; }
    .overview { display:grid; grid-template-columns:repeat(4, minmax(150px, 1fr)); gap:10px; }
    .metric {
      border:1px solid var(--line);
      border-radius:8px;
      padding:12px;
      min-height:78px;
      background:#fff;
    }
    .metric strong { display:block; font-size:20px; line-height:1.15; overflow-wrap:anywhere; }
    .metric span { display:block; margin-top:6px; color:var(--muted); font-size:12px; }
    .workflow { display:grid; grid-template-columns:repeat(3, minmax(220px, 1fr)); gap:14px; }
    .lane { display:grid; gap:10px; align-content:start; }
    .lane-title {
      display:flex;
      align-items:center;
      justify-content:space-between;
      padding-bottom:7px;
      border-bottom:1px solid var(--line);
    }
    .nodes { display:grid; gap:8px; }
    .node {
      border:1px solid var(--line);
      border-radius:8px;
      padding:10px;
      background:#fff;
      min-height:104px;
    }
    .node.health-warn, .node.activity-stuck { border-color:#e0a849; background:#fffaf0; }
    .node.health-error, .node.activity-error { border-color:#d58a95; background:#fff5f6; }
    .node.activity-active { border-color:#8ab6ea; background:#f6fbff; }
    .node-top { display:flex; align-items:center; justify-content:space-between; gap:8px; }
    .node-title { display:flex; align-items:center; gap:8px; min-width:0; }
    .node-title strong { font-size:14px; overflow-wrap:anywhere; }
    .badges { display:flex; align-items:center; gap:5px; flex-wrap:wrap; justify-content:flex-end; }
    .badge {
      border:1px solid var(--line);
      border-radius:999px;
      padding:3px 7px;
      color:var(--muted);
      font-size:11px;
      line-height:1;
      white-space:nowrap;
    }
    .badge.health-ok { color:var(--ok); border-color:#a8d7d4; background:#f1fbfa; }
    .badge.health-warn { color:var(--warn); border-color:#ecc47e; background:#fff8e8; }
    .badge.health-error, .badge.health-offline { color:var(--offline); border-color:#e6a5ae; background:#fff1f3; }
    .badge.activity-active { color:var(--active); border-color:#9cc2ed; background:#eef6ff; }
    .badge.activity-stuck { color:var(--warn); border-color:#e0a849; background:#fff4d8; }
    .badge.activity-error { color:var(--offline); border-color:#e6a5ae; background:#fff1f3; }
    .diagnostics { display:grid; grid-template-columns:minmax(220px,.75fr) 1.25fr; gap:12px; }
    .phase-card, .alert-list {
      border:1px solid var(--line);
      border-radius:8px;
      background:#fff;
      padding:12px;
      min-height:88px;
    }
    .phase-card strong { display:block; font-size:18px; overflow-wrap:anywhere; }
    .phase-card span { display:block; margin-top:6px; color:var(--muted); font-size:12px; }
    .alert-list { display:grid; gap:8px; align-content:start; }
    .alert-item {
      display:grid;
      grid-template-columns:92px 1fr;
      gap:8px;
      align-items:start;
      border:1px solid var(--line);
      border-radius:8px;
      padding:9px;
      background:#fbfcfd;
      font-size:12px;
    }
    .alert-item.error { border-color:#d58a95; background:#fff5f6; }
    .alert-item.warn { border-color:#e0a849; background:#fffaf0; }
    .alert-item.offline { border-color:#d58a95; background:#fff5f6; }
    .alert-item strong { font-size:13px; }
    .alert-item span { color:var(--muted); overflow-wrap:anywhere; }
    .detail {
      margin-top:8px;
      color:var(--muted);
      font-size:12px;
      line-height:1.35;
      min-height:32px;
      overflow-wrap:anywhere;
    }
    .counter-row {
      margin-top:8px;
      display:flex;
      gap:6px;
      flex-wrap:wrap;
    }
    .counter {
      background:var(--soft);
      border-radius:6px;
      padding:4px 6px;
      font-size:11px;
      color:#33434b;
    }
    .two-col { display:grid; grid-template-columns:1.2fr .8fr; gap:16px; }
    table { width:100%; border-collapse:collapse; font-size:12px; }
    th, td { text-align:left; border-bottom:1px solid var(--line); padding:8px 6px; vertical-align:top; overflow-wrap:anywhere; }
    th { color:var(--muted); font-weight:650; background:#fafcfd; }
    .kv-grid { display:grid; grid-template-columns:repeat(4, minmax(110px, 1fr)); gap:8px; }
    .kv {
      border:1px solid var(--line);
      border-radius:8px;
      padding:9px;
      min-height:58px;
      background:#fff;
    }
    .kv span { display:block; color:var(--muted); font-size:11px; }
    .kv strong { display:block; margin-top:5px; font-size:14px; overflow-wrap:anywhere; }
    pre {
      margin:0;
      max-height:280px;
      overflow:auto;
      white-space:pre-wrap;
      word-break:break-word;
      border:1px solid var(--line);
      border-radius:8px;
      padding:10px;
      background:#fbfcfd;
      color:#223139;
      font-size:12px;
      line-height:1.45;
    }
    .empty {
      border:1px dashed var(--line);
      border-radius:8px;
      padding:16px;
      color:var(--muted);
      background:#fbfcfd;
      font-size:13px;
    }
    @media (max-width: 980px) {
      header { padding:18px; flex-direction:column; }
      main { padding:14px 18px 28px; }
      .overview, .workflow, .two-col, .kv-grid, .diagnostics { grid-template-columns:1fr; }
      input { min-width:0; width:100%; }
      .toolbar { justify-content:flex-start; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>AI Robot Workflow Monitor</h1>
      <p>只读展示感知、跟踪、语音响应和服务连接状态。</p>
    </div>
    <div class="toolbar">
      <span class="status-pill"><span id="statusDot" class="dot"></span><span id="overallText">等待数据</span></span>
      <button class="secondary" type="button" onclick="refreshNow()">刷新</button>
    </div>
  </header>
  <main>
    <div id="tokenPanel" class="token-panel">
      <strong>需要 admin token</strong>
      <input id="tokenInput" type="password" placeholder="ai-robot-gateway-20260620" autocomplete="off">
      <button type="button" onclick="saveToken()">保存并刷新</button>
      <span id="tokenHint" class="muted">token 保存在 localStorage.adminToken。</span>
    </div>

    <section class="panel">
      <div class="section-head">
        <div>
          <h2>Overview</h2>
          <p>每 2 秒自动刷新一次。</p>
        </div>
        <span id="lastUpdated" class="muted">未刷新</span>
      </div>
      <div id="overview" class="overview"></div>
    </section>

    <section class="panel">
      <div class="section-head">
        <div>
          <h2>Diagnostics</h2>
          <p>当前阶段和异常环节会优先暴露。</p>
        </div>
      </div>
      <div class="diagnostics">
        <div id="activeStage" class="phase-card"></div>
        <div id="alerts" class="alert-list"></div>
      </div>
    </section>

    <section class="panel">
      <div class="section-head">
        <div>
          <h2>Workflow</h2>
          <p>从摄像头到语音输出的完整只读链路。</p>
        </div>
      </div>
      <div id="workflow" class="workflow"></div>
    </section>

    <div class="two-col">
      <section class="panel">
        <div class="section-head">
          <h2>Models</h2>
          <span id="modelStatus" class="muted"></span>
        </div>
        <div id="models"></div>
      </section>
      <section class="panel">
        <div class="section-head">
          <h2>Devices</h2>
          <span id="deviceStatus" class="muted"></span>
        </div>
        <div id="devices"></div>
      </section>
    </div>

    <section class="panel">
      <div class="section-head">
        <h2>Runtime Counters</h2>
        <span class="muted">来自边缘设备 status.monitoring。</span>
      </div>
      <div id="runtime" class="kv-grid"></div>
    </section>

    <section class="panel">
      <div class="section-head">
        <h2>Recent Logs</h2>
        <span class="muted">仅展示 registry 中缓存的设备日志。</span>
      </div>
      <pre id="logs">等待数据</pre>
    </section>
  </main>
  <script>
    const TOKEN_KEY = "adminToken";
    const LEGACY_TOKEN_KEY = "aiRobotAdminToken";
    let lastPayload = null;
    let refreshing = false;

    function token() {
      return localStorage.getItem(TOKEN_KEY) || localStorage.getItem(LEGACY_TOKEN_KEY) || "";
    }

    function authHeaders() {
      const value = token();
      return value ? {"Authorization": "Bearer " + value} : {};
    }

    function saveToken() {
      const value = document.getElementById("tokenInput").value.trim();
      if (value) {
        localStorage.setItem(TOKEN_KEY, value);
      }
      refreshNow();
    }

    async function refreshNow() {
      if (refreshing) return;
      refreshing = true;
      try {
        const response = await fetch("/api/v1/monitor/summary", {headers: authHeaders()});
        if (response.status === 401) {
          showTokenPanel(true, "token 无效或缺失");
          setOverall("offline", "需要 admin token");
          return;
        }
        const text = await response.text();
        if (!response.ok) {
          throw new Error(text || ("HTTP " + response.status));
        }
        const payload = JSON.parse(text);
        lastPayload = payload;
        showTokenPanel(false, "");
        render(payload);
      } catch (error) {
        setOverall("warning", "刷新失败");
        document.getElementById("logs").textContent = String(error);
      } finally {
        refreshing = false;
      }
    }

    function showTokenPanel(visible, message) {
      const panel = document.getElementById("tokenPanel");
      panel.classList.toggle("visible", visible);
      document.getElementById("tokenHint").textContent = message || "token 保存在 localStorage.adminToken。";
    }

    function render(payload) {
      const workflow = payload.workflow || {};
      const overall = workflow.overall || {};
      setOverall(overall.status || "warning", statusLabel(overall.status) + " / " + (overall.message || "-"));
      document.getElementById("lastUpdated").textContent = "更新时间 " + fmtTime(payload.timestamp_ms);
      renderOverview(payload);
      renderDiagnostics(payload);
      renderWorkflow(workflow.groups || []);
      renderModels(payload.models || {});
      renderDevices(payload.devices || []);
      renderRuntime(workflow);
      renderLogs(workflow.recent_logs || []);
    }

    function setOverall(status, text) {
      const dot = document.getElementById("statusDot");
      dot.className = "dot " + (status || "warning");
      document.getElementById("overallText").textContent = text;
    }

    function renderOverview(payload) {
      const workflow = payload.workflow || {};
      const server = payload.server || {};
      const models = payload.models || {};
      const configured = payload.configured || {};
      const devices = payload.devices || [];
      const stage = (workflow.active_stage || {});
      const alerts = workflow.alerts || [];
      const online = devices.filter(device => device && device.online).length;
      const cells = [
        ["总体状态", statusLabel(workflow.overall && workflow.overall.status)],
        ["在线设备", online + " / " + devices.length],
        ["当前阶段", stageLabel(stage)],
        ["异常环节", alerts.length ? alerts.length + " 个" : "无"],
        ["模型数量", String((models.items || []).length)],
        ["RAGFlow Chat", configured.ragflow_chat_id || "-"],
        ["ASR 模型", configured.asr_model || "-"],
        ["TTS Provider", configured.tts_provider || "xinference"],
        ["TTS 模型", configured.tts_model || "-"],
        ["当前设备", (workflow.overall && workflow.overall.active_device_id) || "等待连接"],
      ];
      document.getElementById("overview").innerHTML = cells.map(([label, value]) => `
        <div class="metric"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>
      `).join("");
    }

    function renderDiagnostics(payload) {
      const workflow = payload.workflow || {};
      const stage = workflow.active_stage || {};
      const alerts = workflow.alerts || [];
      const elapsed = stage.elapsed_ms ? Math.round(stage.elapsed_ms / 100) / 10 + "s" : "-";
      document.getElementById("activeStage").innerHTML = `
        <strong>${escapeHtml(stageLabel(stage))}</strong>
        <span>活动状态：${escapeHtml(activityLabel(stage.activity || "idle"))}</span>
        <span>耗时：${escapeHtml(elapsed)}</span>
        <span>${escapeHtml(stage.message || "")}</span>
      `;
      if (!alerts.length) {
        document.getElementById("alerts").innerHTML = '<div class="empty">当前没有检测到异常环节</div>';
        return;
      }
      document.getElementById("alerts").innerHTML = alerts.map(alert => `
        <div class="alert-item ${escapeHtml(alert.level || "warn")}">
          <strong>${escapeHtml(alert.label || alert.node_id || "-")}</strong>
          <span>${escapeHtml(alert.message || alert.level || "-")}</span>
        </div>
      `).join("");
    }

    function renderWorkflow(groups) {
      const target = document.getElementById("workflow");
      if (!groups.length) {
        target.innerHTML = '<div class="empty">等待边缘设备连接</div>';
        return;
      }
      target.innerHTML = groups.map(group => `
        <div class="lane">
          <div class="lane-title"><h3>${escapeHtml(group.label || group.id || "-")}</h3><span class="muted">${(group.nodes || []).length} 节点</span></div>
          <div class="nodes">${(group.nodes || []).map(renderNode).join("")}</div>
        </div>
      `).join("");
    }

    function renderNode(node) {
      const status = node.status || "warning";
      const health = node.health || healthFromStatus(status);
      const activity = node.activity || activityFromStatus(status);
      const marker = activity === "active" || activity === "stuck" || activity === "error" ? activity : health;
      const counters = Object.entries(node.counters || {})
        .filter(([, value]) => value !== null && value !== undefined && value !== "")
        .map(([key, value]) => `<span class="counter">${escapeHtml(key)}=${escapeHtml(String(value))}</span>`)
        .join("");
      return `
        <article class="node health-${escapeHtml(health)} activity-${escapeHtml(activity)}">
          <div class="node-top">
            <div class="node-title"><span class="dot ${escapeHtml(marker)}"></span><strong>${escapeHtml(node.label || node.id || "-")}</strong></div>
            <div class="badges">
              <span class="badge health-${escapeHtml(health)}">${escapeHtml(healthLabel(health))}</span>
              <span class="badge activity-${escapeHtml(activity)}">${escapeHtml(activityLabel(activity))}</span>
            </div>
          </div>
          <div class="detail">${escapeHtml(node.detail || node.event_text || "-")}</div>
          <div class="counter-row">${counters}</div>
        </article>
      `;
    }

    function renderModels(models) {
      const items = models.items || [];
      document.getElementById("modelStatus").textContent = models.error ? models.error : items.length + " loaded";
      if (!items.length) {
        document.getElementById("models").innerHTML = `<div class="empty">${escapeHtml(models.error || "没有加载模型")}</div>`;
        return;
      }
      document.getElementById("models").innerHTML = `
        <table>
          <thead><tr><th>模型</th><th>类型</th><th>状态</th></tr></thead>
          <tbody>${items.map(model => {
            const id = model.id || model.model || model.model_name || model.model_uid || model.name || "-";
            const type = model.model_type || model.type || model.object || "-";
            const state = model.status || model.state || "loaded";
            return `<tr><td>${escapeHtml(String(id))}</td><td>${escapeHtml(String(type))}</td><td>${escapeHtml(String(state))}</td></tr>`;
          }).join("")}</tbody>
        </table>
      `;
    }

    function renderDevices(devices) {
      document.getElementById("deviceStatus").textContent = devices.length ? devices.length + " registered" : "waiting";
      if (!devices.length) {
        document.getElementById("devices").innerHTML = '<div class="empty">等待边缘设备连接</div>';
        return;
      }
      document.getElementById("devices").innerHTML = `
        <table>
          <thead><tr><th>设备</th><th>在线</th><th>最近心跳</th></tr></thead>
          <tbody>${devices.map(device => `
            <tr>
              <td>${escapeHtml(String(device.device_id || "-"))}</td>
              <td>${device.online ? "online" : "offline"}</td>
              <td>${escapeHtml(fmtTime(device.last_seen_ms))}</td>
            </tr>
          `).join("")}</tbody>
        </table>
      `;
    }

    function renderRuntime(workflow) {
      const counters = workflow.counters || {};
      const queues = workflow.queues || {};
      const session = workflow.session || {};
      const rows = [
        ["session", session.state || "-"],
        ["reason", session.reason || "-"],
        ["turn_index", String(session.turn_index || 0)],
        ["vision_queue", String(queues.vision ?? "-")],
        ["conversation_queue", String(queues.conversation ?? "-")],
        ["tts_queue", String(queues.tts ?? "-")],
        ["turns_ok", String(counters.turns_ok || 0)],
        ["turns_failed", String(counters.turns_failed || 0)],
        ["utterances", String(counters.utterances || 0)],
        ["playback_chunks", String(counters.playback_chunks || 0)],
        ["vision_present", String(counters.vision_present || 0)],
        ["vision_absent", String(counters.vision_absent || 0)],
      ];
      document.getElementById("runtime").innerHTML = rows.map(([key, value]) => `
        <div class="kv"><span>${escapeHtml(key)}</span><strong>${escapeHtml(value)}</strong></div>
      `).join("");
    }

    function renderLogs(logs) {
      const target = document.getElementById("logs");
      target.textContent = logs.length ? logs.join("\\n") : "暂无 registry 设备日志";
    }

    function statusLabel(status) {
      const labels = {
        healthy: "正常",
        warning: "警告",
        offline: "离线",
        ok: "正常",
        active: "运行中",
        idle: "空闲",
        warn: "警告",
        error: "错误",
        stuck: "卡住",
      };
      return labels[status] || "未知";
    }

    function healthLabel(health) {
      const labels = {
        ok: "健康正常",
        warn: "健康警告",
        error: "健康错误",
        offline: "离线",
      };
      return labels[health] || "健康未知";
    }

    function activityLabel(activity) {
      const labels = {
        active: "运行中",
        idle: "空闲",
        waiting: "等待",
        stuck: "卡住",
        error: "错误",
        offline: "离线",
      };
      return labels[activity] || "状态未知";
    }

    function stageLabel(stage) {
      if (!stage || !stage.node_id) return "空闲";
      const names = {
        microphone: "接收语音",
        listening: "监听窗口",
        asr: "ASR 识别",
        llm_ragflow: "LLM/RAGFlow",
        tts: "TTS 合成",
        speaker: "扬声器播放",
        tracking: "视觉跟踪",
      };
      const base = names[stage.node_id] || stage.node_id;
      const state = activityLabel(stage.activity || "idle");
      return base + " / " + state;
    }

    function healthFromStatus(status) {
      if (status === "warn" || status === "warning" || status === "stuck") return "warn";
      if (status === "error") return "error";
      if (status === "offline") return "offline";
      return "ok";
    }

    function activityFromStatus(status) {
      if (status === "active" || status === "idle" || status === "stuck" || status === "error" || status === "offline") return status;
      return "idle";
    }

    function fmtTime(value) {
      if (!value) return "-";
      const date = new Date(Number(value));
      if (Number.isNaN(date.getTime())) return "-";
      return date.toLocaleString();
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    document.getElementById("tokenInput").addEventListener("keydown", event => {
      if (event.key === "Enter") saveToken();
    });
    refreshNow();
    setInterval(refreshNow, 2000);
  </script>
</body>
</html>"""


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
