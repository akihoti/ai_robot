from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse

from ..config import EdgeConfig, load_config
from .commands import CommandRejected, run_command
from .status import collect_edge_status, read_recent_logs

CONFIG_UPDATE_KEYS = {
    "server",
    "runtime",
    "camera",
    "vision",
    "microphone",
    "voice",
    "wake_word",
    "vad",
    "speaker",
    "servo",
    "tracking",
    "admin",
}


@dataclass(frozen=True)
class ParameterField:
    path: str
    label: str
    group: str
    input_type: str
    description: str
    options: tuple[str, ...] = ()
    min_value: float | int | None = None
    max_value: float | int | None = None
    step: float | int | None = None


PARAMETER_FIELDS: tuple[ParameterField, ...] = (
    ParameterField("runtime.mode", "运行模式", "基础", "select", "当前边缘端运行模式", ("simulated", "cpu", "npu")),
    ParameterField("runtime.log_level", "日志级别", "基础", "select", "运行日志详细程度", ("DEBUG", "INFO", "WARNING", "ERROR")),
    ParameterField("camera.enabled", "启用摄像头", "基础", "bool", "控制摄像头链路是否启用"),
    ParameterField("camera.fps", "摄像头 FPS", "基础", "int", "摄像头采样帧率", min_value=1, max_value=60, step=1),
    ParameterField("vision.detector", "视觉检测器", "视觉", "select", "人/脸检测后端", ("simulated", "cpu", "auto", "acl", "yolov5-face-om")),
    ParameterField("vision.stable_frames", "欢迎稳定帧数", "视觉", "int", "连续检测到人的帧数阈值", min_value=1, max_value=20, step=1),
    ParameterField("vision.welcome_cooldown_seconds", "欢迎冷却秒数", "视觉", "float", "两次欢迎之间的最短间隔", min_value=0, max_value=300, step=0.5),
    ParameterField("wake_word.enabled", "启用语音唤醒", "语音", "bool", "是否启用 wake word 检测"),
    ParameterField("wake_word.engine", "唤醒引擎", "语音", "text", "当前唤醒引擎名称"),
    ParameterField("wake_word.keyword_id", "唤醒词 ID", "语音", "text", "当前配置的唤醒词标识"),
    ParameterField("voice.visual_listen_timeout_ms", "视觉唤醒监听时长", "语音", "int", "欢迎后第一次监听窗口时长（毫秒）", min_value=500, max_value=30000, step=100),
    ParameterField("voice.followup_listen_timeout_ms", "追问监听时长", "语音", "int", "回答后追问监听窗口时长（毫秒）", min_value=500, max_value=30000, step=100),
    ParameterField("voice.auto_listen_after_welcome", "欢迎后自动监听", "语音", "bool", "欢迎语播完后是否立即进入监听"),
    ParameterField("voice.speech_interrupt_enabled", "启用插话打断", "语音", "bool", "机器人说话时是否允许用户插话打断"),
    ParameterField("voice.suppress_mic_while_speaking", "说话时抑制麦克风", "语音", "bool", "防止 TTS 被重新录进麦克风"),
    ParameterField("voice.welcome_once_per_session", "单会话只欢迎一次", "语音", "bool", "同一会话内避免重复欢迎"),
    ParameterField("voice.welcome_text", "欢迎语文本", "语音", "text", "视觉/语音唤醒后的默认欢迎语"),
    ParameterField("vad.energy_threshold", "VAD 能量阈值", "VAD", "float", "静音与语音分界阈值", min_value=0.001, max_value=1.0, step=0.001),
    ParameterField("vad.silence_ms", "VAD 静音判停", "VAD", "int", "静音持续多久判定一句话结束", min_value=100, max_value=5000, step=50),
    ParameterField("vad.max_utterance_ms", "最长句长", "VAD", "int", "单次 utterance 最长时长", min_value=1000, max_value=60000, step=100),
    ParameterField("tracking.enabled", "启用头部跟踪", "跟踪", "bool", "是否启用云台/头部跟踪"),
    ParameterField("tracking.tilt_enabled", "启用俯仰跟踪", "跟踪", "bool", "是否启用垂直方向跟踪"),
    ParameterField("tracking.target_stickiness", "目标粘性", "跟踪", "float", "保持当前主目标的倾向", min_value=0, max_value=1, step=0.01),
    ParameterField("tracking.target_lock_timeout_ms", "目标锁定窗口", "跟踪", "int", "短时丢失主目标后的保留时间", min_value=0, max_value=10000, step=50),
    ParameterField("tracking.target_lost_timeout_seconds", "丢目标回中延时", "跟踪", "float", "目标消失后延迟多久执行回中", min_value=0, max_value=60, step=0.1),
    ParameterField("tracking.idle_return_to_center_seconds", "无人回中延时", "跟踪", "float", "无人时多久回到待机朝向", min_value=0, max_value=60, step=0.1),
    ParameterField("servo.enabled", "启用舵机", "设备", "bool", "是否启用真实舵机控制"),
    ParameterField("speaker.enabled", "启用扬声器", "设备", "bool", "是否启用音频播放"),
    ParameterField("admin.allow_remote_ops", "允许远程运维命令", "运维", "bool", "允许后台执行拉取/重启等操作"),
)


def create_edge_admin_app(config_path: str | Path) -> FastAPI:
    app = FastAPI(title="AI Robot Edge Admin", version="0.1.0")

    def current_config() -> EdgeConfig:
        return load_config(config_path)

    async def require_admin(
        authorization: Optional[str] = Header(default=None),
        x_admin_token: Optional[str] = Header(default=None),
    ) -> None:
        config = current_config()
        token = _extract_token(authorization) or x_admin_token
        if config.admin.auth_token == "change-me":
            return
        if token != config.admin.auth_token:
            raise HTTPException(status_code=401, detail="invalid admin token")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return EDGE_ADMIN_HTML

    @app.get("/api/v1/edge/status", dependencies=[Depends(require_admin)])
    async def status() -> dict[str, Any]:
        return collect_edge_status(current_config())

    @app.get("/api/v1/edge/logs", dependencies=[Depends(require_admin)])
    async def logs() -> dict[str, Any]:
        config = current_config()
        return {"lines": read_recent_logs(config.admin.log_path)}

    @app.get("/api/v1/edge/config", dependencies=[Depends(require_admin)])
    async def get_config() -> dict[str, Any]:
        return _read_config_yaml(config_path)

    @app.get("/api/v1/edge/parameters", dependencies=[Depends(require_admin)])
    async def get_parameters() -> dict[str, Any]:
        config = _read_config_yaml(config_path)
        return {
            "fields": [_parameter_field_to_json(field) for field in PARAMETER_FIELDS],
            "values": _parameter_values(config),
        }

    @app.put("/api/v1/edge/parameters", dependencies=[Depends(require_admin)])
    async def update_parameters(payload: dict[str, Any]) -> dict[str, Any]:
        values = payload.get("values")
        if not isinstance(values, dict):
            raise HTTPException(status_code=400, detail="values must be an object")
        patch = _parameter_patch(values)
        updated = update_config_yaml(config_path, patch)
        load_config(config_path)
        return {
            "ok": True,
            "values": _parameter_values(updated),
        }

    @app.put("/api/v1/edge/config", dependencies=[Depends(require_admin)])
    async def update_config(payload: dict[str, Any]) -> dict[str, Any]:
        updated = update_config_yaml(config_path, payload)
        load_config(config_path)
        return {"ok": True, "config": updated}

    @app.get("/api/v1/edge/commands", dependencies=[Depends(require_admin)])
    async def commands() -> dict[str, Any]:
        config = current_config()
        return {
            "allowed_commands": list(config.admin.allowed_commands),
            "allow_remote_ops": config.admin.allow_remote_ops,
        }

    @app.post(
        "/api/v1/edge/tests/{test_name}",
        dependencies=[Depends(require_admin)],
    )
    async def run_test(test_name: str) -> dict[str, Any]:
        command_name = f"test_{test_name}"
        try:
            return await run_command(current_config(), command_name)
        except CommandRejected as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.post(
        "/api/v1/edge/commands/{command_name}",
        dependencies=[Depends(require_admin)],
    )
    async def command(command_name: str) -> dict[str, Any]:
        try:
            return await run_command(current_config(), command_name)
        except CommandRejected as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    return app


def update_config_yaml(path: str | Path, patch: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(patch, dict):
        raise HTTPException(status_code=400, detail="config patch must be an object")
    invalid = sorted(set(patch) - CONFIG_UPDATE_KEYS)
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported config sections: {', '.join(invalid)}",
        )
    data = _read_config_yaml(path)
    for key, value in patch.items():
        if value is None:
            continue
        if not isinstance(value, dict):
            raise HTTPException(status_code=400, detail=f"{key} must be an object")
        section = data.setdefault(key, {})
        if not isinstance(section, dict):
            raise HTTPException(status_code=400, detail=f"{key} is not an object")
        section.update(value)
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data


def _read_config_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="config root is not an object")
    return data


def _extract_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    prefix = "Bearer "
    if authorization.startswith(prefix):
        return authorization[len(prefix) :]
    return authorization


def _parameter_field_to_json(field: ParameterField) -> dict[str, Any]:
    return {
        "path": field.path,
        "label": field.label,
        "group": field.group,
        "input_type": field.input_type,
        "description": field.description,
        "options": list(field.options),
        "min_value": field.min_value,
        "max_value": field.max_value,
        "step": field.step,
    }


def _parameter_values(config_data: dict[str, Any]) -> dict[str, Any]:
    return {field.path: _get_nested(config_data, field.path) for field in PARAMETER_FIELDS}


def _parameter_patch(values: dict[str, Any]) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    for field in PARAMETER_FIELDS:
        if field.path not in values:
            continue
        section, key = field.path.split(".", 1)
        section_patch = patch.setdefault(section, {})
        section_patch[key] = _coerce_parameter_value(field, values[field.path])
    return patch


def _coerce_parameter_value(field: ParameterField, value: Any) -> Any:
    if field.input_type == "bool":
        return bool(value)
    if field.input_type == "int":
        return int(value)
    if field.input_type == "float":
        return float(value)
    return "" if value is None else str(value)


def _get_nested(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


EDGE_ADMIN_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Atlas Edge Console</title>
  <style>
    :root {
      --ink:#172229; --muted:#66747a; --line:#d5dfe3; --accent:#0f8f63; --accent-2:#116149;
      --bg:#f4f7f8; --panel:#ffffff; --warn:#b6541a; --good:#0d7a4f;
    }
    * { box-sizing:border-box; }
    body { margin:0; font-family: ui-sans-serif, "Avenir Next", "Segoe UI", sans-serif; background:var(--bg); color:var(--ink); }
    header { padding:20px 24px 16px; border-bottom:1px solid var(--line); background:#ffffffee; position:sticky; top:0; backdrop-filter:blur(10px); z-index:10; }
    h1 { margin:0; font-size:24px; }
    .toolbar { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-top:12px; }
    .toolbar input[type=text] { min-width:280px; }
    main { display:grid; grid-template-columns:1.25fr 1fr; gap:16px; padding:18px 24px 36px; align-items:start; }
    .stack { display:grid; gap:16px; }
    section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; }
    h2 { margin:0 0 12px; font-size:16px; }
    h3 { margin:0 0 10px; font-size:14px; }
    button { border:1px solid var(--accent); background:var(--accent); color:white; border-radius:6px; padding:8px 12px; cursor:pointer; }
    button.secondary { background:#fff; color:var(--accent-2); }
    button:disabled { opacity:.6; cursor:default; }
    input, select, textarea {
      width:100%; border:1px solid var(--line); border-radius:6px; padding:8px 10px; font:inherit; background:#fff;
    }
    textarea { min-height:120px; resize:vertical; }
    label { display:grid; gap:6px; font-size:13px; color:var(--muted); }
    .muted { color:var(--muted); font-size:13px; }
    .summary-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; }
    .summary-card { border:1px solid var(--line); border-radius:8px; padding:12px; background:#fbfdfd; }
    .summary-card strong { display:block; font-size:20px; color:var(--accent-2); }
    .summary-card span { display:block; margin-top:4px; color:var(--muted); font-size:12px; }
    .two-col { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }
    .param-groups { display:grid; gap:14px; }
    .param-group { border:1px solid var(--line); border-radius:8px; padding:12px; background:#fcfefe; }
    .field-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }
    .field-grid .full { grid-column:1 / -1; }
    .kv { display:grid; gap:8px; }
    .kv div { display:flex; justify-content:space-between; gap:12px; border-bottom:1px dashed #e3eaec; padding-bottom:6px; }
    .kv div:last-child { border-bottom:none; padding-bottom:0; }
    pre { white-space:pre-wrap; background:#f4f8f8; border-radius:6px; padding:10px; overflow:auto; min-height:80px; margin:0; }
    .actions { display:flex; flex-wrap:wrap; gap:8px; }
    .status-ok { color:var(--good); }
    .status-warn { color:var(--warn); }
    @media (max-width: 1100px) { main { grid-template-columns:1fr; } }
    @media (max-width: 760px) { .field-grid, .two-col { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>Atlas Edge Console</h1>
    <p class="muted">边缘端运行监控、参数调整、硬件探测与受控运维操作。</p>
    <div class="toolbar">
      <input id="token" type="text" placeholder="Admin Token（留空表示直接访问）" />
      <label style="width:auto; display:flex; align-items:center; gap:8px; color:var(--muted);">
        <input id="autoRefresh" type="checkbox" checked style="width:auto;" />
        自动刷新
      </label>
      <button onclick="saveToken()">保存 Token</button>
      <button class="secondary" onclick="refreshAll()">立即刷新</button>
    </div>
  </header>
  <main>
    <div class="stack">
      <section>
        <h2>运行总览</h2>
        <div id="summary" class="summary-grid"></div>
      </section>
      <section>
        <h2>运行态监控</h2>
        <div id="runtimeDetails" class="two-col"></div>
      </section>
      <section>
        <h2>常用参数</h2>
        <p class="muted">保存后会直接写回配置文件。涉及生效时机的参数，仍需要按你的部署方式重启边缘端服务。</p>
        <div id="parameterGroups" class="param-groups"></div>
        <div class="toolbar">
          <button onclick="saveParameters()">保存常用参数</button>
          <button class="secondary" onclick="loadParameters()">重载参数</button>
        </div>
        <pre id="parameterResult"></pre>
      </section>
      <section>
        <h2>完整配置</h2>
        <div class="toolbar">
          <button class="secondary" onclick="loadConfig()">刷新配置</button>
        </div>
        <pre id="config"></pre>
      </section>
    </div>
    <div class="stack">
      <section>
        <h2>硬件探测</h2>
        <div class="actions">
          <button onclick="test('camera')">摄像头</button>
          <button onclick="test('microphone')">麦克风</button>
          <button onclick="test('speaker')">扬声器</button>
          <button onclick="command('test_server_connection')">服务端连通性</button>
        </div>
        <pre id="tests"></pre>
      </section>
      <section>
        <h2>运维操作</h2>
        <div class="actions">
          <button onclick="command('logs')">读取服务日志</button>
          <button onclick="command('pull_update')">拉取更新</button>
          <button onclick="command('restart_edge_service')">重启边缘端服务</button>
        </div>
        <pre id="commands"></pre>
      </section>
      <section>
        <h2>近期日志</h2>
        <div class="toolbar">
          <button class="secondary" onclick="loadLogs()">刷新日志</button>
        </div>
        <pre id="logs"></pre>
      </section>
      <section>
        <h2>原始状态 JSON</h2>
        <pre id="status"></pre>
      </section>
    </div>
  </main>
  <script>
    let state = { fields: [], values: {}, status: null, timer: null };
    document.getElementById("token").value = localStorage.getItem("aiRobotEdgeToken") || "";
    function authHeaders() {
      const token = document.getElementById("token").value.trim();
      return token ? {"Authorization":"Bearer " + token} : {};
    }
    function saveToken() {
      localStorage.setItem("aiRobotEdgeToken", document.getElementById("token").value.trim());
    }
    async function api(path, options={}) {
      const res = await fetch(path, {...options, headers:{...authHeaders(), ...(options.headers||{})}});
      const text = await res.text();
      let data;
      try { data = JSON.parse(text); } catch { data = text; }
      if (!res.ok) {
        throw new Error(typeof data === "string" ? data : JSON.stringify(data, null, 2));
      }
      return data;
    }
    function show(id, data) { document.getElementById(id).textContent = JSON.stringify(data, null, 2); }
    function fmtTime(ts) {
      if (!ts) return "—";
      return new Date(ts).toLocaleString();
    }
    function renderSummary(status) {
      const monitoring = status.monitoring || {};
      const session = monitoring.session || {};
      const queues = monitoring.queues || {};
      const counters = monitoring.counters || {};
      const playback = monitoring.playback || {};
      const cards = [
        ["会话状态", session.state || "unknown"],
        ["当前轮次", String(session.turn_index || 0)],
        ["播放状态", playback.active ? "speaking" : "idle"],
        ["监听窗口", (monitoring.listening && monitoring.listening.armed) ? "armed" : "idle"],
        ["视觉队列", String(queues.vision ?? "—")],
        ["TTS 队列", String(queues.tts ?? "—")],
        ["成功轮次", String(counters.turns_ok || 0)],
        ["失败轮次", String(counters.turns_failed || 0)],
      ];
      document.getElementById("summary").innerHTML = cards.map(([label, value]) => `
        <div class="summary-card"><strong>${value}</strong><span>${label}</span></div>
      `).join("");
    }
    function renderRuntime(status) {
      const monitoring = status.monitoring || {};
      const session = monitoring.session || {};
      const last = monitoring.last_events || {};
      const components = monitoring.components || {};
      const queues = monitoring.queues || {};
      const blocks = [
        {
          title: "会话",
          rows: [
            ["状态", session.state || "—"],
            ["原因", session.reason || "—"],
            ["session_id", session.session_id || "—"],
            ["轮次", String(session.turn_index || 0)],
            ["更新时间", fmtTime(session.updated_at_ms)],
          ],
        },
        {
          title: "最近事件",
          rows: [
            ["视觉", last.vision ? `${last.vision.event_type} (${(last.vision.confidence || 0).toFixed(2)})` : "—"],
            ["唤醒词", last.wake_word ? fmtTime(last.wake_word.timestamp_ms) : "—"],
            ["utterance", last.utterance ? `${last.utterance.duration_ms} ms / ${last.utterance.reason}` : "—"],
            ["服务端轮次", last.server_turn ? `${last.server_turn.phase} / ${last.server_turn.success}` : "—"],
            ["播放块", last.playback_chunk ? `${last.playback_chunk.bytes} bytes / x${last.playback_chunk.merged_chunks}` : "—"],
          ],
        },
        {
          title: "组件",
          rows: [
            ["Tracker 降级", String(!!components.tracker_degraded)],
            ["Servo 降级", String(!!components.servo_degraded)],
            ["监听 armed", String(!!(monitoring.listening && monitoring.listening.armed))],
            ["播放 active", String(!!(monitoring.playback && monitoring.playback.active))],
            ["进程启动", fmtTime(monitoring.process && monitoring.process.started_at_ms)],
          ],
        },
        {
          title: "队列",
          rows: Object.entries(queues).map(([k, v]) => [k, String(v)]),
        },
      ];
      document.getElementById("runtimeDetails").innerHTML = blocks.map(block => `
        <div class="param-group">
          <h3>${block.title}</h3>
          <div class="kv">
            ${block.rows.map(([k, v]) => `<div><span>${k}</span><strong>${v}</strong></div>`).join("")}
          </div>
        </div>
      `).join("");
    }
    async function loadStatus(){
      const data = await api("/api/v1/edge/status");
      state.status = data;
      renderSummary(data);
      renderRuntime(data);
      show("status", data);
    }
    async function loadLogs(){ show("logs", await api("/api/v1/edge/logs")); }
    async function loadConfig(){ show("config", await api("/api/v1/edge/config")); }
    async function loadParameters() {
      const data = await api("/api/v1/edge/parameters");
      state.fields = data.fields || [];
      state.values = data.values || {};
      const groups = {};
      for (const field of state.fields) {
        if (!groups[field.group]) groups[field.group] = [];
        groups[field.group].push(field);
      }
      document.getElementById("parameterGroups").innerHTML = Object.entries(groups).map(([group, fields]) => `
        <div class="param-group">
          <h3>${group}</h3>
          <div class="field-grid">
            ${fields.map(renderField).join("")}
          </div>
        </div>
      `).join("");
    }
    function renderField(field) {
      const value = state.values[field.path];
      const full = field.input_type === "text" && String(value || "").length > 24;
      let control = "";
      if (field.input_type === "bool") {
        control = `<select data-path="${field.path}"><option value="true" ${value ? "selected" : ""}>true</option><option value="false" ${!value ? "selected" : ""}>false</option></select>`;
      } else if (field.input_type === "select") {
        control = `<select data-path="${field.path}">${(field.options || []).map(opt => `<option value="${opt}" ${opt === value ? "selected" : ""}>${opt}</option>`).join("")}</select>`;
      } else {
        const type = field.input_type === "text" ? "text" : "number";
        control = `<input data-path="${field.path}" type="${type}" value="${value ?? ""}" ${field.min_value !== null ? `min="${field.min_value}"` : ""} ${field.max_value !== null ? `max="${field.max_value}"` : ""} ${field.step !== null ? `step="${field.step}"` : ""} />`;
      }
      return `<label class="${full ? "full" : ""}">${field.label}${control}<span>${field.description}</span></label>`;
    }
    async function saveParameters() {
      const values = {};
      for (const field of state.fields) {
        const el = document.querySelector(`[data-path="${field.path}"]`);
        if (!el) continue;
        let value = el.value;
        if (field.input_type === "bool") value = value === "true";
        values[field.path] = value;
      }
      const result = await api("/api/v1/edge/parameters", {
        method:"PUT",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({values}),
      });
      document.getElementById("parameterResult").textContent = JSON.stringify(result, null, 2);
      await loadConfig();
      await loadStatus();
      await loadParameters();
    }
    async function test(name){ show("tests", await api(`/api/v1/edge/tests/${name}`, {method:"POST"})); }
    async function command(name){ show("commands", await api(`/api/v1/edge/commands/${name}`, {method:"POST"})); }
    async function refreshAll() {
      await Promise.all([loadStatus(), loadLogs(), loadConfig(), loadParameters()]);
    }
    function setupAutoRefresh() {
      if (state.timer) clearInterval(state.timer);
      state.timer = setInterval(() => {
        if (document.getElementById("autoRefresh").checked) {
          loadStatus().catch(err => show("status", {error:String(err)}));
          loadLogs().catch(err => show("logs", {error:String(err)}));
        }
      }, 4000);
    }
    document.getElementById("autoRefresh").addEventListener("change", setupAutoRefresh);
    refreshAll().catch(err => show("status", {error:String(err)}));
    setupAutoRefresh();
  </script>
</body>
</html>"""
