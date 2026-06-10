from __future__ import annotations

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
    "wake_word",
    "vad",
    "speaker",
    "servo",
    "admin",
}


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

    @app.put("/api/v1/edge/config", dependencies=[Depends(require_admin)])
    async def update_config(payload: dict[str, Any]) -> dict[str, Any]:
        updated = update_config_yaml(config_path, payload)
        load_config(config_path)
        return {"ok": True, "config": updated}

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


EDGE_ADMIN_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Atlas Edge Console</title>
  <style>
    :root { --ink:#182326; --muted:#68767a; --line:#d4dddd; --accent:#116149; --bg:#f5f8f7; --panel:#fff; }
    body { margin:0; font-family: ui-sans-serif, "Avenir Next", "Segoe UI", sans-serif; background:linear-gradient(180deg,#edf5f1,#fbfcfb); color:var(--ink); }
    header { padding:22px 28px 14px; border-bottom:1px solid var(--line); background:#fffffff0; }
    h1 { margin:0; font-size:26px; letter-spacing:0; }
    main { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:14px; padding:20px 28px 36px; }
    section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; }
    h2 { margin:0 0 10px; font-size:16px; }
    button { border:1px solid var(--accent); background:var(--accent); color:white; border-radius:6px; padding:8px 12px; margin:4px 4px 4px 0; cursor:pointer; }
    pre { white-space:pre-wrap; background:#f0f4f2; border-radius:6px; padding:10px; overflow:auto; min-height:64px; }
    .muted { color:var(--muted); font-size:13px; }
  </style>
</head>
<body>
  <header>
    <h1>Atlas 200I DK A2 Edge Console</h1>
    <p class="muted">Local hardware status, probes, configuration, and controlled operations.</p>
  </header>
  <main>
    <section><h2>Status</h2><button onclick="loadStatus()">Refresh</button><pre id="status"></pre></section>
    <section><h2>Logs</h2><button onclick="loadLogs()">Refresh</button><pre id="logs"></pre></section>
    <section>
      <h2>Hardware Tests</h2>
      <button onclick="test('camera')">Camera</button>
      <button onclick="test('microphone')">Microphone</button>
      <button onclick="test('speaker')">Speaker</button>
      <pre id="tests"></pre>
    </section>
    <section>
      <h2>Operations</h2>
      <button onclick="command('logs')">Journal Logs</button>
      <button onclick="command('test_server_connection')">Server Test</button>
      <button onclick="command('pull_update')">Pull Update</button>
      <button onclick="command('restart_edge_service')">Restart Service</button>
      <pre id="commands"></pre>
    </section>
    <section><h2>Config</h2><button onclick="loadConfig()">Refresh</button><pre id="config"></pre></section>
  </main>
  <script>
    const token = localStorage.getItem("aiRobotEdgeToken") || "";
    const headers = token ? {"Authorization":"Bearer " + token} : {};
    async function api(path, options={}) {
      const res = await fetch(path, {...options, headers:{...headers, ...(options.headers||{})}});
      const text = await res.text();
      try { return JSON.parse(text); } catch { return text; }
    }
    function show(id, data) { document.getElementById(id).textContent = JSON.stringify(data, null, 2); }
    async function loadStatus(){ show("status", await api("/api/v1/edge/status")); }
    async function loadLogs(){ show("logs", await api("/api/v1/edge/logs")); }
    async function loadConfig(){ show("config", await api("/api/v1/edge/config")); }
    async function test(name){ show("tests", await api(`/api/v1/edge/tests/${name}`, {method:"POST"})); }
    async function command(name){ show("commands", await api(`/api/v1/edge/commands/${name}`, {method:"POST"})); }
    loadStatus(); loadLogs(); loadConfig();
  </script>
</body>
</html>"""
