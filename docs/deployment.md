# Deployment

## 1. Server Preparation

The AI Robot server hosts the web console and aggregates Ragflow and Xinference.
The target host is `10.88.129.172`; use SSH keys or host-local secret
management for access credentials instead of committing passwords.

Current upstream services:

- RAGFlow: `http://10.88.129.172:9381`
- Xinference: `http://10.88.129.172:9997`

Install system packages:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-dev git
```

Clone the repository to `/opt/ai_robot`, then run:

```bash
sudo APP_DIR=/opt/ai_robot scripts/install_server.sh
```

Edit `/etc/ai-robot/server.yaml`:

- set `http.admin_token`;
- set `ragflow.base_url` to `http://10.88.129.172:9381`;
- set `xinference.base_url` to `http://10.88.129.172:9997`;
- set `ragflow.api_key` and `xinference.api_key` if those services require auth;
- set `edge.bearer_tokens` for every edge device.

Enable and inspect:

```bash
sudo systemctl enable --now ai-robot-server
sudo journalctl -u ai-robot-server -f
```

Open:

```text
http://10.88.129.172:8010/admin
```

Restart the standalone Xinference deployment when needed:

```bash
docker compose -p xinference -f D:\software\ragflow-main\docker\docker-compose-xinference-only.yml up -d
```

## 2. Atlas Edge Preparation

The Atlas 200I DK A2 edge device is expected at `10.90.67.45` with SSH on port
`2222`. Use SSH keys or a local password manager for the deployment credential.

Install system packages:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-dev git alsa-utils
```

Clone the repository to `/opt/ai_robot`, then run:

```bash
sudo APP_DIR=/opt/ai_robot scripts/install_edge.sh
```

Edit `/etc/ai-robot/edge.yaml` before enabling the service:

- set `device_id`;
- set `server.websocket_url`;
- set `server.bearer_token`;
- set `wake_word.keyword_id`;
- select `vision.detector`: `acl`, `cpu`, `auto`, or `simulated`;
- use `vision.detector: yolov5-face-om` and set `vision.face_model_path` for
  Ascend NPU face detection;
- confirm camera, microphone, and speaker device names.
- set `admin.auth_token`;
- keep `admin.allowed_commands` limited to approved operations.

Enable and inspect:

```bash
sudo systemctl enable --now ai-robot-edge
sudo systemctl enable --now ai-robot-edge-admin
sudo journalctl -u ai-robot-edge -f
```

Open:

```text
http://10.90.67.45:8090/
```

## 3. Local Simulation

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[audio,servo,vision]"
cp config/edge.example.yaml config/edge.yaml
python -m ai_robot_edge --config config/edge.yaml
```

The app will try to connect to the configured server when simulated audio
produces an utterance. Use a test WebSocket server or disable microphone capture
when only validating the camera/welcome loop.

## 4. Acceptance Checklist

- Service starts under systemd.
- Server console opens at `/admin`.
- Edge console opens on port `8090`.
- Server health shows Ragflow and Xinference connector status.
- Edge device appears in the server device list after the management WebSocket connects.
- Allowlisted remote commands queue from the server and return progress/result frames.
- Logs show camera worker startup and periodic vision events.
- Person detection triggers one welcome action per cooldown interval.
- Wake word detection starts VAD.
- VAD emits utterance metadata.
- WebSocket client sends `session.start`, audio chunks, and `audio.end`.
- TTS chunks reach the speaker implementation.
- Action intents reach the Noop servo controller.
