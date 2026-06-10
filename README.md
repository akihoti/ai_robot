# AI Robot

This repository contains the current edge and server implementation for the
robot system.

- `src/ai_robot_edge`: Atlas 200I DK A2 edge runtime
- `src/ai_robot_server`: server-side management and aggregation service
- `config/`: edge and server configuration templates
- `deploy/`: systemd units
- `scripts/`: deployment and model conversion utilities

## Main Implementation

### Edge Runtime

- Local microphone, wake-word, VAD, camera, speaker, and action coordination
- WebSocket client for connecting the edge device to the server
- Edge admin panel for status, logs, config inspection, hardware probes, and
  allowlisted remote operations
- Real SongJia serial gimbal driver for Atlas-connected servos
- Face detection with YOLOv5-face OM on Ascend NPU
- Face-to-gimbal horizontal tracking logic for robot head following

### Server Runtime

- FastAPI-based management service
- Admin console at `/admin`
- Edge device registry and WebSocket session management
- Ragflow and Xinference connector aggregation
- Model, knowledge-base, and remote-command API surface

## Quick Start

### Edge

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[audio,servo,vision]"
cp config/edge.example.yaml config/edge.yaml
python -m ai_robot_edge --config config/edge.yaml
```

### Edge Admin

```bash
ai-robot-edge-admin --config config/edge.yaml
```

### Server

```bash
cp config/server.example.yaml config/server.yaml
ai-robot-server --config config/server.yaml
```

## Face Tracking

The repository keeps one operational tracking entry script:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
/root/.conda/envs/ai-robot/bin/python -u scripts/run_face_gimbal_tracking.py --live --no-tilt-enabled --source /dev/video1
```

This script uses the edge config, loads the YOLOv5-face `.om` model, reads the
camera stream, and drives the horizontal head servo in real time.

## Deployment

See `docs/deployment.md` for deployment on the Atlas edge device and the server.

Do not commit SSH passwords, access tokens, or model secrets into the
repository.
