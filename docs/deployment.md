# Edge Deployment

## 1. Orange Pi Preparation

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
- select `vision.detector`: `rknn`, `cpu`, `auto`, or `simulated`;
- confirm camera, microphone, and speaker device names.

Enable and inspect:

```bash
sudo systemctl enable --now ai-robot-edge
sudo journalctl -u ai-robot-edge -f
```

## 2. Local Simulation

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
cp config/edge.example.yaml config/edge.yaml
python -m ai_robot_edge --config config/edge.yaml
```

The app will try to connect to the configured server when simulated audio
produces an utterance. Use a test WebSocket server or disable microphone capture
when only validating the camera/welcome loop.

## 3. Acceptance Checklist

- Service starts under systemd.
- Logs show camera worker startup and periodic vision events.
- Person detection triggers one welcome action per cooldown interval.
- Wake word detection starts VAD.
- VAD emits utterance metadata.
- WebSocket client sends `session.start`, audio chunks, and `audio.end`.
- TTS chunks reach the speaker implementation.
- Action intents reach the Noop servo controller.
