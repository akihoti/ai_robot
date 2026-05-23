# AI Robot Edge Service

This repository contains the edge service for an intelligent voice interaction
robot. The edge runs on an Orange Pi class device and coordinates local camera,
microphone, speaker, and future servo hardware.

The first version is a Python background process. It performs local wake-word
and VAD handling, detects people in the camera stream, triggers welcome
interactions, streams utterances to a Windows server over WebSocket, plays TTS
audio, and dispatches optional action intents.

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
cp config/edge.example.yaml config/edge.yaml
python -m ai_robot_edge --config config/edge.yaml
```

The sample config runs in simulated mode by default. Replace
`wake_word.keyword_id`, `server.bearer_token`, and device settings before
deploying to hardware.

## Documentation

- `docs/edge_design.md`: edge architecture and runtime flows.
- `docs/server_api_contract.md`: WebSocket protocol expected from the server.
- `docs/build_plan.md`: staged implementation and validation plan.

## GitHub Push Note

The local GitHub token for `akihoti` was invalid during initial implementation.
Local commits can be pushed after re-authentication:

```bash
gh auth login -h github.com
git push -u origin main
```
