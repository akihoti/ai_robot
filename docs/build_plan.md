# Build Plan

## 1. Commit Strategy

Work is committed directly on `main`, as requested. Each functional block gets
one small commit. Before every commit, inspect `git status` and stage only files
belonging to that block.

Remote pushes require a valid GitHub credential for `akihoti/ai_robot`.

## 2. Milestones

1. Documentation skeleton: edge design, server contract, and build plan.
2. Project skeleton: Python package, config sample, README, dependencies, and
   entry point.
3. Device abstractions: camera, microphone, speaker, servo, and simulated
   devices.
4. Vision pipeline: camera loop, person detector adapter, CPU/NPU/simulated
   selection.
5. Welcome logic: stable presence detection, cooldown, welcome event, and
   optional welcome motion intent.
6. Audio pipeline: microphone capture, configurable wake word, VAD segmentation,
   and utterance buffering.
7. Server client: authenticated WebSocket session, binary audio streaming,
   response frame handling, reconnects, and errors.
8. Playback and actions: TTS playback queue, action dispatch, Noop servo.
9. Deployment: systemd service, venv install script, configuration guide, and
   acceptance checklist.
10. Server console: FastAPI app, Ragflow/Xinference adapters, device registry,
    remote command queue, and static admin UI.
11. Edge admin: local FastAPI app, Atlas status, hardware probes, config API,
    allowlisted operations, and management WebSocket status reporting.

## 3. Acceptance Tests

- Simulated mode runs on a development machine without camera or microphone.
- Person-present simulation triggers exactly one welcome event per cooldown.
- Missing wake-word configuration prevents startup with a clear error.
- VAD emits an utterance after speech followed by silence.
- WebSocket client sends the documented frame order.
- TTS chunks are queued and passed to the configured speaker.
- Action intents are logged by `NoopServoController`.
- systemd template starts the configured Python module.
- Server `/admin` renders and reports degraded connector status when upstreams
  are not configured.
- Edge `/` renders and exposes status, logs, config, tests, and allowlisted
  operations.
- Remote commands are rejected unless allowlisted on both sides.

## 4. Risks

- GitHub authentication is currently invalid on this machine; remote push must
  wait until `gh auth login -h github.com` or equivalent credential repair.
- RKNN/NPU model choice is not final. The detector interface must keep model
  loading replaceable.
- Real servo control is blocked by missing hardware and electrical details.
- Wake-word engine choice is not final. The first implementation uses an
  interface and simulated detector, with a clear slot for a real engine.
