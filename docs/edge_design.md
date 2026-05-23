# Edge Service Design

## 1. Goals

The edge service runs on an Orange Pi class device and owns the local,
real-time interaction loop:

- capture camera frames and microphone audio;
- detect a person in view and trigger a welcome interaction;
- wait for a configurable local wake word;
- use local VAD to cut one spoken utterance;
- stream the utterance to the Windows service over WebSocket;
- play streamed TTS audio returned by the service;
- dispatch optional motion intents to a servo controller abstraction.

The first implementation targets RK3588/NPU devices, while keeping CPU and
simulated-device fallbacks so development can continue without final hardware.

## 2. Non-goals for v1

- No local ASR, LLM, RAG, or TTS implementation.
- No full gesture or action recognition.
- No real servo hardware driver until the board, power, PWM/GPIO, and protocol
  details are confirmed.
- No local HTTP management API. The service is a background process managed by
  systemd.

## 3. Runtime Architecture

The service is a Python background process composed of small async workers:

1. `Config` loads YAML settings and environment overrides.
2. `CameraWorker` reads frames, downsamples if needed, and sends frames to a
   detector.
3. `PersonDetector` detects people through an RKNN/NPU adapter when available,
   with CPU or simulated fallback.
4. `InteractionCoordinator` applies presence stability and cooldown rules.
5. `MicrophoneWorker` captures PCM16 mono audio.
6. `WakeWordDetector` waits for a configured wake word model or keyword id.
7. `VadSegmenter` cuts a single utterance after wake-up.
8. `ConversationClient` streams the utterance to the server and receives ASR,
   LLM, TTS, and action frames.
9. `Speaker` plays TTS audio chunks.
10. `ActionDispatcher` maps server action intents to a servo controller.

Workers communicate with typed events and bounded queues. The coordinator owns
the high-level state, preventing overlapping welcome playback and voice sessions.

## 4. Main Flows

### 4.1 Person welcome

1. Camera captures frames at the configured FPS.
2. Detector emits `VisionEvent(person_detected=True, confidence=...)`.
3. Coordinator requires N consecutive positive frames before treating the user
   as present.
4. If welcome cooldown has expired, coordinator sends `event.vision` to the
   server and requests or plays the configured welcome response.
5. Optional `welcome_motion` action is dispatched to the servo abstraction.
6. Cooldown suppresses repeated greetings while the person remains nearby.

### 4.2 Voice conversation

1. Microphone continuously captures short PCM frames.
2. Wake-word detector consumes frames until the configured wake word is detected.
3. VAD starts buffering speech after wake-up and ends on silence or max duration.
4. Client opens a WebSocket session with Bearer token authentication.
5. Client sends `session.start`, streams binary audio chunks, and sends
   `audio.end`.
6. Server returns partial/final ASR, partial/final LLM text, TTS audio chunks,
   and optional action intents.
7. Speaker plays audio chunks in order; ActionDispatcher handles motion intents.
8. Session ends or reconnects according to the retry policy.

## 5. State and Concurrency

- `idle`: watching camera and wake word.
- `welcoming`: playing or requesting welcome response; voice wake remains active
  unless configured otherwise.
- `listening`: wake word detected, VAD is collecting the utterance.
- `conversing`: audio has been sent and TTS/actions are being received.
- `recovering`: server or device error; service logs and returns to idle when
  possible.

Long-running workers must handle cancellation and close hardware resources on
shutdown.

## 6. Servo TODO

Real servo control is intentionally blocked until these are confirmed:

- exact Orange Pi model and OS image;
- GPIO/PWM availability and pin mapping;
- whether a dedicated servo controller board is used;
- power supply current budget and common ground design;
- signal level requirements and level shifting;
- number of channels;
- angle limits and safe neutral positions;
- max speed/acceleration limits;
- physical collision boundaries.

Until then, all action intents are logged and acknowledged by
`NoopServoController`.
