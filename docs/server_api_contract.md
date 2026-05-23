# Server API Contract

## 1. Ownership

The Windows server owns ASR, RAG, LLM, and TTS. The expected stack is FastAPI,
Xinference, and Ragflow. The edge service owns wake word, VAD, camera events,
audio playback, and local motion dispatch.

## 2. WebSocket Endpoint

```text
ws://<server-host>:<port>/api/v1/edge/sessions/{device_id}
```

Required header:

```text
Authorization: Bearer <token>
```

The server must reject missing or invalid tokens with a WebSocket close code or
an `error` frame before closing.

## 3. Client Frames

JSON text frames use this common envelope:

```json
{
  "type": "session.start",
  "request_id": "uuid",
  "timestamp_ms": 1710000000000,
  "payload": {}
}
```

Binary frames carry raw audio bytes and are associated with the most recent
`audio.chunk` metadata frame.

### `session.start`

```json
{
  "device_id": "orange-pi-001",
  "audio": {
    "encoding": "pcm_s16le",
    "sample_rate": 16000,
    "channels": 1
  },
  "wake_word_id": "configured-wake-word",
  "context": {
    "last_vision_event": "person_present"
  }
}
```

### `audio.chunk`

Sent before a binary audio frame.

```json
{
  "sequence": 1,
  "duration_ms": 200,
  "encoding": "pcm_s16le"
}
```

### `audio.end`

```json
{
  "total_chunks": 12,
  "reason": "vad_silence"
}
```

### `event.vision`

```json
{
  "event": "person_present",
  "confidence": 0.86,
  "source": "camera",
  "cooldown_active": false
}
```

### `session.cancel`

```json
{
  "reason": "user_interrupt"
}
```

## 4. Server Frames

### `asr.partial` and `asr.final`

```json
{
  "text": "你好",
  "confidence": 0.91
}
```

### `llm.partial` and `llm.final`

```json
{
  "text": "你好，我在。",
  "rag_sources": []
}
```

### `tts.chunk`

Sent before a binary audio frame.

```json
{
  "sequence": 1,
  "encoding": "pcm_s16le",
  "sample_rate": 16000,
  "channels": 1,
  "is_final": false
}
```

### `action.intent`

```json
{
  "name": "nod",
  "parameters": {
    "intensity": 0.5
  }
}
```

Supported v1 names:

- `look_at_user`
- `nod`
- `idle`
- `welcome_motion`

### `error`

```json
{
  "code": "asr_failed",
  "message": "ASR failed",
  "retryable": true,
  "speak_text": "我刚刚没有听清楚。"
}
```

## 5. Audio Defaults

- Edge upload: mono PCM16, 16 kHz.
- TTS return: PCM16 or WAV chunks. PCM16 is preferred for lower latency.
- The server should tolerate short utterances, silence-only utterances, and
  reconnects after network drops.

## 6. Reliability

- The server should send ping/pong or application heartbeat at least every 20s.
- The edge client will reconnect with backoff.
- Duplicate `request_id` values may happen after reconnect and should be
  treated idempotently where possible.
- If the server cannot produce TTS, it should still return `llm.final` and an
  `error` frame with `retryable=false` for the TTS part.
