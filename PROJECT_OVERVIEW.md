# AI Robot Edge Service - 项目知识库

## 1. 项目概述

### 1.1 项目定位
这是一个运行在 **Orange Pi 类设备**（如 RK3588/NPU）上的 **Python 边缘服务**，负责智能语音交互机器人的本地实时交互循环。

### 1.2 核心使命
协调本地摄像头、麦克风、扬声器和未来的伺服硬件，实现：
- 人员检测与欢迎交互
- 本地唤醒词与 VAD 处理
- 通过 WebSocket 与 Windows 服务器通信
- TTS 音频播放
- 动作意图分发

### 1.3 架构原则
- **本地处理**：唤醒词、VAD、人员检测在边缘完成
- **云端协作**：ASR、LLM、RAG、TTS 在 Windows 服务器完成
- **异步设计**：多个小的 async worker 通过队列通信
- **设备抽象**：支持模拟模式，开发无需真实硬件

---

## 2. 目录结构

```
ai_robot/
├── src/ai_robot_edge/          # 源代码
│   ├── __main__.py             # 入口点
│   ├── app.py                  # 主应用（Worker 编排）
│   ├── config.py               # 配置加载与校验
│   ├── events.py               # 事件类型定义
│   ├── actions.py              # 动作分发器
│   ├── playback.py             # TTS 播放 Worker
│   ├── audio/                  # 音频处理
│   │   ├── worker.py           # 麦克风采集 Worker
│   │   ├── wake_word.py        # 唤醒词检测
│   │   └── vad.py              # VAD 分段
│   ├── vision/                 # 视觉处理
│   │   ├── worker.py           # 摄像头采集 Worker
│   │   └── detector.py         # 人员检测器
│   ├── interaction/            # 交互协调
│   │   └── coordinator.py      # 状态管理与欢迎逻辑
│   ├── server/                 # 服务器通信
│   │   └── client.py           # WebSocket 客户端
│   └── devices/                # 设备抽象
│       ├── base.py             # 设备接口
│       ├── factory.py          # 设备工厂
│       └── simulated.py        # 模拟设备实现
├── config/                     # 配置文件
├── tests/                      # 测试
└── docs/                       # 文档
```

---

## 3. 核心组件

### 3.1 Worker 列表

| Worker | 职责 | 输入 | 输出 |
|--------|------|------|------|
| `CameraWorker` | 摄像头帧采集 | 摄像头设备 | `VisionEvent` 队列 |
| `PersonDetector` | 人员检测 | 图像帧 | `VisionEvent` |
| `AudioWorker` | 麦克风采集 | 麦克风设备 | `Utterance` 队列 |
| `WakeWordDetector` | 唤醒词检测 | 音频帧 | 唤醒事件 |
| `EnergyVadSegmenter` | VAD 语音分段 | 音频帧 | `Utterance` |
| `InteractionCoordinator` | 交互状态管理 | `VisionEvent` | `ActionIntent` |
| `ConversationWorker` | 会话管理 | `Utterance` | 发送到服务器 |
| `ConversationClient` | WebSocket 通信 | 服务器响应 | TTS 帧、动作意图 |
| `PlaybackWorker` | TTS 播放 | TTS 队列 | 扬声器输出 |
| `ActionDispatcher` | 动作分发 | `ActionIntent` | 伺服控制器 |

### 3.2 关键类

- **`EdgeApp`**：主应用，编排所有 Worker
- **`EdgeConfig`**：配置数据类（所有配置项）
- **`InteractionCoordinator`**：状态机管理
- **`ConversationClient`**：WebSocket 协议实现

---

## 4. 状态机

```
idle ──(人员检测)──→ welcoming ──(播放完成)──→ idle
 │                      │
 │                      └──(唤醒词)──→ listening
 │                                        │
 │                                        └──(VAD完成)──→ conversing
 │                                                          │
 │                                                          └──(会话结束/错误)──→ recovering ──→ idle
```

| 状态 | 描述 |
|------|------|
| `idle` | 监控摄像头和唤醒词 |
| `welcoming` | 播放欢迎语，唤醒词仍可触发 |
| `listening` | 唤醒词已检测，VAD 正在收集语音 |
| `conversing` | 音频已发送，正在接收 TTS/动作 |
| `recovering` | 服务器或设备错误，恢复后返回 idle |

---

## 5. 数据流

### 5.1 人员欢迎流程

1. 摄像头以配置的 FPS 捕获帧
2. 检测器输出 `VisionEvent(person_detected=True)`
3. Coordinator 要求连续 N 帧确认人员存在
4. 如果欢迎冷却时间已过，触发欢迎交互
5. 可选的 `welcome_motion` 动作发送到伺服控制器

### 5.2 语音对话流程

1. 麦克风持续捕获 PCM16 单声道音频
2. 唤醒词检测器监听直到检测到唤醒词
3. VAD 在唤醒后开始缓冲语音，在静音或最大时长时结束
4. Client 打开 WebSocket 会话（Bearer 认证）
5. 发送 `session.start` → 二进制音频块 → `audio.end`
6. 服务器返回 ASR、LLM、TTS、动作帧
7. 扬声器播放音频，动作分发器处理运动意图

---

## 6. WebSocket 协议

### 6.1 端点

```
ws://<server-host>:<port>/api/v1/edge/sessions/{device_id}
```

### 6.2 认证

```
Authorization: Bearer <token>
```

### 6.3 客户端帧

| 帧类型 | 用途 |
|--------|------|
| `session.start` | 开始会话 |
| `audio.chunk` | 音频块元数据（在二进制帧前） |
| `audio.end` | 音频流结束 |
| `event.vision` | 视觉事件（人员检测） |
| `session.cancel` | 取消会话 |

### 6.4 服务器帧

| 帧类型 | 用途 |
|--------|------|
| `asr.partial/final` | ASR 识别结果 |
| `llm.partial/final` | LLM 响应 |
| `tts.chunk` | TTS 音频块元数据 |
| `action.intent` | 动作意图 |
| `error` | 错误信息 |

### 6.5 音频格式

- 上传：单声道 PCM16，16 kHz
- TTS 返回：PCM16 或 WAV 块

---

## 7. 配置说明

配置文件为 YAML 格式，所有配置项定义在 `EdgeConfig` 数据类中：

```yaml
device_id: "orange-pi-001"
server:
  websocket_url: "ws://server:8000/api/v1/edge/sessions/{device_id}"
  bearer_token: "your-token"
runtime:
  mode: "simulated"  # simulated | hardware
  log_level: "INFO"
camera:
  enabled: true
  source: 0
  width: 640
  height: 480
  fps: 10
vision:
  detector: "simulated"
  person_threshold: 0.55
  stable_frames: 3
  welcome_cooldown_seconds: 30
microphone:
  enabled: true
  sample_rate: 16000
  channels: 1
wake_word:
  enabled: true
  engine: "simulated"
  keyword_id: "your-keyword"
vad:
  energy_threshold: 0.015
  silence_ms: 800
  max_utterance_ms: 10000
speaker:
  enabled: true
servo:
  enabled: false
  controller: "noop"
```

---

## 8. 开发指南

### 8.1 快速开始

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
cp config/edge.example.yaml config/edge.yaml
python -m ai_robot_edge --config config/edge.yaml
```

### 8.2 模拟模式

默认配置运行在模拟模式，无需真实硬件。部署到硬件前需要替换：
- `wake_word.keyword_id`
- `server.bearer_token`
- 设备设置

### 8.3 测试

```bash
pytest tests/
```

---

## 9. 部署

使用 `systemd + venv` 部署到 Orange Pi：

```bash
# 创建 systemd 服务文件
/etc/systemd/system/ai-robot-edge.service
```

详细部署步骤见 `docs/deployment.md`。

---

## 10. v1 非目标

- 无本地 ASR、LLM、RAG、TTS
- 无完整手势或动作识别
- 无真实伺服硬件驱动（使用 `NoopServoController`）
- 无本地 HTTP 管理 API

---

## 11. 关键文件索引

| 文件 | 用途 |
|------|------|
| `src/ai_robot_edge/app.py` | 主应用编排 |
| `src/ai_robot_edge/config.py` | 配置定义与加载 |
| `src/ai_robot_edge/events.py` | 事件类型与数据结构 |
| `src/ai_robot_edge/interaction/coordinator.py` | 状态机与欢迎逻辑 |
| `src/ai_robot_edge/server/client.py` | WebSocket 客户端 |
| `src/ai_robot_edge/audio/vad.py` | VAD 分段器 |
| `src/ai_robot_edge/audio/wake_word.py` | 唤醒词检测 |
| `src/ai_robot_edge/vision/detector.py` | 人员检测器 |
| `docs/edge_design.md` | 架构设计文档 |
| `docs/server_api_contract.md` | WebSocket 协议文档 |
