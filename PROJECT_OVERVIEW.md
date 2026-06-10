# AI Robot - 项目概览

## 1. 项目定位

当前仓库面向两部分运行时：

- `ai_robot_edge`：运行在 Atlas 200I DK A2 上的边缘端
- `ai_robot_server`：运行在服务器上的管理与聚合端

目标是把本地感知、基础运动控制、远程会话连接、管理面板和大模型服务接入整合在同一套工程里。

## 2. 当前主要实现

### 2.1 边缘端

- 麦克风、唤醒词、VAD、扬声器播放的本地交互链路
- 摄像头采集与人员检测工作流
- WebSocket 会话客户端
- 动作分发与设备抽象层
- Atlas 串口舵机控制
- 二维云台控制封装
- 基于 YOLOv5-face OM 的人脸检测
- 基于检测框误差的云台水平跟踪
- 本地 FastAPI 管理面板

### 2.2 服务端

- FastAPI 服务端骨架
- `/admin` 管理入口
- 设备注册与在线状态管理
- Ragflow / Xinference 连接器封装
- 远程命令下发与结果聚合接口

## 3. 关键目录

```text
src/ai_robot_edge/
  admin/               边缘管理面板
  audio/               唤醒词、VAD、音频采集
  devices/             摄像头、云台、设备工厂
  interaction/         交互状态协同
  server/              边缘到服务端通信
  vision/              检测与跟踪

src/ai_robot_server/
  app.py               服务端 FastAPI 入口
  connectors.py        Ragflow / Xinference 适配
  registry.py          设备注册表
  remote.py            远程命令服务
  orchestrator.py      聚合编排逻辑
```

## 4. 目标跟踪实现

当前保留的是正式可运行的水平跟踪链路：

1. 摄像头读取实时画面
2. Ascend NPU 加载 YOLOv5-face `.om`
3. 选择人脸目标并计算画面偏差
4. 跟踪器做误差整形、预测和限幅
5. 舵机驱动通过串口控制头部水平转动

对应核心文件：

- `src/ai_robot_edge/vision/yolov5_face.py`
- `src/ai_robot_edge/vision/tracking.py`
- `src/ai_robot_edge/devices/gimbal.py`
- `scripts/run_face_gimbal_tracking.py`

## 5. 配置与部署

- 边缘配置模板：`config/edge.example.yaml`
- 服务端配置模板：`config/server.example.yaml`
- 边缘部署脚本：`scripts/install_edge.sh`
- 服务端部署脚本：`scripts/install_server.sh`
- systemd 单元：`deploy/`

## 6. 当前边界

- 仓库保留运行脚本、部署脚本和模型转换脚本
- 临时测试脚本、测试目录、缓存产物不再保留
- 模型权重仍建议本地部署，不直接提交仓库

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
