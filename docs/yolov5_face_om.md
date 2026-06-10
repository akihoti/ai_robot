# YOLOv5-face Ascend OM Deployment

This integration uses `deepcam-cn/yolov5-face` `yolov5s-face` weights for face
boxes and five facial landmarks. The export was verified against upstream
commit `152c688d551aefb973b7b589fb0691c93dab3564`.

## Model Files

- PyTorch source weights: `pretrain/yolov5s-face.pt`
- Fixed-shape ONNX: `pretrain/yolov5s-face.onnx`
- Atlas runtime model: `pretrain/yolov5s-face.om`
- Input: FP32 RGB NCHW, `1x3x640x640`, values normalized to `0-1`
- Output: `1x25200x16`

The 16 output values are center-x, center-y, width, height, object confidence,
five landmark coordinate pairs, and face class confidence. NMS runs on the CPU
after the NPU forward pass.

ACL model initialization, inference, and release must remain on the same
thread. The edge camera worker intentionally performs the approximately 18 ms
NPU call on its own event-loop thread instead of a shared thread pool.

## Export ONNX

Create a separate checkout of the upstream repository and install its PyTorch
export dependencies. Then run:

```bash
python scripts/export_yolov5_face.py \
  --repo /path/to/yolov5-face \
  --weights pretrain/yolov5s-face.pt \
  --output pretrain/yolov5s-face.onnx
```

## Convert On Atlas

Run on the Atlas 200I DK A2:

```bash
bash scripts/convert_yolov5_face_atc.sh \
  pretrain/yolov5s-face.onnx \
  pretrain/yolov5s-face
```

The script targets `Ascend310B1` by default. Override `SOC_VERSION` only when
deploying to a different Ascend device.

## Runtime Entry

The repository keeps the face detector as part of the runtime tracking flow.
Use the tracking script to load the OM model, read the camera stream, and drive
the gimbal:

```bash
PYTHONPATH=src python scripts/run_face_gimbal_tracking.py \
  --config config/edge.example.yaml \
  --source 0 \
  --duration 15
```

## Edge Configuration

Set:

```yaml
vision:
  detector: yolov5-face-om
  person_threshold: 0.55
  face_model_path: pretrain/yolov5s-face.om
  face_input_size: 640
  face_iou_threshold: 0.45
  face_device_id: 0
```

The detector implements the existing person-presence interface. Face boxes can
also be passed to `PanTiltTracker` for the target-tracking flow.

## Face-to-Gimbal Tracking Test

The end-to-end test uses the USB camera, NPU detector, nearest-face selection,
tracking dead zone, and SongJia gimbal SDK:

```bash
PYTHONPATH=src python scripts/run_face_gimbal_tracking.py \
  --config config/edge.example.yaml \
  --source 0 \
  --duration 15
```

This is a dry-run by default: it prints the target error, requested movement,
and virtual gimbal position without writing to the serial controller. Add
`--live` only while the physical robot head and cables can be observed. Live
tests center the head before and after tracking.

Physical axis direction depends on camera and servo mounting. Override either
direction during calibration without editing the YAML:

```bash
PYTHONPATH=src python scripts/run_face_gimbal_tracking.py \
  --config config/edge.example.yaml \
  --duration 10 \
  --live \
  --pan-direction -1 \
  --tilt-direction 1
```
