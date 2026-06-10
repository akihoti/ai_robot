from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ..events import CameraFrame
from .detector import DetectionResult


@dataclass(frozen=True)
class FaceDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    landmarks: tuple[tuple[float, float], ...]

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass(frozen=True)
class LetterboxInfo:
    ratio: float
    pad_x: float
    pad_y: float
    original_width: int
    original_height: int


class OmRunner(Protocol):
    def infer(self, tensor: np.ndarray) -> np.ndarray:
        """Run one NCHW FP32 tensor and return the YOLO prediction tensor."""

    def close(self) -> None:
        """Release NPU resources."""


class AclLiteOmRunner:
    """Small adapter around the AclLite runtime shipped on Atlas devices."""

    def __init__(
        self,
        model_path: str | Path,
        device_id: int = 0,
        acllite_path: str = "/usr/local/Ascend/thirdpart/aarch64/acllite",
    ) -> None:
        path = str(Path(acllite_path))
        if path not in sys.path:
            sys.path.insert(0, path)
        try:
            from acllite_model import AclLiteModel
            from acllite_resource import AclLiteResource
        except ImportError as exc:
            raise RuntimeError(
                f"could not load AclLite on Atlas: {exc}; check CANN and vision dependencies"
            ) from exc

        self._resource = AclLiteResource(device_id)
        self._resource.init()
        self._model = AclLiteModel(str(model_path))

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        outputs = self._model.execute([np.ascontiguousarray(tensor, dtype=np.float32)])
        if not outputs:
            raise RuntimeError("ACL model returned no outputs")
        return np.asarray(outputs[0])

    def close(self) -> None:
        if self._model is not None:
            self._model.destroy()
            self._model = None
        # AclLiteResource releases the device context when it is destroyed.
        self._resource = None


class YoloV5FaceOmDetector:
    """YOLOv5-face detector using an Ascend OM model for NPU inference."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        input_size: int = 640,
        confidence_threshold: float = 0.55,
        iou_threshold: float = 0.45,
        device_id: int = 0,
        runner: OmRunner | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device_id = device_id
        self._runner = runner

    def detect_faces(self, image: np.ndarray) -> list[FaceDetection]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("YOLOv5-face expects a BGR image with shape HxWx3")
        tensor, info = preprocess(image, self.input_size)
        prediction = self._get_runner().infer(tensor)
        return postprocess(
            prediction,
            info,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
        )

    async def detect(self, frame: CameraFrame) -> DetectionResult:
        faces = self.detect_faces(np.asarray(frame.data))
        confidence = max((face.confidence for face in faces), default=0.0)
        return DetectionResult(bool(faces), confidence, "yolov5-face-om")

    def close(self) -> None:
        if self._runner is not None:
            self._runner.close()
            self._runner = None

    def _get_runner(self) -> OmRunner:
        if self._runner is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"YOLOv5-face OM model not found: {self.model_path}")
            self._runner = AclLiteOmRunner(self.model_path, self.device_id)
        return self._runner


def preprocess(image: np.ndarray, input_size: int) -> tuple[np.ndarray, LetterboxInfo]:
    import cv2

    height, width = image.shape[:2]
    ratio = min(input_size / height, input_size / width)
    resized_width = int(round(width * ratio))
    resized_height = int(round(height * ratio))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (input_size - resized_width) / 2
    pad_y = (input_size - resized_height) / 2
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    tensor = padded[:, :, ::-1].transpose(2, 0, 1)
    tensor = np.ascontiguousarray(tensor, dtype=np.float32)[None] / 255.0
    return tensor, LetterboxInfo(ratio, pad_x, pad_y, width, height)


def postprocess(
    prediction: np.ndarray,
    info: LetterboxInfo,
    *,
    confidence_threshold: float,
    iou_threshold: float,
) -> list[FaceDetection]:
    rows = np.asarray(prediction, dtype=np.float32)
    if rows.ndim == 3:
        rows = rows[0]
    if rows.ndim != 2 or rows.shape[1] < 16:
        raise ValueError(f"unexpected YOLOv5-face output shape: {prediction.shape}")

    scores = rows[:, 4] * rows[:, 15]
    rows = rows[scores >= confidence_threshold]
    scores = scores[scores >= confidence_threshold]
    if not len(rows):
        return []

    boxes = _xywh_to_xyxy(rows[:, :4])
    keep = _nms(boxes, scores, iou_threshold)
    detections: list[FaceDetection] = []
    for index in keep:
        box = _scale_points(boxes[index].reshape(2, 2), info).reshape(-1)
        landmarks = _scale_points(rows[index, 5:15].reshape(5, 2), info)
        detections.append(
            FaceDetection(
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3]),
                confidence=float(scores[index]),
                landmarks=tuple((float(x), float(y)) for x, y in landmarks),
            )
        )
    return detections


def draw_faces(image: np.ndarray, faces: list[FaceDetection]) -> np.ndarray:
    import cv2

    output = image.copy()
    colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255))
    for face in faces:
        cv2.rectangle(
            output,
            (round(face.x1), round(face.y1)),
            (round(face.x2), round(face.y2)),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            output,
            f"{face.confidence:.2f}",
            (round(face.x1), max(18, round(face.y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        for point, color in zip(face.landmarks, colors):
            cv2.circle(output, (round(point[0]), round(point[1])), 2, color, -1)
    return output


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    result = boxes.copy()
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return result


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        intersection_width = np.maximum(0, np.minimum(x2[current], x2[rest]) - np.maximum(x1[current], x1[rest]))
        intersection_height = np.maximum(0, np.minimum(y2[current], y2[rest]) - np.maximum(y1[current], y1[rest]))
        intersection = intersection_width * intersection_height
        union = areas[current] + areas[rest] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = rest[iou <= threshold]
    return keep


def _scale_points(points: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    result = points.copy()
    result[:, 0] = np.clip((result[:, 0] - info.pad_x) / info.ratio, 0, info.original_width)
    result[:, 1] = np.clip((result[:, 1] - info.pad_y) / info.ratio, 0, info.original_height)
    return result
