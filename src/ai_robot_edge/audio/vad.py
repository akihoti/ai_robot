from __future__ import annotations

from collections import deque
from array import array
from pathlib import Path
from collections.abc import Callable
from typing import Protocol

import numpy as np

from ..config import VadConfig
from ..events import AudioFrame, Utterance


def frame_energy(frame: AudioFrame) -> float:
    if not frame.data:
        return 0.0
    samples = array("h")
    samples.frombytes(frame.data)
    if not samples:
        return 0.0
    square_sum = sum((sample / 32768.0) ** 2 for sample in samples)
    return (square_sum / len(samples)) ** 0.5


class VadSegmenter(Protocol):
    def accept(self, frame: AudioFrame) -> Utterance | None:
        """Accept one PCM16 audio frame and optionally return a completed utterance."""

    def reset(self) -> None:
        """Reset all in-progress speech state."""


class Pcm16Resampler:
    def __init__(self, target_sample_rate: int) -> None:
        self.target_sample_rate = target_sample_rate

    def resample(self, frame: AudioFrame) -> AudioFrame:
        if frame.channels != 1:
            raise ValueError("only mono PCM16 frames can be resampled")
        if frame.sample_rate == self.target_sample_rate:
            return frame
        samples = np.frombuffer(frame.data, dtype=np.int16)
        if len(samples) == 0:
            return AudioFrame(
                data=b"",
                sample_rate=self.target_sample_rate,
                channels=1,
                timestamp_ms=frame.timestamp_ms,
                sequence=frame.sequence,
            )

        if frame.sample_rate % self.target_sample_rate == 0:
            step = frame.sample_rate // self.target_sample_rate
            output = samples[::step]
        else:
            output_count = max(
                1,
                int(round(len(samples) * self.target_sample_rate / frame.sample_rate)),
            )
            source_positions = np.arange(len(samples), dtype=np.float32)
            target_positions = np.linspace(0, len(samples) - 1, output_count)
            output = np.interp(target_positions, source_positions, samples).astype(
                np.int16
            )

        return AudioFrame(
            data=np.asarray(output, dtype=np.int16).tobytes(),
            sample_rate=self.target_sample_rate,
            channels=1,
            timestamp_ms=frame.timestamp_ms,
            sequence=frame.sequence,
        )


class EnergyVadSegmenter:
    def __init__(self, config: VadConfig) -> None:
        self.config = config
        self._pre_roll: deque[AudioFrame] = deque()
        self._frames: list[AudioFrame] = []
        self._speech_started = False
        self._silence_ms = 0
        self._duration_ms = 0

    def accept(self, frame: AudioFrame) -> Utterance | None:
        self._append_pre_roll(frame)
        energy = frame_energy(frame)
        voiced = energy >= self.config.energy_threshold

        if not self._speech_started:
            if voiced:
                self._speech_started = True
                self._frames.extend(self._pre_roll)
                self._pre_roll.clear()
            else:
                return None

        self._frames.append(frame)
        self._duration_ms += frame.duration_ms
        self._silence_ms = 0 if voiced else self._silence_ms + frame.duration_ms

        if self._duration_ms >= self.config.max_utterance_ms:
            return self._finish("max_duration")
        if self._silence_ms >= self.config.silence_ms:
            return self._finish("vad_silence")
        return None

    def reset(self) -> None:
        self._pre_roll.clear()
        self._frames.clear()
        self._speech_started = False
        self._silence_ms = 0
        self._duration_ms = 0

    def _append_pre_roll(self, frame: AudioFrame) -> None:
        self._pre_roll.append(frame)
        max_frames = max(1, self.config.pre_roll_ms // max(1, frame.duration_ms))
        while len(self._pre_roll) > max_frames:
            self._pre_roll.popleft()

    def _finish(self, reason: str) -> Utterance:
        utterance = Utterance(frames=list(self._frames), reason=reason)
        self.reset()
        return utterance


class ProbabilityVadSegmenter:
    def __init__(
        self,
        config: VadConfig,
        *,
        probability_provider: Callable[[AudioFrame], float],
    ) -> None:
        self.config = config
        self.probability_provider = probability_provider
        self._pre_roll: deque[AudioFrame] = deque()
        self._frames: list[AudioFrame] = []
        self._speech_started = False
        self._silence_ms = 0
        self._speech_ms = 0
        self._duration_ms = 0
        self._cooldown_ms_remaining = 0

    def accept(self, frame: AudioFrame) -> Utterance | None:
        self._append_pre_roll(frame)
        if self._cooldown_ms_remaining > 0:
            self._cooldown_ms_remaining = max(
                0,
                self._cooldown_ms_remaining - frame.duration_ms,
            )
            return None

        probability = float(self.probability_provider(frame))
        voiced = probability >= self.config.threshold
        silence = probability < self.config.negative_threshold

        if not self._speech_started:
            if not voiced:
                return None
            self._speech_started = True
            self._frames.extend(self._pre_roll)
            self._pre_roll.clear()
            self._duration_ms = sum(item.duration_ms for item in self._frames)
            self._speech_ms = frame.duration_ms
            self._silence_ms = 0
            return None

        self._frames.append(frame)
        self._duration_ms += frame.duration_ms
        if voiced:
            self._speech_ms += frame.duration_ms
            self._silence_ms = 0
        elif silence:
            self._silence_ms += frame.duration_ms

        if self._duration_ms >= self.config.max_utterance_ms:
            return self._finish_or_drop("max_duration")
        if self._silence_ms >= self.config.silence_ms:
            return self._finish_or_drop("vad_silence")
        return None

    def reset(self) -> None:
        self._pre_roll.clear()
        self._frames.clear()
        self._speech_started = False
        self._silence_ms = 0
        self._speech_ms = 0
        self._duration_ms = 0
        self._cooldown_ms_remaining = 0

    def _append_pre_roll(self, frame: AudioFrame) -> None:
        self._pre_roll.append(frame)
        max_frames = max(1, self.config.pre_roll_ms // max(1, frame.duration_ms))
        while len(self._pre_roll) > max_frames:
            self._pre_roll.popleft()

    def _finish_or_drop(self, reason: str) -> Utterance | None:
        if self._speech_ms < self.config.min_speech_ms:
            self._drop_current_segment()
            return None
        utterance = Utterance(frames=list(self._frames), reason=reason)
        self.reset()
        return utterance

    def _drop_current_segment(self) -> None:
        self._pre_roll.clear()
        self._frames.clear()
        self._speech_started = False
        self._silence_ms = 0
        self._speech_ms = 0
        self._duration_ms = 0
        self._cooldown_ms_remaining = self.config.cooldown_ms


class SileroOnnxModel:
    WINDOW_SAMPLES = 512

    def __init__(self, model_path: str, sample_rate: int) -> None:
        import onnxruntime as ort  # type: ignore

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Silero VAD model not found: {model_path}")
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=providers,
        )
        self.sample_rate = sample_rate
        self._context_size = 64 if sample_rate == 16000 else 32
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._context_size), dtype=np.float32)

    def predict(self, samples: np.ndarray) -> float:
        if len(samples) != self.WINDOW_SAMPLES:
            raise ValueError("Silero VAD expects 512 samples for 16kHz input")
        chunk = np.asarray(samples, dtype=np.float32).reshape(1, -1)
        model_input = np.concatenate([self._context, chunk], axis=1)
        outputs = self.session.run(
            None,
            {
                "input": model_input,
                "state": self._state,
                "sr": np.array(self.sample_rate, dtype=np.int64),
            },
        )
        probability = float(np.asarray(outputs[0]).reshape(-1)[0])
        self._state = np.asarray(outputs[1], dtype=np.float32)
        self._context = model_input[:, -self._context_size :]
        return probability


class SileroOnnxVadSegmenter:
    def __init__(self, config: VadConfig) -> None:
        self.config = config
        self._resampler = Pcm16Resampler(config.sample_rate)
        self._model = SileroOnnxModel(config.model_path, config.sample_rate)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._segmenter = ProbabilityVadSegmenter(
            config,
            probability_provider=self._speech_probability,
        )

    def accept(self, frame: AudioFrame) -> Utterance | None:
        return self._segmenter.accept(frame)

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._model.reset()
        self._segmenter.reset()

    def _speech_probability(self, frame: AudioFrame) -> float:
        vad_frame = self._resampler.resample(frame)
        samples = (
            np.frombuffer(vad_frame.data, dtype=np.int16).astype(np.float32) / 32768.0
        )
        if len(samples) == 0:
            return 0.0
        self._buffer = np.concatenate([self._buffer, samples])
        probabilities: list[float] = []
        while len(self._buffer) >= SileroOnnxModel.WINDOW_SAMPLES:
            chunk = self._buffer[: SileroOnnxModel.WINDOW_SAMPLES]
            self._buffer = self._buffer[SileroOnnxModel.WINDOW_SAMPLES :]
            probabilities.append(self._model.predict(chunk))
        return max(probabilities, default=0.0)


def build_vad_segmenter(config: VadConfig) -> VadSegmenter:
    engine = config.engine.lower()
    if engine in {"silero", "silero_onnx"}:
        return SileroOnnxVadSegmenter(config)
    if engine == "energy":
        return EnergyVadSegmenter(config)
    raise ValueError(f"unsupported vad.engine: {config.engine}")
