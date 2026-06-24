from __future__ import annotations

import unittest

import numpy as np

from ai_robot_edge.audio.vad import Pcm16Resampler, ProbabilityVadSegmenter
from ai_robot_edge.config import VadConfig
from ai_robot_edge.events import AudioFrame


class ProbabilityVadSegmenterTests(unittest.TestCase):
    def test_finishes_speech_after_negative_threshold_silence(self) -> None:
        probabilities = iter([0.05, 0.62, 0.70, 0.12, 0.10])
        segmenter = ProbabilityVadSegmenter(
            _vad_config(min_speech_ms=60, silence_ms=60),
            probability_provider=lambda _frame: next(probabilities),
        )

        utterance = None
        for sequence in range(5):
            utterance = segmenter.accept(_frame(sequence))

        self.assertIsNotNone(utterance)
        assert utterance is not None
        self.assertEqual(utterance.reason, "vad_silence")
        self.assertGreaterEqual(utterance.duration_ms, 120)

    def test_drops_short_noise_and_enters_cooldown(self) -> None:
        probabilities = iter([0.8, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1])
        segmenter = ProbabilityVadSegmenter(
            _vad_config(min_speech_ms=90, silence_ms=60, cooldown_ms=120),
            probability_provider=lambda _frame: next(probabilities),
        )

        utterances = [segmenter.accept(_frame(sequence)) for sequence in range(7)]

        self.assertTrue(all(utterance is None for utterance in utterances))


class Pcm16ResamplerTests(unittest.TestCase):
    def test_resamples_48k_mono_frame_to_16k_without_changing_duration(self) -> None:
        samples = np.arange(480, dtype=np.int16)
        frame = AudioFrame(
            data=samples.tobytes(),
            sample_rate=48000,
            channels=1,
            sequence=1,
        )
        resampler = Pcm16Resampler(target_sample_rate=16000)

        output = resampler.resample(frame)

        self.assertEqual(output.sample_rate, 16000)
        self.assertEqual(output.channels, 1)
        self.assertEqual(output.duration_ms, frame.duration_ms)
        self.assertEqual(len(output.data), 160 * 2)


def _vad_config(
    *,
    min_speech_ms: int,
    silence_ms: int,
    cooldown_ms: int = 0,
) -> VadConfig:
    return VadConfig(
        energy_threshold=0.01,
        silence_ms=silence_ms,
        max_utterance_ms=2000,
        pre_roll_ms=30,
        engine="silero_onnx",
        model_path="pretrain/silero_vad.onnx",
        sample_rate=16000,
        threshold=0.45,
        negative_threshold=0.30,
        min_speech_ms=min_speech_ms,
        cooldown_ms=cooldown_ms,
    )


def _frame(sequence: int) -> AudioFrame:
    samples = np.full(480, 1000, dtype=np.int16)
    return AudioFrame(
        data=samples.tobytes(),
        sample_rate=16000,
        channels=1,
        sequence=sequence,
    )


if __name__ == "__main__":
    unittest.main()
