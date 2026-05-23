import unittest

from ai_robot_edge.audio.vad import EnergyVadSegmenter
from ai_robot_edge.config import VadConfig
from ai_robot_edge.events import AudioFrame


class VadTests(unittest.TestCase):
    def test_vad_emits_after_speech_and_silence(self):
        vad = EnergyVadSegmenter(
            VadConfig(
                energy_threshold=0.01,
                silence_ms=60,
                max_utterance_ms=1000,
                pre_roll_ms=30,
            )
        )
        speech = AudioFrame(data=(b"\x00\x20" * 480), sample_rate=16000)
        silence = AudioFrame(data=(b"\x00\x00" * 480), sample_rate=16000)

        self.assertIsNone(vad.accept(speech))
        self.assertIsNone(vad.accept(speech))
        self.assertIsNone(vad.accept(silence))
        utterance = vad.accept(silence)

        self.assertIsNotNone(utterance)
        assert utterance is not None
        self.assertEqual(utterance.reason, "vad_silence")
        self.assertGreaterEqual(utterance.duration_ms, 90)
