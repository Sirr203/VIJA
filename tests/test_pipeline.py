import unittest

from app.services.base import SpeechRecognizer, Translator
from app.services.pipeline import TwoStepInterpreter


class FakeASR(SpeechRecognizer):
    def __init__(self, text):
        self.text = text

    def transcribe(self, audio, mime, language):
        return self.text


class FakeMT(Translator):
    def __init__(self):
        self.calls = []

    def translate(self, text, src, dst, context=None):
        self.calls.append((text, src, dst, context))
        return f"[{dst}] {text}"


class PipelineTests(unittest.TestCase):
    def test_passes_text_and_context_to_translator(self):
        mt = FakeMT()
        r = TwoStepInterpreter(FakeASR("xin chào"), mt).interpret(b"x", "audio/wav", "vi", "ja", ["câu trước"])
        self.assertEqual((r.original, r.translated), ("xin chào", "[ja] xin chào"))
        self.assertEqual(mt.calls[0][3], ["câu trước"])
        self.assertIn("total", r.timings_ms)

    def test_silence_skips_translation(self):
        mt = FakeMT()
        r = TwoStepInterpreter(FakeASR(""), mt).interpret(b"x", "audio/wav", "vi", "ja")
        self.assertEqual(r.original, "")
        self.assertEqual(mt.calls, [])  # không tốn request dịch


if __name__ == "__main__":
    unittest.main()
