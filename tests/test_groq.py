import unittest

from app.services.groq import ApiError, GroqRecognizer, GroqTranslator, clean_transcript
from app.services.limits import with_retry


class FakeHTTP:
    def __init__(self, response):
        self.response, self.calls = response, []

    def post(self, path, **kw):
        self.calls.append((path, kw))
        return self.response


class CleanTranscriptTests(unittest.TestCase):
    def test_drops_hallucinated_and_looping_segments(self):
        result = {"segments": [
            {"text": " Xin chào.", "no_speech_prob": 0.01, "avg_logprob": -0.2, "compression_ratio": 1.1},
            {"text": " Hãy subscribe cho kênh", "no_speech_prob": 0.9, "avg_logprob": -1.5, "compression_ratio": 1.0},
            {"text": " la la la la la la", "no_speech_prob": 0.1, "avg_logprob": -0.5, "compression_ratio": 3.0},
        ]}
        self.assertEqual(clean_transcript(result), "Xin chào.")

    def test_keeps_segment_when_only_one_signal_is_bad(self):
        # High no_speech but good logprob: may be quiet speech, retain
        result = {"segments": [{"text": " Tôi đây", "no_speech_prob": 0.8, "avg_logprob": -0.3}]}
        self.assertEqual(clean_transcript(result), "Tôi đây")

    def test_falls_back_to_text_without_segments(self):
        self.assertEqual(clean_transcript({"text": " ok "}), "ok")
        self.assertEqual(clean_transcript({}), "")


class RecognizerTests(unittest.TestCase):
    def test_sends_iso_language_model_and_extension(self):
        http = FakeHTTP({"text": "xin chào"})
        asr = GroqRecognizer(http=http, model="whisper-large-v3", prompt="Thân Phú Sĩ")
        self.assertEqual(asr.transcribe(b"RIFF", "audio/wav", "Vietnamese"), "xin chào")
        path, kw = http.calls[0]
        self.assertEqual(path, "/audio/transcriptions")
        self.assertEqual(kw["data"]["language"], "vi")
        self.assertEqual(kw["data"]["model"], "whisper-large-v3")
        self.assertEqual(kw["data"]["prompt"], "Thân Phú Sĩ")
        self.assertEqual(kw["files"]["file"][0], "audio.wav")

    def test_m4a_mime_gets_m4a_extension(self):
        http = FakeHTTP({"text": "x"})
        GroqRecognizer(http=http).transcribe(b"x", "audio/mp4a-latm", "Japanese")
        self.assertEqual(http.calls[0][1]["files"]["file"][0], "audio.m4a")
        self.assertEqual(http.calls[0][1]["data"]["language"], "ja")
        self.assertNotIn("prompt", http.calls[0][1]["data"])

    def test_language_specific_prompt_and_fallback(self):
        http = FakeHTTP({"text": "x"})
        asr = GroqRecognizer(http=http, prompt="Chung", prompt_vi="Riêng VI", prompt_ja="Riêng JA")
        
        # Tiếng Việt dùng prompt_vi
        asr.transcribe(b"x", "audio/wav", "Vietnamese")
        self.assertEqual(http.calls[-1][1]["data"]["prompt"], "Riêng VI")

        # Tiếng Nhật dùng prompt_ja
        asr.transcribe(b"x", "audio/wav", "Japanese")
        self.assertEqual(http.calls[-1][1]["data"]["prompt"], "Riêng JA")

        # Ngôn ngữ khác rơi về prompt chung
        asr.transcribe(b"x", "audio/wav", "English")
        self.assertEqual(http.calls[-1][1]["data"]["prompt"], "Chung")

        # Không có prompt nào thì không gửi field 'prompt'
        asr_empty = GroqRecognizer(http=http, prompt="", prompt_vi="", prompt_ja="")
        asr_empty.transcribe(b"x", "audio/wav", "Vietnamese")
        self.assertNotIn("prompt", http.calls[-1][1]["data"])


class TranslatorTests(unittest.TestCase):
    def reply(self, content):
        return {"choices": [{"message": {"content": content}}]}

    def test_strips_think_block_and_sends_context(self):
        http = FakeHTTP(self.reply("<think>nghĩ...</think>こんにちは"))
        out = GroqTranslator(http=http, model="llama-3.3-70b-versatile").translate("xin chào", "Vietnamese", "Japanese", ["câu trước"])
        self.assertEqual(out, "こんにちは")
        body = http.calls[0][1]["json"]
        self.assertIn("câu trước", body["messages"][1]["content"])
        self.assertNotIn("reasoning_effort", body)

    def test_gpt_oss_gets_low_reasoning(self):
        http = FakeHTTP(self.reply("ok"))
        GroqTranslator(http=http, model="openai/gpt-oss-120b").translate("a", "Vietnamese", "Japanese")
        self.assertEqual(http.calls[0][1]["json"]["reasoning_effort"], "low")

    def test_passes_tone_to_prompt(self):
        http = FakeHTTP(self.reply("こんにちは"))
        translator = GroqTranslator(http=http, tone="polite")
        translator.translate("xin chào", "Vietnamese", "Japanese")
        sys_msg = http.calls[0][1]["json"]["messages"][0]["content"]
        self.assertIn("always polite", sys_msg)

        # Tham số truyền trực tiếp trong translate() ghi đè thuộc tính tone của instance
        translator.translate("xin chào", "Vietnamese", "Japanese", tone="casual")
        sys_msg2 = http.calls[1][1]["json"]["messages"][0]["content"]
        self.assertIn("always casual", sys_msg2)


class RetryAfterTests(unittest.TestCase):
    def test_honors_retry_after_header(self):
        slept, calls = [], []

        def fn():
            calls.append(1)
            if len(calls) == 1:
                raise ApiError(429, "slow down", retry_after="2")
            return "ok"

        self.assertEqual(with_retry(fn, sleep=slept.append), "ok")
        self.assertTrue(2.0 <= slept[0] <= 2.5)

    def test_bad_retry_after_is_ignored(self):
        self.assertIsNone(ApiError(429, "x", retry_after="abc").retry_after)


if __name__ == "__main__":
    unittest.main()
