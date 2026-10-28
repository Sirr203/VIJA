import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.factory import build_interpreter
from app.services.groq import GroqRecognizer, GroqTranslator
from app.services.pipeline import TwoStepInterpreter

try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


class FactoryDefaultsTests(unittest.TestCase):
    @patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test"}, clear=True)
    def test_default_providers_are_groq(self):
        # When ASR_PROVIDER, MT_PROVIDER, MT_FALLBACK are unset -> defaults to Groq, fallback none
        interp = build_interpreter()
        self.assertIsInstance(interp, TwoStepInterpreter)
        self.assertIsInstance(interp.recognizer, GroqRecognizer)
        self.assertIsInstance(interp.translator, GroqTranslator)
        self.assertEqual(interp.tone, "auto")

    @patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test"})
    def test_factory_passes_tone(self):
        interp = build_interpreter(tone="polite")
        self.assertEqual(interp.tone, "polite")
        self.assertEqual(interp.translator.tone, "polite")

    @unittest.skipUnless(HAS_GEMINI, "google-genai is not installed in this environment")
    def test_gemini_provider_when_installed(self):
        with patch.dict(os.environ, {"ASR_PROVIDER": "gemini", "MT_PROVIDER": "gemini", "GEMINI_API_KEY": "dummy_key"}):
            from app.services.gemini import GeminiRecognizer, GeminiTranslator
            interp = build_interpreter()
            self.assertIsInstance(interp.recognizer, GeminiRecognizer)
            self.assertIsInstance(interp.translator, GeminiTranslator)


class HealthzEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_healthz_response_format(self):
        res = self.client.get("/healthz")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("ok"))
        self.assertIn("asr_provider", data)
        self.assertIn("asr_model", data)
        self.assertIn("mt_provider", data)
        self.assertIn("mt_model", data)
        self.assertIn("mt_fallback", data)
        self.assertIsInstance(data.get("has_password"), bool)
        self.assertIsInstance(data.get("online_count"), int)

        # Ensure password or API key is not leaked
        raw_text = res.text
        password = os.getenv("ACCESS_PASSWORD", "")
        if password:
            self.assertNotIn(password, raw_text)

    def test_static_files_served(self):
        res1 = self.client.get("/static/handsfree.js")
        self.assertEqual(res1.status_code, 200)
        self.assertIn("class Segmenter", res1.text)

        res2 = self.client.get("/handsfree.js")
        self.assertEqual(res2.status_code, 200)
        self.assertIn("class Segmenter", res2.text)


if __name__ == "__main__":
    unittest.main()
