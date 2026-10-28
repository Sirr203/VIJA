import json
import logging
import os
import time

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore

from pydantic import BaseModel

from app.services import prompts
from app.services.base import Interpretation, Interpreter, SpeechRecognizer, Translator
from app.services.limits import RateLimiter, with_retry

log = logging.getLogger("vija.gemini")

# Model names may change over time; verify in Google AI Studio if 404 occurs
ASR_MODEL = os.getenv("ASR_MODEL", "gemini-3.5-flash")
MT_MODEL = os.getenv("MT_MODEL", "gemini-3.5-flash")
RPM = int(os.getenv("GEMINI_RPM", "5"))  # Free tier default: 5 RPM/model
TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "30000"))
THINKING = os.getenv("GEMINI_THINKING", "minimal").upper()


def _thinking(model: str):
    if not THINKING or THINKING in ("OFF", "NONE") or types is None:
        return None
    try:
        level = getattr(types.ThinkingLevel, THINKING, None)
        if level:
            return types.ThinkingConfig(thinking_level=level)
    except Exception:
        pass
    return None


# Rate limits are per model; share a rate limiter for each model name
_limiters: dict[str, RateLimiter] = {}


def _limiter(model: str) -> RateLimiter:
    return _limiters.setdefault(model, RateLimiter(RPM))


class _Caller:
    """Invokes generate_content with rate limiting and retry handling."""

    def __init__(self, client=None, model: str = MT_MODEL, max_wait: float | None = None):
        if genai is None or types is None:
            raise ImportError(
                "google-genai is not installed. To use Gemini, run: pip install -r requirements-gemini.txt"
            )
        if client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key or api_key in ("dan_key_moi_vao_day", "your_gemini_api_key_here"):
                raise ValueError("GEMINI_API_KEY is not set in environment or .env")
            self.client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=TIMEOUT_MS),
            )
        else:
            self.client = client
        self.model = model
        self.max_wait = max_wait

    def generate(self, contents, config) -> str:
        config.thinking_config = _thinking(self.model)
        waited = 0.0

        def call():
            nonlocal waited
            waited += _limiter(self.model).acquire(self.max_wait)
            return self.client.models.generate_content(
                model=self.model, contents=contents, config=config
            )

        t0 = time.perf_counter()
        text = (with_retry(call).text or "").strip()
        log.info("gemini %s: total %.1fs, rate limit wait %.1fs", self.model, time.perf_counter() - t0, waited)
        return text


class GeminiRecognizer(SpeechRecognizer):
    def __init__(self, client=None, model: str = ASR_MODEL):
        self._c = _Caller(client, model)

    def transcribe(self, audio: bytes, mime: str, language: str) -> str:
        if types is None:
            raise ImportError("google-genai is not installed.")
        return self._c.generate(
            [prompts.transcribe_prompt(language), types.Part.from_bytes(data=audio, mime_type=mime)],
            types.GenerateContentConfig(temperature=0),
        )


class GeminiTranslator(Translator):
    def __init__(
        self,
        client=None,
        model: str = MT_MODEL,
        max_wait: float | None = None,
        tone: str | None = None,
    ):
        self._c = _Caller(client, model, max_wait)
        self.tone = tone or os.getenv("TONE", "auto")

    def translate(self, text, src, dst, context=None, tone: str | None = None):
        content = text
        if context:
            content = "Previous sentences: " + " / ".join(context) + "\n\nTranslate: " + text
        active_tone = tone or self.tone
        if types is None:
            raise ImportError("google-genai is not installed.")
        return self._c.generate(
            content,
            types.GenerateContentConfig(
                system_instruction=prompts.translate_system(src, dst, tone=active_tone),
                temperature=0,
            ),
        )


class _Out(BaseModel):
    original: str
    translated: str


class GeminiOneStepInterpreter(Interpreter):
    """Single call: audio -> {original, translated}. Faster with fewer requests."""

    def __init__(self, client=None, model: str = ASR_MODEL, tone: str | None = None):
        self._c = _Caller(client, model)
        self.tone = tone or os.getenv("TONE", "auto")

    def interpret(self, audio, mime, src, dst, context=None, tone: str | None = None):
        if types is None:
            raise ImportError("google-genai is not installed.")
        parts = []
        if context:
            parts.append("Previous sentences: " + " / ".join(context))
        parts.append(types.Part.from_bytes(data=audio, mime_type=mime))
        active_tone = tone or self.tone
        t0 = time.perf_counter()
        raw = self._c.generate(
            parts,
            types.GenerateContentConfig(
                system_instruction=prompts.one_step_system(src, dst, tone=active_tone),
                temperature=0,
                response_mime_type="application/json",
                response_schema=_Out,
            ),
        )
        data = json.loads(raw)
        total = int((time.perf_counter() - t0) * 1000)
        return Interpretation(data["original"].strip(), data["translated"].strip(), {"total": total})
