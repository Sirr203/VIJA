"""Speech recognition (Whisper) and translation (LLM) via Groq. OpenAI-compatible API."""
import logging
import os
import re

from app.services import prompts
from app.services.base import SpeechRecognizer, Translator
from app.services.limits import RateLimiter, with_retry

log = logging.getLogger("vija.groq")

BASE_URL = "https://api.groq.com/openai/v1"
TIMEOUT_S = 20.0

DEFAULT_ASR_MODEL = "whisper-large-v3"
DEFAULT_MT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_ASR_RPM = 20
DEFAULT_MT_RPM = 30

# Legacy aliases for backward compatibility
ASR_MODEL = DEFAULT_ASR_MODEL
MT_MODEL = DEFAULT_MT_MODEL

# Whisper expects ISO-639-1 language codes; pipeline interface provides English names
_ISO = {"Vietnamese": "vi", "Japanese": "ja", "English": "en"}


class ApiError(Exception):
    """HTTP API error; contains `code` attribute to guide with_retry decisions."""

    def __init__(self, code: int, message: str, retry_after: str | float | None = None):
        super().__init__(f"HTTP {code}: {message}")
        self.code = code
        try:
            self.retry_after = float(retry_after) if retry_after is not None else None
        except ValueError:
            self.retry_after = None


class GroqHTTP:
    """Lightweight wrapper around httpx for test mocking."""

    def __init__(self, api_key: str | None = None, timeout: float | None = None):
        import httpx  # Lazy import so tests can run without httpx installed

        self._httpx = httpx
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY is not set in environment or .env")
        actual_timeout = timeout if timeout is not None else float(os.getenv("GROQ_TIMEOUT_S", str(TIMEOUT_S)))
        self._client = httpx.Client(
            base_url=BASE_URL, headers={"Authorization": f"Bearer {key}"}, timeout=actual_timeout
        )

    def post(self, path: str, *, data=None, files=None, json=None) -> dict:
        try:
            r = self._client.post(path, data=data, files=files, json=json)
        except self._httpx.TransportError as exc:  # Timeout, network drop...
            raise ApiError(503, f"{type(exc).__name__}") from exc
        if r.status_code >= 400:
            raise ApiError(r.status_code, r.text[:200], r.headers.get("retry-after"))
        return r.json()


def _extension(mime: str) -> str:
    m = mime.lower()
    for key, ext in (("wav", ".wav"), ("mp4", ".m4a"), ("m4a", ".m4a"), ("aac", ".m4a"),
                     ("mpeg", ".mp3"), ("mp3", ".mp3"), ("ogg", ".ogg"),
                     ("webm", ".webm"), ("flac", ".flac")):
        if key in m:
            return ext
    return ".wav"


def clean_transcript(result: dict) -> str:
    """Filter out Whisper hallucinations during silence or noise."""
    segments = result.get("segments")
    if not segments:
        return (result.get("text") or "").strip()
    kept = []
    for s in segments:
        no_speech = s.get("no_speech_prob", 0) > 0.6 and s.get("avg_logprob", 0) < -1.0
        looping = s.get("compression_ratio", 0) > 2.4  # Repetition loop
        if no_speech or looping:
            log.info("dropping hallucinated segment: %r", s.get("text", "")[:60])
            continue
        kept.append(s.get("text", "").strip())
    return " ".join(t for t in kept if t).strip()


class GroqRecognizer(SpeechRecognizer):
    def __init__(self, http: GroqHTTP | None = None, model: str | None = None,
                 prompt: str | None = None, rpm: int | None = None,
                 prompt_vi: str | None = None, prompt_ja: str | None = None):
        self.http = http or GroqHTTP()
        self.model = model or os.getenv("GROQ_ASR_MODEL", DEFAULT_ASR_MODEL)
        self.prompt = prompt
        self.prompt_vi = prompt_vi
        self.prompt_ja = prompt_ja
        actual_rpm = rpm if rpm is not None else int(os.getenv("GROQ_ASR_RPM", str(DEFAULT_ASR_RPM)))
        self._limiter = RateLimiter(actual_rpm)

    def _resolve_prompt(self, language: str) -> str:
        lang_key = _ISO.get(language, language.lower())
        if lang_key == "vi":
            p = self.prompt_vi or os.getenv("GROQ_ASR_PROMPT_VI")
            if p:
                return p.strip()
        elif lang_key == "ja":
            p = self.prompt_ja or os.getenv("GROQ_ASR_PROMPT_JA")
            if p:
                return p.strip()
        p = self.prompt if self.prompt is not None else os.getenv("GROQ_ASR_PROMPT", "")
        return (p or "").strip()

    def transcribe(self, audio: bytes, mime: str, language: str) -> str:
        data = {"model": self.model, "response_format": "verbose_json", "temperature": "0"}
        if language in _ISO:
            data["language"] = _ISO[language]
        prompt = self._resolve_prompt(language)
        if prompt:
            data["prompt"] = prompt
        files = {"file": (f"audio{_extension(mime)}", audio, mime)}

        def call():
            self._limiter.acquire()
            return self.http.post("/audio/transcriptions", data=data, files=files)

        return clean_transcript(with_retry(call))


class GroqTranslator(Translator):
    def __init__(
        self,
        http: GroqHTTP | None = None,
        model: str | None = None,
        rpm: int | None = None,
        tone: str | None = None,
    ):
        self.http = http or GroqHTTP()
        self.model = model or os.getenv("GROQ_MT_MODEL", DEFAULT_MT_MODEL)
        actual_rpm = rpm if rpm is not None else int(os.getenv("GROQ_MT_RPM", str(DEFAULT_MT_RPM)))
        self._limiter = RateLimiter(actual_rpm)
        self.tone = tone or os.getenv("TONE", "auto")

    def translate(self, text, src, dst, context=None, tone: str | None = None):
        user = text
        if context:
            user = "Previous sentences: " + " / ".join(context) + "\n\nTranslate: " + text
        active_tone = tone or self.tone
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": prompts.translate_system(src, dst, tone=active_tone)},
                {"role": "user", "content": user},
            ],
        }
        if self.model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = "low"  # model suy luận: giữ ở mức thấp cho nhanh

        def call():
            self._limiter.acquire()
            return self.http.post("/chat/completions", json=body)

        out = with_retry(call)["choices"][0]["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
