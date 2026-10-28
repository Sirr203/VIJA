import os

from app.services.base import Interpreter, SpeechRecognizer, Translator


def _recognizer(provider: str) -> SpeechRecognizer:
    if provider == "groq":
        from app.services.groq import GroqRecognizer
        return GroqRecognizer()
    if provider == "gemini":
        from app.services.gemini import GeminiRecognizer
        return GeminiRecognizer()
    raise ValueError(f"Invalid ASR_PROVIDER: {provider}")


def _translator(provider: str, max_wait: float | None = None, tone: str | None = None) -> Translator:
    if provider == "groq":
        from app.services.groq import GroqTranslator
        return GroqTranslator(tone=tone)
    if provider == "gemini":
        from app.services.gemini import GeminiTranslator
        return GeminiTranslator(max_wait=max_wait, tone=tone)
    raise ValueError(f"Invalid MT_PROVIDER: {provider}")


def build_interpreter(mode: str | None = None, tone: str | None = None) -> Interpreter:
    """Select service providers via environment variables. Lazy import allows testing without SDKs.

    MODE          two_step (default) | one_step (Gemini only)
    ASR_PROVIDER  groq (default) | gemini
    MT_PROVIDER   groq (default) | gemini
    MT_FALLBACK   none (default) | groq | gemini  -- fallback when primary translator fails or queues too long
    MT_MAX_WAIT   maximum seconds to wait on rate limit before falling back
    TONE          auto (default) | polite | casual
    """
    mode = mode or os.getenv("MODE", "two_step")
    tone = tone or os.getenv("TONE", "auto")
    if mode == "one_step":
        from app.services.gemini import GeminiOneStepInterpreter
        return GeminiOneStepInterpreter(tone=tone)
    if mode != "two_step":
        raise ValueError(f"Invalid MODE: {mode}")

    from app.services.pipeline import TwoStepInterpreter

    asr_provider = os.getenv("ASR_PROVIDER", "groq").lower()
    mt_provider = os.getenv("MT_PROVIDER", "groq").lower()
    fallback = os.getenv("MT_FALLBACK", "none").lower()
    use_fallback = fallback != "none" and fallback != mt_provider
    max_wait = float(os.getenv("MT_MAX_WAIT", "3")) if use_fallback else None

    translator = _translator(mt_provider, max_wait=max_wait, tone=tone)
    if use_fallback:
        from app.services.fallback import FallbackTranslator
        translator = FallbackTranslator(translator, _translator(fallback, tone=tone), tone=tone)
    return TwoStepInterpreter(_recognizer(asr_provider), translator, tone=tone)
