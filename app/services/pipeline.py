import time

from app.services.base import Interpretation, Interpreter, SpeechRecognizer, Translator


class TwoStepInterpreter(Interpreter):
    """Composes an arbitrary SpeechRecognizer with a Translator."""

    def __init__(self, recognizer: SpeechRecognizer, translator: Translator, tone: str | None = None):
        self.recognizer, self.translator = recognizer, translator
        self.tone = tone or getattr(translator, "tone", "auto")

    def interpret(self, audio, mime, src, dst, context=None, tone: str | None = None):
        t0 = time.perf_counter()
        original = self.recognizer.transcribe(audio, mime, src)
        t1 = time.perf_counter()
        if not original:
            return Interpretation("", "", {"asr": int((t1 - t0) * 1000), "total": int((t1 - t0) * 1000)})
        active_tone = tone or self.tone
        try:
            translated = self.translator.translate(original, src, dst, context, tone=active_tone)
        except TypeError:
            translated = self.translator.translate(original, src, dst, context)
        t2 = time.perf_counter()
        return Interpretation(
            original,
            translated,
            {"asr": int((t1 - t0) * 1000), "mt": int((t2 - t1) * 1000), "total": int((t2 - t0) * 1000)},
        )
