import logging

from app.services.base import Translator

log = logging.getLogger("vija.fallback")


class FallbackTranslator(Translator):
    """Attempt primary translator; on failure (429, 503, queue timeout...), fallback to secondary."""

    def __init__(self, primary: Translator, secondary: Translator, tone: str | None = None):
        self.primary, self.secondary = primary, secondary
        self.tone = tone or getattr(primary, "tone", "auto")

    def _call(self, trans: Translator, text: str, src: str, dst: str, context, tone):
        try:
            return trans.translate(text, src, dst, context, tone=tone)
        except TypeError:
            return trans.translate(text, src, dst, context)

    def translate(self, text, src, dst, context=None, tone: str | None = None):
        active_tone = tone or self.tone
        try:
            return self._call(self.primary, text, src, dst, context, active_tone)
        except Exception as exc:
            log.warning("Primary translator failed (%s: %s), falling back to secondary", type(exc).__name__, exc)
            return self._call(self.secondary, text, src, dst, context, active_tone)
