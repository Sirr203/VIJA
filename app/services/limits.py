"""Rate limiting and retry logic for 429/5xx responses. No AI library dependencies."""
import logging
import random
import re
import threading
import time
from collections import deque
from typing import Callable, TypeVar

log = logging.getLogger("vija.limits")
T = TypeVar("T")


class WaitTooLong(Exception):
    """Queue wait time too long; signals caller to attempt fallback."""


class RateLimiter:
    """Sliding window rate limiter: at most max_calls per period seconds."""

    def __init__(self, max_calls: int, period: float = 60.0, clock=time.monotonic, sleep=time.sleep):
        self.max_calls, self.period = max_calls, period
        self._clock, self._sleep = clock, sleep
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self, max_wait: float | None = None) -> float:
        """Block until an execution slot is available. Returns seconds waited."""
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                while self._calls and now - self._calls[0] >= self.period:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return waited
                wait = self.period - (now - self._calls[0])
                if max_wait is not None and waited + wait > max_wait:
                    raise WaitTooLong(f"Need to wait {wait:.1f}s more")
            log.info("Rate limit: waiting %.1fs", wait)
            self._sleep(wait)
            waited += wait


RETRYABLE_CODES = {429, 500, 502, 503, 504}


def is_retryable(exc: Exception) -> bool:
    return getattr(exc, "code", None) in RETRYABLE_CODES


def retry_delay_hint(exc: Exception) -> float | None:
    """Extract 'Please retry in 1.3s' hint if present in Gemini/Groq error."""
    retry_after = getattr(exc, "retry_after", None)  # Retry-After header
    if retry_after:
        return float(retry_after)
    m = re.search(r"retry in ([\d.]+)s", str(exc))
    return float(m.group(1)) if m else None


def with_retry(
    fn: Callable[[], T], attempts: int = 3, max_wait: float = 20.0, sleep=time.sleep
) -> T:
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if not is_retryable(exc) or i == attempts - 1:
                raise
            delay = retry_delay_hint(exc) or (2 ** i)
            if delay > max_wait:  # E.g. daily quota exceeded: waiting is futile
                raise
            delay += random.uniform(0, 0.5)
            log.warning("HTTP %s error, retrying in %.1fs (attempt %d/%d)", getattr(exc, "code", "?"), delay, i + 1, attempts)
            sleep(delay)
    raise RuntimeError("unreachable")
