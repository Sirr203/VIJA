"""Security utilities for WebSocket: password checks, IP extraction, and rate limiting."""
import asyncio
import hmac
import json
import threading
import time
from collections import deque
from typing import Any, Callable


def password_ok(expected: str, given: Any) -> bool:
    """Verify password using hmac.compare_digest. Empty expected string always succeeds."""
    if not expected:
        return True
    if not isinstance(given, str):
        return False
    return hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))


def client_ip(headers: Any, peer_host: str | None, trust_proxy: bool) -> str:
    """Extract client IP. Only inspects cf-connecting-ip / x-forwarded-for when trust_proxy=True."""
    if trust_proxy and headers:
        lookup: dict[str, str] = {}
        if isinstance(headers, dict):
            lookup = {k.lower(): str(v) for k, v in headers.items()}
        elif hasattr(headers, "items"):
            lookup = {k.lower(): str(v) for k, v in headers.items()}
        elif isinstance(headers, (list, tuple)):
            for item in headers:
                if len(item) == 2:
                    k = item[0].decode("latin1").lower() if isinstance(item[0], bytes) else str(item[0]).lower()
                    v = item[1].decode("latin1") if isinstance(item[1], bytes) else str(item[1])
                    lookup[k] = v

        for header_name in ("cf-connecting-ip", "x-forwarded-for"):
            val = lookup.get(header_name)
            if val:
                first = val.split(",")[0].strip()
                if first:
                    return first

    return peer_host or "unknown"


class SlidingWindow:
    """Sliding window call rate limiter per key, automatically purging empty keys."""

    def __init__(self, limit: int, period: float, clock: Callable[[], float] = time.monotonic):
        self.limit = limit
        self.period = period
        self._clock = clock
        self._calls: dict[Any, deque[float]] = {}
        self._lock = threading.Lock()

    def _clean(self, key: Any, now: float) -> deque[float] | None:
        q = self._calls.get(key)
        if q is None:
            return None
        while q and now - q[0] >= self.period:
            q.popleft()
        if not q:
            self._calls.pop(key, None)
            return None
        return q

    def count(self, key: Any) -> int:
        """Call count within the current active window."""
        with self._lock:
            now = self._clock()
            q = self._clean(key, now)
            return len(q) if q else 0

    def add(self, key: Any) -> None:
        """Record an additional call event."""
        with self._lock:
            now = self._clock()
            self._clean(key, now)
            q = self._calls.setdefault(key, deque())
            q.append(now)

    def allow(self, key: Any) -> bool:
        """Check rate limit. If within allowance, records event and returns True."""
        with self._lock:
            now = self._clock()
            q = self._clean(key, now)
            if q is not None and len(q) >= self.limit:
                return False
            if q is None:
                q = self._calls.setdefault(key, deque())
            q.append(now)
            return True

    def forget(self, key: Any) -> None:
        """Remove key from tracking."""
        with self._lock:
            self._calls.pop(key, None)


async def read_hello(ws: Any, timeout: float = 5.0) -> dict | None:
    """Read the first incoming text message from WebSocket.
    
    Must be a JSON object with type="hello" and at most 1024 characters.
    Returns None on invalid format, timeout, or disconnect.
    """
    try:
        if hasattr(ws, "receive"):
            msg = await asyncio.wait_for(ws.receive(), timeout=timeout)
            if not isinstance(msg, dict) or msg.get("type") == "websocket.disconnect":
                return None
            raw = msg.get("text")
        elif hasattr(ws, "receive_text"):
            raw = await asyncio.wait_for(ws.receive_text(), timeout=timeout)
        else:
            return None

        if not isinstance(raw, str) or len(raw) > 1024:
            return None
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get("type") != "hello":
            return None
        return data
    except Exception:
        return None

