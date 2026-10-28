"""Quick validation and audio analysis of WAV audio data before making API calls."""
import io
import math
import sys
import wave
from array import array


def analyze_wav(data: bytes) -> tuple[float, float]:
    """Returns (duration in seconds, RMS level in dBFS) for 16-bit mono WAV audio.

    Corrupted data or invalid formats are treated as silence: (0.0, -120.0).
    """
    try:
        with wave.open(io.BytesIO(data)) as w:
            if w.getsampwidth() != 2:
                return 0.0, -120.0
            rate = w.getframerate()
            frames = w.readframes(w.getnframes())
    except (wave.Error, EOFError):
        return 0.0, -120.0

    samples = array("h")
    samples.frombytes(frames[: len(frames) // 2 * 2])
    if sys.byteorder == "big":
        samples.byteswap()
    if not samples:
        return 0.0, -120.0

    duration = len(samples) / rate
    sub = samples[::4]  # Decimate for faster computation; accurate enough for energy estimation
    rms = math.sqrt(sum(s * s for s in sub) / len(sub)) / 32768
    return duration, (20 * math.log10(rms) if rms > 0 else -120.0)
