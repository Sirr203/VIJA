import io
import math
import struct
import unittest
import wave

from app.audio import analyze_wav


def make_wav(seconds, amplitude, rate=16000):
    n = int(seconds * rate)
    frames = b"".join(
        struct.pack("<h", int(amplitude * 32767 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(n)
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return buf.getvalue()


class AudioTests(unittest.TestCase):
    def test_loud_tone(self):
        dur, db = analyze_wav(make_wav(1.0, 0.3))
        self.assertAlmostEqual(dur, 1.0, places=2)
        self.assertTrue(-16 < db < -11, db)  # sin biên độ 0.3 có RMS ≈ -13.5 dBFS

    def test_silence_and_short_clip(self):
        dur, db = analyze_wav(make_wav(1.0, 0.0))
        self.assertLess(db, -100)
        dur, _ = analyze_wav(make_wav(0.2, 0.3))
        self.assertAlmostEqual(dur, 0.2, places=2)

    def test_garbage_is_treated_as_silence(self):
        self.assertEqual(analyze_wav(b"not a wav"), (0.0, -120.0))


if __name__ == "__main__":
    unittest.main()
