import unittest

from app.services.base import Translator
from app.services.fallback import FallbackTranslator


class Fake(Translator):
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, 0

    def translate(self, text, src, dst, context=None):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


class FallbackTests(unittest.TestCase):
    def test_primary_ok_secondary_untouched(self):
        a, b = Fake("A"), Fake("B")
        self.assertEqual(FallbackTranslator(a, b).translate("x", "vi", "ja"), "A")
        self.assertEqual(b.calls, 0)

    def test_primary_error_uses_secondary(self):
        a, b = Fake(error=RuntimeError("503")), Fake("B")
        self.assertEqual(FallbackTranslator(a, b).translate("x", "vi", "ja"), "B")

    def test_both_fail_raises_secondary_error(self):
        a, b = Fake(error=RuntimeError("a")), Fake(error=ValueError("b"))
        with self.assertRaises(ValueError):
            FallbackTranslator(a, b).translate("x", "vi", "ja")


if __name__ == "__main__":
    unittest.main()
