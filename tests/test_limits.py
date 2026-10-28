import unittest

from app.services.limits import RateLimiter, is_retryable, retry_delay_hint, with_retry


class FakeErr(Exception):
    def __init__(self, code, msg=""):
        super().__init__(msg)
        self.code = code


class RetryTests(unittest.TestCase):
    def test_retries_then_succeeds(self):
        calls, slept = [], []

        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise FakeErr(429, "Please retry in 1.5s.")
            return "ok"

        self.assertEqual(with_retry(fn, attempts=3, sleep=slept.append), "ok")
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(1.5 <= s <= 2.0 for s in slept))

    def test_non_retryable_raises_immediately(self):
        calls = []

        def fn():
            calls.append(1)
            raise FakeErr(400)

        with self.assertRaises(FakeErr):
            with_retry(fn, sleep=lambda s: None)
        self.assertEqual(len(calls), 1)

    def test_gives_up_when_wait_too_long(self):
        def fn():
            raise FakeErr(429, "Please retry in 3600s.")

        with self.assertRaises(FakeErr):
            with_retry(fn, max_wait=20, sleep=lambda s: None)

    def test_helpers(self):
        self.assertTrue(is_retryable(FakeErr(503)))
        self.assertFalse(is_retryable(ValueError()))
        self.assertIsNone(retry_delay_hint(Exception("no hint")))
        self.assertAlmostEqual(retry_delay_hint(Exception("Please retry in 1.3s.")), 1.3)


class RateLimiterTests(unittest.TestCase):
    def test_blocks_when_window_full(self):
        now = [0.0]
        slept = []

        def sleep(s):
            slept.append(s)
            now[0] += s

        rl = RateLimiter(2, period=60, clock=lambda: now[0], sleep=sleep)
        self.assertEqual(rl.acquire(), 0.0)
        self.assertEqual(rl.acquire(), 0.0)
        waited = rl.acquire()  # slot thứ 3 phải chờ hết cửa sổ
        self.assertAlmostEqual(waited, 60.0)
        self.assertEqual(len(slept), 1)


if __name__ == "__main__":
    unittest.main()
