import unittest

from app.services.limits import RateLimiter, WaitTooLong


class MaxWaitTests(unittest.TestCase):
    def test_raises_instead_of_sleeping_when_wait_exceeds_max(self):
        now, slept = [0.0], []
        rl = RateLimiter(1, period=60, clock=lambda: now[0], sleep=slept.append)
        rl.acquire(max_wait=3)
        with self.assertRaises(WaitTooLong):
            rl.acquire(max_wait=3)
        self.assertEqual(slept, [])  # không ngủ

    def test_short_wait_is_allowed(self):
        now = [0.0]

        def sleep(s):
            now[0] += s

        rl = RateLimiter(1, period=2, clock=lambda: now[0], sleep=sleep)
        rl.acquire(max_wait=5)
        self.assertAlmostEqual(rl.acquire(max_wait=5), 2.0)


if __name__ == "__main__":
    unittest.main()
