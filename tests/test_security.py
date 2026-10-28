import asyncio
import unittest

from app.security import (
    SlidingWindow,
    client_ip,
    password_ok,
    read_hello,
)


class SecurityTests(unittest.TestCase):
    def test_password_ok(self):
        # expected rỗng thì luôn đúng
        self.assertTrue(password_ok("", "anything"))
        self.assertTrue(password_ok("", None))

        # given không phải string thì sai
        self.assertFalse(password_ok("secret", None))
        self.assertFalse(password_ok("secret", 12345))

        # Đúng và sai mật khẩu
        self.assertTrue(password_ok("vija2026", "vija2026"))
        self.assertFalse(password_ok("vija2026", "wrong"))

    def test_client_ip(self):
        headers = {
            "cf-connecting-ip": "203.0.113.195",
            "x-forwarded-for": "198.51.100.1, 192.0.2.1",
        }
        # trust_proxy=False: không đọc proxy header
        self.assertEqual(client_ip(headers, "127.0.0.1", trust_proxy=False), "127.0.0.1")

        # trust_proxy=True: ưu tiên cf-connecting-ip
        self.assertEqual(client_ip(headers, "127.0.0.1", trust_proxy=True), "203.0.113.195")

        # trust_proxy=True: x-forwarded-for lấy IP đầu tiên
        self.assertEqual(
            client_ip({"x-forwarded-for": "198.51.100.1, 192.0.2.1"}, "127.0.0.1", trust_proxy=True),
            "198.51.100.1",
        )

        # trust_proxy=True nhưng không có header: dùng peer_host
        self.assertEqual(client_ip({}, "10.0.0.2", trust_proxy=True), "10.0.0.2")

    def test_sliding_window_rate_limiter(self):
        now = [100.0]
        sw = SlidingWindow(limit=3, period=60.0, clock=lambda: now[0])

        self.assertTrue(sw.allow("user1"))
        self.assertTrue(sw.allow("user1"))
        self.assertTrue(sw.allow("user1"))
        self.assertFalse(sw.allow("user1"))  # lần 4 bị từ chối
        self.assertEqual(sw.count("user1"), 3)

        # user2 vẫn được allow
        self.assertTrue(sw.allow("user2"))
        self.assertEqual(sw.count("user2"), 1)

        # Quá thời gian 60s
        now[0] += 61.0
        self.assertEqual(sw.count("user1"), 0)
        self.assertNotIn("user1", sw._calls)  # tự xóa khóa rỗng

        # Forget
        sw.add("user3")
        self.assertEqual(sw.count("user3"), 1)
        sw.forget("user3")
        self.assertEqual(sw.count("user3"), 0)


class MockWebSocket:
    def __init__(self, msg):
        self._msg = msg

    async def receive(self):
        if isinstance(self._msg, Exception):
            raise self._msg
        return self._msg


class ReadHelloTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_hello(self):
        ws = MockWebSocket({"type": "websocket.receive", "text": '{"type": "hello", "token": "abc"}'})
        data = await read_hello(ws, timeout=1.0)
        self.assertEqual(data, {"type": "hello", "token": "abc"})

    async def test_invalid_type_or_format(self):
        ws1 = MockWebSocket({"type": "websocket.receive", "text": '{"type": "chat"}'})
        self.assertIsNone(await read_hello(ws1, timeout=1.0))

        ws2 = MockWebSocket({"type": "websocket.receive", "text": "not json"})
        self.assertIsNone(await read_hello(ws2, timeout=1.0))

        ws3 = MockWebSocket({"type": "websocket.disconnect"})
        self.assertIsNone(await read_hello(ws3, timeout=1.0))

    async def test_message_too_long(self):
        long_text = '{"type": "hello", "token": "' + "a" * 1050 + '"}'
        ws = MockWebSocket({"type": "websocket.receive", "text": long_text})
        self.assertIsNone(await read_hello(ws, timeout=1.0))


if __name__ == "__main__":
    unittest.main()

