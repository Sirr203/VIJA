import unittest

from app.rooms import Peer, RoomManager


class RoomTests(unittest.TestCase):
    def setUp(self):
        self.rm = RoomManager()

    def test_two_different_languages_can_join(self):
        a, b = Peer(None, "vi"), Peer(None, "ja")
        room, old = self.rm.join("1", a)
        self.assertIsNone(old)
        room2, old = self.rm.join("1", b)
        self.assertIsNone(old)
        self.assertIs(room, room2)
        self.assertIs(room.other(a), b)

    def test_same_language_replaces_old_connection(self):
        old_peer, new_peer = Peer(None, "vi"), Peer(None, "vi")
        self.rm.join("1", old_peer)
        room, old = self.rm.join("1", new_peer)
        self.assertIs(old, old_peer)
        self.assertEqual(room.peers, [new_peer])

    def test_old_connection_leaving_does_not_drop_new_one(self):
        old_peer, new_peer = Peer(None, "vi"), Peer(None, "vi")
        room, _ = self.rm.join("1", old_peer)
        self.rm.join("1", new_peer)
        self.rm.leave(room, old_peer)  # kết nối cũ đóng muộn
        self.assertEqual(room.peers, [new_peer])

    def test_empty_room_is_removed_and_history_is_capped(self):
        a = Peer(None, "vi")
        room, _ = self.rm.join("1", a)
        for i in range(10):
            room.remember(str(i))
        self.assertEqual(room.context(), ["4", "5", "6", "7", "8", "9"])
        self.assertEqual(self.rm.get("1"), room)
        self.rm.leave(room, a)
        self.assertIsNone(self.rm.get("1"))
        room2, _ = self.rm.join("1", Peer(None, "ja"))
        self.assertIsNot(room, room2)  # phòng cũ đã bị xóa, lịch sử không rò sang phòng mới


if __name__ == "__main__":
    unittest.main()
