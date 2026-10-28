"""Room and peer session management. No AI/ML dependencies."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import WebSocket

LANG_NAMES = {"vi": "Vietnamese", "ja": "Japanese"}


@dataclass(eq=False)
class Peer:
    ws: WebSocket
    lang: str

    async def send(self, data: dict) -> None:
        try:
            await self.ws.send_json(data)
        except Exception:  # Connection already closed
            pass


class Room:
    def __init__(self, code: str):
        self.code = code
        self.peers: list[Peer] = []
        self._history: deque[str] = deque(maxlen=6)  # Conversation context for translation

    def other(self, peer: Peer) -> Peer | None:
        return next((p for p in self.peers if p is not peer), None)

    def context(self) -> list[str]:
        return list(self._history)

    def remember(self, text: str) -> None:
        self._history.append(text)

    async def broadcast(self, data: dict) -> None:
        for p in list(self.peers):
            await p.send(data)


class RoomManager:
    def __init__(self):
        self._rooms: dict[str, Room] = {}

    def get(self, code: str) -> Room | None:
        return self._rooms.get(code)

    def join(self, code: str, peer: Peer) -> tuple[Room, Peer | None]:
        if code not in self._rooms:
            room = Room(code)
            self._rooms[code] = room
        else:
            room = self._rooms[code]

        old = next((p for p in room.peers if p.lang == peer.lang), None)
        if old:
            room.peers.remove(old)
        room.peers.append(peer)
        return room, old

    def leave(self, room: Room, peer: Peer) -> None:
        if peer in room.peers:
            room.peers.remove(peer)
        if not room.peers:
            self._rooms.pop(room.code, None)
