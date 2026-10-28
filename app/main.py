"""Bidirectional conversation translation server (Vietnamese <-> Japanese)."""
import asyncio
import logging
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Automatically load environment variables from .env
load_dotenv()

from app.audio import analyze_wav
from app.rooms import LANG_NAMES, Peer, Room, RoomManager
from app.security import (
    SlidingWindow,
    client_ip,
    password_ok,
    read_hello,
)
from app.services.factory import build_interpreter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("vija")

DEFAULT_ROOM = "default"
MAX_AUDIO_BYTES = 5 * 1024 * 1024
ACCESS_PASSWORD = os.getenv("ACCESS_PASSWORD", "")
MIN_SPEECH_S = float(os.getenv("MIN_SPEECH_S", "0.4"))
MIN_SPEECH_DBFS = float(os.getenv("MIN_SPEECH_DBFS", "-48"))
MAX_AUDIO_S = float(os.getenv("MAX_AUDIO_S", "30"))
AUDIO_PER_MIN = int(os.getenv("AUDIO_PER_MIN", "15"))
AUTH_MAX_FAILS = int(os.getenv("AUTH_MAX_FAILS", "5"))
AUTH_FAIL_WINDOW_S = float(os.getenv("AUTH_FAIL_WINDOW_S", "600"))
TRUST_PROXY = os.getenv("TRUST_PROXY", "0") == "1"

if not ACCESS_PASSWORD:
    log.warning("WARNING: ACCESS_PASSWORD is not set. Anyone can connect without a password!")

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="vija")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
rooms = RoomManager()
try:
    interpreter = build_interpreter()
except Exception as exc:
    log.warning("Interpreter not initialized at startup: %s", exc)
    interpreter = None
auth_failures = SlidingWindow(limit=AUTH_MAX_FAILS, period=AUTH_FAIL_WINDOW_S)
audio_rate = SlidingWindow(limit=AUDIO_PER_MIN, period=60.0)


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/handsfree.js")
async def handsfree_js():
    return FileResponse(STATIC / "handsfree.js", media_type="application/javascript")


@app.get("/healthz")
async def healthz():
    asr_provider = os.getenv("ASR_PROVIDER", "groq")
    asr_model = (
        getattr(getattr(interpreter, "recognizer", None), "model", None)
        or os.getenv("GROQ_ASR_MODEL")
        or os.getenv("ASR_MODEL")
        or "whisper-large-v3"
    )
    mt_provider = os.getenv("MT_PROVIDER", "groq")
    mt_obj = getattr(interpreter, "translator", None)
    mt_model = (
        getattr(mt_obj, "model", None)
        or os.getenv("GROQ_MT_MODEL")
        or os.getenv("MT_MODEL")
        or "llama-3.3-70b-versatile"
    )
    mt_fallback = os.getenv("MT_FALLBACK", "none")

    online_count = 0
    default_room = rooms.get(DEFAULT_ROOM)
    if default_room:
        online_count = len(default_room.peers)

    return {
        "ok": True,
        "asr_provider": asr_provider,
        "asr_model": asr_model,
        "mt_provider": mt_provider,
        "mt_model": mt_model,
        "mt_fallback": mt_fallback,
        "has_password": bool(ACCESS_PASSWORD),
        "online_count": online_count,
    }


async def handle_audio(room: Room, peer: Peer, audio: bytes) -> None:
    other = room.other(peer)
    if other is None:
        await peer.send({"type": "error", "text": "Waiting for the other peer to connect."})
        return

    if not audio_rate.allow(peer):
        await peer.send({"type": "error", "text": "Rate limit exceeded. Please wait a moment."})
        return

    duration, level = await asyncio.to_thread(analyze_wav, audio)
    log.info("audio %.1fs %.1f dBFS", duration, level)
    if duration > MAX_AUDIO_S:
        await peer.send({"type": "error", "text": "Audio recording is too long."})
        return
    if duration < MIN_SPEECH_S or level < MIN_SPEECH_DBFS:
        await peer.send({"type": "status", "text": "No speech detected."})
        return

    await peer.send({"type": "status", "text": "Processing..."})
    try:
        global interpreter
        if interpreter is None:
            interpreter = build_interpreter()
        result = await asyncio.to_thread(
            interpreter.interpret, audio, "audio/wav",
            LANG_NAMES[peer.lang], LANG_NAMES[other.lang], room.context(),
        )
    except Exception as exc:
        log.exception("pipeline failed: %s", exc)
        await peer.send({"type": "error", "text": "Processing error, please try again."})
        return
    if not result.original:
        await peer.send({"type": "status", "text": "No speech detected."})
        return

    log.info("room=%s %s->%s timings_ms=%s", room.code, peer.lang, other.lang, result.timings_ms)
    room.remember(result.original)
    await room.broadcast({
        "type": "message",
        "id": uuid.uuid4().hex[:8],
        "speaker": peer.lang,
        "original": result.original,
        "translated": result.translated,
        "timings_ms": result.timings_ms,
    })
    await peer.send({"type": "status", "text": ""})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, lang: str = "vi"):
    await ws.accept()
    ip = client_ip(ws.headers, ws.client.host if ws.client else None, TRUST_PROXY)

    # 1. Check if IP is currently rate-limited/locked
    if auth_failures.count(ip) >= AUTH_MAX_FAILS:
        await ws.send_json({"type": "fatal", "text": "Too many failed attempts. Please try again later."})
        await ws.close()
        return

    # 2. Check valid language
    if lang not in LANG_NAMES:
        await ws.send_json({"type": "fatal", "text": "Invalid language. Choose 'vi' or 'ja'."})
        await ws.close()
        return

    # 3. Read first hello message
    hello = await read_hello(ws)
    if not hello:
        await ws.send_json({"type": "fatal", "text": "Invalid handshake request."})
        await ws.close()
        return

    # 4. Check password token
    token = hello.get("token")
    if not password_ok(ACCESS_PASSWORD, token):
        auth_failures.add(ip)
        await ws.send_json({"type": "fatal", "text": "Incorrect password."})
        await ws.close()
        return

    peer = Peer(ws, lang)
    room, old = rooms.join(DEFAULT_ROOM, peer)

    if old:
        await old.send({"type": "fatal", "text": "Session replaced by a newer connection."})
        await old.ws.close()

    log.info("join lang=%s ip=%s", lang, ip)
    await room.broadcast({"type": "presence", "count": len(room.peers)})
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):
                await handle_audio(room, peer, msg["bytes"])
    finally:
        audio_rate.forget(peer)
        rooms.leave(room, peer)
        log.info("leave lang=%s", lang)
        await room.broadcast({"type": "presence", "count": len(room.peers)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
