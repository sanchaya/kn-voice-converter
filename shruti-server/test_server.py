"""Protocol test. Needs shruti-kannada.cact in this folder (or SHRUTI_MODEL) and ffmpeg.
Run: python -m pytest -q test_server.py   (Kannada accuracy is NOT tested here: use a real recording.)"""
import io
import math
import struct
import wave

import server


def tone(sec=2, rate=44100):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(rate)  # stereo 44.1k: must be converted
        w.writeframes(b"".join(struct.pack("<hh", s, s) for s in
                               (int(8000 * math.sin(2 * math.pi * 220 * i / rate)) for i in range(rate * sec))))
    return buf.getvalue()


def test_models_and_health():
    c = server.app.test_client()
    assert c.get("/v1/models").get_json()["data"][0]["id"] == "shruti-kannada"
    assert c.get("/health").get_json()["status"] == "ok"


def test_transcribe_returns_text_field():
    c = server.app.test_client()
    r = c.post("/v1/audio/transcriptions", data={"file": (io.BytesIO(tone()), "a.wav"), "model": "x", "language": "kn"})
    assert r.status_code == 200 and isinstance(r.get_json()["text"], str)


def test_bad_audio_and_missing_file():
    c = server.app.test_client()
    assert c.post("/v1/audio/transcriptions", data={"file": (io.BytesIO(b"nope"), "a.wav")}).status_code == 400
    assert c.post("/v1/audio/transcriptions", data={}).status_code == 400
