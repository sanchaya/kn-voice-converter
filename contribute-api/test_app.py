import io
import wave

import pytest

from app import create_app


def wav(seconds=2, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(rate * seconds))
    return buf.getvalue()


@pytest.fixture
def c(tmp_path):
    return create_app(tmp_path).test_client()


def join(c):
    r = c.post("/v1/consent", json={"adult": True, "version": "draft-1", "licence": "CC0-1.0"})
    return {"Authorization": "Bearer " + r.get_json()["token"]}


def upload(c, h, data=None):
    return c.post("/v1/clips", headers=h, data={"audio": (io.BytesIO(data or wav()), "a.wav"), "prompt_id": "1"})


def test_consent_requires_adult(c):
    assert c.post("/v1/consent", json={"version": "x", "licence": "CC0-1.0"}).status_code == 400


def test_auth_required(c):
    assert c.post("/v1/clips").status_code == 401


def test_prompt_and_upload(c):
    h = join(c)
    assert c.get("/v1/prompts/next?task=read").get_json()["text"]
    assert upload(c, h).status_code == 201


@pytest.mark.parametrize("data", [wav(rate=44100), wav(seconds=0.2), b"not a wav"])
def test_bad_audio_rejected(c, data):
    assert upload(c, join(c), data).status_code == 400


def test_validation_flow(c):
    a, b, d = join(c), join(c), join(c)
    clip = upload(c, a).get_json()["id"]
    assert c.get("/v1/validate/next", headers=a).status_code == 404  # never your own clip
    assert c.get(f"/v1/clips/{clip}/audio", headers=a).status_code == 404
    assert c.get("/v1/validate/next", headers=b).get_json()["id"] == clip
    assert c.get(f"/v1/clips/{clip}/audio", headers=b).status_code == 200
    assert c.post("/v1/validate", headers=a, json={"clip_id": clip, "vote": "correct"}).status_code == 400
    assert c.post("/v1/validate", headers=b, json={"clip_id": clip, "vote": "correct"}).get_json()["status"] == "pending"
    assert c.post("/v1/validate", headers=b, json={"clip_id": clip, "vote": "correct"}).status_code == 409
    assert c.post("/v1/validate", headers=d, json={"clip_id": clip, "vote": "correct"}).get_json()["status"] == "accepted"
    assert c.get("/v1/stats").get_json()["hours"]["accepted"] > 0


def test_delete_removes_audio(c, tmp_path):
    h = join(c)
    upload(c, h)
    assert c.delete("/v1/contributions", headers=h).get_json()["deleted_clips"] == 1
    assert c.get("/v1/stats").get_json()["clips"] == 0
    assert not list((tmp_path / "audio").glob("*.wav"))
