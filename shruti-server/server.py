"""Local OpenAI-compatible transcription server for Shruti Kannada (Udgam Labs, Apache-2.0).

Lets OpenWhispr (Settings -> Transcription -> Custom) use the on-device Shruti model.
Audio never leaves this machine: it listens on 127.0.0.1 only.
Needs: ffmpeg on PATH, `pip install -r requirements.txt`, and shruti-kannada.cact.
shortcut: one request at a time and the whole file is transcribed at once; fine for dictation,
use a queue and chunking before serving long recordings or several users.
"""
import os
import subprocess
import tempfile

os.environ.setdefault("CACTUS_TELEMETRY", "0")  # the package sends anonymous usage counts by default

from flask import Flask, jsonify, request
from cactus import Whistle

MODEL_PATH = os.environ.get("SHRUTI_MODEL", "shruti-kannada.cact")
MODEL_ID = "shruti-kannada"
MAX_BYTES = 100 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES
model = Whistle(MODEL_PATH)


def to_wav16k(src, dst):
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, "-ac", "1", "-ar", "16000", "-f", "wav", dst]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode:
        raise ValueError(r.stderr.strip()[:300] or "ffmpeg could not read the audio")


@app.get("/v1/models")
def models():
    return jsonify(object="list", data=[{"id": MODEL_ID, "object": "model"}])


@app.post("/v1/audio/transcriptions")
def transcribe():
    f = request.files.get("file")
    if not f:
        return jsonify(error={"message": "file is required"}), 400
    with tempfile.TemporaryDirectory() as d:
        src, wav = os.path.join(d, "in"), os.path.join(d, "clip.wav")
        f.save(src)
        try:
            to_wav16k(src, wav)
        except (ValueError, subprocess.TimeoutExpired) as e:
            return jsonify(error={"message": str(e)}), 400
        # language=None: Shruti detects it; "kn" from OpenWhispr is not in the base engine's language list
        text = (model.transcribe(wav).get("text") or "").strip()
    if request.form.get("response_format") in ("text", "srt", "vtt"):
        return text, 200, {"Content-Type": "text/plain; charset=utf-8"}
    return jsonify(text=text)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


if __name__ == "__main__":
    # 127.0.0.1 only: do not expose a dictation server to the network
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8765")))
