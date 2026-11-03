"""Contribute API for the Kannada voice collection (runs at dani-api.sanchaya.net).

All data stays on this server: SQLite + WAV files under DATA_DIR. No third-party services.
shortcut: SQLite is single-writer and no rate limiting, fine for a pilot; move to Postgres
and add rate limits at the Coolify proxy before a public campaign.
"""
import os
import secrets
import sqlite3
import time
import uuid
import wave
from pathlib import Path

from flask import Flask, abort, g, jsonify, request, send_file

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "https://dani.sanchaya.net")
MAX_BYTES = 5 * 1024 * 1024  # 60 s of 16 kHz mono 16-bit is ~1.9 MB
MIN_SEC, MAX_SEC = 1.0, 60.0
VOTES = {"correct", "wrong", "unusable"}
VOTES_NEEDED = 2  # agreeing validators to accept or reject a clip

SCHEMA = """
CREATE TABLE IF NOT EXISTS contributors (
  token TEXT PRIMARY KEY, consent_version TEXT NOT NULL, licence TEXT NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS prompts (
  id INTEGER PRIMARY KEY, task TEXT NOT NULL, text TEXT NOT NULL, UNIQUE(task, text));
CREATE TABLE IF NOT EXISTS clips (
  id TEXT PRIMARY KEY, token TEXT NOT NULL, prompt_id INTEGER, transcript TEXT,
  district TEXT, dialect TEXT, seconds REAL NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS votes (
  clip_id TEXT NOT NULL, token TEXT NOT NULL, vote TEXT NOT NULL, created REAL NOT NULL,
  PRIMARY KEY (clip_id, token));
"""


def create_app(data_dir=None):
    app = Flask(__name__)
    root = Path(data_dir or DATA_DIR)
    (root / "audio").mkdir(parents=True, exist_ok=True)
    db_path = root / "contribute.db"

    def db():
        if "db" not in g:
            g.db = sqlite3.connect(db_path)
            g.db.row_factory = sqlite3.Row
        return g.db

    @app.teardown_appcontext
    def close_db(_):
        conn = g.pop("db", None)
        if conn:
            conn.close()

    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        seed = Path(__file__).with_name("prompts.tsv")
        if seed.exists():
            for line in seed.read_text(encoding="utf-8").splitlines():
                if "\t" in line and not line.startswith("#"):
                    task, text = line.split("\t", 1)
                    conn.execute("INSERT OR IGNORE INTO prompts(task, text) VALUES (?, ?)", (task, text.strip()))

    @app.after_request
    def cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
        resp.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
        resp.headers["Vary"] = "Origin"
        return resp

    @app.route("/v1/<path:_>", methods=["OPTIONS"])
    def preflight(_):
        return "", 204

    def me():
        token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        if not token or not db().execute("SELECT 1 FROM contributors WHERE token=?", (token,)).fetchone():
            abort(401)
        return token

    def status(clip_id):
        rows = db().execute("SELECT vote, COUNT(*) n FROM votes WHERE clip_id=? GROUP BY vote", (clip_id,)).fetchall()
        n = {r["vote"]: r["n"] for r in rows}
        if n.get("correct", 0) >= VOTES_NEEDED:
            return "accepted"
        if n.get("wrong", 0) + n.get("unusable", 0) >= VOTES_NEEDED:
            return "rejected"
        return "pending"

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/consent")
    def consent():
        body = request.get_json(silent=True) or {}
        if body.get("adult") is not True or not body.get("version") or not body.get("licence"):
            abort(400, "adult confirmation, consent version and licence are required")
        token = secrets.token_urlsafe(24)
        db().execute("INSERT INTO contributors VALUES (?,?,?,?)", (token, body["version"], body["licence"], time.time()))
        db().commit()
        return {"token": token}

    @app.get("/v1/prompts/next")
    def next_prompt():
        task = request.args.get("task", "read")
        row = db().execute("SELECT id, text FROM prompts WHERE task=? ORDER BY RANDOM() LIMIT 1", (task,)).fetchone()
        if not row:
            abort(404, "no prompts for this task")
        return {"id": row["id"], "text": row["text"], "task": task}

    @app.post("/v1/clips")
    def upload_clip():
        token = me()
        f = request.files.get("audio")
        if not f:
            abort(400, "audio file required")
        data = f.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            abort(413, "clip too large")
        clip_id = uuid.uuid4().hex
        path = root / "audio" / f"{clip_id}.wav"
        path.write_bytes(data)
        try:
            with wave.open(str(path)) as w:
                ok = (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
                seconds = w.getnframes() / w.getframerate()
        except (wave.Error, EOFError):
            ok, seconds = False, 0
        if not ok:
            path.unlink()
            abort(400, "audio must be 16 kHz mono 16-bit WAV")
        if not MIN_SEC <= seconds <= MAX_SEC:
            path.unlink()
            abort(400, f"clip must be {MIN_SEC:g} to {MAX_SEC:g} seconds")
        form = request.form
        db().execute(
            "INSERT INTO clips VALUES (?,?,?,?,?,?,?,?)",
            (clip_id, token, form.get("prompt_id", type=int), form.get("transcript", "").strip() or None,
             form.get("district", "")[:80] or None, form.get("dialect", "")[:80] or None, seconds, time.time()))
        db().commit()
        return {"id": clip_id, "seconds": round(seconds, 2)}, 201

    @app.get("/v1/validate/next")
    def next_to_validate():
        token = me()
        rows = db().execute(
            """SELECT c.id, c.transcript, p.text AS prompt FROM clips c LEFT JOIN prompts p ON p.id=c.prompt_id
               WHERE c.token != ? AND c.id NOT IN (SELECT clip_id FROM votes WHERE token=?)
               ORDER BY c.created LIMIT 50""", (token, token)).fetchall()
        for r in rows:
            if status(r["id"]) == "pending":
                return {"id": r["id"], "text": r["transcript"] or r["prompt"]}
        abort(404, "nothing to validate right now")

    @app.get("/v1/clips/<clip_id>/audio")
    def clip_audio(clip_id):
        token = me()
        row = db().execute("SELECT token FROM clips WHERE id=?", (clip_id,)).fetchone()
        if not row or row["token"] == token:
            abort(404)
        return send_file(root / "audio" / f"{clip_id}.wav", mimetype="audio/wav")

    @app.post("/v1/validate")
    def validate():
        token = me()
        body = request.get_json(silent=True) or {}
        clip = db().execute("SELECT token FROM clips WHERE id=?", (body.get("clip_id"),)).fetchone()
        if body.get("vote") not in VOTES or not clip or clip["token"] == token:
            abort(400, "valid clip_id (not your own) and vote required")
        try:
            db().execute("INSERT INTO votes VALUES (?,?,?,?)", (body["clip_id"], token, body["vote"], time.time()))
        except sqlite3.IntegrityError:
            abort(409, "already voted")
        db().commit()
        return {"status": status(body["clip_id"])}

    @app.get("/v1/stats")
    def stats():
        clips = db().execute("SELECT id, seconds FROM clips").fetchall()
        by = {"accepted": 0.0, "pending": 0.0, "rejected": 0.0}
        for c in clips:
            by[status(c["id"])] += c["seconds"]
        return {"clips": len(clips), "hours": {k: round(v / 3600, 3) for k, v in by.items()}}

    @app.delete("/v1/contributions")
    def delete_mine():
        token = me()
        ids = [r["id"] for r in db().execute("SELECT id FROM clips WHERE token=?", (token,))]
        for i in ids:
            (root / "audio" / f"{i}.wav").unlink(missing_ok=True)
            db().execute("DELETE FROM votes WHERE clip_id=?", (i,))
        db().execute("DELETE FROM clips WHERE token=?", (token,))
        db().execute("DELETE FROM votes WHERE token=?", (token,))
        db().execute("DELETE FROM contributors WHERE token=?", (token,))
        db().commit()
        return {"deleted_clips": len(ids)}

    return app


app = create_app() if os.environ.get("DATA_DIR") else None
