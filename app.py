#!/usr/bin/env python3
"""
ದನಿ ಕನ್ನಡ — Kannada Speech-to-Text Server
  Live mic recording  → chunked Whisper → streaming Kannada transcript
  File upload         → full transcription

Run:
    python app.py
    python app.py --port 8998 --model large-v3

Endpoints:
    GET  /              Web UI
    POST /transcribe    Upload file → JSON
    WS   /ws/transcribe Live mic chunks → streaming JSON
    GET  /health
"""

import argparse, base64, json, os, sys, tempfile, time, threading
from pathlib import Path
from flask import Flask, request, jsonify

try:
    from flask_sock import Sock
    HAS_SOCK = True
except ImportError:
    HAS_SOCK = False

SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".ogg",
                     ".aac", ".wma", ".webm", ".mp4"}

app        = Flask(__name__)
sock       = Sock(app) if HAS_SOCK else None
model      = None
model_lock = threading.Lock()

# ── Transcription backend ─────────────────────────────────────────────────────
# BACKEND = "mlx"   → mlx-whisper  (Apple Silicon M1-M4, uses Metal GPU — recommended)
# BACKEND = "api"   → OpenAI Whisper API (cloud, requires OPENAI_API_KEY)
# BACKEND = "local" → openai-whisper on CPU (slow, may freeze Mac)
BACKEND    = "local"
api_client = None
mlx_model_repo = None   # HuggingFace repo string for mlx-whisper

KANNADA_PROMPT = "ಕನ್ನಡ ಭಾಷೆಯಲ್ಲಿ ಮಾತನಾಡಿದ ಪಠ್ಯ:"


def do_transcribe(audio_path: str) -> tuple[str, float]:
    """Transcribe audio file. Returns (text, duration_seconds)."""
    if BACKEND == "mlx":
        import mlx_whisper
        with model_lock:   # prevent concurrent Metal GPU access
            result = mlx_whisper.transcribe(
                audio_path,
                path_or_hf_repo=mlx_model_repo,
                language="kn",
                initial_prompt=KANNADA_PROMPT,
            )
        text = result["text"].strip()
        dur  = result["segments"][-1]["end"] if result.get("segments") else 0
        return text, dur

    elif BACKEND == "api":
        with open(audio_path, "rb") as f:
            resp = api_client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="kn",
                prompt=KANNADA_PROMPT,
                response_format="verbose_json",
            )
        text = resp.text.strip()
        dur  = getattr(resp, "duration", 0) or 0
        return text, dur

    else:  # local
        with model_lock:
            result = model.transcribe(
                audio_path, language="kn", task="transcribe",
                fp16=False, initial_prompt=KANNADA_PROMPT,
            )
        text = result["text"].strip()
        dur  = result["segments"][-1]["end"] if result["segments"] else 0
        return text, dur


# ── Load branding assets ──────────────────────────────────────────────────────
def load_assets():
    path = Path(__file__).parent / "assets.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    # Fallback: transparent 1×1 PNG
    blank = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    return {"favicon_32": blank, "app_icon": blank, "chrome_192": blank, "sanchaya_logo": blank}

ASSETS = load_assets()


# ── HTML ──────────────────────────────────────────────────────────────────────
def build_html():
    A = ASSETS
    return """<!DOCTYPE html>
<html lang="kn">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ದನಿ ಕನ್ನಡ — ಧ್ವನಿ ಪಠ್ಯ</title>
  <link rel="icon" type="image/png" href="{favicon_32}">
  <link rel="apple-touch-icon" sizes="180x180" href="{app_icon}">
  <link rel="icon" type="image/png" sizes="192x192" href="{chrome_192}">
  <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Kannada:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    /* ── Sanchaya design tokens (matches fonts.sanchaya.net) ── */
    :root {{
      --primary:     #4361ee;
      --primary-dk:  #3f37c9;
      --primary-grad: linear-gradient(135deg, #4361ee 0%, #3f37c9 100%);
      --text-color1: #1a1a2e;
      --text-color3: #4361ee;
      --span-color:  #4a4a68;
      --bg-color:    #f8f9fa;
      --card-bg:     #ffffff;
      --border:      #e0e0e0;
      --card-shadow: 0 4px 20px rgba(0,0,0,.08);
      --card-shadow-hover: 0 8px 30px rgba(67,97,238,.15);
      --hover-color: rgba(67,97,238,.08);
      --green:  #2ecc71; --green-dk: #27ae60;
      --red:    #e74c3c;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans Kannada', sans-serif;
      background: var(--bg-color);
      color: var(--text-color1);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }}

    /* ── NAV  (matches sanchaya.net: white bg, centred logo, 80px tall) ── */
    nav {{
      width: 100%;
      min-height: 80px;
      background: var(--card-bg);
      box-shadow: 0 2px 10px rgba(0,0,0,.04);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 12px 24px;
      position: relative;
    }}
    .center-nav {{
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 4px;
    }}
    /* width:auto so the 400×109 landscape logo isn't squashed */
    .nav-logo {{ height: 40px; width: auto; display: block; }}
    .nav-tagline {{
      font-size: 13px;
      color: var(--span-color);
      font-weight: 500;
      font-family: 'Noto Sans Kannada', sans-serif;
    }}
    .nav-right {{
      position: absolute;
      right: 24px;
      display: flex;
      align-items: center;
      gap: 16px;
    }}
    .nav-link {{
      font-size: 13px;
      color: var(--span-color);
      text-decoration: none;
      padding: 6px 10px;
      border-radius: 6px;
      transition: background .2s, color .2s;
    }}
    .nav-link:hover {{ background: var(--hover-color); color: var(--primary); }}

    /* ── LAYOUT ── */
    main {{ flex: 1; display: flex; justify-content: center; padding: 32px 16px; }}
    .card {{
      background: var(--card-bg);
      border-radius: 16px;
      box-shadow: var(--card-shadow);
      width: 100%;
      max-width: 720px;
      overflow: hidden;
      transition: box-shadow .2s;
    }}
    .card:hover {{ box-shadow: var(--card-shadow-hover); }}

    /* ── TABS ── */
    .tabs {{ display: flex; border-bottom: 1px solid var(--border); }}
    .tab {{
      flex: 1; padding: 14px; text-align: center; cursor: pointer; font-size: 15px;
      font-weight: 500; color: var(--span-color); border: none; background: none;
      font-family: inherit; transition: background .2s, color .2s;
    }}
    .tab.active {{
      color: var(--primary);
      border-bottom: 2px solid var(--primary);
      margin-bottom: -1px;
    }}
    .tab:hover:not(.active) {{ background: var(--hover-color); color: var(--primary); }}
    .panel {{ display: none; padding: 28px 28px 24px; }}
    .panel.active {{ display: block; }}

    /* ── MIC BUTTON ── */
    .mic-row {{ display: flex; align-items: center; gap: 16px; margin-bottom: 20px; }}
    .mic-btn {{
      width: 64px; height: 64px; border-radius: 50%; border: none; cursor: pointer;
      background: linear-gradient(135deg, var(--primary), var(--primary-dk));
      color: #fff; font-size: 26px; display: flex; align-items: center;
      justify-content: center; transition: transform .15s; flex-shrink: 0;
    }}
    .mic-btn:hover {{ transform: scale(1.06); }}
    .mic-btn.recording {{
      background: linear-gradient(135deg, var(--red), #c0392b);
      animation: pulse 1.4s infinite;
    }}
    @keyframes pulse {{
      0%   {{ box-shadow: 0 0 0 0   rgba(231,76,60,.5); }}
      70%  {{ box-shadow: 0 0 0 14px rgba(231,76,60,0); }}
      100% {{ box-shadow: 0 0 0 0   rgba(231,76,60,0); }}
    }}
    .mic-info h2 {{ font-size: 17px; color: var(--navy); margin-bottom: 4px; }}
    .mic-info p  {{ font-size: 13px; color: #999; }}

    /* ── WAVEFORM ── */
    canvas#waveform {{
      width: 100%; height: 56px; border-radius: 8px;
      background: var(--navy); margin-bottom: 18px; display: block;
    }}

    /* ── TIMELINE ── */
    .timeline-box {{
      background: var(--bg-color);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 16px;
      margin-bottom: 18px;
    }}
    .timeline-row {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
    .tl-label {{
      font-size: 11px; font-weight: 600; color: var(--primary);
      text-transform: uppercase; letter-spacing: .6px;
    }}
    .tl-counters {{ display: flex; gap: 20px; }}
    .tl-counter {{ text-align: center; }}
    .tl-counter .val {{
      font-size: 20px; font-weight: 700; color: var(--text-color1);
      font-variant-numeric: tabular-nums;
    }}
    .tl-counter .lbl {{ font-size: 11px; color: var(--span-color); margin-top: 1px; }}
    .tl-bar-wrap {{ position: relative; height: 10px; background: #dde1f9;
                    border-radius: 5px; overflow: hidden; }}
    .tl-bar-recorded {{ position: absolute; height: 100%; background: rgba(67,97,238,.25);
                         border-radius: 5px; transition: width .5s; }}
    .tl-bar-done {{ position: absolute; height: 100%; background: var(--primary);
                    border-radius: 5px; transition: width .8s; }}
    .tl-bar-proc {{ position: absolute; height: 100%; width: 0; background: rgba(67,97,238,.55);
                    animation: blink 1s ease-in-out infinite; }}
    @keyframes blink {{ 0%,100% {{ opacity:.4; }} 50% {{ opacity:1; }} }}
    .tl-legend {{ display: flex; gap: 14px; margin-top: 8px; }}
    .tl-legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 11px; color: #888; }}
    .tl-legend-dot {{ width: 10px; height: 10px; border-radius: 2px; }}

    /* ── CHUNK STATUS ── */
    .chunk-status {{ font-size: 12px; color: #bbb; margin-bottom: 14px;
                     min-height: 18px; display: flex; align-items: center; gap: 6px; }}
    .dot-spin {{ width: 8px; height: 8px; border-radius: 50%;
                 border: 2px solid rgba(67,97,238,.3); border-top-color: var(--primary);
                 animation: spin .7s linear infinite; flex-shrink: 0; }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

    /* ── TRANSCRIPT ── */
    .transcript-label {{ font-size: 12px; font-weight: 600; color: var(--primary);
                          text-transform: uppercase; letter-spacing: .4px; margin-bottom: 8px; }}
    .transcript-box {{
      background: var(--bg-color); border: 1px solid var(--border); border-radius: 8px;
      padding: 12px 16px; font-size: 18px; line-height: 1.9; min-height: 120px;
      color: var(--text-color1);
      user-select: text; -webkit-user-select: text;
      max-height: 360px; overflow-y: auto;
      display: flex; flex-direction: column; gap: 2px;
    }}
    .transcript-row {{
      display: flex; align-items: baseline; gap: 10px;
      padding: 3px 0; border-bottom: 1px solid transparent;
      transition: color 1.5s;
    }}
    .transcript-row.chunk-new {{ color: var(--primary); }}
    .transcript-row.chunk-new .ts {{ color: var(--primary); opacity: .9; }}
    .ts {{
      font-size: 11px; color: #aaa; font-variant-numeric: tabular-nums;
      white-space: nowrap; flex-shrink: 0; padding-top: 3px;
      font-family: ui-monospace, monospace;
    }}
    .chunk-text {{ flex: 1; }}
    .session-sep {{
      font-size: 11px; color: #bbb; text-align: center;
      padding: 8px 0 4px; letter-spacing: .04em;
    }}
    .pending-dots {{
      color: var(--primary); opacity: .5;
      animation: blink .9s step-start infinite;
    }}
    @keyframes blink {{ 0%,100%{{opacity:.5}} 50%{{opacity:1}} }}
    /* interim row — live partial result while user is still speaking */
    .interim-row {{ opacity: .55; font-style: italic; }}
    .interim-text {{ color: var(--span-color); }}

    /* ── UPLOAD ── */
    .upload-zone {{
      border: 2px dashed var(--border); border-radius: 10px; padding: 36px 20px;
      text-align: center; cursor: pointer; margin-bottom: 16px;
      transition: border-color .2s, background .2s;
    }}
    .upload-zone:hover, .upload-zone.drag-over {{
      border-color: var(--primary); background: rgba(67,97,238,.04);
    }}
    .upload-icon {{ font-size: 38px; margin-bottom: 10px; }}
    .upload-zone p {{ color: var(--text); font-size: 15px; margin-bottom: 4px; }}
    .formats {{ color: #aaa; font-size: 12px; }}
    input[type=file] {{ display: none; }}
    .file-tag {{
      display: none; align-items: center; gap: 8px;
      background: rgba(67,97,238,.06); border: 1px solid rgba(67,97,238,.2);
      border-radius: 8px; padding: 10px 14px; font-size: 14px;
      color: var(--primary); margin-bottom: 16px;
    }}
    .file-tag.show {{ display: flex; }}

    /* ── BUTTONS ── */
    .btn {{
      width: 100%; padding: 13px; border: none; border-radius: 8px;
      font-size: 15px; font-family: inherit; cursor: pointer;
      transition: opacity .2s; font-weight: 500;
    }}
    .btn-primary {{ background: linear-gradient(135deg, var(--primary), var(--primary-dk)); color: #fff; }}
    .btn-primary:hover:not(:disabled) {{ opacity: .9; }}
    .btn-primary:disabled {{ background: #ccc; cursor: not-allowed; }}
    .btn-copy  {{ background: var(--green); color: #fff; margin-top: 10px; }}
    .btn-copy:hover {{ background: var(--green-dk); }}
    .btn-clear {{ background: #f0f0f0; color: #666; }}
    .btn-clear:hover {{ background: #e0e0e0; }}
    .btn-row {{ display: flex; gap: 10px; margin-top: 12px; }}
    .btn-row .btn {{ flex: 1; margin-top: 0; }}

    /* ── STATUS / RESULT ── */
    .status {{ text-align: center; font-size: 13px; color: #999; margin-top: 10px; min-height: 20px; }}
    .status .err {{ color: var(--red); }}
    .result-upload {{ display: none; margin-top: 22px; }}

    /* ── FOOTER (light, matching sanchaya.net) ── */
    footer {{
      background: var(--card-bg);
      border-top: 1px solid var(--border);
      color: var(--span-color);
      text-align: center;
      padding: 24px 16px;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 12px;
    }}
    /* sanchaya_logo is 450×451 (square) — show at 36px height */
    .footer-logo {{ height: 36px; width: auto; opacity: .85; }}
    .footer-links {{ display: flex; gap: 24px; }}
    .footer-links a {{
      color: var(--span-color); text-decoration: none; font-size: 13px;
      transition: color .2s;
    }}
    .footer-links a:hover {{ color: var(--primary); }}
    .footer-copy {{ font-size: 12px; color: #bbb; }}

    @media (max-width: 480px) {{
      .panel {{ padding: 20px 16px; }}
      .tl-counters {{ gap: 10px; }}
    }}
  </style>
</head>
<body>

<!-- ── Navbar ── -->
<nav>
  <div class="center-nav">
    <a href="https://sanchaya.org" target="_blank">
      <img class="nav-logo" src="{app_icon}" alt="ದನಿ ಕನ್ನಡ">
    </a>
    <span class="nav-tagline">ಧ್ವನಿ → ಕನ್ನಡ ಪಠ್ಯ</span>
  </div>
  <div class="nav-right">
    <a class="nav-link" href="https://sanchaya.org" target="_blank">ಸಂಚಯ</a>
    <a class="nav-link" href="https://sanchifoundation.org" target="_blank">ಸಂಚಿ ಫೌಂಡೇಶನ್</a>
  </div>
</nav>

<main>
<div class="card">

  <div class="tabs">
    <button class="tab active" onclick="switchTab('live')">🎙️ ನೇರ ರೆಕಾರ್ಡ್</button>
    <button class="tab"        onclick="switchTab('upload')">📂 ಕಡತ ಅಪ್ಲೋಡ್</button>
  </div>

  <!-- ── LIVE TAB ── -->
  <div class="panel active" id="panel-live">

    <div class="mic-row">
      <button class="mic-btn" id="micBtn" onclick="toggleRecording()">🎙️</button>
      <div class="mic-info">
        <h2 id="micStatus">ರೆಕಾರ್ಡ್ ಮಾಡಲು ಕ್ಲಿಕ್ ಮಾಡಿ</h2>
        <p  id="micHint">ಮೈಕ್ ಅನ್ನು ಬ್ರೌಸರ್ ಬಳಸಲು ಅನುಮತಿ ನೀಡಿ</p>
      </div>
    </div>

    <canvas id="waveform"></canvas>

    <div class="timeline-box">
      <div class="timeline-row">
        <span class="tl-label">ಸಮಯದ ಸ್ಥಿತಿ</span>
        <div class="tl-counters">
          <div class="tl-counter">
            <div class="val" id="tl-recorded">0:00</div>
            <div class="lbl">ರೆಕಾರ್ಡ್</div>
          </div>
          <div class="tl-counter">
            <div class="val" id="tl-transcribed">0:00</div>
            <div class="lbl">ಪಠ್ಯ ಆದದ್ದು</div>
          </div>
          <div class="tl-counter">
            <div class="val" id="tl-lag">0s</div>
            <div class="lbl">ವ್ಯತ್ಯಾಸ</div>
          </div>
        </div>
      </div>
      <div class="tl-bar-wrap">
        <div class="tl-bar-recorded" id="bar-recorded" style="width:0%"></div>
        <div class="tl-bar-done"     id="bar-done"     style="width:0%"></div>
        <div class="tl-bar-proc"     id="bar-proc"     style="width:0%;left:0%"></div>
      </div>
      <div class="tl-legend">
        <div class="tl-legend-item">
          <div class="tl-legend-dot" style="background:rgba(67,97,238,.25)"></div>ರೆಕಾರ್ಡ್
        </div>
        <div class="tl-legend-item">
          <div class="tl-legend-dot" style="background:var(--primary)"></div>ಪಠ್ಯ ಪೂರ್ಣ
        </div>
        <div class="tl-legend-item">
          <div class="tl-legend-dot" style="background:rgba(67,97,238,.55)"></div>ಪ್ರಕ್ರಿಯೆ
        </div>
      </div>
    </div>

    <div class="chunk-status" id="chunkStatus"></div>

    <div class="transcript-label">ಕನ್ನಡ ಪಠ್ಯ (ನೇರ)</div>
    <div class="transcript-box" id="liveTranscript"></div>

    <div class="btn-row">
      <button class="btn btn-copy"  onclick="copyLive()">📋 ನಕಲಿಸಿ</button>
      <button class="btn btn-clear" onclick="clearLive()">🗑️ ಅಳಿಸಿ</button>
    </div>
  </div>

  <!-- ── UPLOAD TAB ── -->
  <div class="panel" id="panel-upload">
    <label for="fileInput">
      <div class="upload-zone" id="dropZone">
        <div class="upload-icon">🎵</div>
        <p>ಇಲ್ಲಿ ಕ್ಲಿಕ್ ಮಾಡಿ ಅಥವಾ ಕಡತವನ್ನು ಎಳೆದು ಬಿಡಿ</p>
        <p class="formats">MP3 · WAV · M4A · FLAC · OGG · AAC · WebM</p>
      </div>
    </label>
    <input type="file" id="fileInput"
           accept=".mp3,.wav,.m4a,.flac,.ogg,.aac,.wma,.webm,.mp4">
    <div class="file-tag" id="fileTag">
      <span>📄</span><span id="fileName"></span>
    </div>
    <button class="btn btn-primary" id="transcribeBtn"
            disabled onclick="transcribeFile()">ಪಠ್ಯಕ್ಕೆ ಪರಿವರ್ತಿಸಿ</button>
    <div class="status" id="uploadStatus"></div>
    <div class="result-upload" id="resultUpload">
      <div class="transcript-label" style="margin-top:20px">ಕನ್ನಡ ಪಠ್ಯ</div>
      <div class="transcript-box" id="uploadOutput"></div>
      <button class="btn btn-copy" onclick="copyUpload()">📋 ಪಠ್ಯ ನಕಲಿಸಿ</button>
    </div>
  </div>

</div>
</main>

<!-- ── Footer ── -->
<footer>
  <img class="footer-logo" src="{sanchaya_logo}" alt="Sanchaya">
  <div class="footer-links">
    <a href="https://sanchaya.org"        target="_blank">ಸಂಚಯ</a>
    <a href="https://sanchifoundation.org" target="_blank">ಸಂಚಿ ಫೌಂಡೇಶನ್</a>
  </div>
  <div class="footer-copy">© 2026 Sanchaya &amp; Sanchi Foundation</div>
</footer>

<script>
// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(name) {{
  const isLive = name === 'live';
  document.querySelectorAll('.tab').forEach((t,i)   => t.classList.toggle('active', i === (isLive?0:1)));
  document.querySelectorAll('.panel').forEach((p,i) => p.classList.toggle('active', i === (isLive?0:1)));
}}

function fmtTime(sec) {{
  sec = Math.floor(sec);
  return Math.floor(sec/60) + ':' + String(sec % 60).padStart(2,'0');
}}

// ── Waveform ──────────────────────────────────────────────────────────────────
let analyser = null, waveAF = null;
const wfCanvas = document.getElementById('waveform');
const wfCtx    = wfCanvas.getContext('2d');

function startWaveform(stream) {{
  const actx = new AudioContext();
  analyser = actx.createAnalyser();
  analyser.fftSize = 256;
  actx.createMediaStreamSource(stream).connect(analyser);
  drawWave();
}}
function stopWaveform() {{
  if (waveAF) cancelAnimationFrame(waveAF);
  wfCtx.clearRect(0, 0, wfCanvas.width, wfCanvas.height);
  analyser = null;
}}
function drawWave() {{
  if (!analyser) return;
  waveAF = requestAnimationFrame(drawWave);
  const buf = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteTimeDomainData(buf);
  const W = wfCanvas.clientWidth, H = wfCanvas.clientHeight;
  wfCanvas.width = W; wfCanvas.height = H;
  wfCtx.fillStyle = '#1a1a2e';
  wfCtx.fillRect(0, 0, W, H);
  wfCtx.lineWidth = 2;
  wfCtx.strokeStyle = '#4361ee';
  wfCtx.beginPath();
  const slice = W / buf.length;
  let x = 0;
  for (let i = 0; i < buf.length; i++) {{
    const y = (buf[i] / 128) * H / 2;
    i === 0 ? wfCtx.moveTo(x, y) : wfCtx.lineTo(x, y);
    x += slice;
  }}
  wfCtx.stroke();
}}

// ── Timeline state ────────────────────────────────────────────────────────────
let recordedSec = 0, transcribedSec = 0, processingNow = false;
const CHUNK_SECS = 5;   // shorter = faster feedback on M4
let recordTimer = null;

function updateTimeline() {{
  document.getElementById('tl-recorded').textContent    = fmtTime(recordedSec);
  document.getElementById('tl-transcribed').textContent = fmtTime(transcribedSec);
  const lag = Math.max(0, recordedSec - transcribedSec);
  document.getElementById('tl-lag').textContent = lag + 's';

  const maxSec  = Math.max(recordedSec, 1);
  const recPct  = Math.min(100, (recordedSec    / maxSec) * 100);
  const donePct = Math.min(100, (transcribedSec / maxSec) * 100);

  document.getElementById('bar-recorded').style.width = recPct  + '%';
  document.getElementById('bar-done').style.width     = donePct + '%';

  const procEl = document.getElementById('bar-proc');
  if (processingNow) {{
    procEl.style.left  = donePct + '%';
    procEl.style.width = Math.min(8, recPct - donePct) + '%';
  }} else {{
    procEl.style.width = '0%';
  }}
}}

// ── Web Speech API recording ──────────────────────────────────────────────────
let recognition  = null;
let micStream    = null;
let isRecording  = false;
let interimRow   = null;   // live partial-result row

const SR = window.SpeechRecognition || window.webkitSpeechRecognition;

async function toggleRecording() {{
  isRecording ? stopRec() : startRec();
}}

async function startRec() {{
  if (!SR) {{
    setMicStatus('ಬ್ರೌಸರ್ ಬೆಂಬಲವಿಲ್ಲ',
      'Chrome ಅಥವಾ Safari ಬಳಸಿ — Firefox Speech API ಬೆಂಬಲಿಸುವುದಿಲ್ಲ');
    return;
  }}

  // Get mic stream for waveform visualisation only
  try {{
    micStream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
  }} catch(e) {{
    setMicStatus('ಮೈಕ್ ಅನುಮತಿ ನಿರಾಕರಿಸಲಾಗಿದೆ', 'ಬ್ರೌಸರ್‌ನಲ್ಲಿ ಮೈಕ್ ಅನುಮತಿ ನೀಡಿ');
    return;
  }}

  isRecording = true;
  recordedSec = 0; transcribedSec = 0;
  appendSessionMarker();
  updateTimeline();
  startWaveform(micStream);

  document.getElementById('micBtn').classList.add('recording');
  document.getElementById('micBtn').textContent = '⏹️';
  setMicStatus('ರೆಕಾರ್ಡ್ ನಡೆಯುತ್ತಿದೆ…', 'ನಿಲ್ಲಿಸಲು ಮತ್ತೆ ಕ್ಲಿಕ್ ಮಾಡಿ');
  recordTimer = setInterval(() => {{ recordedSec++; updateTimeline(); }}, 1000);

  // Start Web Speech API
  recognition = new SR();
  recognition.lang = 'kn-IN';
  recognition.continuous      = true;
  recognition.interimResults  = true;
  recognition.maxAlternatives = 1;

  recognition.onresult = (event) => {{
    let interim = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {{
      const r = event.results[i];
      if (r.isFinal) {{
        const text = r[0].transcript.trim();
        if (text) {{
          // Remove the interim row, add a permanent row
          if (interimRow) {{ interimRow.remove(); interimRow = null; }}
          appendLive(text);
          transcribedSec = recordedSec;
          updateTimeline();
        }}
      }} else {{
        interim += r[0].transcript;
      }}
    }}
    // Update the live interim row
    if (interim) {{
      const box = document.getElementById('liveTranscript');
      if (!interimRow) {{
        interimRow = document.createElement('div');
        interimRow.className = 'transcript-row interim-row';
        box.appendChild(interimRow);
      }}
      interimRow.innerHTML =
        '<span class="ts">' + fmtClock() + '</span>'
        + '<span class="chunk-text interim-text">' + interim + '</span>';
      box.scrollTop = box.scrollHeight;
    }}
  }};

  recognition.onerror = (e) => {{
    if (e.error === 'no-speech') return;   // normal silence — ignore
    if (e.error === 'aborted')  return;   // we stopped it — ignore
    setMicStatus('ದೋಷ: ' + e.error, 'ಪುಟ ರಿಫ್ರೆಶ್ ಮಾಡಿ ಮತ್ತು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ');
  }};

  recognition.onend = () => {{
    // Auto-restart while still recording (browser stops after silence)
    if (isRecording) recognition.start();
  }};

  recognition.start();
  setChunkStatus('');
}}

function stopRec() {{
  isRecording = false;
  clearInterval(recordTimer);
  if (recognition) {{ recognition.abort(); recognition = null; }}
  if (micStream)   micStream.getTracks().forEach(t => t.stop());
  if (interimRow)  {{ interimRow.remove(); interimRow = null; }}
  stopWaveform();
  document.getElementById('micBtn').classList.remove('recording');
  document.getElementById('micBtn').textContent = '🎙️';
  setMicStatus('ರೆಕಾರ್ಡ್ ಮಾಡಲು ಕ್ಲಿಕ್ ಮಾಡಿ', 'ರೆಕಾರ್ಡಿಂಗ್ ನಿಲ್ಲಿಸಲಾಗಿದೆ');
  setChunkStatus('');
}}

function fmtClock() {{
  const n = new Date();
  return n.toLocaleTimeString('kn-IN', {{ hour: '2-digit', minute: '2-digit', second: '2-digit' }});
}}

function appendSessionMarker() {{
  const box = document.getElementById('liveTranscript');
  // Skip marker if box is empty (very first session)
  if (!box.innerHTML.trim()) return;
  const sep = document.createElement('div');
  sep.className = 'session-sep';
  sep.textContent = '── ಹೊಸ ರೆಕಾರ್ಡಿಂಗ್ ' + fmtClock() + ' ──';
  box.appendChild(sep);
}}

function appendLive(text) {{
  const box = document.getElementById('liveTranscript');

  // Promote the interim row to a final row, or create a fresh one
  let row;
  if (interimRow) {{
    row = interimRow;
    interimRow = null;
    row.innerHTML = '';
  }} else {{
    row = document.createElement('div');
    box.appendChild(row);
  }}

  row.className = 'transcript-row chunk-new';

  const ts = document.createElement('span');
  ts.className   = 'ts';
  ts.textContent = fmtClock();

  const txt = document.createElement('span');
  txt.className   = 'chunk-text';
  txt.textContent = text;

  row.appendChild(ts);
  row.appendChild(txt);
  setTimeout(() => row.classList.remove('chunk-new'), 1800);
  box.scrollTop = box.scrollHeight;
}}

function setMicStatus(h, p) {{
  document.getElementById('micStatus').textContent = h;
  document.getElementById('micHint').textContent   = p;
}}
function setChunkStatus(msg) {{
  const el = document.getElementById('chunkStatus');
  el.innerHTML = msg ? '<div class="dot-spin"></div>' + msg : '';
}}
function copyLive() {{
  // Copy only the Kannada text, not the timestamps
  const rows = document.querySelectorAll('#liveTranscript .chunk-text');
  const text = Array.from(rows).map(r => r.textContent).join(' ').trim();
  navigator.clipboard.writeText(text);
}}
function clearLive() {{
  document.getElementById('liveTranscript').innerHTML = '';
  recordedSec = 0; transcribedSec = 0; updateTimeline();
}}

// ── File upload ───────────────────────────────────────────────────────────────
const fileInput = document.getElementById('fileInput');
const dropZone  = document.getElementById('dropZone');

fileInput.addEventListener('change', () => {{
  if (fileInput.files[0]) showFile(fileInput.files[0].name);
}});
function showFile(name) {{
  document.getElementById('fileName').textContent = name;
  document.getElementById('fileTag').classList.add('show');
  document.getElementById('transcribeBtn').disabled = false;
}}
dropZone.addEventListener('dragover',  e => {{ e.preventDefault(); dropZone.classList.add('drag-over'); }});
dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {{
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) {{ const dt = new DataTransfer(); dt.items.add(file); fileInput.files = dt.files; showFile(file.name); }}
}});

async function transcribeFile() {{
  const file = fileInput.files[0]; if (!file) return;
  const btn    = document.getElementById('transcribeBtn');
  const status = document.getElementById('uploadStatus');
  btn.disabled = true;
  btn.textContent = 'ಪ್ರಕ್ರಿಯೆ…';
  status.textContent = 'ಧ್ವನಿ ಪ್ರಕ್ರಿಯೆ ಆಗುತ್ತಿದೆ, ದಯವಿಟ್ಟು ಕಾಯಿರಿ…';
  document.getElementById('resultUpload').style.display = 'none';
  const form = new FormData(); form.append('audio', file);
  try {{
    const resp = await fetch('/transcribe', {{ method: 'POST', body: form }});
    const data = await resp.json();
    if (data.error) {{
      status.innerHTML = '<span class="err">ದೋಷ: ' + data.error + '</span>';
    }} else {{
      document.getElementById('uploadOutput').textContent = data.text;
      document.getElementById('resultUpload').style.display = 'block';
      status.textContent = '✓ ' + fmtTime(data.duration_seconds) +
                           ' · ' + data.characters + ' ಅಕ್ಷರ · ' +
                           data.processing_seconds + 's';
    }}
  }} catch(e) {{
    status.innerHTML = '<span class="err">ಸಂಪರ್ಕ ದೋಷ: ' + e.message + '</span>';
  }}
  btn.disabled = false; btn.textContent = 'ಪಠ್ಯಕ್ಕೆ ಪರಿವರ್ತಿಸಿ';
}}
function copyUpload() {{
  navigator.clipboard.writeText(document.getElementById('uploadOutput').textContent);
}}
</script>
</body>
</html>""".format(
        favicon_32    = A["favicon_32"],
        app_icon      = A["app_icon"],
        chrome_192    = A["chrome_192"],
        sanchaya_logo = A["sanchaya_logo"],
    )


HTML = build_html()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return HTML


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file. Field name must be 'audio'."}), 400
    audio_file = request.files["audio"]
    ext = Path(audio_file.filename).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        return jsonify({"error": f"Unsupported format '{ext}'."}), 400
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name
    try:
        t0 = time.time()
        text, duration = do_transcribe(tmp_path)
        return jsonify({
            "text": text, "characters": len(text),
            "duration_seconds": duration,
            "processing_seconds": round(time.time() - t0, 2),
            "language": "kn",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


if HAS_SOCK:
    @sock.route("/ws/transcribe")
    def ws_transcribe(ws):
        while True:
            raw = ws.receive()
            if raw is None:
                break
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if msg.get("type") != "audio_chunk":
                continue

            audio_bytes = base64.b64decode(msg["data"])
            chunk_idx   = msg.get("chunk_index", 0)
            chunk_dur   = msg.get("chunk_duration", CHUNK_SECS)
            print(f"[WS] chunk {chunk_idx+1} received — {len(audio_bytes):,} bytes", flush=True)

            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name
            try:
                print(f"[WS] chunk {chunk_idx+1} → transcribing…", flush=True)
                t0 = time.time()
                text, actual_dur = do_transcribe(tmp_path)
                if not actual_dur:
                    actual_dur = chunk_dur
                elapsed = round(time.time() - t0, 1)
                print(f"[WS] chunk {chunk_idx+1} done in {elapsed}s → '{text[:60]}'", flush=True)
                ws.send(json.dumps({
                    "type": "transcript", "text": text,
                    "chunk_index": chunk_idx, "chunk_duration": actual_dur,
                }))
            except Exception as e:
                print(f"[WS] chunk {chunk_idx+1} ERROR: {e}", flush=True)
                ws.send(json.dumps({"type": "error", "message": str(e)}))
            finally:
                os.unlink(tmp_path)


CHUNK_SECS = 5   # must match JS constant


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    global model

    parser = argparse.ArgumentParser(
        description="ದನಿ ಕನ್ನಡ — Kannada Speech-to-Text Server"
    )
    parser.add_argument("--port",  type=int, default=8998)
    parser.add_argument("--host",  default="0.0.0.0")
    parser.add_argument(
        "--model", default="turbo",
        choices=["tiny", "base", "small", "medium", "large", "large-v3", "turbo"],
    )
    parser.add_argument(
        "--model-dir", default=None,
        help="Directory to store/load Whisper model weights "
             "(default: ~/.cache/whisper).",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="OpenAI API key for cloud Whisper. "
             "Can also be set via OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--mlx", action="store_true", default=True,
        help="Use mlx-whisper (Apple Silicon Metal GPU). Default on M-series Macs.",
    )
    parser.add_argument(
        "--no-mlx", dest="mlx", action="store_false",
        help="Disable mlx-whisper and use CPU (not recommended on Apple Silicon).",
    )
    args = parser.parse_args()

    if not HAS_SOCK:
        print("WARNING: flask-sock not installed — live mic disabled.")
        print("         Run: python3 -m pip install flask-sock")

    # ── Choose backend ────────────────────────────────────────────────────────
    global BACKEND, api_client, model, mlx_model_repo

    # Priority: mlx > api-key > local
    if args.mlx:
        try:
            import mlx_whisper  # noqa: F401
            # Map --model names to mlx-community HuggingFace repos
            MLX_REPOS = {
                "tiny":      "mlx-community/whisper-tiny-mlx",
                "base":      "mlx-community/whisper-base-mlx",
                "small":     "mlx-community/whisper-small-mlx",
                "medium":    "mlx-community/whisper-medium-mlx",
                "large":     "mlx-community/whisper-large-v3-mlx",
                "large-v3":  "mlx-community/whisper-large-v3-mlx",
                "turbo":     "mlx-community/whisper-large-v3-turbo",
            }
            mlx_model_repo = MLX_REPOS.get(args.model, "mlx-community/whisper-large-v3-mlx")
            BACKEND = "mlx"
            print(f"✓ mlx-whisper backend — Apple Silicon Metal GPU")
            print(f"  Model: {mlx_model_repo}")
            print(f"  (model downloads on first use, cached in ~/.cache/huggingface/)")
        except ImportError:
            print("mlx-whisper not installed. Installing now…")
            os.system(f"{sys.executable} -m pip install mlx-whisper -q")
            try:
                import mlx_whisper  # noqa: F401
                MLX_REPOS = {
                    "tiny":      "mlx-community/whisper-tiny-mlx",
                    "base":      "mlx-community/whisper-base-mlx",
                    "small":     "mlx-community/whisper-small-mlx",
                    "medium":    "mlx-community/whisper-medium-mlx",
                    "large":     "mlx-community/whisper-large-v3-mlx",
                    "large-v3":  "mlx-community/whisper-large-v3-mlx",
                    "turbo":     "mlx-community/whisper-large-v3-turbo",
                }
                mlx_model_repo = MLX_REPOS.get(args.model, "mlx-community/whisper-large-v3-mlx")
                BACKEND = "mlx"
                print(f"✓ mlx-whisper installed and ready")
            except ImportError:
                print("WARNING: mlx-whisper install failed — falling back to local CPU mode")

    if BACKEND == "local":
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            try:
                from openai import OpenAI
                api_client = OpenAI(api_key=api_key)
                api_client.models.list()
                BACKEND = "api"
                print("✓ OpenAI Whisper API mode")
            except Exception as e:
                print(f"WARNING: OpenAI API failed ({e}) — falling back to local model")

    if BACKEND == "local":
        print("⚠️  Local CPU mode — may be slow and freeze the Mac")
        try:
            import whisper
        except ImportError:
            print("Error: run: python3 -m pip install openai-whisper")
            sys.exit(1)
        model_dir = args.model_dir
        if model_dir:
            Path(model_dir).mkdir(parents=True, exist_ok=True)
        print(f"Loading Whisper '{args.model}' model…")
        model = whisper.load_model(args.model, download_root=model_dir or None)
        print("Model loaded.")

    print(f"\nReady → http://localhost:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
