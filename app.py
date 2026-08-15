#!/usr/bin/env python3
"""
ನುಡಿಯಕ್ಷರ — Kannada Speech-to-Text Server
  Live mic  → Web Speech API (browser-native, kn-IN, no server needed)
  File upload → server-side Whisper transcription (mlx / api / local)

Run:
    python app.py                        # CPU backend (slow)
    python app.py --backend mlx          # Apple Silicon Metal GPU (recommended)
    python app.py --backend api          # OpenAI cloud API (needs OPENAI_API_KEY)
    python app.py --port 8998 --model large-v3

Endpoints:
    GET  /              Main UI (two-card layout: live + upload)
    GET  /about         About page (Sanchaya mission, tech, community appeal)
    POST /transcribe    Upload audio file → JSON {text, duration_seconds, ...}
    GET  /health        {"status": "ok", "model_loaded": bool}

Browser notes:
    Live recording uses Web Speech API — works on Chrome and Safari (desktop/Android).
    iOS Safari does not support Kannada (kn-IN) in Apple Dictation → service-not-allowed.
    Firefox does not implement Web Speech API.
"""

import argparse, base64, json, os, sys, tempfile, time, threading
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

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
  <title>ನುಡಿಯಕ್ಷರ — ಧ್ವನಿ ಪಠ್ಯ</title>
  <link rel="icon" type="image/png" href="{favicon_32}">
  <link rel="apple-touch-icon" href="{app_icon}">
  <link rel="icon" type="image/png" sizes="192x192" href="{chrome_192}">
  <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Kannada:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --primary:    #4361ee;
      --primary-dk: #3f37c9;
      --text-color1:#1a1a2e;
      --span-color: #4a4a68;
      --bg-color:   #f8f9fa;
      --card-bg:    #ffffff;
      --border:     #e0e0e0;
      --card-shadow:0 4px 20px rgba(0,0,0,.08);
      --red:        #e74c3c;
      --green:      #2ecc71;
      --green-dk:   #27ae60;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Noto Sans Kannada', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg-color); color: var(--text-color1);
      min-height: 100vh; display: flex; flex-direction: column;
    }}
    /* NAV */
    nav {{
      width: 100%; min-height: 80px; background: var(--card-bg);
      box-shadow: 0 2px 10px rgba(0,0,0,.04);
      display: flex; align-items: center; justify-content: center;
      padding: 12px 24px; position: relative;
    }}
    .center-nav {{ display: flex; flex-direction: column; align-items: center; gap: 4px; }}
    .nav-logo    {{ height: 40px; width: auto; display: block; }}
    .nav-tagline {{ font-size: 13px; color: var(--span-color); letter-spacing: .04em; }}
    .nav-right   {{
      position: absolute; right: 24px; top: 50%; transform: translateY(-50%);
      display: flex; gap: 16px; align-items: center;
    }}
    .nav-link {{ color: var(--span-color); text-decoration: none; font-size: 13px;
                 padding: 6px 10px; border-radius: 6px; transition: background .2s, color .2s; }}
    .nav-link:hover {{ background: rgba(67,97,238,.08); color: var(--primary); }}
    /* MAIN */
    main {{ flex: 1; display: flex; flex-direction: column; align-items: center; padding: 40px 16px; gap: 28px; }}
    .card {{
      background: var(--card-bg); border-radius: 16px;
      box-shadow: var(--card-shadow); padding: 32px;
      width: 100%; max-width: 760px;
    }}
    .card-label {{
      font-size: 11px; font-weight: 700; letter-spacing: .1em;
      text-transform: uppercase; color: var(--primary); margin-bottom: 20px;
    }}
    /* MIC */
    .mic-area {{ display: flex; align-items: center; gap: 20px; margin-bottom: 28px; }}
    .mic-btn {{
      width: 72px; height: 72px; border-radius: 50%; border: none; cursor: pointer;
      background: linear-gradient(135deg, var(--primary), var(--primary-dk));
      font-size: 28px; color: #fff; flex-shrink: 0;
      box-shadow: 0 4px 14px rgba(67,97,238,.35); transition: transform .15s;
    }}
    .mic-btn:hover {{ transform: scale(1.06); }}
    .mic-btn.recording {{
      background: linear-gradient(135deg, var(--red), #c0392b);
      animation: pulse 1.4s ease-in-out infinite;
    }}
    @keyframes pulse {{
      0%,100% {{ box-shadow: 0 4px 14px rgba(231,76,60,.4); }}
      50%      {{ box-shadow: 0 4px 28px rgba(231,76,60,.75); }}
    }}
    .mic-info h3 {{ font-size: 17px; font-weight: 600; margin-bottom: 4px; }}
    .mic-info p  {{ font-size: 13px; color: var(--span-color); }}
    /* WAVEFORM */
    canvas#waveform {{
      width: 100%; height: 56px; border-radius: 8px;
      background: #1a1a2e; margin-bottom: 20px; display: none;
    }}
    canvas#waveform.active {{ display: block; }}
    /* TRANSCRIPT */
    .transcript-label {{ font-size: 13px; font-weight: 600; color: var(--primary); margin-bottom: 8px; }}
    .transcript-box {{
      background: var(--bg-color); border: 1px solid var(--border); border-radius: 8px;
      padding: 12px 16px; font-size: 18px; line-height: 1.9; min-height: 160px;
      color: var(--text-color1); user-select: text; -webkit-user-select: text;
      max-height: 400px; overflow-y: auto; display: flex; flex-direction: column; gap: 2px;
    }}
    .transcript-row {{ display: flex; align-items: baseline; gap: 10px; padding: 3px 0; transition: color 1.5s; }}
    .transcript-row.chunk-new {{ color: var(--primary); }}
    .ts {{ font-size: 11px; color: #aaa; white-space: nowrap; flex-shrink: 0; font-family: ui-monospace, monospace; }}
    .chunk-text {{ flex: 1; }}
    .interim-row {{ opacity: .55; font-style: italic; }}
    .session-sep {{ font-size: 11px; color: #bbb; text-align: center; padding: 8px 0 4px; }}
    /* BUTTONS */
    .btn-row {{ display: flex; gap: 10px; margin-top: 14px; }}
    .btn {{
      flex: 1; padding: 11px 0; border-radius: 10px; border: none; cursor: pointer;
      font-size: 14px; font-family: inherit; font-weight: 500; transition: opacity .15s;
    }}
    .btn:hover:not(:disabled) {{ opacity: .88; }}
    .btn:disabled {{ background: #ccc !important; cursor: not-allowed; }}
    .btn-primary {{ background: linear-gradient(135deg, var(--primary), var(--primary-dk)); color: #fff; }}
    .btn-green   {{ background: var(--green); color: #fff; }}
    .btn-green:hover {{ background: var(--green-dk) !important; opacity: 1 !important; }}
    .btn-clear   {{ background: #f0f0f5; color: var(--span-color); }}
    .status-bar  {{ margin-top: 10px; font-size: 12px; color: var(--span-color); min-height: 18px; text-align: center; }}
    .err {{ color: var(--red); }}
    /* UPLOAD CARD */
    .upload-zone {{
      border: 2px dashed var(--border); border-radius: 10px; padding: 36px 20px;
      text-align: center; cursor: pointer; margin-bottom: 16px;
      transition: border-color .2s, background .2s;
    }}
    .upload-zone:hover, .upload-zone.drag-over {{
      border-color: var(--primary); background: rgba(67,97,238,.04);
    }}
    .upload-icon {{ font-size: 40px; margin-bottom: 10px; }}
    .upload-zone p {{ font-size: 15px; margin-bottom: 4px; }}
    .formats {{ color: #aaa; font-size: 12px; }}
    input[type=file] {{ display: none; }}
    .file-tag {{
      display: none; align-items: center; gap: 8px;
      background: rgba(67,97,238,.06); border: 1px solid rgba(67,97,238,.2);
      border-radius: 8px; padding: 10px 14px; font-size: 14px;
      color: var(--primary); margin-bottom: 16px;
    }}
    .file-tag.show {{ display: flex; }}
    .result-upload {{ display: none; margin-top: 20px; }}
    /* FOOTER */
    footer {{
      background: var(--card-bg); border-top: 1px solid var(--border);
      color: var(--span-color); text-align: center; padding: 24px 16px;
      display: flex; flex-direction: column; align-items: center; gap: 12px;
      margin-top: auto;
    }}
    .footer-logo  {{ height: 36px; width: auto; opacity: .85; }}
    .footer-links {{ display: flex; gap: 24px; flex-wrap: wrap; justify-content: center; }}
    .footer-links a {{ color: var(--span-color); text-decoration: none; font-size: 13px; }}
    .footer-links a:hover {{ color: var(--primary); }}
    .footer-copy  {{ font-size: 12px; color: #bbb; }}
    @media (max-width: 520px) {{ .nav-right {{ display: none; }} }}
  </style>
</head>
<body>

<nav>
  <div class="center-nav">
    <a href="/"><img class="nav-logo" src="{app_icon}" alt="ನುಡಿಯಕ್ಷರ"></a>
    <span class="nav-tagline">ಧ್ವನಿ → ಕನ್ನಡ ಪಠ್ಯ</span>
  </div>
  <div class="nav-right">
    <a class="nav-link" href="/about">ನಮ್ಮ ಬಗ್ಗೆ</a>
    <a class="nav-link" href="https://sanchaya.org/support-us/" target="_blank">ಬೆಂಬಲಿಸಿ</a>
    <a class="nav-link" href="https://sanchaya.org" target="_blank">ಸಂಚಯ</a>
    <a class="nav-link" href="https://sanchifoundation.org" target="_blank">ಸಂಚಿ ಫೌಂಡೇಶನ್</a>
  </div>
</nav>

<main>

  <!-- LIVE RECORDING -->
  <div class="card">
    <div class="card-label">🎙️ ನೇರ ರೆಕಾರ್ಡ್ — ಬ್ರೌಸರ್ ನೇರ, ಯಾವುದೇ ಸರ್ವರ್ ಬೇಡ</div>
    <div class="mic-area">
      <button class="mic-btn" id="micBtn" onclick="toggleRecording()">🎙️</button>
      <div class="mic-info">
        <h3 id="micStatus">ರೆಕಾರ್ಡ್ ಮಾಡಲು ಕ್ಲಿಕ್ ಮಾಡಿ</h3>
        <p  id="micHint">Chrome ಅಥವಾ Safari ಬಳಸಿ</p>
      </div>
    </div>
    <canvas id="waveform"></canvas>
    <div class="transcript-label">ಕನ್ನಡ ಪಠ್ಯ (ನೇರ)</div>
    <div class="transcript-box" id="liveTranscript"></div>
    <div class="btn-row">
      <button class="btn btn-green" onclick="copyLive()">📋 ನಕಲಿಸಿ</button>
      <button class="btn btn-clear" onclick="clearLive()">🗑️ ಅಳಿಸಿ</button>
    </div>
    <div class="status-bar" id="liveStatus"></div>
  </div>

  <!-- FILE UPLOAD -->
  <div class="card">
    <div class="card-label">📂 ಕಡತ ಅಪ್‌ಲೋಡ್ — AI Whisper ಮಾದರಿ (ಸರ್ವರ್)</div>
    <label for="fileInput">
      <div class="upload-zone" id="dropZone">
        <div class="upload-icon">🎵</div>
        <p>ಇಲ್ಲಿ ಕ್ಲಿಕ್ ಮಾಡಿ ಅಥವಾ ಕಡತವನ್ನು ಎಳೆದು ಬಿಡಿ</p>
        <p class="formats">MP3 · WAV · M4A · FLAC · OGG · AAC · WebM</p>
      </div>
    </label>
    <input type="file" id="fileInput" accept=".mp3,.wav,.m4a,.flac,.ogg,.aac,.wma,.webm,.mp4">
    <div class="file-tag" id="fileTag">
      <span>📄</span><span id="fileName"></span>
    </div>
    <div class="btn-row">
      <button class="btn btn-primary" id="transcribeBtn" disabled onclick="transcribeFile()">
        ಪಠ್ಯಕ್ಕೆ ಪರಿವರ್ತಿಸಿ
      </button>
    </div>
    <div class="status-bar" id="uploadStatus"></div>
    <div class="result-upload" id="resultUpload">
      <div class="transcript-label" style="margin-top:20px">ಕನ್ನಡ ಪಠ್ಯ</div>
      <div class="transcript-box" id="uploadOutput"></div>
      <div class="btn-row" style="margin-top:10px">
        <button class="btn btn-green" onclick="copyUpload()">📋 ಪಠ್ಯ ನಕಲಿಸಿ</button>
      </div>
    </div>
  </div>

</main>

<footer>
  <img class="footer-logo" src="{sanchaya_logo}" alt="Sanchaya">
  <div class="footer-links">
    <a href="/about">ನಮ್ಮ ಬಗ್ಗೆ</a>
    <a href="https://sanchaya.org/support-us/" target="_blank">ಬೆಂಬಲಿಸಿ</a>
    <a href="https://sanchaya.org" target="_blank">ಸಂಚಯ</a>
    <a href="https://sanchifoundation.org" target="_blank">ಸಂಚಿ ಫೌಂಡೇಶನ್</a>
  </div>
  <div class="footer-copy">© 2026 Sanchaya &amp; Sanchi Foundation · Creative Commons Attribution 4.0</div>
</footer>

<script>
function fmtClock(){{return new Date().toLocaleTimeString('kn-IN',{{hour:'2-digit',minute:'2-digit',second:'2-digit'}});}}
function fmtTime(sec){{sec=Math.floor(sec);return Math.floor(sec/60)+':'+ String(sec%60).padStart(2,'0');}}
let analyser=null,waveAF=null;
const wfCanvas=document.getElementById('waveform');
const wfCtx=wfCanvas.getContext('2d');
function startWaveform(stream){{
  try{{const actx=new AudioContext();analyser=actx.createAnalyser();analyser.fftSize=256;
    actx.createMediaStreamSource(stream).connect(analyser);wfCanvas.classList.add('active');drawWave();}}catch(e){{}}
}}
function stopWaveform(){{
  if(waveAF)cancelAnimationFrame(waveAF);
  wfCtx.clearRect(0,0,wfCanvas.width,wfCanvas.height);wfCanvas.classList.remove('active');analyser=null;
}}
function drawWave(){{
  if(!analyser)return;waveAF=requestAnimationFrame(drawWave);
  const buf=new Uint8Array(analyser.frequencyBinCount);analyser.getByteTimeDomainData(buf);
  const W=wfCanvas.clientWidth,H=wfCanvas.clientHeight;wfCanvas.width=W;wfCanvas.height=H;
  wfCtx.fillStyle='#1a1a2e';wfCtx.fillRect(0,0,W,H);
  wfCtx.lineWidth=2;wfCtx.strokeStyle='#4361ee';wfCtx.beginPath();
  const slice=W/buf.length;let x=0;
  for(let i=0;i<buf.length;i++){{const y=(buf[i]/128)*H/2;i===0?wfCtx.moveTo(x,y):wfCtx.lineTo(x,y);x+=slice;}}
  wfCtx.stroke();
}}
const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
let recognition=null,micStream=null,isRecording=false,interimRow=null;
async function toggleRecording(){{isRecording?stopRec():startRec();}}
async function startRec(){{
  if(!SR){{setLiveStatus('Chrome ಅಥವಾ Safari ಬಳಸಿ — Firefox ಬೆಂಬಲಿಸುವುದಿಲ್ಲ',true);return;}}
  if(/iPad|iPhone|iPod/.test(navigator.userAgent)){{setLiveStatus('iOS Safari ಕನ್ನಡ ಧ್ವನಿ ಗುರುತಿಸುವಿಕೆಯನ್ನು ಬೆಂಬಲಿಸುವುದಿಲ್ಲ — Android Chrome ಬಳಸಿ.',true);return;}}
  isRecording=true;appendSessionMarker();
  document.getElementById('micBtn').classList.add('recording');
  document.getElementById('micBtn').textContent='⏹️';
  document.getElementById('micStatus').textContent='ರೆಕಾರ್ಡ್ ನಡೆಯುತ್ತಿದೆ…';
  document.getElementById('micHint').textContent='ನಿಲ್ಲಿಸಲು ಮತ್ತೆ ಕ್ಲಿಕ್ ಮಾಡಿ';
  setLiveStatus('');
  startRecognition();
  try{{micStream=await navigator.mediaDevices.getUserMedia({{audio:true}});startWaveform(micStream);}}
  catch(e){{micStream=null;}}
}}
function startRecognition(){{
  if(!isRecording)return;
  recognition=new SR();
  recognition.lang='kn-IN';recognition.continuous=true;
  recognition.interimResults=true;recognition.maxAlternatives=1;
  recognition.onresult=(event)=>{{
    let interim='';
    for(let i=event.resultIndex;i<event.results.length;i++){{
      const r=event.results[i];
      if(r.isFinal){{const t=r[0].transcript.trim();if(t){{if(interimRow){{interimRow.remove();interimRow=null;}}appendLive(t);}}}}
      else interim+=r[0].transcript;
    }}
    if(interim){{
      const box=document.getElementById('liveTranscript');
      if(!interimRow){{interimRow=document.createElement('div');interimRow.className='transcript-row interim-row';box.appendChild(interimRow);}}
      interimRow.innerHTML='<span class="ts">'+fmtClock()+'</span><span class="chunk-text" style="color:var(--span-color);font-style:italic">'+interim+'</span>';
      box.scrollTop=box.scrollHeight;
    }}
  }};
  recognition.onerror=(e)=>{{if(e.error==='no-speech'||e.error==='aborted')return;if(e.error==='service-not-allowed'){{setLiveStatus('iOS Safari ಕನ್ನಡ ಧ್ವನಿ ಗುರುತಿಸುವಿಕೆಯನ್ನು ಬೆಂಬಲಿಸುವುದಿಲ್ಲ — Android Chrome ಬಳಸಿ.',true);stopRec();return;}}setLiveStatus('ದೋಷ: '+e.error,true);}};
  recognition.onend=()=>{{if(isRecording)setTimeout(()=>{{if(isRecording)startRecognition();}},300);}};
  try{{recognition.start();}}catch(e){{}}
}}
function stopRec(){{
  isRecording=false;
  if(recognition){{try{{recognition.abort();}}catch(e){{}}recognition=null;}}
  if(micStream)micStream.getTracks().forEach(t=>t.stop());
  if(interimRow){{interimRow.remove();interimRow=null;}}
  stopWaveform();
  document.getElementById('micBtn').classList.remove('recording');
  document.getElementById('micBtn').textContent='🎙️';
  document.getElementById('micStatus').textContent='ರೆಕಾರ್ಡ್ ಮಾಡಲು ಕ್ಲಿಕ್ ಮಾಡಿ';
  document.getElementById('micHint').textContent='ರೆಕಾರ್ಡಿಂಗ್ ನಿಲ್ಲಿಸಲಾಗಿದೆ';
}}
function appendSessionMarker(){{
  const box=document.getElementById('liveTranscript');if(!box.innerHTML.trim())return;
  const sep=document.createElement('div');sep.className='session-sep';
  sep.textContent='── ಹೊಸ ರೆಕಾರ್ಡಿಂಗ್ '+fmtClock()+' ──';box.appendChild(sep);
}}
function appendLive(text){{
  const box=document.getElementById('liveTranscript');
  let row;
  if(interimRow){{row=interimRow;interimRow=null;row.innerHTML='';}}
  else{{row=document.createElement('div');box.appendChild(row);}}
  row.className='transcript-row chunk-new';
  const ts=document.createElement('span');ts.className='ts';ts.textContent=fmtClock();
  const txt=document.createElement('span');txt.className='chunk-text';txt.textContent=text;
  row.appendChild(ts);row.appendChild(txt);
  setTimeout(()=>row.classList.remove('chunk-new'),1800);box.scrollTop=box.scrollHeight;
}}
function setLiveStatus(msg,isErr){{
  const el=document.getElementById('liveStatus');
  el.innerHTML=msg?(isErr?'<span class="err">'+msg+'</span>':msg):'';
}}
function copyLive(){{
  const rows=document.querySelectorAll('#liveTranscript .chunk-text');
  const text=Array.from(rows).map(r=>r.textContent).join(' ').trim();
  navigator.clipboard.writeText(text).then(()=>{{setLiveStatus('✓ ನಕಲಿಸಲಾಗಿದೆ');setTimeout(()=>setLiveStatus(''),2000);}});
}}
function clearLive(){{document.getElementById('liveTranscript').innerHTML='';setLiveStatus('');}}
const fileInput=document.getElementById('fileInput');
const dropZone=document.getElementById('dropZone');
fileInput.addEventListener('change',()=>{{if(fileInput.files[0])showFile(fileInput.files[0].name);}});
function showFile(name){{
  document.getElementById('fileName').textContent=name;
  document.getElementById('fileTag').classList.add('show');
  document.getElementById('transcribeBtn').disabled=false;
}}
dropZone.addEventListener('dragover',e=>{{e.preventDefault();dropZone.classList.add('drag-over');}});
dropZone.addEventListener('dragleave',()=>dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop',e=>{{
  e.preventDefault();dropZone.classList.remove('drag-over');
  const file=e.dataTransfer.files[0];
  if(file){{const dt=new DataTransfer();dt.items.add(file);fileInput.files=dt.files;showFile(file.name);}}
}});
async function transcribeFile(){{
  const file=fileInput.files[0];if(!file)return;
  const btn=document.getElementById('transcribeBtn');
  const status=document.getElementById('uploadStatus');
  btn.disabled=true;btn.textContent='ಪ್ರಕ್ರಿಯೆ…';
  status.textContent='ಧ್ವನಿ ಪ್ರಕ್ರಿಯೆ ಆಗುತ್ತಿದೆ, ದಯವಿಟ್ಟು ಕಾಯಿರಿ…';
  document.getElementById('resultUpload').style.display='none';
  const form=new FormData();form.append('audio',file);
  try{{
    const resp=await fetch('/transcribe',{{method:'POST',body:form}});
    const data=await resp.json();
    if(data.error){{status.innerHTML='<span class="err">ದೋಷ: '+data.error+'</span>';}}
    else{{
      document.getElementById('uploadOutput').textContent=data.text;
      document.getElementById('resultUpload').style.display='block';
      status.textContent='✓ '+fmtTime(data.duration_seconds)+' · '+data.characters+' ಅಕ್ಷರ · '+data.processing_seconds+'s';
    }}
  }}catch(e){{status.innerHTML='<span class="err">ಸಂಪರ್ಕ ದೋಷ: '+e.message+'</span>';}}
  btn.disabled=false;btn.textContent='ಪಠ್ಯಕ್ಕೆ ಪರಿವರ್ತಿಸಿ';
}}
function copyUpload(){{navigator.clipboard.writeText(document.getElementById('uploadOutput').textContent);}}
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


@app.route("/about")
def about():
    here = Path(__file__).parent
    about_file = here / "about.html"
    if about_file.exists():
        return send_from_directory(str(here), "about.html")
    return "<h1>About page not found — run from the project directory.</h1>", 404


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
        description="ನುಡಿಯಕ್ಷರ — Kannada Speech-to-Text Server"
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
