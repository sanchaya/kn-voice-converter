#!/usr/bin/env python3
"""
Kannada Speech-to-Text — Flask Web Server
Provides a REST API and simple web UI for audio transcription.

Run:
    python app.py
    python app.py --port 8080 --model large-v3

API endpoint:
    POST /transcribe   (multipart: audio file)
    GET  /health
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify, render_template_string

SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".webm", ".mp4"}

app = Flask(__name__)
model = None  # Loaded once at startup


# ── HTML UI with Sanchaya branding ────────────────────────────────────────────
HTML_PAGE = """
<!DOCTYPE html>
<html lang="kn">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ಧ್ವನಿ → ಪಠ್ಯ — ಫಾಂಟ್ಸ್ ಸಂಚಯ</title>

  <!-- Sanchaya favicons -->
  <link rel="icon" type="image/png" href="https://fonts.sanchaya.net/img/favicon-32x32.png">
  <link rel="apple-touch-icon" sizes="180x180" href="https://fonts.sanchaya.net/img/apple-touch-icon.png">
  <link rel="icon" type="image/png" sizes="192x192" href="https://fonts.sanchaya.net/img/android-chrome-192x192.png">
  <meta name="msapplication-TileImage" content="https://fonts.sanchaya.net/img/mstile-310x310.png">

  <!-- Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Kannada:wght@400;500;600&display=swap" rel="stylesheet">

  <style>
    :root {
      --primary:    #4361ee;
      --primary-dk: #3f37c9;
      --navy:       #1a1a2e;
      --bg:         #f8f9fa;
      --card-bg:    #ffffff;
      --text:       #4a4a68;
      --border:     #e0e0e0;
      --success:    #2ecc71;
      --success-dk: #27ae60;
      --shadow:     0 4px 20px rgba(0,0,0,0.08);
      --shadow-hov: 0 8px 30px rgba(67,97,238,0.15);
      --radius:     12px;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Noto Sans Kannada', 'Noto Sans', -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    /* ── Navbar ── */
    nav {
      background: var(--navy);
      padding: 0 24px;
      height: 56px;
      display: flex;
      align-items: center;
      gap: 12px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    nav a { display: flex; align-items: center; gap: 10px; text-decoration: none; }
    nav img.nav-logo { height: 32px; width: 32px; border-radius: 6px; }
    nav .brand-name { color: #fff; font-size: 17px; font-weight: 600; letter-spacing: 0.2px; }
    nav .brand-sub  { color: rgba(255,255,255,0.5); font-size: 13px; margin-left: 4px; }
    nav .nav-spacer { flex: 1; }
    nav .nav-link   { color: rgba(255,255,255,0.7); font-size: 13px; text-decoration: none; }
    nav .nav-link:hover { color: #fff; }

    /* ── Main ── */
    main {
      flex: 1;
      display: flex;
      justify-content: center;
      align-items: flex-start;
      padding: 40px 16px;
    }

    .card {
      background: var(--card-bg);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 36px 32px;
      width: 100%;
      max-width: 680px;
      transition: box-shadow 0.2s;
    }

    .card-header {
      display: flex;
      align-items: center;
      gap: 14px;
      margin-bottom: 8px;
    }
    .card-icon {
      width: 48px; height: 48px;
      background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dk) 100%);
      border-radius: 12px;
      display: flex; align-items: center; justify-content: center;
      font-size: 24px;
      flex-shrink: 0;
    }
    h1 { font-size: 22px; color: var(--navy); font-weight: 600; }
    .subtitle { color: var(--text); font-size: 14px; margin-bottom: 28px; margin-top: 4px; line-height: 1.5; }

    /* ── Upload zone ── */
    .upload-zone {
      border: 2px dashed var(--border);
      border-radius: var(--radius);
      padding: 36px 20px;
      text-align: center;
      cursor: pointer;
      transition: border-color 0.2s, background 0.2s;
      margin-bottom: 16px;
    }
    .upload-zone:hover, .upload-zone.drag-over {
      border-color: var(--primary);
      background: rgba(67,97,238,0.04);
    }
    .upload-icon { font-size: 38px; margin-bottom: 10px; }
    .upload-zone p { color: var(--text); font-size: 15px; margin-bottom: 4px; }
    .upload-zone .formats { color: #aaa; font-size: 12px; }
    input[type=file] { display: none; }

    .file-chosen {
      display: none;
      align-items: center;
      gap: 8px;
      background: rgba(67,97,238,0.06);
      border: 1px solid rgba(67,97,238,0.2);
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 14px;
      color: var(--primary);
      margin-bottom: 16px;
    }
    .file-chosen.show { display: flex; }

    /* ── Buttons ── */
    .btn {
      width: 100%;
      padding: 13px;
      border: none;
      border-radius: 8px;
      font-size: 16px;
      font-family: inherit;
      cursor: pointer;
      transition: background 0.2s, opacity 0.2s;
      font-weight: 500;
    }
    .btn-primary {
      background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dk) 100%);
      color: #fff;
    }
    .btn-primary:hover:not(:disabled) { opacity: 0.9; }
    .btn-primary:disabled { background: #ccc; cursor: not-allowed; }
    .btn-copy {
      background: var(--success);
      color: #fff;
      margin-top: 10px;
    }
    .btn-copy:hover { background: var(--success-dk); }

    /* ── Status ── */
    .status {
      text-align: center;
      font-size: 13px;
      color: #999;
      margin-top: 10px;
      min-height: 20px;
    }
    .status .error { color: #e74c3c; }

    /* ── Result ── */
    .result { display: none; margin-top: 24px; }
    .result-label {
      font-size: 13px;
      font-weight: 600;
      color: var(--primary);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 8px;
    }
    .output-box {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 18px;
      font-size: 19px;
      line-height: 1.9;
      min-height: 90px;
      white-space: pre-wrap;
      color: var(--navy);
      user-select: text;
      -webkit-user-select: text;
    }

    /* ── Spinner ── */
    .spinner {
      display: inline-block;
      width: 16px; height: 16px;
      border: 2px solid rgba(255,255,255,0.4);
      border-top-color: #fff;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      vertical-align: middle;
      margin-right: 8px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Footer ── */
    footer {
      background: var(--navy);
      color: rgba(255,255,255,0.5);
      text-align: center;
      font-size: 13px;
      padding: 16px;
    }
    footer a { color: rgba(255,255,255,0.7); text-decoration: none; }
    footer a:hover { color: #fff; }
    footer img.footer-logo { height: 20px; vertical-align: middle; margin-right: 6px; opacity: 0.8; }

    @media (max-width: 480px) {
      .card { padding: 24px 18px; }
      h1 { font-size: 19px; }
    }
  </style>
</head>
<body>

<!-- Navbar -->
<nav>
  <a href="https://fonts.sanchaya.net/" target="_blank">
    <img class="nav-logo" src="https://fonts.sanchaya.net/img/app-icon-light.png" alt="ಫಾಂಟ್ಸ್ ಸಂಚಯ">
    <span class="brand-name">ಫಾಂಟ್ಸ್ ಸಂಚಯ</span>
  </a>
  <span class="brand-sub">/ ಧ್ವನಿ → ಪಠ್ಯ</span>
  <span class="nav-spacer"></span>
  <a class="nav-link" href="https://fonts.sanchaya.net/" target="_blank">fonts.sanchaya.net</a>
</nav>

<!-- Main content -->
<main>
  <div class="card">
    <div class="card-header">
      <div class="card-icon">🎙️</div>
      <div>
        <h1>ಧ್ವನಿಯಿಂದ ಕನ್ನಡ ಪಠ್ಯ</h1>
      </div>
    </div>
    <p class="subtitle">ಧ್ವನಿ ಕಡತವನ್ನು ಅಪ್ಲೋಡ್ ಮಾಡಿ — Unicode ಕನ್ನಡ ಪಠ್ಯವಾಗಿ ಪಡೆಯಿರಿ</p>

    <label for="fileInput">
      <div class="upload-zone" id="dropZone">
        <div class="upload-icon">🎵</div>
        <p>ಇಲ್ಲಿ ಕ್ಲಿಕ್ ಮಾಡಿ ಅಥವಾ ಕಡತವನ್ನು ಎಳೆದು ಬಿಡಿ</p>
        <p class="formats">MP3 · WAV · M4A · FLAC · OGG · AAC · WebM</p>
      </div>
    </label>
    <input type="file" id="fileInput" accept=".mp3,.wav,.m4a,.flac,.ogg,.aac,.wma,.webm,.mp4">

    <div class="file-chosen" id="fileChosen">
      <span>📄</span>
      <span id="fileName"></span>
    </div>

    <button class="btn btn-primary" id="transcribeBtn" disabled onclick="transcribe()">
      ಪಠ್ಯಕ್ಕೆ ಪರಿವರ್ತಿಸಿ
    </button>
    <div class="status" id="status"></div>

    <div class="result" id="result">
      <div class="result-label">ಕನ್ನಡ ಪಠ್ಯ</div>
      <div class="output-box" id="output"></div>
      <button class="btn btn-copy" id="copyBtn" onclick="copyText()">📋 ಪಠ್ಯ ನಕಲಿಸಿ</button>
    </div>
  </div>
</main>

<!-- Footer -->
<footer>
  <img class="footer-logo" src="https://fonts.sanchaya.net/img/sanchaya-logo.png" alt="Sanchaya">
  © 2026 <a href="https://sanchaya.org" target="_blank">ಸಂಚಯ</a> ಮತ್ತು
  <a href="https://sanchifoundation.org" target="_blank">ಸಂಚಿ ಫೌಂಡೇಶನ್</a>
</footer>

<script>
  const fileInput = document.getElementById('fileInput');
  const dropZone  = document.getElementById('dropZone');

  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) showFile(fileInput.files[0].name);
  });

  function showFile(name) {
    document.getElementById('fileName').textContent = name;
    document.getElementById('fileChosen').classList.add('show');
    document.getElementById('transcribeBtn').disabled = false;
  }

  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) {
      const dt = new DataTransfer();
      dt.items.add(file);
      fileInput.files = dt.files;
      showFile(file.name);
    }
  });

  async function transcribe() {
    const file = fileInput.files[0];
    if (!file) return;

    const btn    = document.getElementById('transcribeBtn');
    const status = document.getElementById('status');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>ಪ್ರಕ್ರಿಯೆ ನಡೆಯುತ್ತಿದೆ...';
    status.textContent = 'ಧ್ವನಿ ಪ್ರಕ್ರಿಯೆ ಆಗುತ್ತಿದೆ, ದಯವಿಟ್ಟು ಕಾಯಿರಿ…';
    document.getElementById('result').style.display = 'none';

    const form = new FormData();
    form.append('audio', file);

    try {
      const resp = await fetch('/transcribe', { method: 'POST', body: form });
      const data = await resp.json();
      if (data.error) {
        status.innerHTML = '<span class="error">ದೋಷ: ' + data.error + '</span>';
      } else {
        document.getElementById('output').textContent = data.text;
        document.getElementById('result').style.display = 'block';
        status.textContent = `✓ ${data.duration_seconds.toFixed(1)}s ಧ್ವನಿ · ${data.characters} ಅಕ್ಷರಗಳು · ${data.processing_seconds}s ಸಮಯ`;
      }
    } catch (err) {
      status.innerHTML = '<span class="error">ಸಂಪರ್ಕ ದೋಷ: ' + err.message + '</span>';
    }

    btn.disabled = false;
    btn.textContent = 'ಪಠ್ಯಕ್ಕೆ ಪರಿವರ್ತಿಸಿ';
  }

  function copyText() {
    const text = document.getElementById('output').textContent;
    navigator.clipboard.writeText(text).then(() => {
      const btn = document.getElementById('copyBtn');
      btn.textContent = '✓ ನಕಲಾಯಿತು!';
      setTimeout(() => btn.textContent = '📋 ಪಠ್ಯ ನಕಲಿಸಿ', 2000);
    });
  }
</script>
</body>
</html>
"""


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file in request. Use field name 'audio'."}), 400

    audio_file = request.files["audio"]
    if not audio_file.filename:
        return jsonify({"error": "Empty filename."}), 400

    ext = Path(audio_file.filename).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        return jsonify({"error": f"Unsupported format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"}), 400

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        import time
        start = time.time()
        result = model.transcribe(
            tmp_path,
            language="kn",
            task="transcribe",
            fp16=False,
        )
        elapsed = time.time() - start
        text = result["text"].strip()

        # Estimate audio duration from Whisper segments
        duration = result["segments"][-1]["end"] if result["segments"] else 0

        return jsonify({
            "text": text,
            "characters": len(text),
            "duration_seconds": duration,
            "processing_seconds": round(elapsed, 2),
            "language": "kn",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    global model

    parser = argparse.ArgumentParser(description="Kannada Speech-to-Text Web Server")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument(
        "--model", default="medium",
        choices=["tiny", "base", "small", "medium", "large", "large-v3"],
        help="Whisper model (default: medium)"
    )
    args = parser.parse_args()

    try:
        import whisper
    except ImportError:
        print("Error: openai-whisper not installed. Run: pip install openai-whisper")
        sys.exit(1)

    print(f"Loading Whisper model '{args.model}'...")
    model = whisper.load_model(args.model)
    print(f"Model ready. Starting server on http://{args.host}:{args.port}\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
