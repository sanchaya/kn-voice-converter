# ದನಿ ಕನ್ನಡ — Kannada Speech-to-Text

**ದನಿ ಕನ್ನಡ** (Dani Kannada) converts spoken Kannada to Unicode Kannada text. Built by [Sanchaya](https://sanchaya.org) and [Sanchi Foundation](https://sanchifoundation.org) as a free, open-source tool for Kannada language preservation and research.

🌐 **Live demo:** [dani.sanchaya.net](https://dani.sanchaya.net)

---

## Features

| Feature | How | Works on |
|---|---|---|
| Live mic recording | Browser Web Speech API (`kn-IN`) | Android Chrome, desktop Chrome/Safari |
| Audio file upload | Server-side OpenAI Whisper / mlx-whisper | Local server only |
| CLI batch transcription | `transcribe.py` | Any machine with Python |

> **iOS Safari:** Apple's Dictation service does not support Kannada. Use Android Chrome or desktop browsers for live recording.

---

## Quick Start

### Option 1 — GitHub Pages (live recording only, no install)

Open [dani.sanchaya.net](https://dani.sanchaya.net) in Chrome or Safari. No installation needed.

### Option 2 — Local server (live recording + file upload)

**Requirements:** Python 3.9+, Apple Silicon Mac recommended for file upload.

```bash
# Clone
git clone https://github.com/sanchaya/kn-voice-converter.git
cd kn-voice-converter

# Install dependencies
pip install -r requirements.txt

# Run (Apple Silicon — uses Metal GPU)
python app.py --backend mlx

# Run (CPU fallback)
python app.py --backend local

# Run on custom port
python app.py --port 8998
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

### Option 3 — CLI transcription

```bash
# Single file
python transcribe.py audio.mp3

# Batch folder
python transcribe.py ./recordings/ --batch --output ./transcripts/

# Choose model size
python transcribe.py audio.wav --model large-v3
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (dani.sanchaya.net)               │
│                                                                   │
│   Live recording  ──►  Web Speech API (kn-IN)  ──►  Transcript  │
│                         [browser-native, no server]              │
│                                                                   │
│   File upload     ──►  Flask POST /transcribe  ──►  Transcript  │
│                         [requires local server]                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Local Server (app.py)                        │
│                                                                   │
│   GET  /           ──►  Main UI (build_html())                  │
│   GET  /about      ──►  about.html (Sanchaya info)              │
│   POST /transcribe ──►  do_transcribe() → JSON                  │
│   GET  /health     ──►  {"status": "ok"}                        │
└─────────────────────────────────────────────────────────────────┘
```

### Transcription backends

The server supports three backends, set via `BACKEND` in `app.py` or `--backend` flag:

| Backend | Flag | Best for | Notes |
|---|---|---|---|
| `mlx` | `--backend mlx` | Apple Silicon (M1–M4) | Uses Metal GPU + Neural Engine via `mlx-whisper` |
| `api` | `--backend api` | Cloud / accuracy | Requires `OPENAI_API_KEY` env var |
| `local` | `--backend local` | Any machine | CPU-only, slow on large files |

### Model sizes (mlx backend)

| Model | Flag | Speed | Accuracy |
|---|---|---|---|
| turbo | `--model turbo` | Fast | Good **(default)** |
| large-v3 | `--model large-v3` | Slow | Best |
| small | `--model small` | Fastest | Lower |

---

## Technology

- **[Web Speech API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API)** — browser-native speech recognition, `lang="kn-IN"`, no server needed for live recording
- **[OpenAI Whisper](https://github.com/openai/whisper)** — open-source (Apache 2.0) multilingual speech recognition model
- **[mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)** — Apple MLX port, runs Whisper on Metal GPU and Neural Engine
- **[Flask](https://flask.palletsprojects.com/) + [flask-sock](https://flask-sock.readthedocs.io/)** — Python web server with WebSocket support
- **Kannada prompt engineering** — `initial_prompt="ಕನ್ನಡ ಭಾಷೆಯಲ್ಲಿ ಮಾತನಾಡಿದ ಪಠ್ಯ:"` biases Whisper toward Unicode Kannada output
- **GitHub Pages** — static hosting for the live demo; all assets (logo, icons) embedded as base64 data URIs in a single `index.html`

---

## Project Structure

```
kannada-speech-to-text/
├── app.py              # Flask server — live UI + file upload endpoint
├── transcribe.py       # CLI tool — single file and batch transcription
├── index.html          # Static GitHub Pages app (live recording only)
├── about.html          # About page — Sanchaya mission, tech, support
├── assets.json         # Base64-encoded logos and icons
├── requirements.txt    # Python dependencies
├── CNAME               # dani.sanchaya.net (GitHub Pages custom domain)
└── push_to_github.sh   # Helper script to push to GitHub
```

---

## Browser Compatibility

| Browser | Live Recording | Notes |
|---|---|---|
| Android Chrome | ✅ | Best mobile experience |
| Desktop Chrome | ✅ | Full support including waveform |
| Desktop Safari | ✅ | Works on macOS |
| iOS Safari | ❌ | Kannada not in Apple Dictation language list |
| iOS Chrome | ❌ | Uses WebKit — same limitation as iOS Safari |
| Firefox | ❌ | Web Speech API not supported |

---

## Known Issues

- **iOS Safari** — `service-not-allowed` error because Apple's Dictation does not include Kannada. Cannot be fixed in software.
- **Local server over HTTP on mobile** — browsers block microphone on non-HTTPS. Use the GitHub Pages version (`https://dani.sanchaya.net`) instead.
- **Firefox** — Web Speech API is not implemented. File upload via local server works.

---

## Supporting Sanchaya

ಸಂಚಯದ ಕೆಲಸಗಳಿಗೆ ಸಮುದಾಯದ ಕಾಣಿಕೆಗಳೇ ಬಲ.

File upload requires a server running Whisper (~₹3,000–5,000/month). Community funding will make this available to everyone for free.

👉 [sanchaya.org/support-us](https://sanchaya.org/support-us/)

---

## License

Code: [MIT License](LICENSE)  
Content: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)

© 2026 Sanchaya & Sanchi Foundation
