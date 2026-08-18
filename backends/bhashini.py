"""
Bhashini ULCA / Dhruva API backend — cloud ASR, no GPU needed.

Registration: https://bhashini.gov.in/ulca
Set env vars:  BHASHINI_USER_ID  and  BHASHINI_API_KEY

Service IDs for Kannada (kn):
  ai4bharat/conformer-kn-gpu--t4   (IndicConformer, best)
  ai4bharat/whisper-medium-kn--gpu  (Whisper fine-tuned)
"""

import base64, json, os, subprocess, tempfile
import requests

DHRUVA_URL = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"

# Default service — override via BHASHINI_SERVICE_ID env var
DEFAULT_SERVICE = "ai4bharat/conformer-kn-gpu--t4"


def _to_wav16k(audio_path: str) -> str:
    """Convert any audio file to 16 kHz mono WAV using ffmpeg."""
    out = audio_path + "_16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", "-f", "wav", out],
        capture_output=True, check=True,
    )
    return out


def transcribe(audio_path: str) -> tuple[str, float]:
    """Transcribe using Bhashini Dhruva cloud API."""
    user_id  = os.environ.get("BHASHINI_USER_ID", "")
    api_key  = os.environ.get("BHASHINI_API_KEY", "")
    service  = os.environ.get("BHASHINI_SERVICE_ID", DEFAULT_SERVICE)

    if not user_id or not api_key:
        raise RuntimeError(
            "Set BHASHINI_USER_ID and BHASHINI_API_KEY environment variables.\n"
            "Register free at https://bhashini.gov.in/ulca"
        )

    wav_path = _to_wav16k(audio_path)
    try:
        with open(wav_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
    finally:
        try: os.unlink(wav_path)
        except: pass

    payload = {
        "pipelineTasks": [{
            "taskType": "asr",
            "config": {
                "language":    {"sourceLanguage": "kn"},
                "serviceId":   service,
                "audioFormat": "wav",
                "samplingRate": 16000,
            },
        }],
        "inputData": {
            "audio": [{"audioContent": audio_b64}]
        },
    }

    headers = {
        "userID":      user_id,
        "ulcaApiKey":  api_key,
        "Authorization": api_key,
        "Content-Type": "application/json",
    }

    resp = requests.post(DHRUVA_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    text = data["pipelineResponse"][0]["output"][0]["source"].strip()
    return text, 0.0
