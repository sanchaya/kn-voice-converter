"""
AI4Bharat IndicWav2Vec backend — runs locally on CPU (no GPU needed).
Model: ai4bharat/indicwav2vec-kannada  (~1.2 GB)

Install: pip install transformers torch soundfile librosa
"""

import os, subprocess, tempfile
import numpy as np

_processor = None
_model     = None
MODEL_ID   = os.environ.get("INDICWAV2VEC_MODEL", "ai4bharat/indicwav2vec-kannada")


def _load():
    global _processor, _model
    if _model is not None:
        return
    print(f"[IndicWav2Vec] Loading {MODEL_ID}…")
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    _processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    _model     = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    _model.eval()
    print("[IndicWav2Vec] Ready.")


def _to_array(audio_path: str) -> np.ndarray:
    """Load audio as 16 kHz mono float32 numpy array."""
    import soundfile as sf
    import librosa
    speech, sr = sf.read(audio_path)
    if speech.ndim > 1:
        speech = speech.mean(axis=1)       # stereo → mono
    if sr != 16000:
        speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
    return speech.astype(np.float32)


def transcribe(audio_path: str) -> tuple[str, float]:
    """Transcribe using local IndicWav2Vec model."""
    import torch
    _load()

    speech = _to_array(audio_path)
    dur    = len(speech) / 16000

    inputs = _processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = _model(**inputs).logits

    ids  = torch.argmax(logits, dim=-1)
    text = _processor.batch_decode(ids)[0].strip()
    return text, dur
