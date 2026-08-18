"""
AI4Bharat IndicConformer backend — best accuracy, needs GPU for real-time.
Model: ai4bharat/indicconformer_kn  (via NVIDIA NeMo)

Install: pip install nemo_toolkit[asr]  (large, ~3 GB env)
         OR: pip install 'git+https://github.com/NVIDIA/NeMo.git#egg=nemo_toolkit[asr]'
"""

import os, tempfile

_model    = None
MODEL_ID  = os.environ.get("INDICCONFORMER_MODEL", "ai4bharat/indicconformer_kn")


def _load():
    global _model
    if _model is not None:
        return
    print(f"[IndicConformer] Loading {MODEL_ID}…")
    import nemo.collections.asr as nemo_asr
    _model = nemo_asr.models.ASRModel.from_pretrained(MODEL_ID)
    _model.eval()
    print("[IndicConformer] Ready.")


def transcribe(audio_path: str) -> tuple[str, float]:
    """Transcribe using local IndicConformer NeMo model."""
    _load()
    results = _model.transcribe([audio_path])
    text    = (results[0] if isinstance(results[0], str) else results[0].text).strip()
    return text, 0.0
