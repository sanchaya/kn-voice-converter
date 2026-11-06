# Shruti Kannada local server

Runs Udgam Labs' [Shruti Kannada](https://huggingface.co/udgamlabs/shruti-kannada) (Apache-2.0, 29.7 MB, offline)
behind an OpenAI-compatible `/v1/audio/transcriptions` endpoint, so OpenWhispr's **Custom** transcription provider can use it.
It listens on `127.0.0.1` only; audio never leaves your machine.

## Run
    brew install ffmpeg                      # or your package manager
    pip install -r requirements.txt
    hf download udgamlabs/shruti-kannada shruti-kannada.cact --local-dir .
    python server.py                         # http://127.0.0.1:8765

`SHRUTI_MODEL` sets the model path and `PORT` the port. Telemetry from the cactus package is switched off by default.

## Use in OpenWhispr
In Settings, choose the **Custom** transcription provider and set the base URL to `http://127.0.0.1:8765/v1`.
(OpenWhispr accepts plain `http` for local addresses. If it insists on an API key, enter any placeholder: the server ignores it.
Exact menu labels not checked against the running app.)

## Test
    SHRUTI_MODEL=shruti-kannada.cact python -m pytest -q test_server.py
This checks the protocol and format conversion only. Kannada accuracy needs a real recording.

## Known limits
- Accuracy per Udgam's own report: about 33% word error rate on read Kannada (FLEURS) and 57% on conversational Kannada-English.
- One request at a time; the whole recording is transcribed at once.
