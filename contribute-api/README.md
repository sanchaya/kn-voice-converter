# Contribute API

Backend for `/contribute/` (Kannada voice collection). Runs at `dani-api.sanchaya.net` on Coolify.
All data stays on Sanchaya's server: SQLite plus WAV files in `DATA_DIR`. No third-party services.

## Deploy on Coolify
1. New resource, Dockerfile build, base directory `contribute-api`, port 8000, domain `dani-api.sanchaya.net`.
2. Add a persistent volume mounted at `/data` (audio and the database live there). Back it up off-server.
3. Env: `ALLOWED_ORIGIN=https://dani.sanchaya.net` (default).

## Run locally
    pip install -r requirements.txt
    DATA_DIR=./data gunicorn -w1 -b 127.0.0.1:8000 app:app     # the page uses http://localhost:8000 on localhost
    python -m pytest -q                                         # API tests
    node ../contribute/wav.test.js                              # recorder encoder test

## Before a public launch
- Replace the placeholder prompts in `prompts.tsv` with reviewed, freely licensed text.
- Have a lawyer and a Kannada speaker review the consent text in `contribute/index.html` and bump `CONSENT_VERSION`.
- Add rate limiting at the Coolify proxy; move to Postgres if write load grows.
- Not built yet: admin dashboard, exports, age/gender metadata, spontaneous and code-mixed prompt screens, offline queueing.
