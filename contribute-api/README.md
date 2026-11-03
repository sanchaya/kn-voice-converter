# Contribute API

Backend for `/contribute/` (Kannada voice collection). Runs at `dani-api.sanchaya.net` on Coolify.
All data stays on Sanchaya's server: SQLite plus WAV files in `DATA_DIR`. No third-party services.

## Deploy on Coolify
1. DNS: A record `dani-api` pointing at the server.
2. New resource, public repository `sanchaya/kn-voice-converter`, branch `main`, build pack **Docker Compose**, base directory `/contribute-api`, compose file `docker-compose.yaml`.
3. In the resource, set the domain for the `api` service to `https://dani-api.sanchaya.net:8000` (the port after the domain tells Coolify where to route).
4. Deploy. The volume, port and environment come from the compose file; keep one replica (SQLite is single-writer).
5. Check `https://dani-api.sanchaya.net/health`. Back up the `kn-contribute-data` volume off-server.

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
