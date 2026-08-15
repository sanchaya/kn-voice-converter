# Contributing to ನುಡಿಯಕ್ಷರ

Thank you for your interest in contributing to Kannada language technology.

## Ways to contribute

- **Test on devices** — especially Android phones, report browser/OS version and what happened
- **Improve Kannada accuracy** — tune `initial_prompt`, test dialect variations, report misrecognitions
- **Add language support** — the architecture supports any `lang` code; other Indian languages welcome
- **Server funding** — [support Sanchaya](https://sanchaya.org/support-us/) to enable the file upload feature publicly

## Development setup

```bash
git clone https://github.com/sanchaya/kn-voice-converter.git
cd kn-voice-converter
pip install -r requirements.txt
python app.py --backend mlx   # Apple Silicon
python app.py --backend local  # CPU fallback
```

## Updating the static GitHub Pages version

`index.html` is self-contained — all assets (logos, icons) are embedded as base64 data URIs loaded from `assets.json`. When you change `app.py`'s `build_html()`, mirror the same changes in `index.html`.

The `about.html` page is served both by Flask (`GET /about`) and by GitHub Pages (`/about.html`). Navigation links use `/` as the home URL, which works for both.

## Reporting issues

Please include:
- Browser name and version
- OS / device
- The exact error message (e.g. `service-not-allowed`, `not-allowed`)
- Whether you're on HTTP (local) or HTTPS (dani.sanchaya.net)
