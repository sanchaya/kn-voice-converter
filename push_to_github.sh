#!/bin/bash
# Run this from your terminal inside the project folder:
#   bash push_to_github.sh
#
# Prerequisites:
#   - git installed
#   - GitHub access (SSH key or HTTPS token) for github.com/sanchaya

set -e

REPO_URL="https://github.com/sanchaya/kn-voice-converter.git"
# Or use SSH: git@github.com:sanchaya/kn-voice-converter.git

echo "=== Kannada Voice Converter — GitHub Push ==="
echo ""

# Clean up any stale lock from the sandbox (safe to delete if no git process is running)
if [ -f .git/index.lock ]; then
  echo "Removing stale .git/index.lock..."
  rm -f .git/index.lock
fi

# Init if needed
if [ ! -d .git ]; then
  git init
  git branch -m main
fi

# Config (update if needed)
git config user.email "omshivaprakash@gmail.com"
git config user.name "Om Shivaprakash"

# Stage everything
git add .

# Commit
git commit -m "Initial commit: Kannada speech-to-text tool with Sanchaya branding

- transcribe.py : CLI tool — single file and batch transcription
- app.py        : Flask web server with fonts.sanchaya.net branding
- requirements.txt, setup_and_test.sh

Uses OpenAI Whisper (medium model) with Kannada (kn) language.
Supports MP3, WAV, M4A, FLAC, OGG, AAC, WebM, MP4." 2>/dev/null || \
  echo "(Nothing new to commit — already committed)"

# Add remote (skip if already exists)
git remote get-url origin &>/dev/null || git remote add origin "$REPO_URL"

# Push
echo ""
echo "Pushing to $REPO_URL ..."
git push -u origin main

echo ""
echo "✓ Done! View at: https://github.com/sanchaya/kn-voice-converter"
