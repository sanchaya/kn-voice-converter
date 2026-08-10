#!/bin/bash
# Kannada Speech-to-Text — Quick setup & test script
# Run: bash setup_and_test.sh

set -e

echo "=== Kannada Speech-to-Text Setup ==="
echo ""

# Check Python
python3 --version || { echo "Python 3 required"; exit 1; }

# Check ffmpeg (required by Whisper)
if ! command -v ffmpeg &>/dev/null; then
    echo "Installing ffmpeg..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install ffmpeg
    elif command -v apt-get &>/dev/null; then
        sudo apt-get install -y ffmpeg
    else
        echo "Please install ffmpeg manually: https://ffmpeg.org/download.html"
        exit 1
    fi
fi
echo "✓ ffmpeg found: $(ffmpeg -version 2>&1 | head -1)"

# Install Python deps
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt -q
echo "✓ Dependencies installed"

# Quick smoke test
echo ""
echo "Testing import..."
python3 -c "import whisper; print('✓ whisper', whisper.__version__ if hasattr(whisper,'__version__') else 'ok')"
python3 -c "import flask; print('✓ flask', flask.__version__)"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Usage:"
echo "  CLI:    python3 transcribe.py your_audio.mp3"
echo "  Server: python3 app.py"
echo "  Batch:  python3 transcribe.py ./audio_folder/ --batch"
