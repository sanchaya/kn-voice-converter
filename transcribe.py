#!/usr/bin/env python3
"""
Kannada Speech-to-Text Transcriber
Converts audio files to Unicode Kannada text using OpenAI Whisper.

Usage:
    python transcribe.py audio.mp3
    python transcribe.py audio.wav --output result.txt
    python transcribe.py ./audio_folder/ --batch --output ./transcripts/
    python transcribe.py audio.mp3 --model large --verbose
"""

import argparse
import os
import sys
import time
from pathlib import Path

SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".webm", ".mp4"}

def load_model(model_size: str):
    try:
        import whisper
    except ImportError:
        print("Error: openai-whisper not installed. Run: pip install openai-whisper")
        sys.exit(1)

    print(f"Loading Whisper model '{model_size}'...")
    model = whisper.load_model(model_size)
    print("Model loaded.\n")
    return model


def transcribe_file(model, audio_path: Path, verbose: bool = False) -> str:
    """Transcribe a single audio file to Kannada Unicode text."""
    print(f"Transcribing: {audio_path.name}")
    start = time.time()

    result = model.transcribe(
        str(audio_path),
        language="kn",           # Force Kannada
        task="transcribe",       # Transcribe (not translate)
        verbose=verbose,
        fp16=False,              # CPU-safe; remove if you have a GPU
    )

    elapsed = time.time() - start
    text = result["text"].strip()
    print(f"  Done in {elapsed:.1f}s — {len(text)} characters\n")
    return text


def process_batch(model, input_dir: Path, output_dir: Path, verbose: bool):
    """Process all audio files in a directory."""
    audio_files = [f for f in sorted(input_dir.iterdir()) if f.suffix.lower() in SUPPORTED_FORMATS]

    if not audio_files:
        print(f"No supported audio files found in {input_dir}")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(audio_files)} file(s) to transcribe.\n")

    for audio_path in audio_files:
        text = transcribe_file(model, audio_path, verbose)
        out_path = output_dir / (audio_path.stem + ".txt")
        out_path.write_text(text, encoding="utf-8")
        print(f"  Saved → {out_path}\n")

    print(f"Batch complete. Transcripts saved to: {output_dir}")


def process_single(model, audio_path: Path, output_path: Path | None, verbose: bool):
    """Process a single audio file."""
    text = transcribe_file(model, audio_path, verbose)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"Saved → {output_path}")
    else:
        print("=" * 60)
        print("KANNADA TRANSCRIPTION:")
        print("=" * 60)
        print(text)
        print("=" * 60)

    return text


def main():
    parser = argparse.ArgumentParser(
        description="Convert audio to Unicode Kannada text using Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe.py speech.mp3
  python transcribe.py speech.wav --output out.txt
  python transcribe.py ./audio_folder/ --batch --output ./transcripts/
  python transcribe.py speech.mp3 --model large-v3

Models (larger = more accurate, slower):
  tiny, base, small, medium, large, large-v3
  Recommended: 'medium' for Kannada (default)
        """
    )
    parser.add_argument("input", help="Audio file path or directory (for --batch)")
    parser.add_argument("--output", "-o", help="Output file path or directory (for --batch)")
    parser.add_argument(
        "--model", "-m",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large", "large-v3"],
        help="Whisper model size (default: medium)"
    )
    parser.add_argument("--batch", "-b", action="store_true", help="Process all audio files in a directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show word-level timing")

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: '{input_path}' does not exist.")
        sys.exit(1)

    model = load_model(args.model)

    if args.batch or input_path.is_dir():
        output_dir = Path(args.output) if args.output else input_path / "transcripts"
        process_batch(model, input_path, output_dir, args.verbose)
    else:
        if input_path.suffix.lower() not in SUPPORTED_FORMATS:
            print(f"Error: Unsupported format '{input_path.suffix}'. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}")
            sys.exit(1)
        output_path = Path(args.output) if args.output else None
        process_single(model, input_path, output_path, args.verbose)


if __name__ == "__main__":
    main()
