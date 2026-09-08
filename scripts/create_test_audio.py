#!/usr/bin/env python3
"""
Generate or download sample test audio files in Bangla and English for quick testing.
"""

import sys
from pathlib import Path

def generate_samples():
    try:
        from gtts import gTTS
    except ImportError:
        print("Installing gTTS for test audio generation...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "gTTS"], check=True)
        from gtts import gTTS

    test_audio_dir = Path("data/test/audio")
    test_audio_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        ("data/test/audio/sample_test1.mp3", "আমি বাংলায় কথা বলি।", "bn"),
        ("data/test/audio/sample_test2.mp3", "This is a test speech recognition file.", "en")
    ]

    for path_str, text, lang in samples:
        out_path = Path(path_str)
        if not out_path.exists():
            print(f"Synthesizing sample audio ({lang}): '{text}' -> {out_path}")
            tts = gTTS(text=text, lang=lang)
            tts.save(str(out_path))
        else:
            print(f"Sample audio already exists: {out_path}")

    print("\nSample test audio files are ready in data/test/audio/!")

if __name__ == "__main__":
    generate_samples()
