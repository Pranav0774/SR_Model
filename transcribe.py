"""Command line: python transcribe.py audio.wav [--language auto|en|hi|ta] [--engine r2t2|indic|whisper]"""
import argparse
import json
import sys

from stt import Transcriber


def main():
    ap = argparse.ArgumentParser(description="Convert an audio file to text.")
    ap.add_argument("audio")
    ap.add_argument("--language", default="auto", choices=["auto", "en", "hi", "ta"])
    ap.add_argument("--engine", choices=["r2t2", "indic", "whisper"], help="override the configured route")
    ap.add_argument("--translate", action="store_true", help="also translate Hindi/Tamil to English")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = ap.parse_args()

    try:
        result = Transcriber().transcribe_file(args.audio, args.language, args.engine, args.translate)
    except (ValueError, RuntimeError) as e:
        sys.exit(f"error: {e}")

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["text"])
        if result.get("translation"):
            print("\nEnglish: " + result["translation"])
        print(f"[{result['language']} | {result['engine']} | "
              f"{result['audio_seconds']}s audio in {result['elapsed_seconds']}s]", file=sys.stderr)


if __name__ == "__main__":
    main()
