import os

os.environ["CT2_CUDA_ALLOCATOR"] = "cub_caching"

import sys
import argparse
import json

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine import ASREngine

# Sentinel printed after JSON is written — the Node.js parent detects this line
# and kills us via child.kill("SIGKILL"), bypassing the CTranslate2+ROCm hang.
ASR_DONE_SENTINEL = "[ASR_DONE]"


def global_exception_handler(exc_type, exc_value, exc_traceback):
    import traceback
    print("FATAL PYTHON ERROR:", file=sys.stderr)
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)
    sys.exit(1)

sys.excepthook = global_exception_handler


def main():
    parser = argparse.ArgumentParser(description="CTranslate2/Whisper ASR CLI for asmr-one-dump")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--prompt", default="", help="ASR context prompt (initial_prompt)")
    parser.add_argument("--output", required=True, help="Path to save JSON output")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--model", default="large-v3-turbo", help="Whisper model size")
    parser.add_argument("--start", type=float, help="Clip start time in seconds")
    parser.add_argument("--end", type=float, help="Clip end time in seconds")
    parser.add_argument("--temperature", type=float, default=0, help="Whisper temperature")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size")
    parser.add_argument("--no-condition", action="store_true", help="Disable condition_on_previous_text")

    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}", flush=True)
        sys.exit(1)

    engine = ASREngine(model_size=args.model, device=args.device)
    full_text, sentences, segments = engine.transcribe_file(
        args.audio,
        prompt=args.prompt,
        clip_start=args.start,
        clip_end=args.end,
        temperature=args.temperature,
        beam_size=args.beam_size,
        condition_on_previous_text=not args.no_condition
    )

    output_data = {
        "full_text": full_text,
        "sentences": sentences,
        "segments": segments,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Successfully saved ASR result to {args.output}", flush=True)

    # Print sentinel and wait — Node.js parent will kill us via SIGKILL.
    # CTranslate2+ROCm hangs on any Python-initiated exit on Windows, so we
    # let the external process manager do the killing.
    print(ASR_DONE_SENTINEL, flush=True)

    # Block forever; Node kills us.
    import time
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
