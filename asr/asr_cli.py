import os

os.environ["CT2_CUDA_ALLOCATOR"] = "cub_caching"

import sys
import argparse
import json

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine import ASREngine, MMSEngine, SenseVoiceEngine, QwenASREngine, GemmaASREngine

# Sentinel printed after JSON is written — the Node.js parent detects this line
# and kills us via child.kill("SIGKILL"), bypassing the CTranslate2+ROCm hang.
ASR_DONE_SENTINEL = "[ASR_DONE]"


def global_exception_handler(exc_type, exc_value, exc_traceback):
    import traceback
    print("FATAL PYTHON ERROR:", file=sys.stderr)
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)
    sys.exit(1)

sys.excepthook = global_exception_handler


def derive_slice_path(base_path, win_start):
    """Derive a per-window audio slice path from a base path and window start time."""
    root, ext = os.path.splitext(base_path)
    return f"{root}_{int(win_start)}s{ext}"


def main():
    parser = argparse.ArgumentParser(description="CTranslate2/Whisper ASR CLI for asmr-one-dump")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--prompt", default="", help="ASR context prompt (initial_prompt)")
    parser.add_argument("--output", required=True, help="Path to save JSON output")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--model", default="large-v3-turbo", help="Whisper model size")
    parser.add_argument("--windows", help="JSON array of [start, end] pairs, e.g. '[[0,10],[20,30]]'. Omit for full-file transcription.")
    parser.add_argument("--temperature", type=float, help="Whisper temperature")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size")
    parser.add_argument("--mix-audio", help="Optional original audio to mix back (for sterile stems)")
    parser.add_argument("--mix-weight", type=float, default=0.07, help="Weight for mix-audio")
    parser.add_argument("--engine", default="whisper", choices=["whisper", "mms", "qwen", "sensevoice", "gemma"], help="ASR engine: whisper, mms, qwen, sensevoice, or gemma")
    parser.add_argument("--mms-lang", default="jpn", help="MMS target language")
    parser.add_argument("--save-audio-slice", help="Save the audio slice(s) used for processing. In multi-window mode, each window is saved as <base>_<start>s<ext>.")

    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}", flush=True)
        sys.exit(1)

    windows = json.loads(args.windows) if args.windows else None

    if args.engine == "mms":
        engine = MMSEngine(device=args.device)

        # Build intervals from --windows (or empty list for full-file)
        intervals = [(s, e) for s, e in windows] if windows else []

        full_text, sentences, segments = engine.transcribe_file(
            args.audio,
            lang=args.mms_lang,
            intervals=intervals,
            mix_audio_path=args.mix_audio,
            mix_weight=args.mix_weight,
            save_audio_slice_path=args.save_audio_slice,
        )

        if windows:
            # Wrap in array format; MMS produces one combined result across all intervals
            output_data = [{"window": windows, "full_text": full_text, "sentences": sentences, "segments": segments}]
        else:
            output_data = {"full_text": full_text, "sentences": sentences, "segments": segments}

    elif args.engine == "qwen":
        engine = QwenASREngine(device=args.device)
        if windows:
            window_results = []
            for win_start, win_end in windows:
                print(f"  -> Window [{win_start:.2f}s - {win_end:.2f}s]", flush=True)
                slice_path = derive_slice_path(args.save_audio_slice, win_start) if args.save_audio_slice else None
                ft, sents, segs = engine.transcribe_file(
                    args.audio,
                    prompt=args.prompt,
                    clip_start=win_start,
                    clip_end=win_end,
                    mix_audio_path=args.mix_audio,
                    mix_weight=args.mix_weight,
                    save_audio_slice_path=slice_path,
                )
                window_results.append({"window": [win_start, win_end], "full_text": ft, "sentences": sents, "segments": segs})
            output_data = window_results
        else:
            full_text, sentences, segments = engine.transcribe_file(
                args.audio,
                prompt=args.prompt,
                mix_audio_path=args.mix_audio,
                mix_weight=args.mix_weight,
                save_audio_slice_path=args.save_audio_slice,
            )
            output_data = {"full_text": full_text, "sentences": sentences, "segments": segments}

    elif args.engine == "sensevoice":
        engine = SenseVoiceEngine(device=args.device)
        if windows:
            window_results = []
            for win_start, win_end in windows:
                print(f"  -> Window [{win_start:.2f}s - {win_end:.2f}s]", flush=True)
                slice_path = derive_slice_path(args.save_audio_slice, win_start) if args.save_audio_slice else None
                ft, sents, segs = engine.transcribe_file(
                    args.audio,
                    prompt=args.prompt,
                    clip_start=win_start,
                    clip_end=win_end,
                    mix_audio_path=args.mix_audio,
                    mix_weight=args.mix_weight,
                    save_audio_slice_path=slice_path,
                )
                window_results.append({"window": [win_start, win_end], "full_text": ft, "sentences": sents, "segments": segs})
            output_data = window_results
        else:
            full_text, sentences, segments = engine.transcribe_file(
                args.audio,
                prompt=args.prompt,
                mix_audio_path=args.mix_audio,
                mix_weight=args.mix_weight,
                save_audio_slice_path=args.save_audio_slice,
            )
            output_data = {"full_text": full_text, "sentences": sentences, "segments": segments}

    elif args.engine == "gemma":
        engine = GemmaASREngine(device=args.device)
        if windows:
            window_results = []
            for win_start, win_end in windows:
                print(f"  -> Window [{win_start:.2f}s - {win_end:.2f}s]", flush=True)
                slice_path = derive_slice_path(args.save_audio_slice, win_start) if args.save_audio_slice else None
                ft, sents, segs = engine.transcribe_file(
                    args.audio,
                    prompt_info=args.prompt,
                    clip_start=win_start,
                    clip_end=win_end,
                    mix_audio_path=args.mix_audio,
                    mix_weight=args.mix_weight,
                    save_audio_slice_path=slice_path,
                )
                window_results.append({"window": [win_start, win_end], "full_text": ft, "sentences": sents, "segments": segs})
            output_data = window_results
        else:
            full_text, sentences, segments = engine.transcribe_file(
                args.audio,
                prompt_info=args.prompt,
                mix_audio_path=args.mix_audio,
                mix_weight=args.mix_weight,
                save_audio_slice_path=args.save_audio_slice,
            )
            output_data = {"full_text": full_text, "sentences": sentences, "segments": segments}

    else:  # whisper
        engine = ASREngine(model_size=args.model, device=args.device)
        if windows:
            window_results = []
            for win_start, win_end in windows:
                print(f"  -> Window [{win_start:.2f}s - {win_end:.2f}s]", flush=True)
                slice_path = derive_slice_path(args.save_audio_slice, win_start) if args.save_audio_slice else None
                ft, sents, segs = engine.transcribe_file(
                    args.audio,
                    prompt=args.prompt,
                    clip_start=win_start,
                    clip_end=win_end,
                    temperature=args.temperature,
                    beam_size=args.beam_size,
                    condition_on_previous_text=True,
                    mix_audio_path=args.mix_audio,
                    mix_weight=args.mix_weight,
                    save_audio_slice_path=slice_path,
                )
                window_results.append({"window": [win_start, win_end], "full_text": ft, "sentences": sents, "segments": segs})
            output_data = window_results
        else:
            full_text, sentences, segments = engine.transcribe_file(
                args.audio,
                prompt=args.prompt,
                temperature=args.temperature,
                beam_size=args.beam_size,
                condition_on_previous_text=True,
                mix_audio_path=args.mix_audio,
                mix_weight=args.mix_weight,
                save_audio_slice_path=args.save_audio_slice,
            )
            output_data = {"full_text": full_text, "sentences": sentences, "segments": segments}

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
