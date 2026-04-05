import os
import sys
import argparse
import json
import torch
import numpy as np
import soundfile as sf

# Use lower-level Demucs 4.0.1 API as demucs.api is not available in the PyPI release
from demucs.pretrained import get_model_from_args
from demucs.apply import apply_model
from demucs.separate import load_track
from demucs.audio import save_audio

def calculate_fine_energy(audio_data, sr, window_size=1.0):
    """Calculate RMS energy and SNR in fixed-size windows."""
    samples_per_window = int(window_size * sr)
    num_windows = int(np.ceil(audio_data.shape[1] / samples_per_window))
    
    results = []
    
    # Ensure audio is 2D (stems, samples)
    # We expect [vocals, other]
    vocal = audio_data[0]
    other = audio_data[1]
    
    for i in range(num_windows):
        start = i * samples_per_window
        end = min((i + 1) * samples_per_window, audio_data.shape[1])
        
        v_slice = vocal[start:end]
        o_slice = other[start:end]
        
        v_rms = float(np.sqrt(np.mean(v_slice**2)))
        o_rms = float(np.sqrt(np.mean(o_slice**2)))
        
        # Calculate SNR in dB
        # Avoid log(0)
        eps = 1e-10
        snr = float(20 * np.log10((v_rms + eps) / (o_rms + eps)))
        
        results.append({
            "start": round(i * window_size, 3),
            "end": round(end / sr, 3),
            "vocal_rms": round(v_rms, 6),
            "other_rms": round(o_rms, 6),
            "snr_db": round(snr, 3)
        })
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Demucs Separation & Energy Analysis CLI")
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    parser.add_argument("--output", required=True, help="Path to save JSON results")
    parser.add_argument("--save-audio", action="store_true", help="Save vocal/other stems to disk")
    parser.add_argument("--audio-output-dir", help="Directory to save audio stems (if --save-audio is on)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--model", default="htdemucs", help="Demucs model name")
    
    # Add dummy arguments to satisfy get_model_from_args parser expectations if needed
    # but we can also just pass a Namespace
    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
        print(f"Error: File not found: {args.audio}", file=sys.stderr)
        sys.exit(1)
        
    # ── Check for existing stems ─────────────────────────────────────────────
    # If -save-audio is on, check the provided dir. Otherwise check JSON dir.
    out_dir = args.audio_output_dir if args.audio_output_dir else os.path.dirname(os.path.abspath(args.output))
    stem = os.path.splitext(os.path.basename(args.audio))[0]
    vocal_path = os.path.join(out_dir, f"{stem}.vocals.wav")
    other_path = os.path.join(out_dir, f"{stem}.other.wav")

    if os.path.exists(vocal_path) and os.path.exists(other_path):
        print(f"Found existing stems, skipping separation: {vocal_path}", flush=True)
        # Load existing stems
        v_data, v_sr = sf.read(vocal_path)
        o_data, o_sr = sf.read(other_path)
        
        # Ensure they are 1D (mono) for energy analysis
        if len(v_data.shape) > 1: v_data = v_data.mean(axis=1)
        if len(o_data.shape) > 1: o_data = o_data.mean(axis=1)
        
        # We need to make sure they have the same length (they should)
        min_len = min(len(v_data), len(o_data))
        v_data = v_data[:min_len]
        o_data = o_data[:min_len]
        
        analysis_audio = np.stack([v_data, o_data])
        samplerate = v_sr
        print(f"Energy analysis using existing stems at {samplerate}Hz...", flush=True)
    else:
        # ── Standard Separation Path ──────────────────────────────────────────
        print(f"Loading Demucs model '{args.model}' on {args.device}...", flush=True)
        
        # Build a minimal Namespace that get_model_from_args expects
        model_args = argparse.Namespace(
            name=args.model,
            repo=None,
            models=None,
            store=None,
        )
        
        try:
            model = get_model_from_args(model_args)
        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            sys.exit(1)
            
        model.to(args.device)
        model.eval()

        print(f"Separating stems for {os.path.basename(args.audio)}...", flush=True)
        # Load audio using Demucs' internal loader (handles ffmpeg/torchaudio)
        wav = load_track(args.audio, model.audio_channels, model.samplerate)
        
        # Normalization (Matches demucs.separate main logic)
        ref = wav.mean(0)
        wav -= ref.mean()
        wav /= ref.std()
        
        # Apply model
        with torch.no_grad():
            sources = apply_model(model, wav[None], device=args.device, shifts=1, split=True, overlap=0.25, progress=True)[0]
        
        # Denormalize
        sources *= ref.std()
        sources += ref.mean()
        
        # Extract stems
        source_names = model.sources
        vocal_idx = source_names.index('vocals')
        stems_mono = sources.mean(dim=1).cpu().numpy()
        vocal_stem = stems_mono[vocal_idx]
        other_indices = [i for i, name in enumerate(source_names) if name != 'vocals']
        other_stem = stems_mono[other_indices].sum(axis=0)
        
        analysis_audio = np.stack([vocal_stem, other_stem])
        samplerate = model.samplerate

        if args.save_audio:
            os.makedirs(out_dir, exist_ok=True)
            # Save using soundfile directly
            print(f"Saving vocal stem to {vocal_path}...", flush=True)
            sf.write(vocal_path, sources[vocal_idx].cpu().numpy().T, model.samplerate)
            
            # Combine other stems for the 'other' output
            other_stems_raw = sources[other_indices].sum(dim=0).cpu().numpy().T
            print(f"Saving 'other' stem to {other_path}...", flush=True)
            sf.write(other_path, other_stems_raw, model.samplerate)

    # ── Final Analysis & JSON ────────────────────────────────────────────────
    print("Calculating energy metrics (1s windows)...", flush=True)
    results = calculate_fine_energy(analysis_audio, samplerate)
    
    output_data = {
        "model": args.model if 'args' in locals() and hasattr(args, 'model') else "htdemucs", # fallback if reuse
        "samplerate": samplerate,
        "windows": results
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
        
    print(f"Demucs analysis complete. Results saved to {args.output}", flush=True)

    # Print sentinel and wait — Node.js parent will kill us via SIGKILL.
    # CTranslate2+ROCm (and sometimes torch ROCm) hangs on any Python-initiated exit on Windows.
    print("[DEMUCS_DONE]", flush=True)
    import time
    while True:
        time.sleep(60)

if __name__ == "__main__":
    main()
