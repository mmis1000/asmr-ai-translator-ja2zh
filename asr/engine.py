import os
from faster_whisper import WhisperModel


class ASREngine:
    def __init__(self, model_size="large-v3-turbo", device="cuda"):
        try:
            print(f"Loading WhisperModel '{model_size}' onto {device} (bfloat16)...", flush=True)
            self.model = WhisperModel(model_size, device=device, compute_type="bfloat16")
            print("Model loaded.", flush=True)
        except Exception:
            import traceback
            print("CRITICAL: Failed to load WhisperModel", flush=True)
            print(traceback.format_exc(), flush=True)
            raise

    def transcribe_file(
        self,
        audio_path,
        prompt="",
        clip_start=None,
        clip_end=None,
        temperature=None,
        beam_size=5,
        condition_on_previous_text=True,
    ):
        print(f"Processing audio: {os.path.basename(audio_path)}", flush=True)

        target_audio = audio_path
        tmp_clip = None
        offset = 0.0

        if clip_start is not None or clip_end is not None:
            start = float(clip_start or 0.0)
            duration = None
            if clip_end is not None:
                duration = float(clip_end) - start
            
            # Physical slicing using ffmpeg is more robust than clip_timestamps
            import tempfile
            fd, tmp_clip = tempfile.mkstemp(suffix=".sliced.wav")
            os.close(fd)
            
            cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", audio_path]
            if duration is not None:
                cmd += ["-t", str(duration)]
            cmd += ["-c", "copy", tmp_clip]
            
            print(f"  -> Slicing audio segment: {start:.2f}s (duration: {duration if duration else 'end'})", flush=True)
            import subprocess
            subprocess.run(cmd, capture_output=True, check=True)
            
            target_audio = tmp_clip
            offset = start

        try:
            kwargs = {
                "beam_size": beam_size,
                "language": "ja",
                "initial_prompt": prompt or None,
                "word_timestamps": True,
                "condition_on_previous_text": condition_on_previous_text,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature

            segments_iter, info = self.model.transcribe(target_audio, **kwargs)

            if segments_iter is None:
                print("Error: model.transcribe returned NoneType segments_iter", flush=True)
                return "", [], []

            print(
                f"Detected language: {info.language} ({info.language_probability:.2f})",
                flush=True,
            )

            all_segments = []
            texts = []

            # We use enumerate(segments_iter) which triggers the actual transcription
            # for each block. We print BEFORE it completes to catch hangs.
            print("DEBUG: Beginning transcription iteration...", flush=True)

            try:
                for i, seg in enumerate(segments_iter):
                    # Offset timestamps back to original file time
                    seg_start = seg.start + offset
                    seg_end = seg.end + offset
                    
                    print(f"  -> Segment {i+1}: [{seg_start:.2f}s -> {seg_end:.2f}s]: \"{seg.text.strip()}\"", flush=True)
                    texts.append(seg.text)

                    words = []
                    if seg.words:
                        for w in seg.words:
                            words.append({
                                "text": w.word,
                                "start_time": round(w.start + offset, 3),
                                "end_time": round(w.end + offset, 3),
                            })

                    all_segments.append({
                        "text": seg.text.strip(),
                        "start_time": round(seg_start, 3),
                        "end_time": round(seg_end, 3),
                        "words": words,
                        "avg_logprob": round(seg.avg_logprob, 4),
                        "compression_ratio": round(seg.compression_ratio, 4),
                        "no_speech_prob": round(seg.no_speech_prob, 4),
                        "temperature": seg.temperature,
                    })
            except Exception as loop_error:
                print(f"ERROR during transcription loop: {loop_error}", flush=True)
                import traceback
                traceback.print_exc()
                # We return what we have so far instead of failing completely
                if not all_segments:
                    raise

            full_text = "".join(texts)
            sentences = [{"text": s["text"], "start_time": s["start_time"], "end_time": s["end_time"]} for s in all_segments]
            total_words = sum(len(s["words"]) for s in all_segments)
            print(f"Done. {len(all_segments)} segments, {total_words} words.", flush=True)

            return full_text, sentences, all_segments

        except Exception:
            import traceback
            print("ASR Engine Crash Traceback:", flush=True)
            print(traceback.format_exc(), flush=True)
            return "", [], []
        finally:
            if tmp_clip and os.path.exists(tmp_clip):
                try:
                    os.remove(tmp_clip)
                except:
                    pass
