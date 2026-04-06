import os
import re
import difflib
import time
from faster_whisper import WhisperModel
import torch
import librosa
from transformers import Wav2Vec2ForCTC, AutoProcessor


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
        mix_audio_path=None,
        mix_weight=0.07,
    ):
        print(f"Processing audio: {os.path.basename(audio_path)}", flush=True)

        target_audio = audio_path
        tmp_clip = None
        offset = 0.0

        if clip_start is not None or clip_end is not None or mix_audio_path is not None:
            start = float(clip_start or 0.0)
            duration = None
            if clip_end is not None:
                duration = float(clip_end) - start
            
            # Physical slicing/mixing using ffmpeg is more robust than clip_timestamps
            import tempfile
            fd, tmp_clip = tempfile.mkstemp(suffix=".processed.wav")
            os.close(fd)
            
            # Base command
            cmd = ["ffmpeg", "-y"]
            
            # Input 1: Main audio (e.g. vocal stem)
            cmd += ["-ss", str(start), "-i", audio_path]
            
            # Input 2: Optional mix audio (e.g. original audio)
            if mix_audio_path:
                cmd += ["-ss", str(start), "-i", mix_audio_path]
            
            if duration is not None:
                cmd += ["-t", str(duration)]
            
            if mix_audio_path:
                # Mix them: [0:a] is main, [1:a] is mix. 
                # weights=1 <weight> means main stays at full volume, 
                # and original is added at <weight> level.
                cmd += ["-filter_complex", f"[0:a][1:a]amix=inputs=2:weights=1 {mix_weight}:dropout_transition=0"]
            
            cmd += ["-c:a", "pcm_s16le", tmp_clip]
            
            label = "Slicing/Mixing" if mix_audio_path else "Slicing"
            print(f"  -> {label} audio segment: {start:.2f}s (duration: {duration if duration else 'end'})", flush=True)
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


class MMSEngine:
    def __init__(self, model_id="facebook/mms-1b-all", device="cuda"):
        self.device = device
        self.model_id = model_id
        try:
            print(f"Loading '{model_id}' onto {device}...", flush=True)
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
            self.current_lang = None
            print("MMS Model loaded.", flush=True)
        except Exception:
            import traceback
            print("CRITICAL: Failed to load MMSEngine", flush=True)
            print(traceback.format_exc(), flush=True)
            raise

    def transcribe_file(
        self,
        audio_path,
        lang="jpn",
        intervals=None, # List of (start, end)
        mix_audio_path=None,
        mix_weight=0.07,
    ):
        if not intervals:
            return "", [], []

        print(f"Processing audio (MMS): {os.path.basename(audio_path)} ({len(intervals)} clusters)", flush=True)
        
        # Load the correct adapter for the target language
        if self.current_lang != lang:
            print(f"  -> Loading MMS adapter for: {lang}", flush=True)
            try:
                self.model.load_adapter(lang)
                self.processor.tokenizer.set_target_lang(lang)
                self.current_lang = lang
            except Exception as e:
                print(f"  -> WARNING: Failed to load adapter for '{lang}': {e}. Output may be garbled.", flush=True)

        all_segments = []
        full_texts = []
        
        import subprocess

        for i, (start, end) in enumerate(intervals):
            dur = end - start
            if dur <= 0: continue
            
            print(f"  -> Cluster {i+1}/{len(intervals)}: {start:.2f}s - {end:.2f}s (dur={dur:.2f}s)...", flush=True)
            
            tmp_wav = f"tmp_mms_cluster_{i}.wav"
            try:
                # Use ffmpeg for reliable slicing and internal mix-down (16kHz mono)
                if mix_audio_path and mix_weight > 0:
                    # Complex filter to slice both and mix
                    filter_complex = f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[vox];"
                    filter_complex += f"[1:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[orig];"
                    filter_complex += f"[vox][orig]amix=inputs=2:weights=1 {mix_weight}:dropout_transition=0"
                    
                    cmd = [
                        "ffmpeg", "-y", "-i", audio_path, "-i", mix_audio_path,
                        "-filter_complex", filter_complex,
                        "-ar", "16000", "-ac", "1", tmp_wav
                    ]
                else:
                    cmd = [
                        "ffmpeg", "-y", "-ss", str(start), "-t", str(dur),
                        "-i", audio_path, "-ar", "16000", "-ac", "1", tmp_wav
                    ]
                
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Load the sliced cluster
                audio, _ = librosa.load(tmp_wav, sr=16000)
                os.remove(tmp_wav)

                inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    logits = self.model(**inputs).logits
                
                ids = torch.argmax(logits, dim=-1)[0]
                text = self.processor.decode(ids).strip()
                
                print(f"     MMS Result: \"{text}\"", flush=True)

                if text:
                    full_texts.append(text)
                    all_segments.append({
                        "text": text,
                        "start_time": round(start, 3),
                        "end_time": round(end, 3),
                        "words": [],
                        "avg_logprob": 0.0,
                        "compression_ratio": 0.0,
                        "no_speech_prob": 0.0,
                        "engine": "mms"
                    })

            except Exception as e:
                print(f"     MMS Cluster Error: {e}", flush=True)
                if os.path.exists(tmp_wav): os.remove(tmp_wav)

        full_text = " ".join(full_texts)
        sentences = [{"text": s["text"], "start_time": s["start_time"], "end_time": s["end_time"]} for s in all_segments]
        
        return full_text, sentences, all_segments


class QwenASREngine:
    def __init__(self, device="cuda"):
        self.device = device
        self.MAX_SEGMENT_SECONDS = 60.0
        try:
            from qwen_asr import Qwen3ASRModel
            print(f"Loading Qwen3-ASR (1.7B) + Forced Aligner (0.6B) onto {device}...", flush=True)
            self.model = Qwen3ASRModel.from_pretrained(
                "Qwen/Qwen3-ASR-1.7B",
                dtype=torch.bfloat16,
                device_map=device,
                attn_implementation="sdpa",
                forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
                forced_aligner_kwargs=dict(
                    dtype=torch.bfloat16,
                    device_map=device,
                    attn_implementation="sdpa",
                ),
                max_inference_batch_size=32,
                max_new_tokens=2048,
            )
            print("Qwen3-ASR Model loaded.", flush=True)
        except Exception:
            import traceback
            print("CRITICAL: Failed to load Qwen3-ASR", flush=True)
            print(traceback.format_exc(), flush=True)
            raise

    def _check_prompt_leakage(self, text, prompt):
        if not prompt or not text: return False
        
        # Normalize for comparison
        clean_prompt = re.sub(r'[。！？．!?、…・\s]', '', prompt)
        clean_text = re.sub(r'[。！？．!?、…・\s]', '', text)
        if not clean_text: return False
        
        # 1. Direct contains (if text is a subset of the prompt)
        if clean_text in clean_prompt and len(clean_text) > 5:
            return True
            
        # 2. Fuzzy similarity (if text is very similar to a part of the prompt)
        sm = difflib.SequenceMatcher(None, clean_text, clean_prompt)
        if sm.ratio() > 0.8 and len(clean_text) > 10:
            return True
            
        return False

    def transcribe_file(
        self,
        audio_path,
        prompt="",
        clip_start=None,
        clip_end=None,
        mix_audio_path=None,
        mix_weight=0.07,
    ):
        from qwen_asr.inference.utils import normalize_audio_input, split_audio_into_chunks, SAMPLE_RATE
        
        print(f"Processing audio (Qwen): {os.path.basename(audio_path)}", flush=True)
        
        target_audio = audio_path
        tmp_clip = None
        offset = 0.0

        if clip_start is not None or clip_end is not None or mix_audio_path is not None:
            start = float(clip_start or 0.0)
            duration = None
            if clip_end is not None:
                duration = float(clip_end) - start
            
            import tempfile
            fd, tmp_clip = tempfile.mkstemp(suffix=".processed.wav")
            os.close(fd)
            
            cmd = ["ffmpeg", "-y"]
            cmd += ["-ss", str(start), "-i", audio_path]
            if mix_audio_path:
                cmd += ["-ss", str(start), "-i", mix_audio_path]
            if duration is not None:
                cmd += ["-t", str(duration)]
            if mix_audio_path:
                cmd += ["-filter_complex", f"[0:a][1:a]amix=inputs=2:weights=1 {mix_weight}:dropout_transition=0"]
            
            cmd += ["-c:a", "pcm_s16le", tmp_clip]
            import subprocess
            subprocess.run(cmd, capture_output=True, check=True)
            
            target_audio = tmp_clip
            offset = start
        
        # Audio length for safety
        try:
            dur_total = librosa.get_duration(filename=target_audio)
        except:
            dur_total = 60.0 # fallback

        try:
            # 1. Normalize and segment audio
            wav = normalize_audio_input(target_audio)
            segments_wavs = split_audio_into_chunks(
                wav=wav,
                sr=SAMPLE_RATE,
                max_chunk_sec=self.MAX_SEGMENT_SECONDS
            )
            
            print(f"  -> Split into {len(segments_wavs)} segments for Qwen processing.", flush=True)

            all_texts = []
            all_timestamps = []

            for i, (seg_wav, offset_sec) in enumerate(segments_wavs):
                print(f"  -> Transcribing segment {i+1}/{len(segments_wavs)} ({seg_wav.shape[0]/SAMPLE_RATE:.2f}s)...", flush=True)
                
                results = self.model.transcribe(
                    audio=(seg_wav, SAMPLE_RATE),
                    context=prompt,
                    language="Japanese",
                    return_time_stamps=True,
                )
                
                res = results[0]
                
                # Check for prompt leakage in this segment
                if prompt and self._check_prompt_leakage(res.text, prompt):
                    print(f"     -> [DROPPED] Detected prompt leakage in segment {i+1}", flush=True)
                    continue

                all_texts.append(res.text)
                
                if res.time_stamps:
                    for it in res.time_stamps.items:
                        # Offset timestamps by segment start
                        all_timestamps.append({
                            "text": it.text,
                            "start_time": round(it.start_time + offset_sec, 3),
                            "end_time": round(it.end_time + offset_sec, 3)
                        })

            full_text = "".join(all_texts)
            if not full_text:
                return "", [], []
            
            # Reassemble into sentences
            sentences_data = self._group_timestamps_by_sentence(full_text, all_timestamps, max_duration=dur_total)
            
            # Final check: discard if entire result is suspiciously similar to prompt or has failed timestamps
            if prompt and self._check_prompt_leakage(full_text, prompt):
                print(f"  !! [HALT] Entire Qwen result identified as prompt leakage. Discarding.", flush=True)
                return "", [], []

            # Format as TranscriptSegment
            all_segments = []
            for s in sentences_data:
                all_segments.append({
                    "text": s["text"],
                    "start_time": round(s["start_time"] + offset, 3),
                    "end_time": round(s["end_time"] + offset, 3),
                    "words": [],
                    "avg_logprob": 0.0,
                    "compression_ratio": 0.0,
                    "no_speech_prob": 0.0,
                    "engine": "qwen"
                })
                
            # Also update sentences_data for return
            ret_sentences = []
            for s in sentences_data:
                s["start_time"] = round(s["start_time"] + offset, 3)
                s["end_time"] = round(s["end_time"] + offset, 3)
                ret_sentences.append(s)

            return full_text, ret_sentences, all_segments

        except Exception:
            import traceback
            print("Qwen ASR Engine Crash Traceback:", flush=True)
            print(traceback.format_exc(), flush=True)
            return "", [], []
        finally:
            if tmp_clip and os.path.exists(tmp_clip):
                try: os.remove(tmp_clip)
                except: pass

    def _group_timestamps_by_sentence(self, text, timestamps, max_duration=60.0):
        if not timestamps: return []
        
        # Split text into sentences
        parts = re.split(r'([。！？．!?、…・]+)', text)
        sentences_str = []
        for i in range(0, len(parts)-1, 2):
            sentences_str.append(parts[i] + parts[i+1])
        if len(parts) % 2 == 1 and parts[-1]:
            sentences_str.append(parts[-1])
            
        sentences = []
        for s in sentences_str:
            if s.strip():
                sentences.append({
                    "text": s.strip(),
                    "start_time": -1.0,
                    "end_time": -1.0,
                    "char_start": 0,
                    "char_end": 0
                })
        
        # Char boundaries in stripped text
        clean_text = ""
        for s in sentences:
            s_clean = re.sub(r'[。！？．!?、…・\s]', '', s["text"])
            s["char_start"] = len(clean_text)
            s["char_end"] = len(clean_text) + len(s_clean)
            clean_text += s_clean
            
        # Map timestamps to char indices
        clean_ts_str = ""
        ts_char_map = []
        for ts in timestamps:
            c_ts = re.sub(r'[。！？．!?、…・\s]', '', ts["text"] if isinstance(ts, dict) else ts.text)
            clean_ts_str += c_ts
            for _ in range(len(c_ts)):
                ts_char_map.append(ts)
                
        # Matching
        sm = difflib.SequenceMatcher(None, clean_text, clean_ts_str)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ('equal', 'replace'):
                for text_idx in range(i1, i2):
                    # Linear mapping of char indices within the block
                    ts_idx = j1 + int((text_idx - i1) * (j2 - j1) / max(1, i2 - i1)) 
                    if ts_idx < len(ts_char_map):
                        mapped_ts = ts_char_map[ts_idx]
                        m_start = mapped_ts["start_time"] if isinstance(mapped_ts, dict) else mapped_ts.start_time
                        m_end = mapped_ts["end_time"] if isinstance(mapped_ts, dict) else mapped_ts.end_time
                        
                        for s in sentences:
                            if s["char_start"] <= text_idx < s["char_end"]:
                                if s["start_time"] == -1.0:
                                    s["start_time"] = m_start
                                s["end_time"] = max(s["end_time"], m_end)
                                break
                                
        # Advanced Fallback: Linear interpolation for unaligned sentences
        # 1. Fill boundaries
        last_end = 0.0
        for i, s in enumerate(sentences):
            if s["start_time"] != -1.0:
                last_end = s["end_time"]
            else:
                # Find next timed segment
                next_start = max_duration
                for j in range(i+1, len(sentences)):
                    if sentences[j]["start_time"] != -1.0:
                        next_start = sentences[j]["start_time"]
                        break
                
                # Interpolate if we have a gap
                s["start_time"] = last_end
                s["end_time"] = next_start # Placeholder, will refine below
        
        # 2. Refine durations based on character length relative to the gap
        # This prevents simple collapse. We distribute the gap time according to char weight.
        i = 0
        while i < len(sentences):
            if sentences[i]["start_time"] == sentences[i]["end_time"] and i < len(sentences)-1:
                 # Check for a "gap group"
                 gap_group = []
                 base_time = sentences[i]["start_time"]
                 end_time = base_time
                 for j in range(i, len(sentences)):
                     if sentences[j]["start_time"] == base_time and sentences[j]["end_time"] == base_time:
                         gap_group.append(sentences[j])
                     else:
                         end_time = sentences[j]["start_time"]
                         break
                 
                 if gap_group:
                     # Distribute duration by chars
                     total_chars = sum(len(s["text"]) for s in gap_group)
                     total_dur = end_time - base_time
                     curr = base_time
                     for s in gap_group:
                         dur = (len(s["text"]) / max(1, total_chars)) * total_dur
                         s["start_time"] = curr
                         s["end_time"] = curr + dur
                         curr += dur
                     i += len(gap_group)
                     continue
            i += 1

        for s in sentences:
            if "char_start" in s: del s["char_start"]
            if "char_end" in s: del s["char_end"]
            
        return sentences
