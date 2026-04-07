import os
import re
import difflib
import time
import json


class ASREngine:
    def __init__(self, model_size="large-v3-turbo", device="cuda"):
        from faster_whisper import WhisperModel
        import torch
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
        save_audio_slice_path=None,
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
            
                # Mix them: [0:a] is main, [1:a] is mix. 
                # weights=1 <weight> means main stays at full volume, 
                # and original is added at <weight> level.
                cmd += ["-filter_complex", f"[0:a][1:a]amix=inputs=2:weights=1 {mix_weight}:dropout_transition=0:duration=shortest"]
            
            cmd += ["-c:a", "pcm_s16le", tmp_clip]
            
            label = "Slicing/Mixing" if mix_audio_path else "Slicing"
            print(f"  -> {label} audio segment: {start:.2f}s (duration: {duration if duration else 'end'})", flush=True)
            import subprocess
            subprocess.run(cmd, capture_output=True, check=True)
            
            target_audio = tmp_clip
            offset = start

        if save_audio_slice_path and target_audio:
            import shutil
            os.makedirs(os.path.dirname(os.path.abspath(save_audio_slice_path)), exist_ok=True)
            shutil.copy2(target_audio, save_audio_slice_path)
            print(f"  -> Saved audio slice to: {save_audio_slice_path}", flush=True)

        import librosa
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
        from transformers import Wav2Vec2ForCTC, AutoProcessor
        import torch
        self.device = device
        self.model_id = model_id
        try:
            print(f"Loading MMS Model '{model_id}' onto {device}...", flush=True)
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
        save_audio_slice_path=None,
    ):
        import torch
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
        
        if save_audio_slice_path:
            # For MMS, since it processes in clusters, we just save the first one or a combined one if possible.
            # But the user usually wants to see a representative fragment.
            # Let's save a placeholder or the first cluster for now to confirm it's working.
            print(f"  -> [INFO] MMS save_audio_slice_path requested. Saving first cluster to {save_audio_slice_path}", flush=True)

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
                    filter_complex += f"[vox][orig]amix=inputs=2:weights=1 {mix_weight}:dropout_transition=0:duration=shortest"
                    
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
                
                # Diagnostic save: Copy first cluster if requested
                if i == 0 and save_audio_slice_path:
                    import shutil
                    os.makedirs(os.path.dirname(os.path.abspath(save_audio_slice_path)), exist_ok=True)
                    shutil.copy2(tmp_wav, save_audio_slice_path)
                    print(f"  -> Saved MMS diagnostic cluster to: {save_audio_slice_path}", flush=True)

                # Load the sliced cluster
                import librosa
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
        import torch
        from qwen_asr import Qwen3ASRModel
        self.device = device
        self.MAX_SEGMENT_SECONDS = 60.0
        try:
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
        save_audio_slice_path=None,
    ):
        import torch
        import librosa
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
                cmd += ["-filter_complex", f"[0:a][1:a]amix=inputs=2:weights=1 {mix_weight}:dropout_transition=0:duration=shortest"]
            
            cmd += ["-c:a", "pcm_s16le", tmp_clip]
            import subprocess
            subprocess.run(cmd, capture_output=True, check=True)
            
            target_audio = tmp_clip
            offset = start
        
        if save_audio_slice_path and target_audio:
            import shutil
            os.makedirs(os.path.dirname(os.path.abspath(save_audio_slice_path)), exist_ok=True)
            shutil.copy2(target_audio, save_audio_slice_path)
            print(f"  -> Saved audio slice to: {save_audio_slice_path}", flush=True)
        
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


class SenseVoiceEngine:
    def __init__(self, device="cuda"):
        self.device = device
        try:
            from funasr import AutoModel
            print(f"Loading SenseVoiceSmall onto {device}...", flush=True)
            self.model = AutoModel(
                model="iic/SenseVoiceSmall",
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device=device,
                hub="ms", # ModelScope
            )
            print("SenseVoiceSmall Model loaded.", flush=True)
        except Exception:
            import traceback
            print("CRITICAL: Failed to load SenseVoiceSmall", flush=True)
            print(traceback.format_exc(), flush=True)
            raise

    def transcribe_file(
        self,
        audio_path,
        prompt="",
        clip_start=None,
        clip_end=None,
        mix_audio_path=None,
        mix_weight=0.07,
        save_audio_slice_path=None,
    ):
        import torch
        import librosa
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
        
        print(f"Processing audio (SenseVoice): {os.path.basename(audio_path)}", flush=True)
        
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
                cmd += ["-filter_complex", f"[0:a][1:a]amix=inputs=2:weights=1 {mix_weight}:dropout_transition=0:duration=shortest"]
            
            cmd += ["-c:a", "pcm_s16le", "-ar", "16000", tmp_clip] # SenseVoice prefers 16kHz
            import subprocess
            subprocess.run(cmd, capture_output=True, check=True)
            
            target_audio = tmp_clip
            offset = start

        if save_audio_slice_path and target_audio:
            import shutil
            os.makedirs(os.path.dirname(os.path.abspath(save_audio_slice_path)), exist_ok=True)
            shutil.copy2(target_audio, save_audio_slice_path)
            print(f"  -> Saved audio slice to: {save_audio_slice_path}", flush=True)

        try:
            # SenseVoice supports "auto", "zh", "en", "yue", "ja", "ko", "nospeech"
            res = self.model.generate(
                input=target_audio,
                cache={},
                language="auto", 
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
            )

            if not res or len(res) == 0:
                return "", [], []

            # SenseVoice returns result in segments if VAD is used
            # The 'text' usually contains everything including events like <|laughter|>
            raw_text = res[0]["text"]
            clean_text = rich_transcription_postprocess(raw_text)
            
            # SenseVoice doesn't give precise per-sentence timestamps in the same way Whisper does
            # if VAD is merged. If merge_vad=False, we get segments.
            # Let's try to get segments by re-running if needed or parsing the result.
            # Actually, funasr AutoModel with fsmn-vad returns a list of segments if we don't merge them.
            
            # For now, let's assume we want segments for the pipeline.
            # re-run with merge_vad=False to get timestamps
            res_segments = self.model.generate(
                input=target_audio,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=False,
            )

            all_segments = []
            texts = []
            
            for i, seg in enumerate(res_segments):
                seg_text = rich_transcription_postprocess(seg["text"])
                if not seg_text.strip():
                    continue
                
                # funasr timestamps are in ms
                ts = seg.get("timestamp") # [[start_ms, end_ms], ...] or [start_ms, end_ms]
                if ts and isinstance(ts, list):
                    # sometimes it's [[s, e]]
                    if isinstance(ts[0], list):
                        s_ms, e_ms = ts[0]
                    else:
                        s_ms, e_ms = ts
                    
                    s_sec = s_ms / 1000.0 + offset
                    e_sec = e_ms / 1000.0 + offset
                else:
                    # Fallback if no timestamps
                    s_sec = offset
                    e_sec = offset + 1.0 # dummy

                texts.append(seg_text)
                all_segments.append({
                    "text": seg_text.strip(),
                    "start_time": round(s_sec, 3),
                    "end_time": round(e_sec, 3),
                    "words": [],
                    "avg_logprob": 0.0,
                    "compression_ratio": 0.0,
                    "no_speech_prob": 0.0,
                    "engine": "sensevoice"
                })

            full_text = "".join(texts)
            sentences = [{"text": s["text"], "start_time": s["start_time"], "end_time": s["end_time"]} for s in all_segments]

            return full_text, sentences, all_segments

        except Exception:
            import traceback
            print("SenseVoice ASR Engine Crash Traceback:", flush=True)
            print(traceback.format_exc(), flush=True)
            return "", [], []
        finally:
            if tmp_clip and os.path.exists(tmp_clip):
                try: os.remove(tmp_clip)
                except: pass

class GemmaASREngine:
    def __init__(self, model_id="google/gemma-4-E4B-it", device="cuda"):
        self.device = device
        self.model_id = model_id
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor
            print(f"Loading {model_id} onto {device} (BF16)...", flush=True)
            # No bitsandbytes as per user request (ROCm Windows issues)
            self.model = AutoModelForMultimodalLM.from_pretrained(
                model_id,
                device_map=device,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            ).eval()
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            print(f"{model_id} loaded.", flush=True)
        except Exception:
            import traceback
            print(f"CRITICAL: Failed to load {model_id}", flush=True)
            print(traceback.format_exc(), flush=True)
            raise

    def transcribe_file(
        self,
        audio_path,
        prompt_info="", 
        clip_start=None,
        clip_end=None,
        mix_audio_path=None,
        mix_weight=0.07,
        save_audio_slice_path=None,
    ):
        import torch
        import librosa
        import transformers 
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        
        print(f"Processing audio (Gemma-4): {os.path.basename(audio_path)}", flush=True)
        
        # 1. Load and resample audio (Gemma multimodal expects 16kHz)
        # librosa handles various formats and resampling automatically
        try:
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as e:
            print(f"Error loading audio: {e}", flush=True)
            return "", [], []

        # Optional: Mix audio
        if mix_audio_path:
            try:
                bg_audio, _ = librosa.load(mix_audio_path, sr=16000, mono=True)
                # Trim or pad bg_audio to match length
                min_len = min(len(audio), len(bg_audio))
                audio[:min_len] = audio[:min_len] + bg_audio[:min_len] * mix_weight
                print(f"Mixed in {os.path.basename(mix_audio_path)} at weight {mix_weight}", flush=True)
            except Exception as e:
                print(f"Failed to mix audio: {e}", flush=True)

        total_duration = len(audio) / 16000.0
        
        # Window parameters (User requested 5s overlap for ASMR)
        CHUNK_SIZE = 30 * 16000
        OVERLAP = 5 * 16000
        STRIDE = CHUNK_SIZE - OVERLAP
        
        all_segments = []
        prev_tail = "null"
        
        # Start and end offsets if provided
        start_sample = 0
        end_sample = len(audio)
        if clip_start is not None:
            start_sample = int(float(clip_start) * 16000)
        if clip_end is not None:
            end_sample = min(int(float(clip_end) * 16000), len(audio))
            
        audio_to_process = audio[start_sample:end_sample]
        global_offset_ms = int(start_sample / 16 * 1000) # relative to global start

        if save_audio_slice_path:
            import tempfile
            import soundfile as sf
            # Since we have the resampled/mixed/sliced audio in memory, we save it directly
            fd, tmp_save = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            sf.write(tmp_save, audio_to_process, 16000)
            import shutil
            os.makedirs(os.path.dirname(os.path.abspath(save_audio_slice_path)), exist_ok=True)
            shutil.copy2(tmp_save, save_audio_slice_path)
            os.remove(tmp_save)
            print(f"  -> Saved Gemma audio fragment to: {save_audio_slice_path}", flush=True)

        for i, start_idx in enumerate(range(0, len(audio_to_process), STRIDE)):
            end_idx = min(start_idx + CHUNK_SIZE, len(audio_to_process))
            chunk = audio_to_process[start_idx:end_idx]
            if len(chunk) < 1600 * 0.5: # Skip very short tail ends
                break

            current_chunk_start_ms = global_offset_ms + int(start_idx / 16)
            
            # Construct Japanese Prompt with Context
            prompt = f"""以下の日本語 ASMR 音声セグメントを文字起こし（ASR）し、日本語の逐字録形式で出力してください。
注意：この音声は長い音声の一部（例：30秒のウィンドウ）です。境界の切断とコンテキスト処理の規則を厳守してください。

音軌：{os.path.basename(audio_path)}
シーン説明：{prompt_info}

現在のセグメント入力パラメータ：
{{
  "chunk_start_ms": {current_chunk_start_ms}, 
  "previous_tail_text": {prev_tail}
}}

文字起こしおよび音声境界処理の規則：
1. 末尾の切断破棄（重要）：現在の音声の末尾で発話が終了しておらず、意味が途切れている場合は、その不完全な文を文字起こしして出力しないでください。その末尾部分は無視し、次のセグメントでの処理に委ねてください。
2. コンテキストの継続：previous_tail_text を参照して、現在の音声の冒頭の意味を理解してください。冒頭に前のセグメントですでに文字起こしされた単語が含まれている場合は、重複して出力しないでください。
3. ノイズおよび言い間違いのフィルタリング：無意味な言い淀みや繰り返しは静かに無視し、完全でスムーズな文字起こし内容のみを出力してください。
4. 擬音語および吐息：日本語の擬音語や呼吸、喘ぎ声（例：あ、ん、はあ、クチュ）をそのまま保持してください。
5. フォーマット制限：text フィールドには日本語の原文のみを出力し、注釈や説明を加えないでください。

出力形式：
自然な意味の区切りに沿って改行（断句）し、JSON 配列形式で出力してください。
タイムスタンプの計算：出力される start と end は、音声内の相対時間に chunk_start_ms を加え、絶対時間（ミリ秒）に変換してください。

[
  {{"text": "<日本語の完全な文>", "start": <絶対開始ms>, "end": <絶対結束ms>}},
  ...
]"""

            # Prepare Inputs
            # The structure follows Gemma-4 multimodal documentation
            formatted_prompt = f"<|user|>\n{prompt}\n<|audio|>\n<|assistant|>\n"
            inputs = self.processor(text=formatted_prompt, audio=chunk, sampling_rate=16000, return_tensors="pt").to(self.device, torch.bfloat16)

            # Generate
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=448, do_sample=True, temperature=0.7, top_p=0.9)
            
            # Decode only the assistant response part
            input_len = inputs.input_ids.shape[1]
            response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)
            
            # Extract JSON from response
            try:
                # Find the JSON array (most likely)
                match = re.search(r"\[\s*\{.*\}\s*\]", response, re.DOTALL)
                chunk_segments = []
                if match:
                    json_str = match.group(0)
                    try:
                        chunk_segments = json.loads(json_str)
                    except json.JSONDecodeError:
                        # Try to handle trailing commas or other common issues
                        # Simplistic fix: remove trailing comma before ]
                        json_str_fixed = re.sub(r",\s*\]", "]", json_str)
                        try:
                            chunk_segments = json.loads(json_str_fixed)
                        except:
                            # Final fallback: find all {...} objects
                            objects = re.findall(r"\{[^{}]*\}", json_str)
                            for obj in objects:
                                try:
                                    chunk_segments.append(json.loads(obj))
                                except:
                                    pass
                
                if chunk_segments and isinstance(chunk_segments, list):
                    for seg in chunk_segments:
                        text = seg.get("text", "").strip()
                        if not text: continue
                        
                        all_segments.append({
                            "text": text,
                            "start_time": seg.get("start", 0) / 1000.0,
                            "end_time": seg.get("end", 0) / 1000.0,
                            "words": [],
                            "avg_logprob": 0.0,
                            "compression_ratio": 0.0,
                            "no_speech_prob": 0.0,
                            "engine": "gemma"
                        })
                    if all_segments:
                        prev_tail = f'"{all_segments[-1]["text"]}"'
                else:
                    print(f"Warning: No valid JSON segments found in chunk {i}. Content: {response[:100]}...", flush=True)
                    prev_tail = "null"
            except Exception as e:
                print(f"Error parsing Gemma output for chunk {i}: {e}\nResponse: {response}", flush=True)
                prev_tail = "null"

        # Final Cleanup and Formatting
        full_text = " ".join([s["text"] for s in all_segments])
        sentences = [{"text": s["text"], "start_time": s["start_time"], "end_time": s["end_time"]} for s in all_segments]
        
        return full_text, sentences, all_segments
