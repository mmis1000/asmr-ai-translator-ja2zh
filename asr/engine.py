import os
from faster_whisper import WhisperModel


class ASREngine:
    def __init__(self, model_size="large-v3-turbo", device="cuda", compute_type="bfloat16"):
        print(f"Loading WhisperModel '{model_size}' onto {device} ({compute_type})...", flush=True)
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Model loaded.", flush=True)

    def transcribe_file(self, audio_path, prompt=""):
        print(f"Processing audio: {os.path.basename(audio_path)}", flush=True)

        segments_iter, info = self.model.transcribe(
            audio_path,
            beam_size=5,
            language="ja",
            initial_prompt=prompt or None,
            word_timestamps=True,
        )

        print(
            f"Detected language: {info.language} ({info.language_probability:.2f})",
            flush=True,
        )

        all_segments = []
        texts = []

        for i, seg in enumerate(segments_iter):
            print(f"  -> Segment {i+1}: [{seg.start:.2f}s -> {seg.end:.2f}s]", flush=True)
            texts.append(seg.text)

            words = []
            if seg.words:
                for w in seg.words:
                    words.append({
                        "text": w.word,
                        "start_time": round(w.start, 3),
                        "end_time": round(w.end, 3),
                    })

            all_segments.append({
                "text": seg.text.strip(),
                "start_time": round(seg.start, 3),
                "end_time": round(seg.end, 3),
                "words": words,
                "avg_logprob": round(seg.avg_logprob, 4),
                "compression_ratio": round(seg.compression_ratio, 4),
                "no_speech_prob": round(seg.no_speech_prob, 4),
                "temperature": seg.temperature,
            })

        full_text = "".join(texts)
        sentences = [{"text": s["text"], "start_time": s["start_time"], "end_time": s["end_time"]} for s in all_segments]
        total_words = sum(len(s["words"]) for s in all_segments)
        print(f"Done. {len(all_segments)} segments, {total_words} words.", flush=True)

        return full_text, sentences, all_segments
