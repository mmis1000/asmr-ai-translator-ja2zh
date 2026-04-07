#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib>=3.8",
#     "numpy>=1.26",
# ]
# ///
"""
Pipeline timeline visualizer — 1 second = 2 pixels (HiDPI)
Usage: uv run visualize_pipeline.py <output-dir> [track-stem-filter]

Generates one PNG per track showing:
  Row 1  Vocal RMS / Other RMS (from demucs.json) — normalized dBFS
  Row 2  SNR dB (from demucs.json) — clipped + normalized
  Row 3  Raw ASR segments before repair (blue=good, red=bad)
  Row 4  Surgical repair newSegments (from surgical.json)
  Row 5  Final merged segments (from transcription.json)
"""

import sys
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── layout constants ─────────────────────────────────────────────────────────
DPI            = 100          # 2px = 1s when figsize_width = duration * 2 / DPI
PX_PER_SEC     = 2            # HiDPI: 2 output pixels per second
ROW_PX         = 60          # pixel height per sub-track row
N_ROWS         = 6
MARGIN_TOP_PX  = 30
MARGIN_BOT_PX  = 20
TOTAL_HEIGHT_PX = N_ROWS * ROW_PX + MARGIN_TOP_PX + MARGIN_BOT_PX

# ── VAD thresholds (mirroring defaults in asmr-translator cli.ts) ──────────
VAD_VOCAL_THRESHOLD = 0.001
VAD_SNR_THRESHOLD   = 2.0

# ── segment resplit (mirrors resplitSegment in transcript-cleaner.ts) ────────
_TERMINAL_RE = re.compile(r'[。！？!?]$')

def resplit_segments(segs: list[dict], gap_sec: float = 2.0) -> list[dict]:
    """Split Whisper segments at internal word gaps, matching cleanTranscript."""
    out = []
    for seg in segs:
        words = seg.get("words") or []
        if len(words) < 2:
            out.append(seg)
            continue
        groups: list[list[dict]] = [[words[0]]]
        for i in range(1, len(words)):
            gap = words[i]["start_time"] - words[i - 1]["end_time"]
            ends_terminal = bool(_TERMINAL_RE.search(words[i - 1]["text"].strip()))
            if gap >= gap_sec or ends_terminal:
                groups.append([words[i]])
            else:
                groups[-1].append(words[i])
        if len(groups) == 1:
            out.append(seg)
        else:
            for grp in groups:
                out.append({**seg,
                             "text": "".join(w["text"] for w in grp),
                             "start_time": grp[0]["start_time"],
                             "end_time": grp[-1]["end_time"],
                             "words": grp})
    return out

# ── quality heuristic (mirrors isGarbled in transcript-cleaner.ts) ───────────
def is_bad(seg: dict) -> bool:
    if seg.get("compression_ratio", 0) > 2.5:
        return True
    if seg.get("avg_logprob", 0) < -1.5:
        return True
    if seg.get("no_speech_prob", 0) > 0.8:
        return True
    if seg.get("mismatch"):
        return True
    return False

# ── normalization helpers ────────────────────────────────────────────────────
def rms_to_dbfs(rms_array: np.ndarray, floor_db: float = -60.0) -> np.ndarray:
    """Convert RMS to dBFS, clamp at floor."""
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 20.0 * np.log10(np.maximum(rms_array, 1e-10))
    return np.clip(db, floor_db, 0.0)

def normalize_to_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)

def snr_to_01(snr_array: np.ndarray,
              lo_db: float = -20.0, hi_db: float = 40.0) -> np.ndarray:
    """Clip SNR to [lo_db, hi_db] then normalize to [0, 1]."""
    clipped = np.clip(snr_array, lo_db, hi_db)
    return (clipped - lo_db) / (hi_db - lo_db)

# ── drawing helpers ───────────────────────────────────────────────────────────
def draw_signal_row(ax, times: np.ndarray, values_01: np.ndarray,
                    color: str, label: str, row_y0: float, row_h: float):
    """Fill area under a normalized signal within a row."""
    y_base = row_y0
    y_vals = row_y0 + values_01 * row_h
    ax.fill_between(times, y_base, y_vals, color=color, alpha=0.7, linewidth=0)
    ax.plot(times, y_vals, color=color, linewidth=0.5, alpha=0.9, label=label)

def draw_segment_row(ax, segments: list[dict],
                     color_fn,          # callable(seg) -> color str
                     row_y0: float, row_h: float,
                     alpha: float = 0.8):
    """Draw filled rectangles for each segment in a row."""
    for seg in segments:
        start = seg.get("start_time", 0)
        end   = seg.get("end_time", start)
        if end <= start:
            end = start + 0.1
        color = color_fn(seg)
        rect = mpatches.FancyBboxPatch(
            (start, row_y0 + 1), end - start, row_h - 2,
            boxstyle="square,pad=0",
            facecolor=color, edgecolor="none", alpha=alpha,
        )
        ax.add_patch(rect)

# ── per-track renderer ────────────────────────────────────────────────────────
def render_track(stem: str, out_dir: Path, output_png: Path):
    # ── load files ────────────────────────────────────────────────────────────
    def load(ext):
        p = out_dir / (stem + ext)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    demucs   = load(".demucs.json")
    raw_asr  = load(".raw-transcription.json")
    trans    = load(".transcription.json")
    surgical = load(".surgical.json")

    if not demucs:
        print(f"  [skip] no demucs.json for {stem}")
        return

    # ── determine total duration from demucs windows ─────────────────────────
    windows     = demucs["windows"]
    duration    = windows[-1]["end"] if windows else 0
    if duration < 1:
        print(f"  [skip] zero duration for {stem}")
        return

    # ── build demucs signal arrays ────────────────────────────────────────────
    t_centers   = np.array([(w["start"] + w["end"]) / 2 for w in windows])
    vocal_rms   = np.array([w["vocal_rms"]  for w in windows])
    other_rms   = np.array([w["other_rms"]  for w in windows])
    snr_db      = np.array([w["snr_db"]     for w in windows])

    # Normalize: RMS → dBFS → [0, 1]  (both relative to the louder of the two)
    vocal_db    = rms_to_dbfs(vocal_rms)
    other_db    = rms_to_dbfs(other_rms)
    global_lo   = min(vocal_db.min(), other_db.min())
    global_hi   = max(vocal_db.max(), other_db.max())
    def norm_db(db):
        if global_hi == global_lo:
            return np.zeros_like(db)
        return np.clip((db - global_lo) / (global_hi - global_lo), 0, 1)

    vocal_01    = norm_db(vocal_db)
    other_01    = norm_db(other_db)

    # Normalize SNR to the actual data range so variation is visible.
    # A fixed [-20, +40] range causes solid fills when ASMR SNR routinely >40 dB.
    snr_lo      = float(np.nanmin(snr_db))
    snr_hi      = float(np.nanmax(snr_db))
    if snr_hi > snr_lo:
        snr_01  = np.clip((snr_db - snr_lo) / (snr_hi - snr_lo), 0.0, 1.0)
    else:
        snr_01  = np.zeros_like(snr_db)

    # ── figure setup ──────────────────────────────────────────────────────────
    fig_w       = max(duration * PX_PER_SEC / DPI, 8.0)  # inches; 2px/s
    fig_h       = TOTAL_HEIGHT_PX / DPI

    fig, ax     = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
    ax.set_xlim(0, duration)
    total_h     = N_ROWS * ROW_PX
    ax.set_ylim(0, total_h + MARGIN_TOP_PX)
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.axis("off")

    # Row y-origins (bottom to top arrangement, drawn top-to-bottom):
    #   row 0 = top → y0 = total_h - ROW_PX; row 4 = bottom → y0 = 0
    def row_y0(row_index: int) -> float:
        """row_index 0 = topmost visual row."""
        return MARGIN_BOT_PX + (N_ROWS - 1 - row_index) * ROW_PX

    ROW_LABELS = [
        "Vocal / Other RMS (dBFS norm.)",
        f"SNR dB ({snr_lo:.0f} … {snr_hi:.0f})",
        "Demucs Speech VAD (Voice Trigger)",
        "Raw ASR  [blue=ok | red=bad]",
        "Post-Repair segments",
        "Final merged segments",
    ]

    # Draw row backgrounds + labels
    for i, label in enumerate(ROW_LABELS):
        y0  = row_y0(i)
        bg  = "#f0f0f0" if i % 2 == 0 else "#e0e8f0"
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, y0), duration, ROW_PX,
            boxstyle="square,pad=0", facecolor=bg, edgecolor="#cccccc",
            linewidth=0.3, zorder=0,
        ))
        ax.text(2, y0 + ROW_PX - 8, label, fontsize=5.5, va="top",
                color="#333333", zorder=10)

    # ── row 0: vocal / other RMS ──────────────────────────────────────────────
    y0_sig = row_y0(0)
    draw_signal_row(ax, t_centers, other_01, "#e07030", "Other",
                    y0_sig + 2, ROW_PX - 4)
    draw_signal_row(ax, t_centers, vocal_01, "#2060c0", "Vocal",
                    y0_sig + 2, ROW_PX - 4)

    # ── row 1: SNR ────────────────────────────────────────────────────────────
    y0_snr = row_y0(1)
    # zero-dB reference line (only when 0 dB falls within the actual data range)
    if snr_lo < 0.0 < snr_hi:
        zero_01 = (0.0 - snr_lo) / (snr_hi - snr_lo)
        ax.axhline(y0_snr + 2 + zero_01 * (ROW_PX - 4), color="#888888",
                   linewidth=0.5, linestyle="--", zorder=5)
    draw_signal_row(ax, t_centers, snr_01, "#7030a0", "SNR",
                    y0_snr + 2, ROW_PX - 4)

    # ── row 2: Demucs VAD (Speech Presence) ───────────────────────────────────
    y0_vad = row_y0(2)
    vad_segments = []
    current_start = None
    for i, win in enumerate(windows):
        # isActive mirrors logic in transcript-cleaner.ts
        is_active = win["vocal_rms"] > VAD_VOCAL_THRESHOLD and win["snr_db"] > VAD_SNR_THRESHOLD
        if is_active and current_start is None:
            current_start = win["start"]
        elif not is_active and current_start is not None:
            vad_segments.append({"start_time": current_start, "end_time": windows[i-1]["end"]})
            current_start = None
    if current_start is not None:
        vad_segments.append({"start_time": current_start, "end_time": windows[-1]["end"]})

    draw_segment_row(ax, vad_segments, lambda s: "#28a745", y0_vad, ROW_PX, alpha=0.5)

    # ── row 3: raw ASR segments ───────────────────────────────────────────────
    y0_raw = row_y0(3)
    if raw_asr:
        raw_segs = resplit_segments(raw_asr.get("segments", []))
        draw_segment_row(ax, raw_segs,
                         lambda s: "#cc2020" if is_bad(s) else "#2060c0",
                         y0_raw, ROW_PX)

    # ── row 4: surgical repair newSegments ────────────────────────────────────
    y0_rep = row_y0(4)
    if surgical:
        for entry in surgical:
            status = entry.get("status", "")
            new_segs = entry.get("newSegments", [])
            # Shade repair range
            r = entry.get("range", {})
            if r:
                rng_color = "#d4f0d4" if status == "success" else "#f0d4d4"
                ax.add_patch(mpatches.FancyBboxPatch(
                    (r["start"], y0_rep), r["end"] - r["start"], ROW_PX,
                    boxstyle="square,pad=0",
                    facecolor=rng_color, edgecolor="#aaaaaa",
                    linewidth=0.3, zorder=1,
                ))
            seg_color = "#1a9e1a" if status == "success" else "#cc7000"
            draw_segment_row(ax, new_segs,
                             lambda s, c=seg_color: c,
                             y0_rep, ROW_PX, alpha=0.85)

    # ── row 5: final transcription.json segments ──────────────────────────────
    y0_fin = row_y0(5)
    if trans:
        final_segs = trans.get("segments", [])
        draw_segment_row(ax, final_segs,
                         lambda s: "#1a5fa0",
                         y0_fin, ROW_PX)
        # Mismatches that were filtered (grey)
        mismatch_segs = trans.get("mismatches", [])
        draw_segment_row(ax, mismatch_segs,
                         lambda s: "#999999",
                         y0_fin, ROW_PX, alpha=0.4)

    # ── x-axis tick marks every 60s ──────────────────────────────────────────
    ax.axis("on")
    ax.set_xlim(0, duration)
    ax.set_ylim(0, total_h + MARGIN_BOT_PX + MARGIN_TOP_PX)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.yaxis.set_visible(False)
    step = 60 if duration > 300 else 30 if duration > 60 else 10
    ticks = np.arange(0, duration, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t//60)}:{int(t%60):02d}" for t in ticks],
                       fontsize=5, rotation=45, ha="right")
    ax.tick_params(axis="x", length=2, width=0.5)

    # ── legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color="#2060c0", label="Vocal RMS"),
        mpatches.Patch(color="#e07030", label="Other RMS"),
        mpatches.Patch(color="#7030a0", label="SNR"),
        mpatches.Patch(color="#28a745", label="Demucs VAD"),
        mpatches.Patch(color="#2060c0", label="ASR ok"),
        mpatches.Patch(color="#cc2020", label="ASR bad"),
        mpatches.Patch(color="#1a9e1a", label="Repair ok"),
        mpatches.Patch(color="#cc7000", label="Repair fail"),
        mpatches.Patch(color="#1a5fa0", label="Final"),
        mpatches.Patch(color="#999999", label="Mismatch"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=5,
               ncol=3, framealpha=0.8, handlelength=1.2, handleheight=0.8)

    # ── title ─────────────────────────────────────────────────────────────────
    fig.suptitle(stem, fontsize=7, x=0.01, ha="left", y=0.98)

    plt.tight_layout(pad=0.2)
    fig.savefig(output_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_png.name}  ({int(duration)}s, {int(duration)}px wide)")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: uv run visualize_pipeline.py <output-dir> [stem-filter]")
        sys.exit(1)

    out_dir    = Path(sys.argv[1])
    stem_filter = sys.argv[2] if len(sys.argv) > 2 else None

    if not out_dir.is_dir():
        print(f"Error: {out_dir} is not a directory")
        sys.exit(1)

    # Discover all stems that have at least a demucs.json
    stems = sorted({
        p.name.replace(".demucs.json", "")
        for p in out_dir.glob("*.demucs.json")
    })

    if stem_filter:
        stems = [s for s in stems if stem_filter in s]

    if not stems:
        print("No *.demucs.json files found.")
        sys.exit(1)

    vis_dir = out_dir / "vis"
    vis_dir.mkdir(exist_ok=True)

    print(f"Rendering {len(stems)} track(s) → {vis_dir}")
    for stem in stems:
        out_png = vis_dir / (stem + ".timeline.png")
        print(f"  {stem}")
        try:
            render_track(stem, out_dir, out_png)
        except Exception as e:
            print(f"  [ERROR] {e}")


if __name__ == "__main__":
    main()
