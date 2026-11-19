"""Join per-sentence TTS clips into narration.wav and write timeline.json.

talking-head / slides: sentences back to back with short gaps.
loom: each [mm:ss] paragraph starts where it belongs in the screen recording;
      the screen is frozen where narration runs long and sped up (max 2x)
      where there is a long silent stretch.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from common import current_job, die, duration, ffmpeg_bin, load_config, run, say  # noqa: E402
from script_format import parse  # noqa: E402

LEAD_IN = 0.35
TAIL = 0.7
MAX_SPEED = 2.0
SPEEDUP_IF_EXTRA = 4.0  # seconds of silent screen before we speed it up


def load_mono(path: Path, sr: int) -> np.ndarray:
    out = subprocess.run([ffmpeg_bin(), "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def trim_silence(x: np.ndarray, sr: int, thresh_db: float = -45.0, pad: float = 0.04) -> np.ndarray:
    if x.size == 0:
        return x
    win = int(sr * 0.01)
    n = len(x) // win
    if n == 0:
        return x
    rms = np.sqrt(np.mean(x[: n * win].reshape(n, win) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    voiced = np.where(db > thresh_db)[0]
    if voiced.size == 0:
        return x
    a = max(0, voiced[0] * win - int(pad * sr))
    b = min(len(x), (voiced[-1] + 1) * win + int(pad * sr))
    return x[a:b]


def build(job, cfg) -> dict:
    sr = cfg["audio"]["sample_rate"]
    gap_s, gap_p = cfg["audio"]["sentence_gap"], cfg["audio"]["paragraph_gap"]
    script = parse(job.p("script.md").read_text())
    manifest = json.loads(job.p("tts", "manifest.json").read_text())
    by_k = {m["k"]: m for m in manifest["sentences"]}

    clips = {}
    for s in script.sentences:
        m = by_k.get(s.k)
        if not m or not job.p("tts", m["file"]).exists():
            die(f"sentence {s.k} has no audio yet — run `bin/ghl voice`")
        clips[s.k] = trim_silence(load_mono(job.p("tts", m["file"]), sr), sr)

    def para_audio(p):
        parts = []
        for j, s in enumerate(p.sentences):
            parts.append(clips[s.k])
            if j < len(p.sentences) - 1:
                parts.append(np.zeros(int(gap_s * sr), np.float32))
        return parts

    placements = []  # (k, start_sec)
    timeline: dict = {"style": job.style, "sample_rate": sr}

    if job.style != "loom":
        t = LEAD_IN
        for pi, p in enumerate(script.paragraphs):
            for j, s in enumerate(p.sentences):
                placements.append((s.k, t))
                t += len(clips[s.k]) / sr + (gap_s if j < len(p.sentences) - 1 else 0)
            t += gap_p if pi < len(script.paragraphs) - 1 else 0
        total = t + TAIL
    else:
        screen = job.p("work", "screen_norm.mp4")
        if not screen.exists():
            die("Loom style needs the screen recording prepared first (bin/ghl screen)")
        S = duration(screen)
        # group paragraphs into slots by anchor
        slots = []
        for p in script.paragraphs:
            if p.anchor is not None or not slots:
                slots.append({"anchor": p.anchor or 0.0, "paras": [p]})
            else:
                slots[-1]["paras"].append(p)
        anchors = [s["anchor"] for s in slots]
        if anchors != sorted(anchors):
            die(f"[mm:ss] timestamps must increase: {anchors}")
        if anchors[-1] >= S:
            die(f"timestamp {anchors[-1]:.1f}s is past the end of the screen recording ({S:.1f}s)")
        plan = []
        out_t = 0.0

        def add_segment(src_a, src_b, need):
            """Screen [src_a, src_b) must cover `need` seconds of narration."""
            nonlocal out_t
            length = src_b - src_a
            speed, freeze = 1.0, 0.0
            if need > length:
                freeze = need - length
            elif length - need > SPEEDUP_IF_EXTRA:
                speed = min(MAX_SPEED, length / max(need + 1.0, 0.1))
            out_len = length / speed + freeze
            plan.append({"src_start": round(src_a, 3), "src_end": round(src_b, 3), "speed": round(speed, 3),
                         "freeze": round(freeze, 3), "out_start": round(out_t, 3), "out_end": round(out_t + out_len, 3)})
            start = out_t
            out_t += out_len
            return start

        if anchors[0] > 0:
            add_segment(0.0, anchors[0], 0.0)
        for i, slot in enumerate(slots):
            src_a = slot["anchor"]
            src_b = anchors[i + 1] if i + 1 < len(slots) else S
            dur = 0.25  # small lead-in inside each slot
            starts = []
            for pi, p in enumerate(slot["paras"]):
                for j, s in enumerate(p.sentences):
                    starts.append((s.k, dur))
                    dur += len(clips[s.k]) / sr + (gap_s if j < len(p.sentences) - 1 else 0)
                dur += gap_p
            seg_start = add_segment(src_a, src_b, dur)
            placements += [(k, seg_start + off) for k, off in starts]
        total = out_t
        timeline["screen_plan"] = plan
        timeline["screen_source_duration"] = S

    audio = np.zeros(int(total * sr) + sr, np.float32)
    sentences = []
    text_by_k = {s.k: s for s in script.sentences}
    for k, start in placements:
        a = int(start * sr)
        clip = clips[k]
        audio[a:a + len(clip)] += clip
        sentences.append({"k": k, "para": text_by_k[k].para, "text": text_by_k[k].text,
                          "start": round(start, 3), "end": round(start + len(clip) / sr, 3)})
    audio = audio[: int(total * sr)]

    # loudness
    meter = pyln.Meter(sr)
    loud = meter.integrated_loudness(audio)
    audio = pyln.normalize.loudness(audio, loud, cfg["audio"]["lufs"]).astype(np.float32)
    peak = float(np.max(np.abs(audio)) or 1)
    if peak > 0.891:  # keep true peak under ~-1 dBFS
        audio *= 0.891 / peak

    sf.write(job.p("narration.wav"), audio, sr, subtype="PCM_24")
    run([ffmpeg_bin(), "-v", "error", "-y", "-i", job.p("narration.wav"), "-ac", "1", "-ar", "16000",
         job.p("kaggle_in", "narration.wav")])

    # slides / topic labels: active from the first sentence of the paragraph that declares them
    slides = []
    first_start = {}
    for s in sentences:
        first_start.setdefault(s["para"], s["start"])
    for p in script.paragraphs:
        if p.slide:
            slides.append({"start": round(max(0.0, first_start[p.i] - 0.2), 3), **p.slide.__dict__})
    for i, sl in enumerate(slides):
        sl["end"] = slides[i + 1]["start"] if i + 1 < len(slides) else round(total, 3)

    timeline.update({"title": script.title, "duration": round(total, 3), "sentences": sentences,
                     "slides": slides, "words": script.word_count})
    job.p("timeline.json").write_text(json.dumps(timeline, indent=2, ensure_ascii=False))
    return timeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job")
    a = ap.parse_args()
    job = current_job(a.job)
    tl = build(job, load_config())
    say(f"narration.wav: {tl['duration']:.1f}s, {len(tl['sentences'])} sentences, {tl['words']} words")
    if tl.get("screen_plan"):
        for seg in tl["screen_plan"]:
            note = []
            if seg["speed"] > 1:
                note.append(f"sped up {seg['speed']}x")
            if seg["freeze"] > 0:
                note.append(f"frozen {seg['freeze']:.1f}s")
            say(f"  screen {seg['src_start']:6.1f}-{seg['src_end']:6.1f}s -> {seg['out_start']:6.1f}-{seg['out_end']:6.1f}s "
                f"{'(' + ', '.join(note) + ')' if note else ''}")


if __name__ == "__main__":
    main()
