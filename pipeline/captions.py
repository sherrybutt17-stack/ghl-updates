"""Word-timed captions, drawn as transparent PNGs (no libass needed).

1. Whisper word timestamps on narration.wav, aligned to the script's spelling
   (falls back to proportional timing per sentence if Whisper is unavailable).
2. Words grouped into short chunks; the word being spoken is highlighted.
3. One PNG per caption state + an ffconcat list per layout, which compose.py
   overlays on the video. Also writes out/<job>.srt for YouTube closed captions.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import current_job, load_config, say  # noqa: E402
from asr import norm_tokens  # noqa: E402

# layout -> (band width, band height, base font size, use background pill)
LAYOUTS = {
    "16x9": (1600, 190, 66, False),
    "9x16": (1000, 300, 84, False),
    "16x9_split": (1360, 170, 58, True),
    "9x16_split": (1000, 230, 70, True),
}


# ---------- timing ----------
def whisper_words(job) -> list[dict]:
    from asr import transcribe
    res = transcribe(job.p("narration.wav"), words=True)
    out = []
    for seg in res.get("segments", []):
        for w in seg.get("words", []):
            out.append({"w": w["word"].strip(), "start": float(w["start"]), "end": float(w["end"])})
    return out


def align_sentence(sent: dict, hyp: list[dict]) -> list[dict]:
    words = sent["text"].split()
    s0, s1 = sent["start"], sent["end"]
    inside = [h for h in hyp if s0 - 0.15 <= (h["start"] + h["end"]) / 2 <= s1 + 0.15]
    times: list[tuple[float, float] | None] = [None] * len(words)
    if inside:
        ref_tok = [(norm_tokens(w) or [""])[0] for w in words]
        hyp_tok = [(norm_tokens(h["w"]) or [""])[0] for h in inside]
        sm = difflib.SequenceMatcher(a=ref_tok, b=hyp_tok, autojunk=False)
        for a, b, n in sm.get_matching_blocks():
            for i in range(n):
                times[a + i] = (inside[b + i]["start"], inside[b + i]["end"])
    # fill gaps by interpolation (by character length) between known neighbours
    i = 0
    while i < len(words):
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < len(words) and times[j] is None:
            j += 1
        left = times[i - 1][1] if i > 0 else s0
        right = times[j][0] if j < len(words) else s1
        right = max(right, left + 0.08 * (j - i))
        lens = [len(w) + 1 for w in words[i:j]]
        tot = sum(lens)
        t = left
        for k, ln in enumerate(lens):
            d = (right - left) * ln / tot
            times[i + k] = (t, t + d)
            t += d
        i = j
    out = []
    for w, (a, b) in zip(words, times):
        a, b = max(a, s0 - 0.05), min(max(b, a + 0.06), s1 + 0.1)
        out.append({"w": w, "start": round(a, 3), "end": round(b, 3)})
    for k in range(1, len(out)):  # monotonic
        if out[k]["start"] < out[k - 1]["start"]:
            out[k]["start"] = out[k - 1]["start"]
    return out


def chunk(words: list[dict], n: int) -> list[list[dict]]:
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= n or (len(cur) >= 2 and w["w"][-1:] in ",.!?;:"):
            chunks.append(cur)
            cur = []
    if cur:
        if chunks and len(cur) == 1 and len(chunks[-1]) < n + 1:
            chunks[-1] += cur
        else:
            chunks.append(cur)
    return chunks


# ---------- drawing ----------
def find_font(cfg, size: int):
    from PIL import ImageFont
    root = Path(__file__).resolve().parent.parent
    for f in sorted((root / "assets" / "brand" / "fonts").glob("*.[ot]tf")):
        return ImageFont.truetype(str(f), size)
    name = cfg["captions"]["font"]
    candidates = [(f"/System/Library/Fonts/{name}.ttc", ("Heavy", "Bold")),
                  ("/System/Library/Fonts/Avenir Next.ttc", ("Heavy", "Bold")),
                  ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", ())]
    for path, styles in candidates:
        if not Path(path).exists():
            continue
        if not styles:
            return ImageFont.truetype(path, size)
        for idx in range(16):
            try:
                f = ImageFont.truetype(path, size, index=idx)
            except OSError:
                break
            if f.getname()[1] in styles:
                return f
        return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def hex_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def draw_state(words: list[str], active: int, layout: str, cfg) -> "Image.Image":
    from PIL import Image, ImageDraw, ImageFilter
    bw, bh, fs, pill = LAYOUTS[layout]
    color, hi, outline = (hex_rgb(cfg["captions"][k]) for k in ("color", "highlight", "outline"))
    size = fs
    while True:
        font = find_font(cfg, size)
        space = font.getlength(" ")
        widths = [font.getlength(w) for w in words]
        lines, cur, cur_w = [], [], 0.0
        for i, w in enumerate(widths):
            add = w + (space if cur else 0)
            if cur and cur_w + add > bw - 80:
                lines.append(cur)
                cur, cur_w = [], 0.0
                add = w
            cur.append(i)
            cur_w += add
        lines.append(cur)
        asc, desc = font.getmetrics()
        lh = int((asc + desc) * 1.08)
        if len(lines) <= 2 and lh * len(lines) <= bh - 30 or size < 30:
            break
        size -= 4
    img = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    total_h = lh * len(lines)
    y0 = (bh - total_h) // 2
    if pill:
        pw = max(sum(widths[i] for i in ln) + space * (len(ln) - 1) for ln in lines) + 60
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([(bw - pw) / 2, y0 - 14, (bw + pw) / 2, y0 + total_h + 10], radius=28, fill=(0, 0, 0, 165))
    # soft shadow layer
    shadow = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    ds = ImageDraw.Draw(shadow)
    d = ImageDraw.Draw(img)
    stroke = max(3, size // 12)
    for li, ln in enumerate(lines):
        lw = sum(widths[i] for i in ln) + space * (len(ln) - 1)
        x = (bw - lw) / 2
        y = y0 + li * lh
        for i in ln:
            if not pill:
                ds.text((x + 3, y + 5), words[i], font=font, fill=(0, 0, 0, 140), stroke_width=stroke, stroke_fill=(0, 0, 0, 140))
            x += widths[i] + space
    if not pill:
        img = Image.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(4)), img)
        d = ImageDraw.Draw(img)
    for li, ln in enumerate(lines):
        lw = sum(widths[i] for i in ln) + space * (len(ln) - 1)
        x = (bw - lw) / 2
        y = y0 + li * lh
        for i in ln:
            fill = hi if i == active else color
            d.text((x, y), words[i], font=font, fill=fill + (255,), stroke_width=0 if pill else stroke,
                   stroke_fill=outline + (255,))
            x += widths[i] + space
    return img


def render_layout(job, cfg, chunks: list[list[dict]], layout: str, total: float):
    from PIL import Image
    d = job.p("work", f"cap_{layout}")
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.png"):
        f.unlink()
    bw, bh, _, _ = LAYOUTS[layout]
    Image.new("RGBA", (bw, bh), (0, 0, 0, 0)).save(d / "blank.png")
    events = []  # (start, end, file)
    upper = cfg["captions"].get("uppercase", False)
    for ci, ch in enumerate(chunks):
        texts = [w["w"].upper() if upper else w["w"] for w in ch]
        for wi, w in enumerate(ch):
            name = f"c{ci:04d}_{wi}.png"
            draw_state(texts, wi, layout, cfg).save(d / name, optimize=False, compress_level=1)
            start = w["start"] if wi else ch[0]["start"]
            end = ch[wi + 1]["start"] if wi + 1 < len(ch) else ch[-1]["end"]
            events.append([start, end, name])
    # hold each chunk until the next one if the gap is short; no overlaps
    for i in range(len(events) - 1):
        nxt = events[i + 1][0]
        if nxt - events[i][1] < 0.35:
            events[i][1] = nxt
        events[i][1] = min(events[i][1], nxt)
    lines = ["ffconcat version 1.0"]
    t = 0.0
    fps = cfg["face"]["fps"]
    q = lambda x: round(round(x * fps) / fps, 4)  # snap to frame grid  # noqa: E731
    for start, end, name in events:
        start, end = q(start), q(end)
        if start > t:
            lines += ["file 'blank.png'", f"duration {start - t:.4f}"]
            t = start
        if end > t:
            lines += [f"file '{name}'", f"duration {end - t:.4f}"]
            t = end
    if total > t:
        lines += ["file 'blank.png'", f"duration {total - t:.4f}"]
    lines += ["file 'blank.png'"]  # concat demuxer needs a final entry
    (d / "list.ffconcat").write_text("\n".join(lines) + "\n")
    return len(events)


def srt(job, chunks_by_sentence: list[list[dict]]):
    def ts(x):
        h, rem = divmod(x, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace(".", ",")
    out, n = [], 0
    for words in chunks_by_sentence:
        for part in [words[i:i + 12] for i in range(0, len(words), 12)]:
            n += 1
            out += [str(n), f"{ts(part[0]['start'])} --> {ts(part[-1]['end'])}", " ".join(w["w"] for w in part), ""]
    p = job.p("out", f"{job.id}.srt")
    p.write_text("\n".join(out))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job")
    ap.add_argument("--no-asr", action="store_true", help="skip Whisper, use proportional timing")
    a = ap.parse_args()
    job = current_job(a.job)
    cfg = load_config()
    tl = json.loads(job.p("timeline.json").read_text())

    hyp = []
    if not a.no_asr:
        try:
            hyp = whisper_words(job)
        except Exception as e:  # noqa: BLE001
            say(f"WARNING: Whisper timing failed ({e}); using proportional timing")
    words_by_sentence = [align_sentence(s, hyp) for s in tl["sentences"]]
    job.p("work", "words.json").write_text(json.dumps(words_by_sentence, indent=1))
    n = cfg["captions"]["words_per_chunk"]
    chunks = [c for ws in words_by_sentence for c in chunk(ws, n)]
    layouts = ["16x9_split", "9x16_split"] if job.style in ("loom", "slides") else ["16x9", "9x16"]
    for lay in layouts:
        k = render_layout(job, cfg, chunks, lay, tl["duration"])
        say(f"captions {lay}: {k} states")
    say(f"subtitles: {srt(job, words_by_sentence)}")
    job.stage_done("captions")


if __name__ == "__main__":
    main()
