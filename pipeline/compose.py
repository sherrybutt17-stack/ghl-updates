"""Compose final videos with ffmpeg.

Styles
  head   : full-screen talking head (16:9 + 9:16), title banner, topic labels, captions, end card
  slides : auto-generated slides + circular face bubble (16:9), slides-over-face split (9:16)
  loom   : your screen recording (retimed to the narration) + face bubble / split, same as slides

Face source: kaggle_out/lipsync_<job>.mp4 (lip-synced). With --preview the raw avatar
clip is looped instead (no lip-sync) so layouts can be checked before using GPU time.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import graphics as G  # noqa: E402
from common import ROOT, current_job, die, ffmpeg, load_config, say, video_info  # noqa: E402

FPS = 25
X264 = ["-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", "-r", str(FPS)]
AAC = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]


class Graph:
    """Small helper to build -i lists and filter_complex chains."""

    def __init__(self):
        self.inputs: list[str] = []
        self.n = 0
        self.chains: list[str] = []
        self.k = 0

    def add(self, *args) -> int:
        self.inputs += [str(a) for a in args]
        self.n += 1
        return self.n - 1

    def image(self, path, dur) -> int:
        return self.add("-loop", "1", "-framerate", FPS, "-t", f"{dur:.3f}", "-i", path)

    def label(self) -> str:
        self.k += 1
        return f"v{self.k}"

    def chain(self, s: str):
        self.chains.append(s)

    def overlay_png(self, cur, path, dur, start, end, x="0", y="0", fade=0.3) -> str:
        i = self.image(path, dur)
        o = self.label()
        self.chain(f"[{i}:v]format=rgba,fade=t=in:st={start:.2f}:d={fade}:alpha=1,"
                   f"fade=t=out:st={max(start, end - fade):.2f}:d={fade}:alpha=1[{o}i]")
        self.chain(f"[{cur}][{o}i]overlay={x}:{y}:enable='between(t,{start:.2f},{end:.2f})'[{o}]")
        return o


def face_crop(cfg, W, H):
    c = cfg["face"].get("crop")
    if c:
        x, y, s = c
    else:  # assume a centred head-and-shoulders shot
        s = int(H * 0.5)
        x, y = (W - s) // 2, int(H * 0.1)
    return int(x), int(y), int(s)


def even(v):
    return int(v) // 2 * 2


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def build_slides_bg(job, cfg, tl) -> Path:
    d = job.p("work", "slides")
    d.mkdir(parents=True, exist_ok=True)
    slides = list(tl["slides"])
    if not slides or slides[0]["start"] > 0.5:
        slides.insert(0, {"start": 0.0, "end": slides[0]["start"] if slides else tl["duration"],
                          "title": tl["title"], "subtitle": cfg["brand"].get("name") or "", "bullets": []})
    lines = ["ffconcat version 1.0"]
    for i, sl in enumerate(slides):
        png = d / f"slide_{i:02d}.png"
        G.slide(sl, cfg["brand"], png)
        start = 0.0 if i == 0 else sl["start"]
        end = slides[i + 1]["start"] if i + 1 < len(slides) else tl["duration"]
        lines += [f"file '{png.name}'", f"duration {max(end - start, 0.04):.3f}"]
    lines.append(f"file 'slide_{len(slides) - 1:02d}.png'")
    lst = d / "slides.ffconcat"
    lst.write_text("\n".join(lines) + "\n")
    out = job.p("work", "bg_slides.mp4")
    ffmpeg(["-f", "concat", "-safe", "0", "-i", lst, "-vf", f"fps={FPS},scale=1920:1080,format=yuv420p",
            "-t", f"{tl['duration']:.3f}", *X264, "-tune", "stillimage", out])
    return out


def build_screen_bg(job, tl) -> Path:
    src = job.p("work", "screen_norm.mp4")
    parts, labels = [], []
    for i, seg in enumerate(tl["screen_plan"]):
        f = (f"[0:v]trim=start={seg['src_start']}:end={seg['src_end']},setpts=(PTS-STARTPTS)/{seg['speed']}")
        if seg["freeze"] > 0:
            f += f",tpad=stop_mode=clone:stop_duration={seg['freeze']}"
        parts.append(f + f",fps={FPS}[s{i}]")
        labels.append(f"[s{i}]")
    fc = ";".join(parts) + f";{''.join(labels)}concat=n={len(labels)}:v=1:a=0,format=yuv420p[out]"
    out = job.p("work", "bg_screen.mp4")
    ffmpeg(["-i", src, "-filter_complex", fc, "-map", "[out]", "-t", f"{tl['duration']:.3f}", *X264, out])
    return out


def overlays_dir(job):
    d = job.p("work", "overlays")
    d.mkdir(parents=True, exist_ok=True)
    return d


def render_head(job, cfg, tl, face: Path, loop: bool, vertical: bool) -> Path:
    D = tl["duration"]
    info = video_info(face)
    W, H = info["width"], info["height"]
    g = Graph()
    fi = g.add(*(["-stream_loop", "-1"] if loop else []), "-i", face)
    ai = g.add("-i", job.p("narration.wav"))
    size = (1080, 1920) if vertical else (1920, 1080)
    if vertical:
        x, y, s = face_crop(cfg, W, H)
        cw = even(H * 9 / 16)
        x0 = even(clamp(x + s / 2 - cw / 2, 0, W - cw))
        g.chain(f"[{fi}:v]crop={cw}:{H}:{x0}:0,scale=1080:1920,setsar=1,fps={FPS},format=yuv420p[v0]")
    else:
        g.chain(f"[{fi}:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1,"
                f"fps={FPS},format=yuv420p[v0]")
    cur = "v0"
    od = overlays_dir(job)
    tag = "v" if vertical else "h"
    banner = od / f"title_{tag}.png"
    G.title_banner(tl["title"], cfg["brand"], banner, size, vertical)
    cur = g.overlay_png(cur, banner, D, 0.2, min(4.0, D - 3.5))
    for i, sl in enumerate(tl["slides"]):
        a = max(sl["start"], 4.3)
        b = min(a + 3.5, sl["end"], D - 3.4)
        if b - a < 1.2:
            continue
        lab = od / f"topic_{i:02d}_{tag}.png"
        G.topic_label(sl["title"], cfg["brand"], lab, size, vertical)
        cur = g.overlay_png(cur, lab, D, a, b)
    cur = add_captions(g, cur, job, "9x16" if vertical else "16x9", size,
                       y=int(1920 * 0.64) if vertical else 1080 - 190 - 55)
    end = od / f"end_{tag}.png"
    G.end_card(cfg["brand"], end, size, vertical)
    cur = g.overlay_png(cur, end, D, max(0.0, D - 3.2), D)
    out = job.p("out", f"{job.id}_head_{'9x16' if vertical else '16x9'}.mp4")
    encode(g, cur, ai, D, out)
    return out


def render_split(job, cfg, tl, face: Path, loop: bool, bg: Path, vertical: bool) -> Path:
    D = tl["duration"]
    info = video_info(face)
    W, H = info["width"], info["height"]
    x, y, s = face_crop(cfg, W, H)
    g = Graph()
    bi = g.add("-i", bg)
    fi = g.add(*(["-stream_loop", "-1"] if loop else []), "-i", face)
    ai = g.add("-i", job.p("narration.wav"))
    od = overlays_dir(job)
    tag = "v" if vertical else "h"
    if not vertical:
        B, M, R = cfg["layout"]["bubble_size"], cfg["layout"]["bubble_margin"], cfg["layout"]["bubble_ring"]
        mask, ring = od / "bubble_mask.png", od / "bubble_ring.png"
        G.bubble_mask(B, mask)
        pad = G.bubble_ring(B, R, ring)
        mi, ri = g.image(mask, D), g.image(ring, D)
        # a little wider than the face box so the bubble shows head + shoulders
        s2 = even(min(H, s * 1.25))
        x2 = even(clamp(x + s / 2 - s2 / 2, 0, W - s2))
        y2 = even(clamp(y + s / 2 - s2 / 2, 0, H - s2))
        g.chain(f"[{bi}:v]fps={FPS},scale=1920:1080,setsar=1,format=yuv420p[v0]")
        g.chain(f"[{fi}:v]crop={s2}:{s2}:{x2}:{y2},scale={B}:{B},fps={FPS},format=rgba[f]")
        g.chain(f"[{mi}:v]format=gray[m];[f][m]alphamerge[fc]")
        g.chain(f"[v0][fc]overlay={M}:{1080 - B - M}[v1]")
        g.chain(f"[v1][{ri}:v]overlay={M - pad}:{1080 - B - M - pad}[v2]")
        cur = "v2"
        if job.style == "loom":
            banner = od / "title_h.png"
            G.title_banner(tl["title"], cfg["brand"], banner, (1920, 1080), False)
            cur = g.overlay_png(cur, banner, D, 0.2, min(4.0, D - 3.5))
        cur = add_captions(g, cur, job, "16x9_split", (1920, 1080), y=1080 - 170 - 50, x=B + 2 * M + 20)
        size = (1920, 1080)
    else:
        sq = even(min(H, W, s * 1.9))
        x2 = even(clamp(x + s / 2 - sq / 2, 0, W - sq))
        y2 = even(clamp(y + s / 2 - sq / 2, 0, H - sq))
        bgc = cfg["brand"]["background"].lstrip("#")
        ci = g.add("-f", "lavfi", "-i", f"color=c=0x{bgc}:s=1080x1920:r={FPS}:d={D:.3f}")
        g.chain(f"[{bi}:v]fps={FPS},scale=1080:608,setsar=1[sc]")
        g.chain(f"[{fi}:v]crop={sq}:{sq}:{x2}:{y2},scale=1080:1080,setsar=1,fps={FPS}[fc]")
        g.chain(f"[{ci}:v][sc]overlay=0:262[v1];[v1][fc]overlay=0:870,format=yuv420p[v2]")
        banner = od / "title_v.png"
        G.title_banner(tl["title"], cfg["brand"], banner, (1080, 1920), True)
        bi2 = g.image(banner, D)
        g.chain(f"[{bi2}:v]format=rgba,crop=1080:262:0:0[bn];[v2][bn]overlay=0:0[v3]")
        cur = add_captions(g, "v3", job, "9x16_split", (1080, 1920), y=1920 - 230 - 150)
        size = (1080, 1920)
    end = od / f"end_{tag}.png"
    G.end_card(cfg["brand"], end, size, vertical)
    cur = g.overlay_png(cur, end, D, max(0.0, D - 3.2), D)
    out = job.p("out", f"{job.id}_{job.style}_{'9x16' if vertical else '16x9'}.mp4")
    encode(g, cur, ai, D, out)
    return out


def add_captions(g: Graph, cur: str, job, layout: str, size, y: int, x: int | None = None) -> str:
    lst = job.p("work", f"cap_{layout}", "list.ffconcat")
    if not lst.exists():
        say(f"  (no captions for {layout} — run `bin/ghl captions`)")
        return cur
    ci = g.add("-f", "concat", "-safe", "0", "-i", lst)
    o = g.label()
    xpos = f"{x}+({size[0] - x}-w)/2" if x is not None else "(W-w)/2"
    g.chain(f"[{ci}:v]fps={FPS},format=rgba[{o}c];[{cur}][{o}c]overlay={xpos}:{y}:eof_action=pass[{o}]")
    return o


def encode(g: Graph, vlabel: str, audio_idx: int, D: float, out: Path):
    say(f"  rendering {out.name} …")
    ffmpeg([*g.inputs, "-filter_complex", ";".join(g.chains), "-map", f"[{vlabel}]", "-map", f"{audio_idx}:a",
            "-t", f"{D:.3f}", *X264, *AAC, "-movflags", "+faststart", out])
    v = video_info(out)
    if abs(v["duration"] - D) > 0.15:
        say(f"  WARNING: {out.name} is {v['duration']:.2f}s, narration is {D:.2f}s")


def post_md(job, tl, outs, cfg):
    src = job.data.get("sources", [])
    lines = [f"# {tl['title']}", "", "## Files"] + [f"- {o.name}" for o in outs] + [
        f"- {job.id}.srt (upload as closed captions)", f"- thumbnail.png", "",
        "## Description (edit before posting)", "",
        f"{tl['title']}. Here's what's new in HighLevel this week:", ""]
    lines += [f"- {s.get('title')}: {s.get('link')}" for s in src] or ["- (add update links)"]
    lines += ["", "#gohighlevel #highlevel #ghl #marketingautomation", "",
              "## Before you post",
              "- [ ] YouTube: Settings → \"Altered or synthetic content\" = **Yes** (AI face/voice).",
              "- [ ] Instagram/Facebook: turn on the **AI info** label when uploading.",
              "- [ ] LinkedIn/TikTok: add \"Made with an AI version of me\" to the caption (TikTok: AI-generated toggle).",
              "- [ ] Watched it once end to end (facts, pronunciation, lip-sync)."]
    job.p("out", "post.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job")
    ap.add_argument("--preview", action="store_true", help="use the raw avatar clip instead of the lip-synced one")
    ap.add_argument("--only", choices=["16x9", "9x16"])
    ap.add_argument("--face", help="override the face video (testing)")
    a = ap.parse_args()
    job = current_job(a.job)
    cfg = load_config()
    tl = json.loads(job.p("timeline.json").read_text())
    synced = job.p("kaggle_out", f"lipsync_{job.id}.mp4")
    if a.face:
        face, loop = Path(a.face).expanduser(), True
        say(f"PREVIEW with test face video: {face}")
    elif synced.exists() and not a.preview:
        face, loop = synced, False
    else:
        face = ROOT / cfg["face"]["avatar"]
        if not face.exists():
            die(f"no face video: {face} (record + run `bin/ghl face-prep` first)")
        if not a.preview:
            die("no lip-synced video yet — run `bin/ghl lipsync`, or use --preview")
        face, loop = face, True
        say("PREVIEW: using the raw avatar clip (mouth will NOT match the voice)")

    outs = []
    verticals = [False, True] if not a.only else [a.only == "9x16"]
    if job.style == "head":
        for v in verticals:
            outs.append(render_head(job, cfg, tl, face, loop, v))
    else:
        bg = build_slides_bg(job, cfg, tl) if job.style == "slides" else build_screen_bg(job, tl)
        for v in verticals:
            outs.append(render_split(job, cfg, tl, face, loop, bg, v))

    # thumbnail from the face video at ~30%
    thumb_src = job.p("work", "thumb_src.png")
    ffmpeg(["-ss", f"{min(tl['duration'] * 0.3, 8):.2f}", *(["-stream_loop", "-1"] if loop else []), "-i", face,
            "-frames:v", "1", thumb_src])
    G.thumbnail(thumb_src, tl["title"], cfg["brand"], job.p("out", "thumbnail.png"))
    post_md(job, tl, outs, cfg)
    if not a.preview:
        job.stage_done("compose")
    say("done:")
    for o in outs + [job.p("out", "thumbnail.png"), job.p("out", "post.md")]:
        say(f"  {o}")


if __name__ == "__main__":
    main()
