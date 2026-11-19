"""Prepare a Loom-style screen recording for a job.

  screen.py ~/Movies/ghl_demo.mov

- normalises it to 1920x1080 @ 25 fps without audio (work/screen_norm.mp4)
- extracts frames at scene changes (+ at least every 4 s) and tiles them into
  labelled 3x3 contact sheets (frames/sheet_NN.jpg) so Claude can see what
  happens when and write a [mm:ss]-timed script.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import current_job, die, ffmpeg, ffmpeg_bin, say, video_info  # noqa: E402

MIN_EVERY = 4.0
MAX_FRAMES = 63


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--job")
    ap.add_argument("--threshold", type=float, default=0.06, help="scene-change sensitivity (lower = more frames)")
    a = ap.parse_args()
    job = current_job(a.job)
    src = Path(a.recording).expanduser()
    if not src.exists():
        die(f"not found: {src}")
    norm = job.p("work", "screen_norm.mp4")
    say("normalising screen recording to 1920x1080 @ 25 fps …")
    ffmpeg(["-i", src, "-an", "-vf", "fps=25,scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,format=yuv420p",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", norm])
    D = video_info(norm)["duration"]

    # scene-change timestamps
    res = subprocess.run([ffmpeg_bin(), "-hide_banner", "-i", str(norm), "-vf",
                          f"scale=480:-2,select='gt(scene,{a.threshold})',showinfo", "-f", "null", "-"],
                         capture_output=True, text=True)
    scenes = [float(m) for m in re.findall(r"pts_time:([\d.]+)", res.stderr)]
    times, t = [0.5], 0.5
    for s in sorted(scenes + [x * MIN_EVERY for x in range(1, int(D / MIN_EVERY) + 1)]):
        if s - times[-1] >= 1.5 and s < D - 0.3:
            times.append(round(s + 0.4, 2))  # a beat after the change, once the UI settles
    while len(times) > MAX_FRAMES:
        times = times[::2]
    fdir = job.p("frames")
    shutil.rmtree(fdir, ignore_errors=True)
    fdir.mkdir(parents=True)
    for i, ts in enumerate(times):
        mm, ss = divmod(ts, 60)
        ffmpeg(["-ss", f"{ts:.2f}", "-i", norm, "-frames:v", "1", "-vf",
                f"scale=640:360,pad=640:400:0:40:color=black", fdir / f"f_{i:03d}.png"])
        _label(fdir / f"f_{i:03d}.png", f"{int(mm):02d}:{ss:04.1f}")
    sheets = []
    for n in range(0, len(times), 9):
        out = fdir / f"sheet_{n // 9 + 1:02d}.jpg"
        lst = fdir / "tile.txt"
        files = [fdir / f"f_{i:03d}.png" for i in range(n, min(n + 9, len(times)))]
        lst.write_text("".join(f"file '{f.name}'\n" for f in files))
        ffmpeg(["-f", "concat", "-safe", "0", "-i", lst, "-vf", "tile=3x3:padding=8:color=white",
                "-frames:v", "1", "-q:v", "3", out])
        sheets.append(out)
    (fdir / "index.json").write_text(json.dumps({"duration": D, "frames": times}, indent=1))
    job.set(screen=str(src))
    say(f"screen recording: {D:.1f}s, {len(times)} key frames in {len(sheets)} contact sheet(s):")
    for s in sheets:
        say(f"  {s}")


def _label(png: Path, text: str):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(png).convert("RGB")
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 28)
    except OSError:
        f = ImageFont.load_default()
    d.text((12, 4), text, font=f, fill=(255, 220, 0))
    im.save(png)


if __name__ == "__main__":
    main()
