"""Turn your raw 2-minute face recording into the base clip used for every video.

  face_prep.py ~/Movies/face_raw.mov            # auto-pick the steadiest window
  face_prep.py raw.mov --start 42 --length 30   # or choose it yourself

Writes assets/face/avatar.mp4 (25 fps, no audio), saves the face crop box to
config.yaml, and makes assets/face/review.jpg so you can check the pick.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import ASSETS, die, ffmpeg, ffmpeg_bin, save_config_value, say, video_info  # noqa: E402

SAMPLE_FPS = 2
SAMPLE_W = 640


def sample_faces(raw: Path, dur: float):
    import cv2
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    info = video_info(raw)
    h = int(round(info["height"] * SAMPLE_W / info["width"] / 2) * 2)
    cmd = [ffmpeg_bin(), "-v", "error", "-i", str(raw), "-vf", f"fps={SAMPLE_FPS},scale={SAMPLE_W}:{h},format=gray",
           "-f", "rawvideo", "-"]
    data = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(data, np.uint8).reshape(-1, h, SAMPLE_W)
    boxes = []
    for f in frames:
        det = cascade.detectMultiScale(f, scaleFactor=1.1, minNeighbors=6, minSize=(SAMPLE_W // 12, SAMPLE_W // 12))
        if len(det):
            x, y, w, hh = max(det, key=lambda d: d[2] * d[3])
            boxes.append((x + w / 2, y + hh / 2, w))
        else:
            boxes.append(None)
    return boxes, SAMPLE_W / info["width"]


def best_window(boxes, length: float, total: float):
    n = int(length * SAMPLE_FPS)
    if len(boxes) < n:
        return 0.0, 1e9
    best = (1e18, 0)
    for i in range(0, len(boxes) - n + 1):
        win = boxes[i:i + n]
        miss = sum(b is None for b in win) / n
        pts = np.array([b for b in win if b is not None], dtype=float)
        if len(pts) < n * 0.6:
            continue
        move = pts[:, :2].std(axis=0).sum() + pts[:, 2].std()
        jitter = np.abs(np.diff(pts[:, :2], axis=0)).mean() if len(pts) > 1 else 0
        a, b = win[0], win[-1]
        loop = (abs(a[0] - b[0]) + abs(a[1] - b[1])) if (a and b) else 50
        score = move + 2 * jitter + 0.5 * loop + 200 * miss
        if i / SAMPLE_FPS < 2:  # avoid the very start (reaching for the phone)
            score += 20
        best = min(best, (score, i))
    return best[1] / SAMPLE_FPS, best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--start", type=float)
    ap.add_argument("--length", type=float, default=30.0)
    a = ap.parse_args()
    raw = Path(a.raw).expanduser()
    if not raw.exists():
        die(f"not found: {raw}")
    info = video_info(raw)
    say(f"source: {info['width']}x{info['height']} @ {info['fps']:.2f} fps, {info['duration']:.1f}s")
    if info["duration"] < a.length + 1:
        die(f"recording is only {info['duration']:.0f}s — need at least {a.length + 1:.0f}s")

    boxes, scale = sample_faces(raw, info["duration"])
    found = sum(b is not None for b in boxes)
    say(f"face found in {found}/{len(boxes)} sampled frames")
    if found < len(boxes) * 0.5:
        die("couldn't see your face clearly in most of the clip — check lighting/framing")
    start = a.start if a.start is not None else best_window(boxes, a.length, info["duration"])[0]
    say(f"using {start:.1f}s – {start + a.length:.1f}s")

    out_h = 1440 if min(info["width"], info["height"]) >= 1440 else 1080
    out = ASSETS / "face" / "avatar.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg(["-ss", f"{start:.2f}", "-t", f"{a.length:.2f}", "-i", raw, "-an",
            "-vf", f"fps=25,scale=-2:{out_h}:flags=lanczos,format=yuv420p",
            "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-movflags", "+faststart", out])
    v = video_info(out)

    # face crop box in avatar.mp4 coordinates
    i0, i1 = int(start * SAMPLE_FPS), int((start + a.length) * SAMPLE_FPS)
    pts = np.array([b for b in boxes[i0:i1] if b is not None], dtype=float)
    k = v["height"] / (info["height"] * scale) if info["height"] else 1
    cx, cy, fw = np.median(pts, axis=0) * k
    size = int(min(v["height"], fw * 1.9)) // 2 * 2
    x = int(np.clip(cx - size / 2, 0, v["width"] - size)) // 2 * 2
    y = int(np.clip(cy - size / 2 + fw * 0.12, 0, v["height"] - size)) // 2 * 2
    save_config_value("face.crop", [x, y, size])
    say(f"face crop saved: x={x} y={y} size={size}")

    # review sheet: 6 frames across the window + the bubble crop
    review = ASSETS / "face" / "review.jpg"
    ffmpeg(["-i", out, "-vf", f"fps=6/{a.length},scale=640:-2,tile=3x2:padding=6:color=white", "-frames:v", "1", review])
    ffmpeg(["-ss", "5", "-i", out, "-vf", f"crop={size}:{size}:{x}:{y},scale=360:360", "-frames:v", "1",
            ASSETS / "face" / "bubble_preview.jpg"])
    say(f"base clip: {out}  ({v['width']}x{v['height']}, {v['duration']:.1f}s)")
    say(f"check: {review} and {ASSETS / 'face' / 'bubble_preview.jpg'}")


if __name__ == "__main__":
    main()
