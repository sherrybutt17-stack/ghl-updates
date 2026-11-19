"""Cut 3 candidate voice references (12-15 s) from your raw voice recording.

  voice_prep.py ~/Desktop/voice_raw.m4a
  voice_prep.py --pick 2        # make candidate 2 the reference (assets/voice/ref.wav)

Chatterbox only listens to the first ~10 s of the reference, so each candidate
starts right at speech and is dense with clean speech.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from common import ASSETS, die, ffmpeg_bin, say  # noqa: E402

SR = 24000
VOICE = ASSETS / "voice"
CAND = VOICE / "candidates"


def load(path: Path) -> np.ndarray:
    out = subprocess.run([ffmpeg_bin(), "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, np.float32).copy()


def frame_db(x: np.ndarray, hop: int) -> np.ndarray:
    n = len(x) // hop
    r = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    return 20 * np.log10(r)


def candidates(x: np.ndarray, length=13.5, count=3):
    hop = SR // 50  # 20 ms
    db = frame_db(x, hop)
    noise = np.percentile(db, 10)
    speech = db > max(noise + 12, -45)
    clip = np.abs(x) > 0.98
    clip_f = np.add.reduceat(clip, np.arange(0, len(clip), hop))[: len(db)] > 0
    n = int(length * 50)
    first10 = int(10 * 50)
    scored = []
    for i in range(0, len(db) - n, 5):  # 100 ms steps
        if not speech[i] or (i > 0 and speech[i - 1] and speech[max(0, i - 10):i].all()):
            continue  # start at a speech onset (after a short pause)
        dens = speech[i:i + first10].mean()
        if clip_f[i:i + n].any():
            continue
        level = db[i:i + n][speech[i:i + n]]
        consistency = level.std() if len(level) else 99
        # long silences inside the first 10 s waste the reference
        gaps = np.diff(np.where(np.concatenate([[1], speech[i:i + first10].astype(int), [1]]) == 1)[0]).max()
        score = dens * 10 - consistency * 0.15 - max(0, gaps - 25) * 0.05 + (noise < -55) * 0.5
        scored.append((score, i))
    scored.sort(reverse=True)
    picked = []
    for s, i in scored:
        if all(abs(i - j) > n for _, j in picked):
            picked.append((s, i))
        if len(picked) == count:
            break
    return [(i / 50, length, s, noise) for s, i in picked]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="?")
    ap.add_argument("--pick", type=int)
    a = ap.parse_args()
    if a.pick:
        src = CAND / f"cand_{a.pick}.wav"
        if not src.exists():
            die(f"no candidate {a.pick} — run voice-prep first")
        shutil.copy(src, VOICE / "ref.wav")
        say(f"voice reference set: {VOICE / 'ref.wav'} (from candidate {a.pick})")
        return
    if not a.raw:
        die("give the raw voice recording path")
    x = load(Path(a.raw).expanduser())
    say(f"recording: {len(x) / SR:.1f}s, peak {20 * np.log10(np.abs(x).max() + 1e-9):.1f} dBFS")
    if len(x) / SR < 20:
        die("recording is too short — read for 60-90 seconds")
    cands = candidates(x)
    if not cands:
        die("couldn't find a clean 13 s stretch (clipping or too many pauses?) — re-record a bit quieter/steadier")
    CAND.mkdir(parents=True, exist_ok=True)
    for f in CAND.glob("cand_*.wav"):
        f.unlink()
    for n, (start, length, score, noise) in enumerate(cands, 1):
        seg = x[int(start * SR): int((start + length) * SR)]
        seg = seg / (np.abs(seg).max() + 1e-9) * 0.7
        fade = int(0.02 * SR)
        seg[:fade] *= np.linspace(0, 1, fade)
        seg[-fade:] *= np.linspace(1, 0, fade)
        sf.write(CAND / f"cand_{n}.wav", seg, SR, subtype="PCM_16")
        say(f"  candidate {n}: {start:6.1f}s – {start + length:.1f}s  (score {score:.1f}, room noise {noise:.0f} dB)")
    say(f"saved to {CAND}. Next: `bin/ghl voice-test` to hear each one cloned.")


if __name__ == "__main__":
    main()
