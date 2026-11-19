"""Run MuseTalk lip-sync on Kaggle's free GPU, driven from the Mac with the kaggle CLI.

  kaggle_lipsync.py --setup     # one-time: upload avatar, build + test the GPU environment
  kaggle_lipsync.py             # lip-sync the current job's narration.wav

Everything on Kaggle is private: 2 private datasets (ghl-avatar, ghl-job) and
2 private notebooks (ghl-musetalk-env, ghl-musetalk-render).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (ROOT, current_job, die, duration, load_config, lock, save_config_value,  # noqa: E402
                    say)

KAGGLE = ROOT / "envs" / "tools" / "bin" / "kaggle"
KDIR = ROOT / "kaggle"
STATE = ROOT / "data" / "kaggle_state.json"
TEMPLATE = KDIR / "musetalk_kaggle.py"


def kg(*args, check=True, quiet=False) -> str:
    env = dict(os.environ, PYTHONWARNINGS="ignore")
    p = subprocess.run([str(KAGGLE), *map(str, args)], capture_output=True, text=True, env=env,
                       stdin=subprocess.DEVNULL)
    out = (p.stdout + p.stderr).strip()
    if check and p.returncode != 0:
        die(f"kaggle {' '.join(map(str, args[:3]))} failed:\n{out[-2000:]}")
    if not quiet and out:
        say("  " + out.splitlines()[-1][:200])
    return out


def state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(s: dict):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2))


def username(cfg) -> str:
    if cfg.get("kaggle_user"):
        return cfg["kaggle_user"]
    if not KAGGLE.exists():
        die("kaggle CLI missing — run `bin/ghl setup`")
    out = kg("config", "view", check=False, quiet=True)
    m = re.search(r"username:\s*(\S+)", out)
    user = m.group(1) if m and m.group(1) != "None" else None
    if not user:
        kj = Path.home() / ".kaggle" / "kaggle.json"
        if kj.exists():
            user = json.loads(kj.read_text()).get("username")
    if not user:
        die("not logged in to Kaggle — run `envs/tools/bin/kaggle auth login` (see SETUP in CLAUDE.md)")
    save_config_value("kaggle_user", user)
    return user


def sha(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_exists(ref: str) -> bool:
    out = kg("datasets", "status", ref, check=False, quiet=True)
    return "ready" in out.lower() or "pending" in out.lower()


def wait_dataset(ref: str, timeout=900):
    t = time.time()
    while time.time() - t < timeout:
        out = kg("datasets", "status", ref, check=False, quiet=True).lower()
        if "ready" in out:
            return
        if "error" in out:
            die(f"dataset {ref} failed to process: {out}")
        time.sleep(10)
    die(f"dataset {ref} still processing after {timeout}s")


def push_dataset(user: str, slug: str, files: dict[str, Path], msg: str):
    d = KDIR / f"ds_{slug}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    for name, src in files.items():
        try:
            os.link(src, d / name)
        except OSError:
            shutil.copy(src, d / name)
    ref = f"{user}/{slug}"
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": slug, "id": ref, "licenses": [{"name": "other"}]}, indent=1))
    if dataset_exists(ref):
        say(f"updating private dataset {ref} …")
        kg("datasets", "version", "-p", d, "-m", msg, "-d", "-q")
    else:
        say(f"creating private dataset {ref} …")
        kg("datasets", "create", "-p", d, "-q")
    time.sleep(5)
    wait_dataset(ref)
    shutil.rmtree(d, ignore_errors=True)
    return ref


def push_kernel(user: str, slug: str, mode: str, job_id: str, datasets: list[str], kernels: list[str], opts: dict):
    d = KDIR / f"k_{slug}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    src = TEMPLATE.read_text().replace("__MODE__", mode).replace("__JOB_ID__", job_id) \
        .replace("__OPTS__", json.dumps(opts))
    (d / "musetalk_kaggle.py").write_text(src)
    meta = {"id": f"{user}/{slug}", "title": slug, "code_file": "musetalk_kaggle.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": datasets, "kernel_sources": kernels, "competition_sources": [], "model_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    say(f"starting Kaggle GPU job {user}/{slug} ({mode}) …")
    kg("kernels", "push", "-p", d, "-t", "3600")
    return f"{user}/{slug}"


def wait_kernel(ref: str, timeout=3600) -> str:
    t, last = time.time(), ""
    time.sleep(20)
    while time.time() - t < timeout:
        out = kg("kernels", "status", ref, check=False, quiet=True).lower()
        st = next((s for s in ("complete", "error", "cancel", "running", "queued") if s in out), out[-60:])
        if st != last:
            say(f"  [{(time.time() - t) / 60:4.1f} min] {st}")
            last = st
        if st in ("complete", "error", "cancel"):
            return st
        time.sleep(45)
    die(f"Kaggle job {ref} still running after {timeout // 60} min — check kaggle.com/code")


def fetch_outputs(ref: str, dest: Path, pattern: str):
    dest.mkdir(parents=True, exist_ok=True)
    kg("kernels", "output", ref, "-p", dest, "-o", "--file-pattern", pattern, quiet=True)


def setup(cfg):
    user = username(cfg)
    avatar = ROOT / cfg["face"]["avatar"]
    st = state()
    files = {}
    if avatar.exists():
        files["avatar.mp4"] = avatar
    for extra in ("79999_iter.pth", "resnet18-5c106cde.pth"):  # optional local copies of face-parsing weights
        p = ROOT / "models" / extra
        if p.exists():
            files[extra] = p
    if not files:
        die("record your face clip first (bin/ghl face-prep) — the setup test uses it")
    sig = sha(avatar) if avatar.exists() else "none"
    if st.get("avatar_sha") != sig or not dataset_exists(f"{user}/ghl-avatar"):
        push_dataset(user, "ghl-avatar", files, "avatar update")
        st["avatar_sha"] = sig
        save_state(st)
    ref = push_kernel(user, "ghl-musetalk-env", "setup", "setup", [f"{user}/ghl-avatar"], [], cfg["face"]["musetalk"])
    result = wait_kernel(ref, timeout=5400)
    out = ROOT / "data" / "kaggle_setup"
    fetch_outputs(ref, out, r"status\.json|log\.txt|setup_test\.mp4")
    s = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else {}
    if result != "complete" or not s.get("ok"):
        die(f"GPU environment build failed — see {out / 'log.txt'}\n{s.get('error', '')}")
    st["env_ready"] = True
    save_state(st)
    say(f"GPU environment ready on {s.get('gpu')} (took {s.get('seconds', 0) / 60:.0f} min). "
        f"Test clip: {out / 'setup_test.mp4'}")


def lipsync(job, cfg):
    user = username(cfg)
    st = state()
    if not st.get("env_ready"):
        die("GPU environment not built yet — run `bin/ghl kaggle-setup` once")
    avatar = ROOT / cfg["face"]["avatar"]
    if st.get("avatar_sha") != sha(avatar):
        push_dataset(user, "ghl-avatar", {"avatar.mp4": avatar}, "avatar update")
        st["avatar_sha"] = sha(avatar)
        save_state(st)
    narr = job.p("kaggle_in", "narration.wav")
    if not narr.exists():
        die("no narration yet — run `bin/ghl voice`")
    jj = job.p("kaggle_in", "job.json")
    jj.write_text(json.dumps({"job_id": job.id}))
    t0 = time.time()
    push_dataset(user, "ghl-job", {"narration.wav": narr, "job.json": jj}, job.id)
    ref = push_kernel(user, "ghl-musetalk-render", "render", job.id,
                      [f"{user}/ghl-avatar", f"{user}/ghl-job"], [f"{user}/ghl-musetalk-env"], cfg["face"]["musetalk"])
    result = wait_kernel(ref)
    out = job.p("kaggle_out")
    fetch_outputs(ref, out, rf"lipsync_{re.escape(job.id)}\.mp4|status\.json|log\.txt")
    s = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else {}
    video = out / f"lipsync_{job.id}.mp4"
    if result != "complete" or not s.get("ok") or not video.exists():
        die(f"lip-sync failed — see {out / 'log.txt'}\n{s.get('error', '')}")
    dv, da = duration(video), duration(narr)
    if abs(dv - da) > 0.25:
        say(f"WARNING: lip-sync video is {dv:.2f}s but narration is {da:.2f}s")
    job.data.setdefault("timings", {})["lipsync_total_s"] = round(time.time() - t0)
    job.data["timings"]["lipsync_gpu_s"] = s.get("seconds")
    job.save()
    job.stage_done("lipsync")
    say(f"lip-synced video: {video} ({dv:.1f}s, {(time.time() - t0) / 60:.0f} min end to end)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--job")
    a = ap.parse_args()
    cfg = load_config()
    with lock("kaggle"):
        if a.setup:
            setup(cfg)
        else:
            job = current_job(a.job)
            if not job.has("approved_voice"):
                die("approve the voice first (`bin/ghl approve-voice`) — GPU time is only spent on approved audio")
            lipsync(job, cfg)


if __name__ == "__main__":
    main()
