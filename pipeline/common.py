"""Shared helpers: paths, config, jobs, subprocess, ffprobe."""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOBS = ROOT / "jobs"
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
CURRENT = JOBS / ".current"

STAGES = ["created", "script", "voice", "approved_voice", "lipsync", "captions", "compose", "done"]


# ---------- config ----------
def _yaml():
    import yaml  # available in both envs
    return yaml


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return _yaml().safe_load(f)


def save_config_value(dotted: str, value) -> None:
    """Update one top-level or nested scalar in config.yaml without losing comments."""
    path = ROOT / "config.yaml"
    text = path.read_text()
    key = dotted.split(".")[-1]
    rendered = json.dumps(value) if not isinstance(value, str) else f'"{value}"'
    new, n = re.subn(rf"(?m)^(\s*{re.escape(key)}:\s*)([^#\n]*?)(\s*(#.*)?)$",
                     lambda m: f"{m.group(1)}{rendered}{m.group(3)}", text, count=1)
    if n == 0:
        raise KeyError(dotted)
    path.write_text(new)


def load_pronunciations() -> dict:
    p = ROOT / "pronunciations.yaml"
    if not p.exists():
        return {}
    return _yaml().safe_load(p.read_text()) or {}


# ---------- network ----------
def prefer_ipv4():
    """This connection's IPv6 route stalls; resolve IPv4 addresses first (per process)."""
    import socket
    if getattr(socket, "_ghl_ipv4_first", False):
        return
    orig = socket.getaddrinfo

    def gai(*args, **kw):
        return sorted(orig(*args, **kw), key=lambda r: r[0] != socket.AF_INET)

    socket.getaddrinfo = gai
    socket._ghl_ipv4_first = True


def hf_offline_first(fn, *args, **kw):
    """Run fn using only the local Hugging Face cache; if something isn't cached, allow downloads."""
    import huggingface_hub.constants as C
    prev = C.HF_HUB_OFFLINE
    C.HF_HUB_OFFLINE = True
    try:
        return fn(*args, **kw)
    except Exception as e:  # noqa: BLE001
        if "offline" not in str(e).lower() and "LocalEntryNotFound" not in type(e).__name__ \
                and "cache" not in str(e).lower():
            raise
        prefer_ipv4()
        C.HF_HUB_OFFLINE = False
        return fn(*args, **kw)
    finally:
        C.HF_HUB_OFFLINE = prev


# ---------- logging / subprocess ----------
def say(msg: str) -> None:
    print(msg, flush=True)


def die(msg: str, code: int = 1):
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def run(cmd: list, *, cwd=None, check=True, capture=False, env=None) -> subprocess.CompletedProcess:
    cmd = [str(c) for c in cmd]
    res = subprocess.run(cmd, cwd=cwd, env=env, text=True,
                         stdout=subprocess.PIPE if capture else None,
                         stderr=subprocess.PIPE if capture else None,
                         stdin=subprocess.DEVNULL)
    if check and res.returncode != 0:
        err = (res.stderr or "")[-3000:] if capture else ""
        die(f"command failed ({res.returncode}): {' '.join(cmd[:6])} …\n{err}")
    return res


def ffmpeg_bin() -> str:
    return load_config()["tools"]["ffmpeg"]


def ffprobe_bin() -> str:
    return load_config()["tools"]["ffprobe"]


def ffmpeg(args: list, **kw):
    return run([ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", *args], **kw)


def probe(path) -> dict:
    res = run([ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_format",
               "-show_streams", str(path)], capture=True)
    return json.loads(res.stdout)


def duration(path) -> float:
    return float(probe(path)["format"]["duration"])


def video_info(path) -> dict:
    info = probe(path)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = (int(x) for x in v["r_frame_rate"].split("/"))
    return {"width": int(v["width"]), "height": int(v["height"]), "fps": num / den,
            "duration": float(info["format"]["duration"])}


# ---------- jobs ----------
def slugify(text: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:maxlen].strip("-") or "update"


class Job:
    def __init__(self, path: Path):
        self.dir = Path(path)
        self.file = self.dir / "job.json"
        self.data = json.loads(self.file.read_text()) if self.file.exists() else {}

    # paths
    def p(self, *parts) -> Path:
        return self.dir.joinpath(*parts)

    @property
    def id(self) -> str:
        return self.data["job_id"]

    @property
    def style(self) -> str:
        return self.data.get("style", "head")

    def save(self):
        self.file.write_text(json.dumps(self.data, indent=2))

    def set(self, **kw):
        self.data.update(kw)
        self.save()

    def stage_done(self, stage: str):
        done = self.data.setdefault("stages_done", [])
        if stage not in done:
            done.append(stage)
        self.data["stage"] = stage
        self.data.setdefault("stage_times", {})[stage] = dt.datetime.now().isoformat(timespec="seconds")
        self.save()

    def has(self, stage: str) -> bool:
        return stage in self.data.get("stages_done", [])

    def invalidate(self, *stages: str):
        done = self.data.get("stages_done", [])
        self.data["stages_done"] = [s for s in done if s not in stages]
        self.save()


def create_job(title: str, style: str, screen: str | None = None) -> Job:
    JOBS.mkdir(exist_ok=True)
    date = dt.date.today().isoformat()
    base = f"{date}-{slugify(title)}"
    d = JOBS / base
    i = 2
    while d.exists():
        d = JOBS / f"{base}-{i}"
        i += 1
    for sub in ("tts", "kaggle_in", "kaggle_out", "frames", "out", "work"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    job = Job(d)
    job.data = {"job_id": d.name, "title": title, "style": style, "screen": screen,
                "created": dt.datetime.now().isoformat(timespec="seconds"), "stages_done": []}
    job.stage_done("created")
    CURRENT.write_text(d.name)
    return job


def current_job(arg: str | None = None) -> Job:
    if arg:
        d = Path(arg)
        if not d.is_absolute():
            d = JOBS / arg if (JOBS / arg).exists() else ROOT / arg
    else:
        if not CURRENT.exists():
            die("no current job — run `bin/ghl new` first")
        d = JOBS / CURRENT.read_text().strip()
    if not (d / "job.json").exists():
        die(f"not a job folder: {d}")
    return Job(d)


@contextlib.contextmanager
def lock(name: str):
    lf = ROOT / f".{name}.lock"
    if lf.exists():
        pid = lf.read_text().strip()
        try:
            os.kill(int(pid), 0)
            die(f"another `{name}` is running (pid {pid}). Delete {lf} if that's wrong.")
        except (ValueError, ProcessLookupError):
            pass  # stale lock
    lf.write_text(str(os.getpid()))
    try:
        yield
    finally:
        lf.unlink(missing_ok=True)
