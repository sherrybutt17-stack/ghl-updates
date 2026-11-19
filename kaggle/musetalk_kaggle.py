# MuseTalk 1.5 lip-sync on a free Kaggle T4.
# This file is a TEMPLATE: pipeline/kaggle_lipsync.py fills in the __PLACEHOLDERS__
# and pushes it as a Kaggle script kernel.
#
#   MODE = "setup"  -> build a Python 3.10 + torch 2.0.1/cu118 + mmcv env with MuseTalk and
#                      its weights, run a 3 s test, save everything as mt_bundle.tar (output)
#   MODE = "render" -> unpack that bundle (or build it if missing), lip-sync the job's
#                      narration onto avatar.mp4, write lipsync_<JOB_ID>.mp4 + status.json
#
# Licences (all allow commercial use): MuseTalk code MIT, weights "any purpose";
# sd-vae-ft-mse MIT; whisper-tiny MIT; DWPose Apache-2.0; face-parsing MIT; face-alignment BSD.
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

MODE = "__MODE__"
JOB_ID = "__JOB_ID__"
OPTS = json.loads('__OPTS__')  # extra_margin, left/right_cheek_width, parsing_mode, batch_size

MUSETALK_COMMIT = "0a89dec45a"
MT = "/tmp/mt"
VENV = f"{MT}/venv"
PY = f"{VENV}/bin/python"
REPO = f"{MT}/MuseTalk"
WORK = "/kaggle/working"
LOG = f"{WORK}/log.txt"
T0 = time.time()


def log(msg):
    line = f"[{time.time() - T0:7.1f}s] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def sh(cmd, cwd=None, env=None, check=True):
    log(f"$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=cwd, env=env, shell=isinstance(cmd, str), stdin=subprocess.DEVNULL,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(LOG, "a") as f:
        f.write(p.stdout[-20000:])
    if check and p.returncode != 0:
        print(p.stdout[-5000:])
        raise RuntimeError(f"command failed ({p.returncode})")
    return p.stdout


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return sorted(hits, key=len)[0] if hits else None


def status(ok, **kw):
    with open(f"{WORK}/status.json", "w") as f:
        json.dump({"ok": ok, "mode": MODE, "job_id": JOB_ID, "seconds": round(time.time() - T0, 1), **kw}, f, indent=2)


# ---------------------------------------------------------------- environment
def build_env():
    log("building MuseTalk environment (one-time, ~10-15 min)")
    os.makedirs(MT, exist_ok=True)
    sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
    uv = shutil.which("uv") or os.path.join(os.path.dirname(sys.executable), "uv")
    env = dict(os.environ, UV_PYTHON_INSTALL_DIR=f"{MT}/py", UV_CACHE_DIR="/tmp/uvcache", UV_HTTP_TIMEOUT="120")
    sh([uv, "venv", "--python", "3.10", VENV], env=env)
    pip = [uv, "pip", "install", "--python", PY]
    sh(pip + ["torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2",
              "--index-url", "https://download.pytorch.org/whl/cu118"], env=env)
    sh(pip + ["--index-strategy", "unsafe-best-match", "numpy==1.23.5", "openmim", "mmengine==0.10.4",
              "mmcv==2.0.1", "-f", "https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html"], env=env)
    sh(pip + ["mmdet==3.1.0"], env=env)
    try:
        sh(pip + ["mmpose==1.1.0"], env=env)
    except RuntimeError:  # chumpy sometimes needs no build isolation
        sh(pip + ["--no-build-isolation", "chumpy==0.70"], env=env)
        sh(pip + ["mmpose==1.1.0"], env=env)
    sh(["git", "clone", "-q", "https://github.com/TMElyralab/MuseTalk.git", REPO])
    sh(["git", "checkout", "-q", MUSETALK_COMMIT], cwd=REPO)
    reqs = [l.strip() for l in open(f"{REPO}/requirements.txt") if l.strip() and not l.startswith("#")]
    reqs = [r for r in reqs if not r.lower().startswith(("tensorflow", "tensorboard", "gradio", "torch"))]
    sh(pip + reqs, env=env)
    download_weights(env)


def download_weights(env):
    m = f"{REPO}/models"
    code = f"""
import os
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
from huggingface_hub import hf_hub_download, snapshot_download
m = {m!r}
for f in ['musetalkV15/musetalk.json', 'musetalkV15/unet.pth']:
    hf_hub_download('TMElyralab/MuseTalk', f, local_dir=m)
snapshot_download('stabilityai/sd-vae-ft-mse', local_dir=m + '/sd-vae', allow_patterns=['config.json', 'diffusion_pytorch_model.bin'])
snapshot_download('openai/whisper-tiny', local_dir=m + '/whisper', allow_patterns=['config.json', 'pytorch_model.bin', 'preprocessor_config.json'])
hf_hub_download('yzd-v/DWPose', 'dw-ll_ucoco_384.pth', local_dir=m + '/dwpose')
print('hf weights ok')
"""
    sh([PY, "-c", code], env=dict(env, HF_ENDPOINT="https://huggingface.co"))
    fp = f"{m}/face-parse-bisent"
    os.makedirs(fp, exist_ok=True)
    # face-parsing weights: prefer a copy uploaded to the ghl-avatar dataset, else Google Drive
    for name, fallback in [("79999_iter.pth", None),
                           ("resnet18-5c106cde.pth", "https://download.pytorch.org/models/resnet18-5c106cde.pth")]:
        src = find(name)
        if src:
            shutil.copy(src, f"{fp}/{name}")
        elif fallback:
            sh(["curl", "-sSL", "-o", f"{fp}/{name}", fallback])
        else:
            sh([PY, "-m", "pip", "install", "-q", "gdown"], check=False)
            sh([f"{VENV}/bin/gdown", "--id", "154JgKpzCPW82qINcVieuPH3fZ2e0P812", "-O", f"{fp}/{name}"])
    log("weights ready")


def save_bundle():
    tar = f"{WORK}/mt_bundle.tar"
    hub = os.path.expanduser("~/.cache/torch")
    paths = ["tmp/mt"] + (["root/.cache/torch"] if os.path.isdir(hub) else [])
    sh(["tar", "-cf", tar, "--exclude=tmp/mt/MuseTalk/results", "-C", "/", *paths])
    log(f"bundle saved: {os.path.getsize(tar) / 1e9:.1f} GB")


def restore_bundle():
    tar = find("mt_bundle.tar")
    if not tar:
        return False
    log(f"restoring environment from {tar}")
    sh(["tar", "-xf", tar, "-C", "/"])
    return os.path.exists(PY)


# ---------------------------------------------------------------- inference
def lipsync(video, audio, name):
    cfg = f"{MT}/job.yaml"
    with open(cfg, "w") as f:
        f.write(f'task_0:\n  video_path: "{video}"\n  audio_path: "{audio}"\n')
    out_dir = f"{MT}/results"
    shutil.rmtree(out_dir, ignore_errors=True)
    cmd = [PY, "-m", "scripts.inference", "--inference_config", cfg, "--result_dir", out_dir,
           "--unet_model_path", "models/musetalkV15/unet.pth", "--unet_config", "models/musetalkV15/musetalk.json",
           "--version", "v15", "--use_float16", "--batch_size", str(OPTS.get("batch_size", 16)),
           "--extra_margin", str(OPTS.get("extra_margin", 10)),
           "--parsing_mode", OPTS.get("parsing_mode", "jaw"),
           "--left_cheek_width", str(OPTS.get("left_cheek_width", 90)),
           "--right_cheek_width", str(OPTS.get("right_cheek_width", 90)),
           "--output_vid_name", f"{name}.mp4"]
    env = dict(os.environ, PYTHONPATH=REPO, HF_ENDPOINT="https://huggingface.co")
    sh(cmd, cwd=REPO, env=env)
    hits = glob.glob(f"{out_dir}/**/{name}.mp4", recursive=True) or glob.glob(f"{out_dir}/**/*.mp4", recursive=True)
    if not hits:
        raise RuntimeError("MuseTalk finished but produced no video")
    return hits[0]


def gpu_info():
    try:
        return sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main():
    open(LOG, "w").close()
    log(f"mode={MODE} job={JOB_ID} gpu={gpu_info()}")
    if MODE == "setup":
        build_env()
        avatar = find("avatar.mp4")
        test_wav = f"{MT}/test.wav"
        sh(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=220:duration=3", "-ar", "16000", test_wav])
        if avatar:
            clip = f"{MT}/test_clip.mp4"
            sh(["ffmpeg", "-y", "-v", "error", "-i", avatar, "-t", "4", "-an", clip])
            out = lipsync(clip, test_wav, "setup_test")
            shutil.copy(out, f"{WORK}/setup_test.mp4")
        save_bundle()
        status(True, gpu=gpu_info(), tested=bool(avatar))
        return

    job_json = find("job.json")
    if not job_json:
        raise RuntimeError("job.json not found in /kaggle/input")
    job = json.load(open(job_json))
    if job.get("job_id") != JOB_ID:
        raise RuntimeError(f"stale job dataset: has {job.get('job_id')!r}, expected {JOB_ID!r} — "
                           "Kaggle served an old dataset version; re-run")
    audio = os.path.join(os.path.dirname(job_json), "narration.wav")
    avatar = find("avatar.mp4")
    if not avatar or not os.path.exists(audio):
        raise RuntimeError(f"missing inputs: avatar={avatar} audio={os.path.exists(audio)}")
    if not restore_bundle():
        build_env()
    out = lipsync(avatar, audio, f"lipsync_{JOB_ID}")
    final = f"{WORK}/lipsync_{JOB_ID}.mp4"
    # re-mux with the narration so the Mac can verify sync; video stream copied as-is
    sh(["ffmpeg", "-y", "-v", "error", "-i", out, "-i", audio, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
        "-c:a", "aac", "-shortest", final])
    status(True, gpu=gpu_info(), output=os.path.basename(final))
    log("done")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        log("FAILED: " + "".join(traceback.format_exception(e))[-4000:])
        status(False, error=str(e)[-2000:])
        raise
