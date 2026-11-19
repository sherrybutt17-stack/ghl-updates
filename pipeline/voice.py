"""Generate the narration for a job: TTS per sentence -> Whisper check -> retry bad ones -> assemble.

  voice.py                 # generate anything missing/changed, check, assemble
  voice.py --redo 3,7      # regenerate sentences 3 and 7 with a new take
  voice.py --no-check      # skip the Whisper accuracy check
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, current_job, die, load_config, load_pronunciations, say  # noqa: E402
from script_format import parse, spoken, text_hash  # noqa: E402

TTS_PY = ROOT / "envs" / "tts" / "bin" / "python"


def run_tts(items: list[dict], job) -> dict:
    if not TTS_PY.exists():
        die("voice environment missing — run `bin/ghl setup`")
    f = job.p("work", "tts_items.json")
    f.write_text(json.dumps(items, indent=1))
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1", TOKENIZERS_PARALLELISM="false")
    proc = subprocess.Popen([str(TTS_PY), str(ROOT / "pipeline" / "voice_tts.py"), "--items", str(f)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                            stdin=subprocess.DEVNULL)
    stats, tail = {}, []
    for line in proc.stdout:
        if line.startswith("STATS "):
            stats = json.loads(line[6:])
        elif not any(s in line for s in ("FutureWarning", "warnings.warn", "UserWarning", "it/s]", "s/it]")):
            print(line, end="", flush=True)
            tail = (tail + [line])[-30:]
    if proc.wait() != 0:
        die("voice generation failed:\n" + "".join(tail))
    return stats


def check(items: list[dict], manifest: dict, threshold: float) -> list[int]:
    from asr import transcribe, word_error_rate
    bad = []
    for it in items:
        m = manifest[it["k"]]
        hyp = transcribe(it["out"]).get("text", "").strip()
        wer = word_error_rate(m["spoken"], hyp)
        m["heard"], m["wer"] = hyp, round(wer, 3)
        flag = "OK " if wer <= threshold else "BAD"
        say(f"  check [{it['k']:>3}] {flag} {wer:4.0%} heard: {hyp[:80]}")
        if wer > threshold:
            bad.append(it["k"])
    gc.collect()
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job")
    ap.add_argument("--redo", default="", help="comma-separated sentence numbers to regenerate")
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    job = current_job(a.job)
    cfg = load_config()
    vcfg = cfg["voice"]
    if not job.p("script.md").exists():
        die(f"no script yet: {job.p('script.md')}")
    script = parse(job.p("script.md").read_text())
    pron = load_pronunciations()
    ref = ROOT / vcfg["ref"]
    ref_sig = f"{ref.stat().st_size}-{int(ref.stat().st_mtime)}" if ref.exists() else "default-voice"
    if not ref.exists():
        say("NOTE: no voice reference yet (assets/voice/ref.wav) — using Chatterbox's built-in voice for now.")

    mpath = job.p("tts", "manifest.json")
    old = {m["k"]: m for m in json.loads(mpath.read_text())["sentences"]} if mpath.exists() else {}
    redo = {int(x) for x in a.redo.split(",") if x.strip()}
    manifest, todo = {}, []
    for s in script.sentences:
        sp = spoken(s.text, pron)
        h = text_hash(sp, ref_sig, vcfg["model"], vcfg["exaggeration"], vcfg["cfg_weight"])
        prev = old.get(s.k)
        attempt = (prev or {}).get("attempt", 0)
        fresh = prev and prev.get("hash") == h and job.p("tts", prev["file"]).exists()
        if fresh and s.k not in redo:
            manifest[s.k] = prev
            continue
        attempt = attempt + 1 if (fresh and s.k in redo) else (attempt if prev and prev.get("hash") == h else 0)
        m = {"k": s.k, "text": s.text, "spoken": sp, "hash": h, "attempt": attempt,
             "file": f"seg_{s.k:03d}.wav"}
        manifest[s.k] = m
        todo.append({"k": s.k, "spoken": sp, "out": str(job.p("tts", m["file"])),
                     "seed": vcfg["seed"] + s.k * 7 + attempt * 1000})

    for k in [k for k in old if k not in manifest]:  # removed sentences
        job.p("tts", old[k]["file"]).unlink(missing_ok=True)

    rounds = 0
    while todo:
        say(f"Generating {len(todo)} sentence(s) in your voice…")
        stats = run_tts(todo, job)
        if stats:
            job.data.setdefault("timings", {})[f"tts_round{rounds + 1}"] = {
                k: stats[k] for k in ("device", "model", "load_s", "audio_s", "gen_s", "rtf")}
            job.save()
            say(f"  ({stats['audio_s']}s of audio in {stats['gen_s']}s on {stats['device']}, "
                f"model load {stats['load_s']}s)")
        if a.no_check:
            break
        bad = check(todo, manifest, vcfg["max_word_error"])
        rounds += 1
        retry = [k for k in bad if manifest[k]["attempt"] < vcfg["max_retries"] - 1]
        todo = []
        for k in retry:
            manifest[k]["attempt"] += 1
            todo.append({"k": k, "spoken": manifest[k]["spoken"], "out": str(job.p("tts", manifest[k]["file"])),
                         "seed": vcfg["seed"] + k * 7 + manifest[k]["attempt"] * 1000})
        if retry:
            say(f"Re-taking {len(retry)} sentence(s) that didn't come out right: {retry}")
        give_up = [k for k in bad if k not in retry]
        if give_up:
            say(f"WARNING: sentences {give_up} still sound off after {vcfg['max_retries']} takes — "
                f"listen to them, rephrase in script.md or add a fix to pronunciations.yaml.")

    mpath.write_text(json.dumps({"sentences": [manifest[k] for k in sorted(manifest)]}, indent=2,
                                ensure_ascii=False))
    import assemble
    tl = assemble.build(job, cfg)
    job.invalidate("approved_voice", "lipsync", "captions", "compose")
    job.stage_done("voice")
    say(f"\nNarration ready: {tl['duration']:.1f}s, {tl['words']} words -> {job.p('narration.wav')}")
    say("Listen to it, then approve with `bin/ghl approve-voice` (or redo sentences with `bin/ghl voice --redo N`).")


if __name__ == "__main__":
    main()
