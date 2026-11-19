"""Chatterbox TTS worker — runs inside envs/tts (separate from the tools env
because Chatterbox pins old torch/transformers versions).

  voice_tts.py --items items.json            # [{k, spoken, out, seed}], uses config voice settings
  voice_tts.py --text "Hi" --out a.wav [--ref r.wav] [--model original|turbo]

Prints one JSON line at the end with timing stats.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, hf_offline_first, load_config, prefer_ipv4, say  # noqa: E402

prefer_ipv4()


def pick_device(pref: str):
    import torch
    if pref in ("auto", "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(kind: str, device: str):
    return hf_offline_first(_load_model, kind, device)


def _load_model(kind: str, device: str):
    import torch
    orig_load = torch.load

    def load_on_device(*a, **kw):  # checkpoints were saved on CUDA
        kw.setdefault("map_location", torch.device(device))
        return orig_load(*a, **kw)

    torch.load = load_on_device
    try:
        if kind == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            return ChatterboxTurboTTS.from_pretrained(device=device)
        if kind == "multilingual":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            return ChatterboxMultilingualTTS.from_pretrained(device=device)
        from chatterbox.tts import ChatterboxTTS
        return ChatterboxTTS.from_pretrained(device=device)
    finally:
        torch.load = orig_load


class Voice:
    def __init__(self, kind: str, ref: str | None, device_pref: str, vcfg: dict):
        self.kind, self.ref, self.vcfg = kind, ref, vcfg
        self.device = pick_device(device_pref)
        t = time.time()
        try:
            self.model = load_model(kind, self.device)
        except Exception as e:  # noqa: BLE001
            if self.device == "cpu":
                raise
            say(f"MPS load failed ({e}); falling back to CPU")
            self.device = "cpu"
            self.model = load_model(kind, self.device)
        self.load_s = time.time() - t
        self.prepared = False

    def _kw(self):
        kw = {}
        if self.kind != "turbo":
            kw.update(exaggeration=self.vcfg["exaggeration"], cfg_weight=self.vcfg["cfg_weight"])
        if self.kind == "multilingual":
            kw["language_id"] = "en"
        return kw

    def set_ref(self, ref: str | None):
        if ref != self.ref:
            self.ref, self.prepared = ref, False

    def generate(self, text: str, seed: int):
        import torch
        torch.manual_seed(seed)
        kw = self._kw()
        if self.ref and not self.prepared:
            prep_kw = {"exaggeration": kw["exaggeration"]} if "exaggeration" in kw else {}
            try:
                self.model.prepare_conditionals(self.ref, **prep_kw)
                self.prepared = True
            except (AttributeError, TypeError):
                kw["audio_prompt_path"] = self.ref
        try:
            wav = self.model.generate(text, **kw)
        except RuntimeError as e:
            if self.device != "mps":
                raise
            say(f"MPS generation failed ({e}); switching to CPU")
            self.__init__(self.kind, self.ref, "cpu", self.vcfg)
            return self.generate(text, seed)
        return wav.detach().cpu(), self.model.sr


def save(wav, sr, out: Path):
    import soundfile as sf
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), wav.squeeze(0).numpy(), sr, subtype="PCM_16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items")
    ap.add_argument("--text")
    ap.add_argument("--out")
    ap.add_argument("--ref")
    ap.add_argument("--model")
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    vcfg = load_config()["voice"]
    kind = a.model or vcfg["model"]
    ref = a.ref if a.ref is not None else (str(ROOT / vcfg["ref"]) if (ROOT / vcfg["ref"]).exists() else None)
    if ref == "":
        ref = None
    v = Voice(kind, ref, vcfg["device"], vcfg)
    say(f"voice model '{kind}' on {v.device} (loaded in {v.load_s:.0f}s), reference: {ref or 'built-in default voice'}")

    items = json.loads(Path(a.items).read_text()) if a.items else [
        {"k": 0, "spoken": a.text, "out": a.out, "seed": a.seed if a.seed is not None else vcfg["seed"]}]
    stats = []
    for it in items:
        if "ref" in it:
            v.set_ref(it["ref"] or None)
        t = time.time()
        wav, sr = v.generate(it["spoken"], it["seed"])
        save(wav, sr, Path(it["out"]))
        secs = wav.shape[-1] / sr
        el = time.time() - t
        stats.append({"k": it["k"], "audio_s": round(secs, 2), "gen_s": round(el, 2)})
        say(f"  [{it['k']:>3}] {secs:5.1f}s audio in {el:5.1f}s  {it['spoken'][:70]}")
    tot_a = sum(s["audio_s"] for s in stats)
    tot_g = sum(s["gen_s"] for s in stats)
    print("STATS " + json.dumps({"device": v.device, "model": kind, "load_s": round(v.load_s, 1),
                                 "audio_s": round(tot_a, 1), "gen_s": round(tot_g, 1),
                                 "rtf": round(tot_g / tot_a, 2) if tot_a else None, "items": stats}), flush=True)


if __name__ == "__main__":
    main()
