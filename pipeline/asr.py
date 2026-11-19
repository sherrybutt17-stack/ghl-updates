"""Speech recognition helpers (mlx-whisper, runs on the Apple GPU)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import hf_offline_first, load_config  # noqa: E402

_NUM = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six",
        "7": "seven", "8": "eight", "9": "nine", "10": "ten"}


def norm_tokens(text: str) -> list[str]:
    t = text.lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return [_NUM.get(w, w) for w in t.split()]


def transcribe(path, words: bool = False) -> dict:
    import mlx_whisper
    model = load_config()["captions"]["asr_model"]
    return hf_offline_first(mlx_whisper.transcribe, str(path), path_or_hf_repo=model, language="en",
                            word_timestamps=words, condition_on_previous_text=False, verbose=None)


def word_error_rate(ref: str, hyp: str) -> float:
    r, h = norm_tokens(ref), norm_tokens(hyp)
    if not r:
        return 0.0
    # Levenshtein on tokens
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1] / len(r)
