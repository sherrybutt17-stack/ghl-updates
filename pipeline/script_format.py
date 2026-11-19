"""Parse script.md (the human-approved script) into paragraphs and sentences.

Format
------
    # Video title                                  <- one title line
    <!-- comments are ignored -->

    [slide] Managed Agents API | Build agents from code; Publish in one call
    [00:12] First up, HighLevel launched the Managed Agents API. ...

    Next paragraph (blank line = new paragraph) ...

    ---                                            <- everything below is notes/sources, not spoken
    Sources: https://...

- `[slide] Title | subtitle; bullet; bullet` starts a new slide (slides style),
  or a topic label (talking-head style), from the next sentence on.
- `[mm:ss]` at the start of a paragraph = where that paragraph belongs in the
  screen recording (Loom style only).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

ANCHOR = re.compile(r"^\[(\d{1,2}):(\d{2}(?:\.\d+)?)\]\s*")
SLIDE = re.compile(r"^\[slide\]\s*(.+)$", re.I)
COMMENT = re.compile(r"<!--.*?-->", re.S)
ABBREV = ("e.g.", "i.e.", "etc.", "vs.", "Mr.", "Mrs.", "Dr.", "St.", "U.S.", "Inc.")
MAX_CHARS = 260


@dataclass
class Slide:
    title: str
    subtitle: str = ""
    bullets: list[str] = field(default_factory=list)


@dataclass
class Sentence:
    k: int            # global index, 1-based
    text: str         # caption spelling
    para: int


@dataclass
class Paragraph:
    i: int
    anchor: float | None
    slide: Slide | None
    sentences: list[Sentence]


@dataclass
class Script:
    title: str
    paragraphs: list[Paragraph]

    @property
    def sentences(self) -> list[Sentence]:
        return [s for p in self.paragraphs for s in p.sentences]

    @property
    def word_count(self) -> int:
        return sum(len(s.text.split()) for s in self.sentences)


def parse_slide(spec: str) -> Slide:
    title, _, rest = spec.partition("|")
    parts = [x.strip() for x in rest.split(";") if x.strip()]
    sub = parts[0] if parts else ""
    return Slide(title.strip(), sub, parts[1:])


def split_sentences(text: str) -> list[str]:
    protected = text
    for i, ab in enumerate(ABBREV):
        protected = protected.replace(ab, f"\x00{i}\x00")
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“(])", protected)
    out = []
    for s in raw:
        for i, ab in enumerate(ABBREV):
            s = s.replace(f"\x00{i}\x00", ab)
        s = s.strip()
        if not s:
            continue
        while len(s) > MAX_CHARS:  # long sentence -> split at a comma/semicolon/dash near the middle
            cut = max((m.end() for m in re.finditer(r"[,;:—–]\s", s[: MAX_CHARS])), default=0)
            if cut < 40:
                break
            out.append(s[:cut].strip().rstrip(",;:—–") + ".")
            s = s[cut:].strip()
            s = s[:1].upper() + s[1:]
        out.append(s)
    return out


def parse(md: str) -> Script:
    md = COMMENT.sub("", md)
    md = re.split(r"(?m)^---\s*$", md, maxsplit=1)[0]
    title = ""
    paragraphs: list[Paragraph] = []
    pending_slide: Slide | None = None
    buf: list[str] = []
    anchor: float | None = None
    k = 0

    def flush():
        nonlocal buf, anchor, pending_slide, k
        text = " ".join(x.strip() for x in buf if x.strip())
        if text:
            sents = []
            for s in split_sentences(text):
                k += 1
                sents.append(Sentence(k, s, len(paragraphs) + 1))
            paragraphs.append(Paragraph(len(paragraphs) + 1, anchor, pending_slide, sents))
            pending_slide = None
        buf, anchor = [], None

    for line in md.splitlines():
        s = line.strip()
        if s.startswith("# ") and not title:
            title = s[2:].strip()
            continue
        if s.startswith("#"):
            continue
        if not s:
            flush()
            continue
        m = SLIDE.match(s)
        if m:
            flush()
            pending_slide = parse_slide(m.group(1))
            continue
        m = ANCHOR.match(s)
        if m and not buf:
            anchor = int(m.group(1)) * 60 + float(m.group(2))
            s = s[m.end():]
        elif m:
            flush()
            anchor = int(m.group(1)) * 60 + float(m.group(2))
            s = s[m.end():]
        buf.append(s)
    flush()
    return Script(title, paragraphs)


def spoken(text: str, pron: dict) -> str:
    """Apply pronunciation fixes for TTS only."""
    out = text
    for word, say in sorted(pron.items(), key=lambda kv: -len(str(kv[0]))):
        w = re.escape(str(word))
        pre = r"(?<![\w])" if str(word)[0].isalnum() else ""
        post = r"(?![\w])" if str(word)[-1].isalnum() else ""
        out = re.sub(pre + w + post, str(say), out)
    return re.sub(r"\s+", " ", out).strip()


def text_hash(*parts) -> str:
    return hashlib.sha1("\x1f".join(str(p) for p in parts).encode()).hexdigest()[:12]


if __name__ == "__main__":
    import json
    import sys
    sc = parse(open(sys.argv[1]).read())
    print(json.dumps({"title": sc.title, "words": sc.word_count,
                      "paragraphs": [{"i": p.i, "anchor": p.anchor, "slide": p.slide.__dict__ if p.slide else None,
                                      "sentences": [s.text for s in p.sentences]} for p in sc.paragraphs]},
                     indent=2, ensure_ascii=False))
