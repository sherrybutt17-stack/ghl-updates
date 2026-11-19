"""Fetch the public GHL changelog RSS and keep a local history.

The feed only holds the latest ~10 posts, so run this every few days
(bin/ghl fetch) to build up history between videos.

  python fetch_changelog.py            # fetch + list unused posts
  python fetch_changelog.py --list     # list unused posts from local history only
  python fetch_changelog.py --mark-used <job_id> <guid> [<guid> ...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import html
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import DATA, load_config, prefer_ipv4, say, slugify  # noqa: E402

ENTRIES = DATA / "changelog" / "entries"
SEEN = DATA / "changelog" / "seen.json"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Safari/605.1.15"


class _Text(HTMLParser):
    BLOCK = {"p", "br", "li", "h1", "h2", "h3", "h4", "div", "tr"}

    def __init__(self):
        super().__init__()
        self.parts, self.links = [], []

    def handle_starttag(self, tag, attrs):
        if tag in self.BLOCK:
            self.parts.append("\n")
        if tag == "li":
            self.parts.append("- ")
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)
        if tag == "img":
            src = dict(attrs).get("src")
            if src:
                self.links.append(src)

    def handle_data(self, data):
        self.parts.append(data)

    def text(self):
        t = html.unescape("".join(self.parts))
        t = re.sub(r"[ \t]+", " ", t)
        return re.sub(r"\n\s*\n+", "\n\n", t).strip()


def html_to_text(s: str):
    p = _Text()
    p.feed(s or "")
    return p.text(), p.links


def fetch(url: str, tries: int = 3, timeout: int = 30) -> bytes:
    prefer_ipv4()
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/rss+xml, */*"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"could not reach {url}: {last}")


def parse(xml_bytes: bytes) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    ns = {"content": "http://purl.org/rss/1.0/modules/content/"}
    items = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        guid = (it.findtext("guid") or link or title).strip()
        body_html = it.findtext("content:encoded", namespaces=ns) or it.findtext("description") or ""
        text, links = html_to_text(body_html)
        pub = it.findtext("pubDate")
        try:
            date = email.utils.parsedate_to_datetime(pub).date().isoformat() if pub else dt.date.today().isoformat()
        except (TypeError, ValueError):
            date = dt.date.today().isoformat()
        cats = [c.text.strip() for c in it.findall("category") if c.text]
        items.append({"guid": guid, "title": html.unescape(title), "link": link, "date": date,
                      "categories": cats, "text": text, "links": links})
    return items


def load_seen() -> dict:
    return json.loads(SEEN.read_text()) if SEEN.exists() else {"used": {}}


def save_entries(items: list[dict]) -> int:
    ENTRIES.mkdir(parents=True, exist_ok=True)
    new = 0
    for e in items:
        f = ENTRIES / f"{e['date']}-{slugify(e['title'], 60)}.json"
        if not f.exists():
            new += 1
        f.write_text(json.dumps(e, indent=2, ensure_ascii=False))
    return new


def all_entries() -> list[dict]:
    if not ENTRIES.exists():
        return []
    out = [json.loads(f.read_text()) for f in ENTRIES.glob("*.json")]
    return sorted(out, key=lambda e: e["date"], reverse=True)


def list_unused(days: int | None = None):
    used = load_seen()["used"]
    cutoff = (dt.date.today() - dt.timedelta(days=days)).isoformat() if days else "0000"
    rows = [e for e in all_entries() if e["guid"] not in used and e["date"] >= cutoff]
    if not rows:
        say("No unused changelog posts in local history.")
        return
    say(f"{len(rows)} unused GHL changelog posts (newest first):")
    for e in rows:
        cats = f" [{', '.join(e['categories'])}]" if e["categories"] else ""
        say(f"  {e['date']}{cats}  {e['title']}")
        say(f"      {e['link']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="only list local history")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--mark-used", nargs="+", metavar=("JOB_ID", "GUID"))
    a = ap.parse_args()

    if a.mark_used:
        job_id, guids = a.mark_used[0], a.mark_used[1:]
        seen = load_seen()
        for g in guids:
            seen["used"][g] = job_id
        SEEN.parent.mkdir(parents=True, exist_ok=True)
        SEEN.write_text(json.dumps(seen, indent=2))
        say(f"marked {len(guids)} posts as used by {job_id}")
        return

    if not a.list:
        cfg = load_config()["changelog"]
        try:
            items = parse(fetch(cfg["rss"]))
            n = save_entries(items)
            say(f"Fetched {len(items)} posts from the GHL changelog ({n} new to local history).")
        except Exception as e:  # noqa: BLE001
            say(f"WARNING: changelog fetch failed ({e}). Using local history only.")
    list_unused(a.days)


if __name__ == "__main__":
    main()
