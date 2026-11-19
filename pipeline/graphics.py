"""Pillow graphics: face-bubble mask/ring, slides, title banner, topic labels, end card, thumbnail."""
from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SS = 4  # supersampling for smooth circles


def font(size: int, weight: str = "Bold"):
    for f in sorted((ROOT / "assets" / "brand" / "fonts").glob("*.[ot]tf")):
        return ImageFont.truetype(str(f), size)
    path = "/System/Library/Fonts/Avenir Next.ttc"
    if Path(path).exists():
        for idx in range(16):
            try:
                f = ImageFont.truetype(path, size, index=idx)
            except OSError:
                break
            if f.getname()[1] == weight:
                return f
        return ImageFont.truetype(path, size)
    return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", size)


def rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def lighten(c, f):
    return tuple(min(255, int(x + (255 - x) * f)) for x in c)


# ---------- face bubble ----------
def bubble_mask(size: int, out: Path):
    big = Image.new("L", (size * SS, size * SS), 0)
    ImageDraw.Draw(big).ellipse([0, 0, size * SS - 1, size * SS - 1], fill=255)
    big.resize((size, size), Image.LANCZOS).save(out)


def bubble_ring(size: int, ring: int, out: Path, color=(255, 255, 255)):
    """Ring + soft drop shadow, canvas = size + 2*pad; circle centred."""
    pad = ring + 24
    c = size + 2 * pad
    img = Image.new("RGBA", (c * SS, c * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # shadow
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).ellipse([(pad - ring) * SS, (pad - ring + 8) * SS, (pad + size + ring) * SS, (pad + size + ring + 8) * SS],
                               fill=(0, 0, 0, 110))
    sh = sh.filter(ImageFilter.GaussianBlur(14 * SS))
    img = Image.alpha_composite(img, sh)
    d = ImageDraw.Draw(img)
    # punch the face area out of the shadow, then draw ring
    d.ellipse([pad * SS, pad * SS, (pad + size) * SS, (pad + size) * SS], fill=(0, 0, 0, 0))
    d.ellipse([(pad - ring) * SS, (pad - ring) * SS, (pad + size + ring) * SS, (pad + size + ring) * SS],
              outline=color + (255,), width=ring * SS)
    img.resize((c, c), Image.LANCZOS).save(out)
    return pad


# ---------- slides ----------
def background(size, brand):
    w, h = size
    bg = rgb(brand["background"])
    acc = rgb(brand["accent"])
    img = Image.new("RGB", size, bg)
    glow = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(glow)
    d.ellipse([int(w * 0.55), int(-h * 0.5), int(w * 1.4), int(h * 0.7)], fill=lighten(acc, 0.05))
    d.ellipse([int(-w * 0.3), int(h * 0.6), int(w * 0.35), int(h * 1.5)], fill=tuple(int(x * 0.6) for x in acc))
    glow = glow.filter(ImageFilter.GaussianBlur(min(w, h) // 5))
    return Image.blend(img, glow, 0.55)


def wrap_lines(text, fnt, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = f"{cur} {w}".strip()
        if fnt.getlength(t) <= max_w:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def slide(sl: dict, brand: dict, out: Path, size=(1920, 1080), kicker: str = "GHL UPDATE"):
    w, h = size
    img = background(size, brand)
    d = ImageDraw.Draw(img)
    acc = rgb(brand["accent"])
    x0 = int(w * 0.07)
    maxw = int(w * 0.62)
    y = int(h * 0.2)
    kf = font(int(h * 0.032), "Demi Bold")
    d.rounded_rectangle([x0, y, x0 + kf.getlength(kicker) + 48, y + int(h * 0.058)], radius=24, fill=acc)
    d.text((x0 + 24, y + int(h * 0.009)), kicker, font=kf, fill=(255, 255, 255))
    y += int(h * 0.1)
    tsize = int(h * 0.095)
    tf = font(tsize, "Bold")
    lines = wrap_lines(sl.get("title", ""), tf, maxw)
    while len(lines) > 2 and tsize > 40:
        tsize -= 6
        tf = font(tsize, "Bold")
        lines = wrap_lines(sl["title"], tf, maxw)
    for ln in lines:
        d.text((x0, y), ln, font=tf, fill=(255, 255, 255))
        y += int(tsize * 1.12)
    if sl.get("subtitle"):
        y += int(h * 0.02)
        sf_ = font(int(h * 0.042), "Medium")
        for ln in wrap_lines(sl["subtitle"], sf_, maxw):
            d.text((x0, y), ln, font=sf_, fill=(205, 214, 235))
            y += int(h * 0.056)
    if sl.get("bullets"):
        y += int(h * 0.035)
        bf = font(int(h * 0.036), "Medium")
        for b in sl["bullets"][:4]:
            d.ellipse([x0, y + int(h * 0.014), x0 + 14, y + int(h * 0.014) + 14], fill=acc)
            for j, ln in enumerate(wrap_lines(b, bf, maxw - 40)):
                d.text((x0 + 34, y), ln, font=bf, fill=(235, 238, 245))
                y += int(h * 0.05)
            y += int(h * 0.012)
    logo = ROOT / brand.get("logo", "")
    if brand.get("logo") and logo.exists():
        lg = Image.open(logo).convert("RGBA")
        lg.thumbnail((int(w * 0.14), int(h * 0.09)))
        img.paste(lg, (w - lg.width - x0 // 2, h - lg.height - int(h * 0.06)), lg)
    img.save(out)


# ---------- overlays (RGBA, full frame) ----------
def title_banner(title: str, brand: dict, out: Path, size=(1920, 1080), vertical=False):
    w, h = size
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    acc = rgb(brand["accent"])
    fs = int((w if vertical else h) * (0.058 if vertical else 0.052))
    f = font(fs, "Bold")
    maxw = int(w * (0.84 if vertical else 0.6))
    lines = wrap_lines(title, f, maxw)[:3]
    lh = int(fs * 1.18)
    bw = int(max(f.getlength(l) for l in lines)) + 70
    bh = lh * len(lines) + 50
    if vertical:
        x, y = (w - bw) // 2, int(h * 0.075)
    else:
        x, y = int(w * 0.045), int(h * 0.07)
    d.rounded_rectangle([x, y, x + bw, y + bh], radius=22, fill=(10, 14, 28, 215))
    d.rounded_rectangle([x, y, x + 12, y + bh], radius=6, fill=acc + (255,))
    for i, ln in enumerate(lines):
        lx = x + (bw - f.getlength(ln)) / 2 if vertical else x + 40
        d.text((lx, y + 25 + i * lh), ln, font=f, fill=(255, 255, 255))
    img.save(out)


def topic_label(text: str, brand: dict, out: Path, size=(1920, 1080), vertical=False):
    w, h = size
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    acc = rgb(brand["accent"])
    fs = int((w if vertical else h) * 0.042)
    f = font(fs, "Demi Bold")
    tw = f.getlength(text)
    bw, bh = int(tw + 64), int(fs * 1.9)
    if vertical:
        x, y = (w - bw) // 2, int(h * 0.075)
    else:
        x, y = int(w * 0.045), int(h * 0.07)
    d.rounded_rectangle([x, y, x + bw, y + bh], radius=bh // 2, fill=acc + (240,))
    d.text((x + 32, y + (bh - fs) / 2 - fs * 0.12), text, font=f, fill=(255, 255, 255))
    img.save(out)


def end_card(brand: dict, out: Path, size=(1920, 1080), vertical=False):
    w, h = size
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    acc = rgb(brand["accent"])
    cta = brand.get("cta") or "Follow for weekly GHL updates"
    name = brand.get("name") or ""
    fs = int((w if vertical else h) * 0.05)
    f = font(fs, "Bold")
    lines = wrap_lines(cta, f, int(w * 0.8))
    lh = int(fs * 1.2)
    sub = font(int(fs * 0.62), "Medium")
    bh = lh * len(lines) + (int(fs * 0.9) if name else 0) + 56
    bw = int(max(f.getlength(l) for l in lines + ([name] if name else []))) + 90
    x = (w - bw) // 2
    y = int(h * (0.12 if vertical else 0.09))
    d.rounded_rectangle([x, y, x + bw, y + bh], radius=26, fill=acc + (235,))
    for i, ln in enumerate(lines):
        d.text(((w - f.getlength(ln)) / 2, y + 26 + i * lh), ln, font=f, fill=(255, 255, 255))
    if name:
        d.text(((w - sub.getlength(name)) / 2, y + 30 + len(lines) * lh), name, font=sub, fill=(235, 240, 255))
    img.save(out)


def thumbnail(frame: Path, title: str, brand: dict, out: Path):
    base = Image.open(frame).convert("RGB")
    base = base.resize((1280, int(base.height * 1280 / base.width)), Image.LANCZOS).crop((0, 0, 1280, 720))
    shade = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ImageDraw.Draw(shade).rectangle([0, 0, 760, 720], fill=(8, 12, 26, 175))
    base = Image.alpha_composite(base.convert("RGBA"), shade.filter(ImageFilter.GaussianBlur(40)))
    d = ImageDraw.Draw(base)
    f = font(84, "Heavy")
    y = 150
    for ln in wrap_lines(title.upper(), f, 680)[:4]:
        d.text((50, y), ln, font=f, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
        y += 96
    d.rounded_rectangle([50, 70, 330, 128], radius=20, fill=rgb(brand["accent"]))
    d.text((72, 78), "GHL UPDATE", font=font(36, "Bold"), fill=(255, 255, 255))
    base.convert("RGB").save(out)
