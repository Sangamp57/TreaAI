"""Shareable observation card rendering (Pillow + Microsoft YaHei)."""
from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import config

W, H = 1080, 1350
PHOTO_H = 700
PAD = 56

LEVEL_COLOR = {
    "National Class I": "#C62828",
    "National Class II": "#EF6C00",
    "Sanyou": "#00695C",
    "Not Listed": "#546E7A",
    # Keep Chinese keys for backward compatibility (profiles store Chinese values).
    "国家一级": "#C62828",
    "国家二级": "#EF6C00",
    "三有": "#00695C",
    "未列入": "#546E7A",
}
LEVEL_EN = {
    "国家一级": "National Class I",
    "国家二级": "National Class II",
    "三有": "Sanyou",
    "未列入": "Not Listed",
}
UNKNOWN_COLOR = "#6A1B9A"


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str]:
    lines, cur = [], ""
    for ch in text:
        if ch == "\n":
            lines.append(cur)
            cur = ""
            continue
        trial = cur + ch
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def _cover(img: Image.Image, w: int, h: int) -> Image.Image:
    img = img.convert("RGB")
    scale = max(w / img.width, h / img.height)
    img = img.resize((int(img.width * scale + 0.5), int(img.height * scale + 0.5)),
                     Image.LANCZOS)
    left = (img.width - w) // 2
    top = (img.height - h) // 2
    return img.crop((left, top, left + w, top + h))


def _badge(draw, x, y, text, color):
    f = _font(config.FONT_BOLD, 34)
    tw = draw.textlength(text, font=f)
    w, h = int(tw) + 44, 60
    draw.rounded_rectangle([x, y, x + w, y + h], radius=30, fill=color)
    draw.text((x + 22, y + 10), text, font=f, fill="white")
    return w


def make_card(image_path: str, profile: dict, *, out_dir: Path | None = None,
              confidence: float | None = None, observed_at: str | None = None,
              location: str = "", attribution: str = "Photo by observer",
              candidates: list[dict] | None = None, uncertain: bool = False) -> Path:
    out_dir = out_dir or config.CARD_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    canvas = Image.new("RGB", (W, H), "#F7F5EF")
    canvas.paste(_cover(Image.open(image_path), W, PHOTO_H), (0, 0))
    draw = ImageDraw.Draw(canvas)

    y = PHOTO_H + 36
    name = profile.get("name_cn", "")
    name_font = _font(config.FONT_BOLD, 66)
    draw.text((PAD, y), name, font=name_font, fill="#1B1B1B")
    name_w = draw.textlength(name, font=name_font)
    level = profile.get("protection_level", "未列入")
    badge_text = "Pending" if uncertain else LEVEL_EN.get(level, level)
    badge_color = UNKNOWN_COLOR if uncertain else LEVEL_COLOR.get(level, "#546E7A")
    _badge(draw, PAD + name_w + 24, y + 16, badge_text, badge_color)

    y += 78
    draw.text((PAD, y), profile.get("scientific_name", ""),
              font=_font(config.FONT_REGULAR, 33), fill="#5F6368")

    if uncertain:
        y += 38
        conf_txt = f"Most likely · TOP-1 {confidence * 100:.0f}%" if confidence is not None else "Most likely"
        draw.text((PAD, y), conf_txt, font=_font(config.FONT_BOLD, 25), fill=UNKNOWN_COLOR)

    y += 46
    draw.text((PAD, y), "Why it matters", font=_font(config.FONT_BOLD, 31), fill="#2E7D32")
    y += 44
    body_font = _font(config.FONT_REGULAR, 32)
    why_lines = _wrap(draw, profile.get("why_important", ""), body_font, W - 2 * PAD)[:2]
    for line in why_lines:
        draw.text((PAD, y), line, font=body_font, fill="#333333")
        y += 42

    if uncertain and candidates:
        y += 14
        draw.text((PAD, y), "Candidates (no forced decision)", font=_font(config.FONT_BOLD, 26),
                  fill=UNKNOWN_COLOR)
        y += 36
        cand_font = _font(config.FONT_REGULAR, 24)
        def _cand_name(c):
            if "name_cn" in c and c["name_cn"]:
                return str(c["name_cn"])
            return str(config.load_profiles().get(c.get("species_code", ""), {}).get("name_cn", c.get("species_code", "?")))
        text = " | ".join(f"{_cand_name(c)} {c['prob'] * 100:.0f}%" for c in candidates[:3])
        for line in _wrap(draw, text, cand_font, W - 2 * PAD)[:1]:
            draw.text((PAD, y), line, font=cand_font, fill="#5F6368")
    else:
        y += 12
        draw.text((PAD, y), "ID tips", font=_font(config.FONT_BOLD, 26), fill="#37474F")
        y += 36
        small_font = _font(config.FONT_REGULAR, 25)
        max_lines = 1 if len(why_lines) >= 2 else 2
        for line in _wrap(draw, profile.get("identification_points", ""), small_font,
                          W - 2 * PAD)[:max_lines]:
            draw.text((PAD, y), line, font=small_font, fill="#546E7A")
            y += 34

    if y < H - 214 - 96:
        y += 14
        draw.text((PAD, y), "Habitat in Yunnan", font=_font(config.FONT_BOLD, 26), fill="#37474F")
        y += 36
        small_font = _font(config.FONT_REGULAR, 25)
        for line in _wrap(draw, profile.get("habitat_yunnan", ""), small_font,
                          W - 2 * PAD)[:2]:
            draw.text((PAD, y), line, font=small_font, fill="#546E7A")
            y += 34

    draw.line([(PAD, H - 214), (W - PAD, H - 214)], fill="#DAD7CE", width=2)
    small = _font(config.FONT_REGULAR, 24)
    footer = [
        (observed_at or datetime.now().strftime("%Y-%m-%d %H:%M")) +
        (f" · {location}" if location else ""),
        profile.get("observe_etiquette", "Observe quietly from a distance; do not feed or disturb."),
        f"Protection basis: {profile.get('protection_list_version', 'National list')}",
        attribution,
        "AI x Ecology · Dianjing Wildlife | For science outreach only, not a formal ID",
    ]
    y = H - 196
    for line in footer:
        for sub in _wrap(draw, line, small, W - 2 * PAD)[:2]:
            draw.text((PAD, y), sub, font=small, fill="#6B6B6B")
            y += 32
        y += 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"card_{profile.get('species_code', 'obs')}_{stamp}_{uuid.uuid4().hex[:4]}.png"
    canvas.save(out, "PNG")
    return out


if __name__ == "__main__":
    profiles = config.load_profiles()
    df = config.load_manifest()
    sample = df[df["split"] == "test"].iloc[0]
    p = profiles[sample["species_code"]]
    path = make_card(str(sample["path"]), p, observed_at="2026-10-25 09:30",
                     location="Kunming · Yunnan University campus", attribution=sample["photo_attribution"])
    print(path)

    # uncertain variant for layout checking
    cands = [{"name_cn": profiles[c]["name_cn"], "scientific_name": profiles[c]["scientific_name"],
              "prob": 0.41 - i * 0.06} for i, c in enumerate([sample["species_code"], "PL03", "IN02"])]
    print(make_card(str(sample["path"]), p, uncertain=True, confidence=0.41, candidates=cands,
                    location="Kunming · Yunnan University campus", attribution="Photo by observer"))
