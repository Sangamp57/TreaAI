"""Dianjing Wildlife · AI Species Recognition & Citizen Science Platform — Gradio app (English UI)."""
from __future__ import annotations

import html
import uuid
from datetime import datetime

import gradio as gr
import pandas as pd

import cards
import config
import records
from infer import Recognizer, is_ready

PROFILES = config.load_profiles()
MANIFEST = config.load_manifest()
CLASSES = config.class_names()

# English UI choices; mapped back to Chinese profile values for filtering.
GROUPS = ["All", "Birds", "Mammals", "Insects", "Plants"]
LEVELS = ["All", "National Class I", "National Class II", "Sanyou", "Not Listed"]
GROUP_TO_CN = {"All": "全部", "Birds": "鸟类", "Mammals": "兽类", "Insects": "昆虫", "Plants": "植物"}
LEVEL_TO_CN = {"All": "全部", "National Class I": "国家一级", "National Class II": "国家二级",
               "Sanyou": "三有", "Not Listed": "未列入"}
LEVEL_EN = {"国家一级": "National Class I", "国家二级": "National Class II",
            "三有": "Sanyou", "未列入": "Not Listed"}
GROUP_EN = {"鸟类": "Birds", "兽类": "Mammals", "昆虫": "Insects", "植物": "Plants"}

_recognizer: Recognizer | None = None


def _level_en(level: str) -> str:
    return LEVEL_EN.get(level, level)


def _persist(image) -> str | None:
    """Write an uploaded PIL image to data/uploads and return the path."""
    if image is None:
        return None
    d = config.DATA / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"upload_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.jpg"
    image.convert("RGB").save(p, "JPEG", quality=95)
    return str(p)


def recognizer() -> Recognizer | None:
    global _recognizer
    if _recognizer is None and is_ready():
        _recognizer = Recognizer.load()
    return _recognizer


def _badge(level: str) -> str:
    color = cards.LEVEL_COLOR.get(level, "#546E7A")
    label = _level_en(level)
    return (f"<span style='background:{color};color:#fff;padding:4px 14px;border-radius:14px;"
            f"font-size:14px;font-weight:600'>{html.escape(label)}</span>")


def _profile_html(code: str, prob: float, uncertain: bool, candidates: list[dict]) -> str:
    p = PROFILES[code]
    level = p["protection_level"]
    alert = ""
    if level in ("国家一级", "国家二级", "三有"):
        alert = ("<div style='background:#FFF3E0;border-left:5px solid #EF6C00;padding:12px 16px;"
                 "border-radius:8px;margin:12px 0;font-size:14px'>🛡️ <b>Protected species</b> · "
                 f"{html.escape(str(p['observe_etiquette']))}</div>")
    conf = f"{prob * 100:.1f}%"
    conf_color = "#2E7D32" if prob >= 0.75 else "#EF6C00" if prob >= 0.55 else "#C62828"
    cand_html = ""
    if uncertain and candidates:
        rows = []
        for c in candidates[:3]:
            cp = PROFILES[c["species_code"]]
            tip = cp.get("similar_species_tip") or cp.get("identification_points", "")
            rows.append(
                f"<tr><td style='padding:8px 10px'><b>{html.escape(str(cp['name_cn']))}</b><br>"
                f"<span style='color:#78909C;font-size:12px'>{html.escape(str(cp['scientific_name']))}</span></td>"
                f"<td style='padding:8px 10px'>{c['prob'] * 100:.1f}%</td>"
                f"<td style='padding:8px 10px;font-size:13px'>{html.escape(str(tip))}</td></tr>")
        cand_html = (
            "<div style='margin-top:16px'><h4 style='margin:0 0 8px'>🤔 Candidates & distinguishing traits (no forced decision)</h4>"
            "<table style='width:100%;border-collapse:collapse;font-size:14px'>"
            "<tr style='background:#F5F5F5'><th style='text-align:left;padding:8px'>Species</th>"
            "<th style='text-align:left;padding:8px'>Similarity</th>"
            "<th style='text-align:left;padding:8px'>How to tell apart</th></tr>" + "".join(rows) + "</table></div>")

    title = "Pending review · most likely" if uncertain else "Identification result"
    pair_html = ""
    pair_id = str(p.get("pair_id") or "").strip()
    if pair_id:
        partner = next((c for c, q in PROFILES.items()
                        if str(q.get("pair_id") or "").strip() == pair_id and c != code), None)
        if partner:
            q = PROFILES[partner]
            tip = q.get("similar_species_tip") or q.get("identification_points", "")
            pair_html = (
                "<div style='margin-top:12px;padding:10px 14px;background:#F3E5F5;"
                "border-left:4px solid #6A1B9A;border-radius:8px;font-size:13.5px'>"
                f"🔍 <b>Easily confused relative</b>: {html.escape(str(q['name_cn']))}"
                f" (<i>{html.escape(str(q['scientific_name']))}</i>)<br>"
                f"<span style='color:#5F6368'>How to tell apart: {html.escape(str(tip))}</span></div>")
    return f"""
<div style="font-family:'Microsoft YaHei',Arial,sans-serif">
  <div style="color:#78909C;font-size:13px;letter-spacing:1px">{title}</div>
  <div style="display:flex;align-items:center;gap:12px;margin:6px 0 2px">
    <span style="font-size:30px;font-weight:700">{html.escape(str(p['name_cn']))}</span>
    {_badge(level)}
    <span style="font-size:14px;color:{conf_color};font-weight:600">TOP-1 {conf}</span>
  </div>
  <div style="color:#607D8B;font-style:italic;font-size:15px">{html.escape(str(p['scientific_name']))}
    &nbsp;·&nbsp;{html.escape(str(p['family_cn']))} · {html.escape(str(GROUP_EN.get(p['taxon_group'], p['taxon_group'])))}</div>
  {alert}
  <table style="width:100%;font-size:14px;margin-top:10px;border-collapse:collapse">
    <tr><td style="width:150px;color:#78909C;padding:5px 0;vertical-align:top">Habitat in Yunnan</td>
        <td style="padding:5px 0">{html.escape(str(p['habitat_yunnan']))}</td></tr>
    <tr><td style="color:#78909C;padding:5px 0;vertical-align:top">ID tips</td>
        <td style="padding:5px 0">{html.escape(str(p['identification_points']))}</td></tr>
    <tr><td style="color:#78909C;padding:5px 0;vertical-align:top">Why it matters</td>
        <td style="padding:5px 0">{html.escape(str(p['why_important']))}</td></tr>
    <tr><td style="color:#78909C;padding:5px 0;vertical-align:top">Listing basis</td>
        <td style="padding:5px 0;font-size:12.5px;color:#607D8B">{html.escape(str(p['protection_list_version']))}</td></tr>
  </table>
  {pair_html}
  {cand_html}
</div>"""


def identify(image, location, count, attribution):
    rec = recognizer()
    if rec is None:
        return ("<div style='padding:16px;background:#FFEBEE;border-radius:8px'>Model is not trained yet. "
                "Please run <code>python src/train.py</code> and <code>src/build_ensemble.py</code> first.</div>",
                None, None)
    if image is None:
        return ("<div style='padding:12px;color:#C62828'>Please upload a photo first.</div>", None, None)

    preds = rec.predict(image, topk=3, tta=True)
    top = preds[0]
    uncertain = top["prob"] < rec.threshold

    if uncertain:
        cands = [{"species_code": p["species_code"], "prob": p["prob"]} for p in preds]
        info = _profile_html(preds[0]["species_code"], preds[0]["prob"], True, cands)
        chosen = preds[0]["species_code"]
    else:
        info = _profile_html(top["species_code"], top["prob"], False, [])
        chosen = top["species_code"]

    state = {"species_code": chosen, "prob": top["prob"], "uncertain": uncertain,
             "candidates": [{"species_code": p["species_code"], "prob": p["prob"]} for p in preds]}

    card_path = None
    if image is not None:
        profile = PROFILES[chosen]
        upload = _persist(image)
        card_path = str(cards.make_card(
            upload, profile,
            confidence=top["prob"], observed_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            location=location or "", attribution=attribution or "Photo by observer",
            candidates=state["candidates"] if uncertain else None, uncertain=uncertain))

    return info, card_path, state


def save_observation(state, image, location, count, note, attribution):
    if not state:
        return "<div style='color:#C62828'>Please run identification first.</div>", gr.update()
    if not (location or "").strip():
        return "<div style='color:#C62828'>Please enter the observation location (e.g. Yunnan University campus).</div>", gr.update()
    code = state["species_code"]
    p = PROFILES[code]
    img_path = records.save_image(_persist(image))
    rec_id = records.add_record(
        observed_at=datetime.now().strftime("%Y-%m-%d %H:%M"), location=location.strip(),
        species_code=code, name_cn=p["name_cn"], scientific_name=p["scientific_name"],
        count=int(count or 1), confidence=state["prob"], uncertain=state["uncertain"],
        note=note or "", image_path=img_path)
    tag = " (pending review)" if state["uncertain"] else ""
    return (f"<div style='background:#E8F5E9;border-left:5px solid #2E7D32;padding:10px 14px;"
            f"border-radius:8px'>Saved {rec_id} · {html.escape(str(p['name_cn']))}{tag}</div>",
            records_frame())


def records_frame():
    rows = records.list_records()
    if not rows:
        return pd.DataFrame(columns=["ID", "Time", "Location", "Species", "Count", "Confidence", "Status"])
    df = pd.DataFrame([{
        "ID": r["id"], "Time": r["observed_at"], "Location": r["location"],
        "Species": f"{r['name_cn']} {r['species_code']}",
        "Count": r["count"],
        "Confidence": "" if r["confidence"] is None else f"{r['confidence'] * 100:.0f}%",
        "Status": "Pending" if r["uncertain"] else "Identified",
    } for r in rows])
    return df


def filter_records(group, level, start, end, keyword):
    group_cn = GROUP_TO_CN.get(group, group)
    level_cn = LEVEL_TO_CN.get(level, level)
    rows = records.list_records(start=start or None, end=end or None, keyword=keyword or "")
    out = []
    for r in rows:
        p = PROFILES.get(r["species_code"], {})
        if group_cn and group_cn != "全部" and p.get("taxon_group") != group_cn:
            continue
        if level_cn and level_cn != "全部" and p.get("protection_level") != level_cn:
            continue
        out.append({
            "ID": r["id"], "Time": r["observed_at"], "Location": r["location"],
            "Species": f"{r['name_cn']} {r['species_code']}",
            "Group": GROUP_EN.get(p.get("taxon_group", ""), p.get("taxon_group", "")),
            "Protection": _level_en(p.get("protection_level", "")),
            "Count": r["count"],
            "Confidence": "" if r["confidence"] is None else f"{r['confidence'] * 100:.0f}%",
            "Status": "Pending" if r["uncertain"] else "Identified", "Note": r["note"],
        })
    if not out:
        return pd.DataFrame(columns=["ID", "Time", "Location", "Species", "Group", "Protection",
                                     "Count", "Confidence", "Status", "Note"])
    return pd.DataFrame(out)


def export_records():
    return str(records.export_csv())


def delete_record(rec_id):
    if rec_id and rec_id.strip():
        records.delete_record(rec_id.strip())
    return filter_records("All", "All", None, None, ""), "Deleted (if it existed)."


def brief_html():
    s = records.stats()
    if not s["records"]:
        return "<div style='padding:24px;color:#78909C'>No observations yet. Go to the Identify tab and save your first record.</div>"
    profiles = PROFILES
    star = s["star_species"]
    star_name = profiles.get(star, {}).get("name_cn", "-") if star else "-"
    total = max(s["by_species"].values())
    bars = []
    for code, n in sorted(s["by_species"].items(), key=lambda kv: -kv[1]):
        p = profiles.get(code, {})
        pct = n / total * 100
        bars.append(
            f"<div style='display:flex;align-items:center;gap:10px;margin:6px 0'>"
            f"<div style='width:150px;font-size:13px'>{html.escape(str(p.get('name_cn', code)))}</div>"
            f"<div style='flex:1;background:#ECEFF1;border-radius:6px;height:16px'>"
            f"<div style='width:{pct:.0f}%;background:#2E7D32;height:16px;border-radius:6px'></div></div>"
            f"<div style='width:28px;text-align:right;font-size:13px'>{n}</div></div>")
    return f"""
<div style="font-family:'Microsoft YaHei',Arial,sans-serif">
  <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:18px">
    <div style="flex:1;min-width:150px;background:#E8F5E9;padding:18px;border-radius:12px">
      <div style="font-size:13px;color:#607D8B">Observations</div>
      <div style="font-size:34px;font-weight:700;color:#2E7D32">{s['records']}</div></div>
    <div style="flex:1;min-width:150px;background:#E3F2FD;padding:18px;border-radius:12px">
      <div style="font-size:13px;color:#607D8B">Species</div>
      <div style="font-size:34px;font-weight:700;color:#1565C0">{s['species']}</div></div>
    <div style="flex:1;min-width:150px;background:#FFF3E0;padding:18px;border-radius:12px">
      <div style="font-size:13px;color:#607D8B">Protected records</div>
      <div style="font-size:34px;font-weight:700;color:#EF6C00">{s['protected']}</div></div>
    <div style="flex:1;min-width:150px;background:#F3E5F5;padding:18px;border-radius:12px">
      <div style="font-size:13px;color:#607D8B">Star species</div>
      <div style="font-size:24px;font-weight:700;color:#6A1B9A">{html.escape(str(star_name))}</div></div>
  </div>
  <h4 style="margin:0 0 8px">Species frequency</h4>
  {''.join(bars)}
</div>"""


def guide_options(group, level):
    group_cn = GROUP_TO_CN.get(group, group)
    level_cn = LEVEL_TO_CN.get(level, level)
    codes = [c for c, p in PROFILES.items()
             if (group_cn == "全部" or p["taxon_group"] == group_cn)
             and (level_cn == "全部" or p["protection_level"] == level_cn)]
    return gr.update(choices=_code_choices(codes), value=codes[0] if codes else None)


def _code_choices(codes):
    return [(f"{c} · {PROFILES[c]['name_cn']}", c) for c in codes]


def guide_html(code):
    if not code:
        return "<div style='color:#78909C'>Please select a species.</div>"
    return _profile_html(code, 1.0, False, []).replace("Identification result", "Species profile")


def guide_gallery(code):
    if not code:
        return []
    files = MANIFEST[MANIFEST["species_code"] == code]["path"].tolist()[:6]
    return [str(f) for f in files]


def demo_list():
    return [f"{r.species_code} | {r.clean_file}" for r in MANIFEST[MANIFEST.split == "test"].itertuples()]


def load_demo(choice):
    if not choice:
        return None
    clean_file = choice.split("|")[1].strip()
    path = config.TEST_DIR / clean_file
    from PIL import Image

    return Image.open(path).convert("RGB") if path.exists() else None


records.init_db()

with gr.Blocks(title="Dianjing Wildlife · AI Species Recognition & Citizen Science Platform",
               theme=gr.themes.Soft(primary_hue="green", neutral_hue="stone"),
               css=".gradio-container{max-width:1180px}") as demo:
    gr.HTML("""
    <div style="font-family:'Microsoft YaHei',Arial,sans-serif;padding:6px 0 18px">
      <div style="font-size:30px;font-weight:700">Dianjing Wildlife
        <span style="font-size:15px;font-weight:400;color:#607D8B">AI Species Recognition & Citizen Science Platform</span></div>
      <div style="color:#78909C;font-size:13.5px;margin-top:6px">
        Snap a photo → identify species & protection level → generate a shareable science card → contribute to the citizen-science database
        &nbsp;|&nbsp;Protection levels follow the National Key Protected Wild Animal List (No. 3, 2021) and related authoritative lists
      </div>
    </div>""")

    model_ready = is_ready()

    with gr.Tab("Identify"):
        with gr.Row():
            with gr.Column(scale=5):
                image = gr.Image(type="pil", label="Upload photo", height=330)
                with gr.Row():
                    demo_dd = gr.Dropdown(choices=demo_list(), label="Or load a test-set example",
                                          value=None, scale=4)
                    load_btn = gr.Button("Load", scale=1)
                with gr.Row():
                    location = gr.Textbox(label="Observation location (free text)",
                                          placeholder="e.g. Yunnan University campus · Ginkgo Avenue", scale=3)
                    count = gr.Number(label="Count", value=1, minimum=1, precision=0, scale=1)
                attribution = gr.Textbox(
                    label="Photo credit (required for public sharing)", value="Photo by observer")
                note = gr.Textbox(label="Notes (optional)", lines=2)
                with gr.Row():
                    btn = gr.Button("Identify & generate card", variant="primary", scale=2)
                    save_btn = gr.Button("Save observation", scale=1)
                save_msg = gr.HTML()
            with gr.Column(scale=6):
                info = gr.HTML("<div style='color:#78909C;padding:20px'>Upload a photo, then click Identify.</div>")
                card_img = gr.Image(label="Science card (right-click to save / long-press to share)", type="filepath",
                                    height=430)

    with gr.Tab("Field Guide"):
        with gr.Row():
            g_group = gr.Dropdown(GROUPS, value="All", label="Group", scale=1)
            g_level = gr.Dropdown(LEVELS, value="All", label="Protection level", scale=1)
            g_code = gr.Dropdown(choices=_code_choices(CLASSES), value=CLASSES[0], label="Species", scale=2)
        g_info = gr.HTML(guide_html(CLASSES[0]))
        g_gal = gr.Gallery(value=guide_gallery(CLASSES[0]), label="Gallery", columns=6,
                           height=220, object_fit="cover")

    with gr.Tab("Observations"):
        with gr.Row():
            f_group = gr.Dropdown(GROUPS, value="All", label="Group", scale=1)
            f_level = gr.Dropdown(LEVELS, value="All", label="Protection level", scale=1)
            f_start = gr.Textbox(label="Start date", placeholder="YYYY-MM-DD", scale=1)
            f_end = gr.Textbox(label="End date", placeholder="YYYY-MM-DD", scale=1)
            f_kw = gr.Textbox(label="Keyword", placeholder="species / location / note", scale=2)
        with gr.Row():
            refresh = gr.Button("Filter", variant="primary")
            export_btn = gr.Button("Export standard dataset CSV")
            del_id = gr.Textbox(label="Delete record ID", scale=2)
            del_btn = gr.Button("Delete", scale=1)
        table = gr.Dataframe(value=records_frame(), interactive=False, wrap=True)
        file_out = gr.File(label="Exported file")

    with gr.Tab("Report"):
        brief = gr.HTML(brief_html())
        brief_btn = gr.Button("Refresh report")

    state = gr.State(None)

    btn.click(identify, inputs=[image, location, count, attribution],
              outputs=[info, card_img, state])
    load_btn.click(load_demo, inputs=[demo_dd], outputs=[image])
    save_btn.click(save_observation, inputs=[state, image, location, count, note, attribution],
                   outputs=[save_msg, table])
    g_group.change(guide_options, inputs=[g_group, g_level], outputs=[g_code])
    g_level.change(guide_options, inputs=[g_group, g_level], outputs=[g_code])
    g_code.change(guide_html, inputs=[g_code], outputs=[g_info])
    g_code.change(guide_gallery, inputs=[g_code], outputs=[g_gal])
    refresh.click(filter_records, inputs=[f_group, f_level, f_start, f_end, f_kw], outputs=[table])
    export_btn.click(export_records, outputs=[file_out])
    del_btn.click(delete_record, inputs=[del_id], outputs=[table, save_msg])
    brief_btn.click(brief_html, outputs=[brief])

    if not model_ready:
        gr.HTML("""<div style="margin-top:14px;padding:12px 16px;background:#FFF8E1;
        border-left:5px solid #F9A825;border-radius:8px;font-size:13.5px">
        Model not ready: run <code>python src/train.py --model convnext_tiny</code> first,
        then <code>python src/build_ensemble.py</code> to generate the ensemble config.</div>""")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
