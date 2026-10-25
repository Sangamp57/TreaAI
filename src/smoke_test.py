"""End-to-end smoke test: identify -> card -> save -> export -> brief.

Uses a throwaway SQLite database (Q1_DB) so the demo data stays clean.
Run after src/build_ensemble.py:  python src/smoke_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_tmp = Path(tempfile.mkdtemp(prefix="q1_smoke_"))
os.environ["Q1_DB"] = str(_tmp / "smoke.db")

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

import app  # noqa: E402
import config  # noqa: E402
import records  # noqa: E402


def main() -> int:
    ok = 0
    rec = app.recognizer()
    assert rec is not None, "ensemble.json missing or checkpoints unreadable"
    print(f"[1] recognizer loaded: {len(rec.members)} member(s), "
          f"temperature={rec.temperature}, threshold={rec.threshold}")
    ok += 1

    df = config.load_manifest()
    row = df[df["split"] == "test"].iloc[3]
    image = Image.open(row["path"]).convert("RGB")
    info, card, state = app.identify(image, "云南大学校园", 1, "测试署名")
    assert state and "species_code" in state
    assert card and Path(card).exists(), "card not rendered"
    w, h = Image.open(card).size
    assert (w, h) == (1080, 1350), (w, h)
    print(f"[2] identify: true={row['species_code']} pred={state['species_code']} "
          f"p={state['prob']:.3f} uncertain={state['uncertain']} card={Path(card).name} {w}x{h}")
    ok += 1

    saved_thr = rec.threshold
    rec.threshold = 0.999  # force the no-forced-decision path
    info_u, card_u, state_u = app.identify(image, "云南大学校园", 1, "测试署名")
    assert state_u["uncertain"] is True
    assert "候选物种" in info_u, "candidate table missing"
    assert "待鉴定" in Path(card_u).name or Path(card_u).exists()
    rec.threshold = saved_thr
    print(f"[3] uncertain path: {len(state_u['candidates'])} candidates, "
          f"table={'候选物种' in info_u}")
    ok += 1

    msg, table = app.save_observation(state, image, "云南大学校园", 1, "smoke test", "测试署名")
    assert "已保存" in msg, msg
    assert len(table) == 1, table
    print(f"[4] save record: {msg.split('<')[0].strip()[:40]}... rows={len(table)}")
    ok += 1

    csv_path = records.export_csv()
    out = pd.read_csv(csv_path, encoding="utf-8-sig")
    for col in ("qc_image", "qc_confidence_band", "qc_status", "qc_note"):
        assert col in out.columns, col
    print(f"[5] export: {csv_path.name} rows={len(out)} qc_cols=ok")
    ok += 1

    stats = records.stats()
    brief = app.brief_html()
    assert stats["records"] == 1 and "观察记录" in brief
    print(f"[6] brief: stats={stats['records']} records / {stats['species']} species, html={len(brief)} chars")
    ok += 1

    filtered = app.filter_records("鸟类", "全部", None, None, "")
    print(f"[7] filter by taxon: {len(filtered)} row(s)")

    print(f"\nSMOKE TEST PASSED ({ok}/6 checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
