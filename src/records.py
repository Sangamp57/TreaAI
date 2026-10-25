"""Citizen-science observation store (SQLite) with filtering, stats and CSV export."""
from __future__ import annotations

import csv
import shutil
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import config

IMAGE_STORE = config.DATA / "observations"
SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id           TEXT PRIMARY KEY,
    observed_at  TEXT NOT NULL,
    location     TEXT NOT NULL,
    species_code TEXT NOT NULL,
    name_cn      TEXT,
    scientific_name TEXT,
    count        INTEGER DEFAULT 1,
    confidence   REAL,
    uncertain    INTEGER DEFAULT 0,
    note         TEXT,
    image_path   TEXT,
    created_at   TEXT NOT NULL
);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.RECORDS_DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    IMAGE_STORE.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)


def save_image(src: str | Path | None) -> str | None:
    if not src:
        return None
    src = Path(src)
    if not src.exists():
        return None
    IMAGE_STORE.mkdir(parents=True, exist_ok=True)
    dest = IMAGE_STORE / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}{src.suffix or '.jpg'}"
    shutil.copy2(src, dest)
    return str(dest)


def add_record(*, observed_at: str, location: str, species_code: str, name_cn: str,
               scientific_name: str, count: int = 1, confidence: float | None = None,
               uncertain: bool = False, note: str = "", image_path: str | None = None) -> str:
    init_db()
    rec_id = datetime.now().strftime("OBS%Y%m%d%H%M%S") + uuid.uuid4().hex[:4].upper()
    with connect() as conn:
        conn.execute(
            "INSERT INTO observations (id, observed_at, location, species_code, name_cn, "
            "scientific_name, count, confidence, uncertain, note, image_path, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (rec_id, observed_at, location, species_code, name_cn, scientific_name,
             int(count), confidence, int(uncertain), note, image_path,
             datetime.now().isoformat(timespec="seconds")),
        )
    return rec_id


def list_records(species_code: str | None = None, taxon_group: str | None = None,
                 start: str | None = None, end: str | None = None,
                 keyword: str = "") -> list[dict]:
    init_db()
    sql = "SELECT o.* FROM observations o WHERE 1=1"
    params: list = []
    if species_code and species_code != "全部":
        sql += " AND o.species_code = ?"
        params.append(species_code)
    if start:
        sql += " AND date(o.observed_at) >= date(?)"
        params.append(start)
    if end:
        sql += " AND date(o.observed_at) <= date(?)"
        params.append(end)
    if keyword:
        sql += " AND (o.name_cn LIKE ? OR o.location LIKE ? OR o.note LIKE ?)"
        params += [f"%{keyword}%"] * 3
    sql += " ORDER BY o.observed_at DESC, o.created_at DESC"
    with connect() as conn:
        rows = [dict(r) for r in conn.execute(sql, params)]
    if taxon_group and taxon_group != "全部":
        profiles = config.load_profiles()
        rows = [r for r in rows if profiles.get(r["species_code"], {}).get("taxon_group") == taxon_group]
    return rows


def delete_record(rec_id: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM observations WHERE id = ?", (rec_id,))


def export_csv() -> Path:
    rows = list_records()
    out = config.OUTPUT / f"observations_{datetime.now():%Y%m%d_%H%M%S}.csv"
    fields = ["id", "observed_at", "location", "species_code", "name_cn", "scientific_name",
              "count", "confidence", "uncertain", "note", "image_path", "created_at",
              "qc_image", "qc_confidence_band", "qc_status", "qc_note"]
    for r in rows:
        conf = r.get("confidence") or 0.0
        r["qc_image"] = "有" if r.get("image_path") else "无"
        r["qc_confidence_band"] = "高" if conf >= 0.75 else "中" if conf >= 0.5 else "低"
        r["qc_status"] = "待复核" if r.get("uncertain") else "正常"
        notes = []
        if not r.get("image_path"):
            notes.append("缺照片")
        if r.get("uncertain"):
            notes.append("低置信度，需人工复核")
        if len(str(r.get("location", ""))) < 2:
            notes.append("地点缺失")
        r["qc_note"] = "；".join(notes)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return out


def stats() -> dict:
    rows = list_records()
    species = {r["species_code"] for r in rows}
    by_species: dict[str, int] = {}
    for r in rows:
        by_species[r["species_code"]] = by_species.get(r["species_code"], 0) + 1
    star = max(by_species.items(), key=lambda kv: kv[1])[0] if by_species else None
    protected = 0
    profiles = config.load_profiles()
    for r in rows:
        level = profiles.get(r["species_code"], {}).get("protection_level", "")
        if level in ("国家一级", "国家二级"):
            protected += 1
    return {
        "records": len(rows),
        "species": len(species),
        "protected": protected,
        "star_species": star,
        "by_species": by_species,
    }


if __name__ == "__main__":
    init_db()
    print("db:", config.RECORDS_DB)
    print(stats())
