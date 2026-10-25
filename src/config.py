"""Project configuration and data-pack discovery for Q1 (滇境生灵)."""
from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
MODELS = PROJECT / "models"
OUTPUT = PROJECT / "output"
DATA = PROJECT / "data"
for d in (MODELS, OUTPUT, DATA):
    d.mkdir(parents=True, exist_ok=True)

WORKSPACE = Path(r"D:\competation")


def _find_pack() -> Path:
    """Locate the Q1 data pack without hard-coding the (messy) folder nesting."""
    marker = "species_profiles.csv"
    hits = sorted(WORKSPACE.glob(f"**/{marker}"))
    if not hits:
        raise FileNotFoundError(f"could not find {marker} under {WORKSPACE}")
    return hits[0].parent.parent.parent  # .../【发放给参赛队伍】物种图像识别数据包


PACK = _find_pack()
CLEAN = PACK / "02_清洗数据"
TRAIN_DIR = CLEAN / "species_images_train"
TEST_DIR = CLEAN / "species_images_test"
HARD_DIR = PACK / "03_评测真值" / "hard_cases" / "images"
MANIFEST = CLEAN / "clean_image_manifest.csv"
PROFILES = CLEAN / "species_profiles" / "species_profiles.csv"
SOURCE_DOCS = PACK / "00_来源与说明"

MODEL_PATH = MODELS / "ensemble.pt"
# Q1_DB lets tests use a throwaway database instead of the demo one.
RECORDS_DB = Path(os.environ.get("Q1_DB", str(DATA / "observations.db")))
CARD_DIR = OUTPUT / "cards"

IMAGE_SIZE = 256
SEED = 42
# Below this TOP-1 probability the app refuses to force a decision.
CONF_THRESHOLD = 0.55

FONT_REGULAR = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"


def load_profiles() -> dict[str, dict]:
    import pandas as pd

    df = pd.read_csv(PROFILES, encoding="utf-8-sig")
    return {row["species_code"]: row.to_dict() for _, row in df.iterrows()}


def load_manifest() -> "pd.DataFrame":
    import pandas as pd

    df = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    df["path"] = df.apply(
        lambda r: (TRAIN_DIR if r["split"] == "train" else TEST_DIR) / r["clean_file"], axis=1
    )
    df["exists"] = df["path"].map(lambda p: p.exists())
    return df


def class_names() -> list[str]:
    df = load_manifest()
    return sorted(df["species_code"].unique())


if __name__ == "__main__":
    df = load_manifest()
    print(f"pack     : {PACK}")
    print(f"images   : {len(df)} (train={sum(df.split == 'train')}, test={sum(df.split == 'test')})")
    print(f"missing  : {sum(~df.exists)}")
    print(f"classes  : {len(class_names())}")
    print(json.dumps({"pack": str(PACK), "model": str(MODEL_PATH)}, ensure_ascii=False, indent=2))
