"""Build models/ensemble.json from trained checkpoints (one per backbone by default)."""
from __future__ import annotations

import argparse
import json

import torch

import config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="*", help="checkpoint filenames; default = best per backbone")
    ap.add_argument("--threshold", type=float, default=config.CONF_THRESHOLD)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    ckpts = {}
    for p in sorted(config.MODELS.glob("*.pt")):
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001
            print(f"skip {p.name}: {e}")
            continue
        key = f"{ck.get('model', p.stem)}_{ck.get('size', '')}"
        score = ck.get("val_top1", 0.0)
        if key not in ckpts or score > ckpts[key][0]:
            ckpts[key] = (score, p, ck)

    if args.members:
        chosen = [p for b in args.members for p in config.MODELS.glob(b)]
    else:
        chosen = [v[1] for v in ckpts.values()]

    if not chosen:
        raise SystemExit("no checkpoints found; train a model first")

    classes = None
    members = []
    for p in chosen:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        classes = classes or ck["classes"]
        assert ck["classes"] == classes, f"class list mismatch in {p.name}"
        members.append({
            "checkpoint": p.name,
            "size": int(ck["size"]),
            "weight": 1.0,
            "val_top1": round(float(ck.get("val_top1", 0.0)), 4),
            "backbone": ck.get("model"),
        })
        print(f"+ {p.name:38s} val_top1={ck.get('val_top1', 0):.4f} size={ck['size']}")

    meta = {"members": members, "classes": classes, "temperature": 1.0,
            "threshold": args.threshold, "val_frac": args.val_frac, "seed": args.seed,
            "created_at": __import__("datetime").datetime.now().isoformat(timespec="seconds")}
    out = config.MODELS / "ensemble.json"
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out} ({len(members)} members)")


if __name__ == "__main__":
    main()
