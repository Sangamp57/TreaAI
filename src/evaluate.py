"""Evaluate the ensemble on the official test split and the hard-case subset."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import config
from data import hard_case_frame, load_split
from infer import ENSEMBLE_JSON, Recognizer


def collect(recognizer: Recognizer, frame: pd.DataFrame, tta: bool = True) -> tuple[np.ndarray, np.ndarray]:
    probs, labels = [], []
    for _, row in frame.iterrows():
        preds = recognizer.predict(Image.open(row["path"]), topk=len(recognizer.classes), tta=tta)
        vec = np.zeros(len(recognizer.classes), dtype=np.float32)
        for p in preds:
            vec[recognizer.classes.index(p["species_code"])] = p["prob"]
        probs.append(vec)
        labels.append(recognizer.classes.index(row["species_code"]))
    return np.stack(probs), np.array(labels)


def metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    top1 = probs.argmax(1)
    top3 = probs.argsort(1)[:, -3:]
    acc1 = float((top1 == labels).mean())
    acc3 = float((top3 == labels[:, None]).any(1).mean())
    return {"top1": acc1, "top3": acc3, "n": int(len(labels))}


def fit_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    """Power scaling p^(1/T) fitted by NLL on the validation split."""
    logp = torch.log(torch.tensor(probs).clamp_min(1e-9))
    y = torch.tensor(labels)
    best_t, best_nll = 1.0, float("inf")
    for t in np.arange(0.5, 3.01, 0.05):
        nll = F.nll_loss(F.log_softmax(logp / t, dim=1), y).item()
        if nll < best_nll:
            best_nll, best_t = nll, float(round(t, 2))
    return best_t


def tune_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """Highest coverage whose accepted predictions are >=95% correct; else 0.55."""
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    best = None
    for t in np.arange(0.30, 0.96, 0.01):
        mask = conf >= t
        if mask.sum() < 0.5 * len(conf):
            continue
        precision = correct[mask].mean()
        if precision >= 0.95:
            best = float(round(t, 2))
    return best if best is not None else 0.55


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tta", default="flip", choices=["flip", "5crop", "none"])
    ap.add_argument("--skip-tune", action="store_true")
    ap.add_argument("--eval-size", type=int, default=None)
    ap.add_argument("--sweep", action="store_true", help="choose the inference size on validation")
    ap.add_argument("--per-member", action="store_true", help="report each member's validation accuracy")
    ap.add_argument("--val-only", action="store_true", help="stop after validation (no test/hard scoring)")
    args = ap.parse_args()

    meta = json.loads(ENSEMBLE_JSON.read_text(encoding="utf-8"))
    tr, va, te, classes = load_split(meta.get("val_frac", 0.2), meta.get("seed", config.SEED))
    print(f"members={len(meta['members'])} classes={len(classes)} "
          f"val={len(va)} test={len(te)}", flush=True)

    if args.sweep:
        best = (-1.0, None)
        for size in (224, 256, 288, 320):
            probe = Recognizer.load(eval_size=size)
            vp, vl = collect(probe, va, tta=True)
            acc = metrics(vp, vl)["top1"]
            print(f"  eval_size={size}: val_top1={acc:.4f}", flush=True)
            if acc > best[0]:
                best = (acc, size)
        meta["eval_size"] = best[1]
        ENSEMBLE_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"chosen eval_size={best[1]} (val_top1={best[0]:.4f})", flush=True)

    eval_size = args.eval_size or meta.get("eval_size")
    rec = Recognizer.load(eval_size=eval_size)

    if args.per_member:
        print("per-member validation:", flush=True)
        for m in meta["members"]:
            solo = Recognizer([m], classes, temperature=1.0, eval_size=eval_size)
            vp, vl = collect(solo, va, tta="flip")
            print(f"  {m['checkpoint']:34s} top1={metrics(vp, vl)['top1']:.4f}", flush=True)

    if not args.skip_tune:
        vprobs, vlabels = collect(rec, va, tta=True)
        temp = fit_temperature(vprobs, vlabels)
        thr = tune_threshold(vprobs, vlabels)
        meta.update({"temperature": temp, "threshold": thr})
        ENSEMBLE_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"validation: top1={metrics(vprobs, vlabels)['top1']:.4f} "
              f"temperature={temp} threshold={thr}", flush=True)
        rec = Recognizer.load(eval_size=eval_size)

    if args.val_only:
        vprobs, vlabels = collect(rec, va, tta=args.tta)
        print(f"ensemble validation: top1={metrics(vprobs, vlabels)['top1']:.4f} "
              f"top3={metrics(vprobs, vlabels)['top3']:.4f}", flush=True)
        return

    tta = args.tta
    results = {}
    for name, frame in (("test", te), ("hard", hard_case_frame())):
        if frame.empty:
            continue
        probs, labels = collect(rec, frame, tta=tta)
        m = metrics(probs, labels)
        m["per_class"] = {}
        for i, c in enumerate(classes):
            mask = labels == i
            if mask.any():
                m["per_class"][c] = float((probs[mask].argmax(1) == labels[mask]).mean())
        thr = float(meta.get("threshold") or config.CONF_THRESHOLD)
        accepted = probs.max(1) >= thr
        m["confident_coverage"] = float(accepted.mean())
        m["confident_accuracy"] = (float((probs.argmax(1) == labels)[accepted].mean())
                                   if accepted.any() else None)
        results[name] = m
        print(f"{name}: top1={m['top1']:.4f} top3={m['top3']:.4f} n={m['n']}", flush=True)
        pd.DataFrame({"true": [classes[i] for i in labels],
                      "pred": [classes[i] for i in probs.argmax(1)],
                      "confidence": probs.max(1).round(4)}).to_csv(
            config.OUTPUT / f"predictions_{name}.csv", index=False, encoding="utf-8-sig")

        cm = pd.DataFrame(0, index=classes, columns=classes)
        for t, p in zip(labels, probs.argmax(1)):
            cm.iloc[t, p] += 1
        cm.to_csv(config.OUTPUT / f"confusion_{name}.csv", encoding="utf-8-sig")

    out = {"members": [m["checkpoint"] for m in meta["members"]],
           "temperature": rec.temperature, "threshold": meta.get("threshold"),
           "tta": tta, **results}
    (config.OUTPUT / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                encoding="utf-8")

    lines = ["# Q1 评测报告", "",
             f"- 集成成员：{', '.join(out['members'])}",
             f"- 温度缩放：{rec.temperature}　低置信阈值：{meta.get('threshold')}",
             f"- TTA：{tta}", "",
             "| 数据集 | 样本 | TOP-1 | TOP-3 |", "|---|---|---|---|"]
    for name in ("test", "hard"):
        if name in results:
            r = results[name]
            lines.append(f"| {'官方测试集' if name == 'test' else '困难子集'} | {r['n']} | "
                         f"{r['top1'] * 100:.1f}% | {r['top3'] * 100:.1f}% |")
    worst = sorted(results["test"]["per_class"].items(), key=lambda kv: kv[1])[:5]
    lines += ["", "## 最弱物种", ""] + [f"- {c}: {v * 100:.0f}%" for c, v in worst] + [""]
    (config.OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
