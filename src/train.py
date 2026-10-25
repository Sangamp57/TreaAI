"""Fine-tune a torchvision backbone on the 33-species Yunnan pack (CPU friendly)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    convnext_tiny,
    efficientnet_b0,
    resnet18,
    resnet50,
)

import config
from data import load_split, make_loader, set_seed

BACKBONES = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
    "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
    "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
}


def build_model(name: str, n_classes: int, pretrained: bool = True) -> nn.Module:
    factory, weights = BACKBONES[name]
    model = factory(weights=weights if pretrained else None)
    if name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, n_classes))
    elif name.startswith("efficientnet"):
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, n_classes))
    elif name.startswith("convnext"):
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, n_classes)
    else:  # pragma: no cover
        raise ValueError(name)
    return model


def head_parameters(model: nn.Module, name: str):
    if name.startswith("resnet"):
        return model.fc.parameters()
    if name.startswith("efficientnet"):
        return model.classifier.parameters()
    return model.classifier[2].parameters()


@torch.no_grad()
def evaluate(model, loader, device, topk=(1, 3)):
    model.eval()
    correct = {k: 0 for k in topk}
    total = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += crit(logits, y).item() * len(y)
        for k in topk:
            pred = logits.topk(k, dim=1).indices
            correct[k] += (pred == y[:, None]).any(dim=1).sum().item()
        total += len(y)
    return {f"top{k}": correct[k] / total for k in topk}, loss_sum / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="convnext_tiny", choices=list(BACKBONES))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=8e-5)
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--init", default="", help="checkpoint to warm-start from (progressive resizing)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = torch.device("cpu")

    tr, va, te, classes = load_split(args.val_frac, args.seed)
    train_loader = make_loader(tr, classes, args.size, True, args.batch_size)
    val_loader = make_loader(va, classes, args.size, False, args.batch_size, shuffle=False)

    model = build_model(args.model, len(classes)).to(device)
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
        print(f"warm start from {Path(args.init).name}: missing={len(missing)} unexpected={len(unexpected)}",
              flush=True)
    head_ids = {id(p) for p in head_parameters(model, args.model)}
    groups = [
        {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": args.lr_backbone},
        {"params": [p for p in model.parameters() if id(p) in head_ids], "lr": args.lr_head},
    ]
    opt = AdamW(groups, weight_decay=args.weight_decay)
    steps = max(1, len(train_loader))
    warm = LinearLR(opt, start_factor=0.1, total_iters=args.warmup * steps)
    cosine = CosineAnnealingLR(opt, T_max=max(1, (args.epochs - args.warmup) * steps), eta_min=1e-6)
    sched = SequentialLR(opt, [warm, cosine], milestones=[args.warmup * steps])

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    name = f"{args.model}_{args.size}" + (f"_{args.tag}" if args.tag else "")
    ckpt_path = config.MODELS / f"{name}.pt"
    log_path = config.OUTPUT / f"train_{name}.json"
    best = -1.0
    history = []

    print(f"== {name}: train={len(tr)} val={len(va)} classes={len(classes)} "
          f"steps/epoch={steps} threads={torch.get_num_threads()}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cpu", enabled=False):
                logits = model(x)
                loss = crit(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            running += loss.item() * len(y)
            seen += len(y)
        metrics, val_loss = evaluate(model, val_loader, device)
        rec = {"epoch": epoch, "loss": running / seen, "val_loss": val_loss,
               **metrics, "sec": round(time.time() - t0, 1)}
        history.append(rec)
        flag = ""
        if metrics["top1"] > best:
            best = metrics["top1"]
            torch.save({"model": args.model, "size": args.size, "classes": classes,
                        "state_dict": model.state_dict(), "val_top1": best,
                        "val_top3": metrics["top3"], "epoch": epoch, "args": vars(args)},
                       ckpt_path)
            flag = " *"
        print(f"epoch {epoch:02d}/{args.epochs} loss={rec['loss']:.3f} "
              f"val_top1={metrics['top1']:.3f} val_top3={metrics['top3']:.3f} "
              f"{rec['sec']}s{flag}", flush=True)
        log_path.write_text(json.dumps({"args": vars(args), "best": best, "history": history},
                                       ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"best val top1={best:.3f} -> {ckpt_path}", flush=True)
    print(json.dumps({"checkpoint": str(ckpt_path), "best_val_top1": best}))


if __name__ == "__main__":
    main()
