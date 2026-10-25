"""Inference: load one or more fine-tuned backbones and predict with flip TTA."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import config
from data import IMAGENET_MEAN, IMAGENET_STD
from train import build_model

ENSEMBLE_JSON = config.MODELS / "ensemble.json"


class Recognizer:
    def __init__(self, members: list[dict], classes: list[str], temperature: float = 1.0,
                 eval_size: int | None = None, threshold: float = config.CONF_THRESHOLD):
        self.classes = classes
        self.temperature = temperature
        self.threshold = threshold
        self.members = []
        for m in members:
            ckpt = torch.load(config.MODELS / m["checkpoint"], map_location="cpu", weights_only=False)
            model = build_model(ckpt["model"], len(ckpt["classes"]), pretrained=False)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            size = int(eval_size or m.get("eval_size") or ckpt["size"])
            tf = transforms.Compose([
                transforms.Resize(int(round(size * 1.14))),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
            tf_resize = transforms.Compose([
                transforms.Resize(int(round(size * 1.14))),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
            self.members.append({"model": model, "size": size, "weight": float(m.get("weight", 1.0)),
                                 "tf": tf, "tf_resize": tf_resize, "name": ckpt["model"]})
        self.classes = classes

    @classmethod
    def load(cls, path: Path | None = None, eval_size: int | None = None) -> "Recognizer":
        path = path or ENSEMBLE_JSON
        meta = json.loads(path.read_text(encoding="utf-8"))
        return cls(meta["members"], meta["classes"], meta.get("temperature", 1.0),
                   eval_size=eval_size, threshold=meta.get("threshold", config.CONF_THRESHOLD))

    @torch.no_grad()
    def predict(self, image: Image.Image, topk: int = 5, tta: bool | str = True) -> list[dict]:
        image = image.convert("RGB")
        mode = "flip" if tta is True else "none" if tta is False else tta
        probs = torch.zeros(len(self.classes))
        total_w = 0.0
        for m in self.members:
            if mode == "5crop":
                full = m["tf_resize"](image)
                crops = transforms.FiveCrop(m["size"])(full)
                batch = torch.stack([v for c in crops for v in (c, torch.flip(c, dims=[2]))])
            else:
                x = m["tf"](image).unsqueeze(0)
                batch = torch.cat([x, torch.flip(x, dims=[3])]) if mode == "flip" else x
            logits = m["model"](batch)
            p = F.softmax(logits / self.temperature, dim=1).mean(dim=0)
            probs += p * m["weight"]
            total_w += m["weight"]
        probs /= max(total_w, 1e-9)
        k = min(topk, len(self.classes))
        vals, idx = probs.topk(k)
        return [{"species_code": self.classes[i], "prob": float(v)} for v, i in zip(vals, idx)]


def is_ready() -> bool:
    return ENSEMBLE_JSON.exists()


if __name__ == "__main__":
    r = Recognizer.load()
    df = config.load_manifest()
    row = df[df["split"] == "test"].iloc[0]
    preds = r.predict(Image.open(row["path"]))
    print(row["species_code"], "->", [(p["species_code"], round(p["prob"], 3)) for p in preds])
