"""
Stage 2: VLM defect DESCRIBER (inference only, no training).

When Stage 1 flags an item as 'defective', this names the defect TYPE using
zero-shot CLIP: the crop is scored against text prompts (config.DEFECT_PROMPTS)
and the best match wins. Because it needs no training, it sidesteps the tiny
per-defect-type data problem.

This script also EVALUATES the describer: it runs on the defective test images
and compares CLIP's predicted subtype against the true subtype from manifest.csv,
giving you a zero-shot type-accuracy number for your results chapter.

OPTIONAL UPGRADE: swap CLIP for a small generative VLM (SmolVLM2 / Qwen3-VL-2B /
Moondream) to produce free-text descriptions instead of a fixed label set. Keep
it inference-only on the T4.

USAGE
-----
    python -m src.vlm.describe --product pill
"""
from __future__ import annotations

import argparse
import csv

import torch
from PIL import Image

import config

try:
    import open_clip
except ImportError as e:
    raise SystemExit("Install open_clip_torch:  pip install open_clip_torch") from e


class CLIPDescriber:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL)

    def describe(self, image_path: str, product: str):
        prompts = config.DEFECT_PROMPTS[product]
        labels = list(prompts.keys())
        texts = self.tokenizer([prompts[k] for k in labels]).to(self.device)
        image = self.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_feat = self.model.encode_image(image)
            txt_feat = self.model.encode_text(texts)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
            sims = (img_feat @ txt_feat.T).softmax(dim=-1).squeeze(0)

        best = int(sims.argmax())
        return labels[best], prompts[labels[best]], float(sims[best])


def evaluate(product: str):
    describer = CLIPDescriber()
    manifest = config.PROCESSED_DIR / "manifest.csv"
    rows = [r for r in csv.DictReader(open(manifest))
            if r["product"] == product and r["split"] == "test"
            and r["binary_label"] == "defective"]

    correct = 0
    print(f"{'true':16} {'predicted':16} {'conf':>5}")
    for r in rows:
        path = config.PROCESSED_DIR / r["filepath"]
        pred_label, desc, conf = describer.describe(str(path), product)
        ok = (pred_label == r["subtype"])
        correct += ok
        flag = "OK " if ok else "  x"
        print(f"{flag} {r['subtype']:13} {pred_label:16} {conf:.2f}  -> \"{desc}\"")

    if rows:
        print(f"\nZero-shot defect-type accuracy ({product}): {correct}/{len(rows)} = {correct/len(rows):.2%}")
        print("Note: combined/pill_type are intrinsically hard zero-shot; discuss this in the thesis.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default="pill", help="pill | capsule")
    args = ap.parse_args()
    evaluate(args.product)


if __name__ == "__main__":
    main()
