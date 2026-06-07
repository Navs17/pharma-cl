"""
Stage 2: VLM defect DESCRIBER (inference only, no training).

Names the defect TYPE of a flagged item via zero-shot CLIP, with two upgrades
over a naive whole-image approach:

  1. Mask-guided cropping: MVTec ground-truth masks localize the defect (often
     <1% of the image). We crop to the defect region (+ margin) so CLIP sees the
     defect instead of just "a pill".
  2. Prompt ensembling: each defect type is described with several templates
     whose text embeddings are averaged, reducing single-prompt wording bias.

Evaluates on the defective test images and reports zero-shot type accuracy
against the true subtype from manifest.csv. Use --no-crop for the full-image
ablation (handy to quote the before/after improvement in the thesis).

USAGE
-----
    python -m src.vlm.describe --product pill
    python -m src.vlm.describe --product capsule --no-crop
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
from PIL import Image

import config

try:
    import open_clip
except ImportError as e:
    raise SystemExit("Install open_clip_torch:  pip install open_clip_torch") from e


# Several templates per class; their text embeddings are averaged (ensembled).
PROMPT_TEMPLATES = [
    "a photo of {d}",
    "a close-up photo of {d}",
    "an image showing {d}",
    "{d}",
]


def crop_to_defect(image: Image.Image, mask_path: str,
                   margin: float = 0.6, min_size: int = 80) -> Image.Image:
    """Crop to the defect region defined by the mask (+ margin for context).
    Falls back to the full image if the mask is missing or empty."""
    if not mask_path:
        return image
    try:
        mask = np.array(Image.open(config.PROCESSED_DIR / mask_path).convert("L"))
    except Exception:
        return image
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return image
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    w, h = x1 - x0, y1 - y0
    pad_x = max(int(w * margin), (min_size - w) // 2, 0)
    pad_y = max(int(h * margin), (min_size - h) // 2, 0)
    W, H = image.size
    box = (max(x0 - pad_x, 0), max(y0 - pad_y, 0),
           min(x1 + pad_x, W), min(y1 + pad_y, H))
    return image.crop(box)


class CLIPDescriber:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL)
        self._text_cache = {}

    def _class_text_features(self, product: str):
        """Ensembled, L2-normalized text features per defect class (cached)."""
        if product in self._text_cache:
            return self._text_cache[product]
        prompts = config.DEFECT_PROMPTS[product]
        labels = list(prompts.keys())
        feats = []
        with torch.no_grad():
            for k in labels:
                texts = [t.format(d=prompts[k]) for t in PROMPT_TEMPLATES]
                tok = self.tokenizer(texts).to(self.device)
                tf = self.model.encode_text(tok)
                tf = tf / tf.norm(dim=-1, keepdim=True)
                feats.append(tf.mean(dim=0))          # average over templates
        mat = torch.stack(feats)
        mat = mat / mat.norm(dim=-1, keepdim=True)
        self._text_cache[product] = (labels, mat)
        return labels, mat

    def describe(self, image_path, product, mask_path="", use_crop=True):
        labels, txt = self._class_text_features(product)
        img = Image.open(image_path).convert("RGB")
        if use_crop:
            img = crop_to_defect(img, mask_path)
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            f = self.model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            sims = (f @ txt.T).softmax(dim=-1).squeeze(0)
        best = int(sims.argmax())
        return labels[best], config.DEFECT_PROMPTS[product][labels[best]], float(sims[best])


def evaluate(product: str, use_crop: bool = True):
    describer = CLIPDescriber()
    rows = [r for r in csv.DictReader(open(config.PROCESSED_DIR / "manifest.csv"))
            if r["product"] == product and r["split"] == "test"
            and r["binary_label"] == "defective"]

    correct = 0
    per_type = {}
    print(f"{'':3}{'true':16}{'predicted':16}{'conf':>5}")
    for r in rows:
        pred, desc, conf = describer.describe(
            config.PROCESSED_DIR / r["filepath"], product,
            mask_path=r.get("mask_path", ""), use_crop=use_crop)
        ok = (pred == r["subtype"])
        correct += ok
        d = per_type.setdefault(r["subtype"], [0, 0])
        d[0] += ok
        d[1] += 1
        print(f"{'OK ' if ok else '  x'}{r['subtype']:15}{pred:16}{conf:.2f}")

    mode = "mask-crop" if use_crop else "full-image"
    if rows:
        print(f"\nZero-shot defect-type accuracy ({product}, {mode}): "
              f"{correct}/{len(rows)} = {correct/len(rows):.2%}")
        print("Per-type:", {k: f"{v[0]}/{v[1]}" for k, v in sorted(per_type.items())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default="pill", help="pill | capsule")
    ap.add_argument("--no-crop", action="store_true",
                    help="disable mask cropping (full-image ablation)")
    args = ap.parse_args()
    evaluate(args.product, use_crop=not args.no_crop)


if __name__ == "__main__":
    main()
