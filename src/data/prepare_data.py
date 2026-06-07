"""
Reorganize the MVTec AD `pill` and `capsule` datasets into clean, binary
(good / defective) train/val/test ImageFolders for continual-learning experiments.

WHY THIS IS NEEDED
------------------
MVTec ships defect images ONLY inside `test/<defect_type>/`, while `train/`
contains good images only. For a supervised binary classifier we need both
classes in every split, so we:

  good       = train/good/*  +  test/good/*
  defective  = test/<every other folder>/*      (subtype recorded for Stage 2)

We then make a stratified train/val/test split (stratified by binary class AND
by defect subtype, so each split sees a fair mix of defect types). A manifest CSV
records the true subtype of every defective image, which the VLM stage uses as
ground truth.

USAGE
-----
    python -m src.data.prepare_data --input data/raw --out data/processed

`--input` may contain either the zip files (pill.zip, capsule.zip) or already
extracted product folders (pill/, capsule/).
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
import zipfile
from pathlib import Path

# Allow running both as a module and as a script
sys.path.append(str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _maybe_extract_zips(input_dir: Path, work_dir: Path) -> Path:
    """If `input_dir` holds zips, extract them into work_dir and return that.
    Otherwise assume product folders already live in input_dir."""
    zips = list(input_dir.glob("*.zip"))
    if not zips:
        return input_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    for z in zips:
        product = z.stem  # pill.zip -> pill
        target = work_dir / product
        if target.exists():
            print(f"  [skip] {product} already extracted")
            continue
        print(f"  [unzip] {z.name} -> {work_dir}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(work_dir)
    return work_dir


def _gather(product_root: Path):
    """Return two lists of (filepath, subtype):
    good items (subtype='good') and defective items (subtype=<defect folder>)."""
    good, defective = [], []

    for split in ("train", "test"):
        split_dir = product_root / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            files = [p for p in class_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
            if class_dir.name == "good":
                good += [(f, "good", None) for f in files]
            else:
                sub = class_dir.name
                # MVTec mask lives at <product>/ground_truth/<subtype>/<stem>_mask<ext>
                for f in files:
                    mask = product_root / "ground_truth" / sub / f"{f.stem}_mask{f.suffix}"
                    defective.append((f, sub, mask if mask.exists() else None))
    return good, defective


def _split(items, ratios, rng):
    """Stratified split (by subtype) of a list of (file, subtype, mask) tuples."""
    by_subtype: dict[str, list] = {}
    for item in items:
        by_subtype.setdefault(item[1], []).append(item)

    train, val, test = [], [], []
    for st, group in by_subtype.items():
        rng.shuffle(group)
        n = len(group)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        # guard tiny subtypes so val/test are never empty when n >= 3
        if n >= 3:
            n_train = min(n_train, n - 2)
            n_val = max(1, min(n_val, n - n_train - 1))
        train += group[:n_train]
        val += group[n_train:n_train + n_val]
        test += group[n_train + n_val:]
    return train, val, test


def _copy(items, product, split, out_dir, manifest_rows):
    for f, subtype, mask in items:
        binary = "good" if subtype == "good" else "defective"
        dest_dir = out_dir / product / split / binary
        dest_dir.mkdir(parents=True, exist_ok=True)
        # keep subtype + original name to avoid collisions and stay traceable
        dest_name = f"{subtype}__{f.name}"
        dest = dest_dir / dest_name
        shutil.copy2(f, dest)

        # Copy the ground-truth mask (if any) into a sibling _masks folder that
        # is OUTSIDE the train/val/test ImageFolder roots, so it is never picked
        # up as a third class. Mask keeps the same basename as its image.
        mask_rel = ""
        if mask is not None:
            mask_dir = out_dir / product / "_masks"
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask_dest = mask_dir / dest_name
            shutil.copy2(mask, mask_dest)
            mask_rel = str(mask_dest.relative_to(out_dir))

        manifest_rows.append(
            {"product": product, "split": split, "binary_label": binary,
             "subtype": subtype, "filepath": str(dest.relative_to(out_dir)),
             "mask_path": mask_rel}
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(config.RAW_DIR),
                    help="dir with the zips OR extracted product folders")
    ap.add_argument("--out", default=str(config.PROCESSED_DIR))
    ap.add_argument("--workdir", default=None,
                    help="where to extract zips (default: <input>/_extracted)")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    input_dir = Path(args.input)
    out_dir = Path(args.out)
    work_dir = Path(args.workdir) if args.workdir else input_dir / "_extracted"

    print("Step 1/3  locate / extract data")
    raw_root = _maybe_extract_zips(input_dir, work_dir)

    manifest_rows: list[dict] = []
    summary = []

    print("Step 2/3  reorganize into binary train/val/test")
    for product in config.PRODUCT_ORDER:
        product_root = raw_root / product
        if not product_root.is_dir():
            print(f"  [warn] product folder not found: {product_root} -- skipping")
            continue
        good, defective = _gather(product_root)
        g_tr, g_va, g_te = _split(good, config.SPLIT_RATIOS, rng)
        d_tr, d_va, d_te = _split(defective, config.SPLIT_RATIOS, rng)

        _copy(g_tr + d_tr, product, "train", out_dir, manifest_rows)
        _copy(g_va + d_va, product, "val", out_dir, manifest_rows)
        _copy(g_te + d_te, product, "test", out_dir, manifest_rows)

        summary.append((product, len(good), len(defective),
                        len(g_tr + d_tr), len(g_va + d_va), len(g_te + d_te)))

    print("Step 3/3  write manifest.csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["product", "split", "binary_label", "subtype", "filepath", "mask_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\n=== SUMMARY ===")
    print(f"{'product':10} {'good':>6} {'defect':>7} {'train':>7} {'val':>6} {'test':>6}")
    for p, ng, nd, ntr, nva, nte in summary:
        print(f"{p:10} {ng:>6} {nd:>7} {ntr:>7} {nva:>6} {nte:>6}")
    print(f"\nManifest: {manifest_path}")
    print(f"Total images placed: {len(manifest_rows)}")


if __name__ == "__main__":
    main()
