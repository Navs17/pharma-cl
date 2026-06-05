"""
Stage 0 baseline: train a STATIC binary classifier on a single product.

This is your sanity check and a reference point. It proves the data pipeline,
model, and training loop work before you add any continual-learning machinery.
Training on all products pooled together (a "joint" model) is the UPPER BOUND
that CL methods are compared against.

USAGE
-----
    python -m src.train_baseline --product pill
    python -m src.train_baseline --product joint      # pill + capsule pooled
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

import config
from src.data.datasets import load_imagefolder, make_loader
from src.models.model import build_model


def get_split(product: str, split: str):
    if product == "joint":
        from torch.utils.data import ConcatDataset
        parts = [load_imagefolder(p, split) for p in config.PRODUCT_ORDER]
        ds = ConcatDataset(parts)
        return ds
    return load_imagefolder(product, split)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default="pill",
                    help="pill | capsule | joint")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS_PER_EXPERIENCE)
    args = ap.parse_args()

    torch.manual_seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  product: {args.product}")

    train_loader = make_loader(get_split(args.product, "train"), train=True)
    val_loader = make_loader(get_split(args.product, "val"), train=False)
    test_loader = make_loader(get_split(args.product, "test"), train=False)

    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=config.LEARNING_RATE,
                                 weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = config.OUTPUT_DIR / f"baseline_{args.product}.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch:02d}  loss {running/len(train_loader.dataset):.4f}  val_acc {val_acc:.3f}")
        if val_acc >= best_val:
            best_val = val_acc
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    test_acc = evaluate(model, test_loader, device)
    print(f"\nBest val_acc {best_val:.3f}  |  test_acc {test_acc:.3f}")
    print(f"Saved: {ckpt}")


if __name__ == "__main__":
    main()
