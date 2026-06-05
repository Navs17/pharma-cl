"""
Stage 1 (thesis core): DOMAIN-INCREMENTAL continual learning.

Experience 0 = pill, Experience 1 = capsule. The model learns pill, then must
learn capsule WITHOUT forgetting pill. We compare several CL strategies and log
accuracy + forgetting (BWT) per experience using Avalanche.

USAGE
-----
    python -m src.continual.run_continual --strategy replay
    python -m src.continual.run_continual --strategy all      # run every method

Strategies: naive | cumulative | ewc | lwf | replay | all
  naive       -> lower bound (shows catastrophic forgetting)
  cumulative  -> upper bound (re-trains on all data seen so far)
  ewc / lwf   -> regularization / distillation
  replay      -> rehearsal buffer (your strong baseline; add DER++ later)
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

import config
from src.data.datasets import load_imagefolder
from src.models.model import build_model

# --- Avalanche imports (names are stable across 0.4-0.6; wrap the one that moved) ---
from avalanche.benchmarks.generators import dataset_benchmark
try:  # API name differs slightly between versions
    from avalanche.benchmarks.utils import make_classification_dataset as wrap_ds
except ImportError:  # newer Avalanche
    from avalanche.benchmarks.utils import as_classification_dataset as wrap_ds

from avalanche.training.supervised import Naive, Cumulative, EWC, LwF, Replay
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    accuracy_metrics, loss_metrics, forgetting_metrics)
from avalanche.logging import InteractiveLogger, CSVLogger


def build_benchmark():
    """One experience per product, in config.PRODUCT_ORDER."""
    train_sets, test_sets = [], []
    for product in config.PRODUCT_ORDER:
        train_sets.append(wrap_ds(load_imagefolder(product, "train")))
        test_sets.append(wrap_ds(load_imagefolder(product, "test")))
    return dataset_benchmark(train_sets, test_sets)


def make_strategy(name, model, optimizer, criterion, evaluator, device):
    common = dict(
        train_mb_size=config.BATCH_SIZE,
        train_epochs=config.EPOCHS_PER_EXPERIENCE,
        eval_mb_size=config.BATCH_SIZE,
        evaluator=evaluator,
        device=device,
    )
    if name == "naive":
        return Naive(model, optimizer, criterion, **common)
    if name == "cumulative":
        return Cumulative(model, optimizer, criterion, **common)
    if name == "ewc":
        return EWC(model, optimizer, criterion,
                   ewc_lambda=config.EWC_LAMBDA, **common)
    if name == "lwf":
        return LwF(model, optimizer, criterion,
                   alpha=config.LWF_ALPHA, temperature=config.LWF_TEMPERATURE, **common)
    if name == "replay":
        return Replay(model, optimizer, criterion,
                      mem_size=config.REPLAY_BUFFER_SIZE, **common)
    raise ValueError(f"unknown strategy: {name}")


def run_one(name, benchmark, device):
    print(f"\n{'='*60}\nStrategy: {name}\n{'='*60}")
    torch.manual_seed(config.SEED)

    model = build_model().to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_logger = CSVLogger(log_folder=str(config.OUTPUT_DIR / f"cl_{name}"))
    evaluator = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        loss_metrics(experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger(), csv_logger],
    )

    strategy = make_strategy(name, model, optimizer, criterion, evaluator, device)

    results = []
    for experience in benchmark.train_stream:
        product = config.PRODUCT_ORDER[experience.current_experience]
        print(f"\n--- Training on experience {experience.current_experience} ({product}) ---")
        strategy.train(experience)
        # Evaluate on the WHOLE test stream so we can see forgetting on earlier products
        results.append(strategy.eval(benchmark.test_stream))

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="replay",
                    help="naive | cumulative | ewc | lwf | replay | all")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    benchmark = build_benchmark()

    names = (["naive", "cumulative", "ewc", "lwf", "replay"]
             if args.strategy == "all" else [args.strategy])
    for name in names:
        run_one(name, benchmark, device)

    print("\nDone. Per-strategy CSV logs are in outputs/cl_<strategy>/.")
    print("Compare final stream accuracy and StreamForgetting across strategies for your results table.")


if __name__ == "__main__":
    main()
