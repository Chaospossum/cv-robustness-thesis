"""
Unified training script for baseline and all augmented strategies.

Usage:
  # Baseline
  python -m src.train --strategy baseline --model resnet18 --seed 0

  # Mixed augmentation
  python -m src.train --strategy mixed --model resnet18 --seed 0

  # Single-type augmentation
  python -m src.train --strategy single --model resnet18 --deg gaussian_blur --seed 0

  # Curriculum augmentation
  python -m src.train --strategy curriculum --model resnet18 --deg gaussian_noise --seed 0

  # Run ALL experiments
  python -m src.train --run-all
"""

import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from src.models import get_model
from src.degradations import DEG_GRID, apply_degradation
from src.curriculum_scheduler import get_curriculum_severity
from src.evaluate import full_robustness_eval, CIFAR10_MEAN, CIFAR10_STD
from src.augmix import augmix_batch, jsd_loss


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
SEEDS = [0, 42, 123, 456, 789]
EPOCHS = 200
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
MILESTONES = [100, 150]
GAMMA = 0.1
MODELS = ["resnet18", "mobilenetv2"]
SINGLE_DEG_TYPES = ["gaussian_blur", "gaussian_noise"]  # only these two per proposal


# ──────────────────────────────────────────────────────────────────────────────
# Seeding
# ──────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────
def get_train_loader(batch_size: int = BATCH_SIZE) -> DataLoader:
    """
    CIFAR-10 training set with geometric augmentation only.
    Returns [0,1] tensors (normalization applied after degradation).
    """
    transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),  # [0,1]
    ])
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    return DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )


def normalize_batch(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Normalize a batch of [0,1] tensors with CIFAR-10 stats."""
    mean = CIFAR10_MEAN.to(device)
    std = CIFAR10_STD.to(device)
    return (images - mean) / std


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation application (per-image, on CPU before normalization)
# ──────────────────────────────────────────────────────────────────────────────
def augment_batch_single(images: torch.Tensor, deg_type: str) -> torch.Tensor:
    """Single-type: apply fixed deg_type with random severity per image."""
    levels = DEG_GRID[deg_type]
    out = []
    for img in images:
        sev_val = random.choice(levels)
        out.append(apply_degradation(img, deg_type, sev_val))
    return torch.stack(out)


def augment_batch_mixed(images: torch.Tensor) -> torch.Tensor:
    """Mixed: random deg_type + random severity per image."""
    deg_types = list(DEG_GRID.keys())
    out = []
    for img in images:
        deg_type = random.choice(deg_types)
        sev_val = random.choice(DEG_GRID[deg_type])
        out.append(apply_degradation(img, deg_type, sev_val))
    return torch.stack(out)


def augment_batch_curriculum(
    images: torch.Tensor, deg_type: str, epoch: int
) -> torch.Tensor:
    """Curriculum: single deg_type, severity ramps with epoch."""
    sev_val = get_curriculum_severity(deg_type, epoch, EPOCHS)
    out = []
    for img in images:
        out.append(apply_degradation(img, deg_type, sev_val))
    return torch.stack(out)


def augment_batch_curriculum_capped(
    images: torch.Tensor, deg_type: str, epoch: int
) -> torch.Tensor:
    """Capped curriculum: severity ramps linearly but caps at medium level."""
    sev_val = get_curriculum_severity(deg_type, epoch, EPOCHS, schedule="capped")
    out = []
    for img in images:
        out.append(apply_degradation(img, deg_type, sev_val))
    return torch.stack(out)


def augment_batch_curriculum_cosine(
    images: torch.Tensor, deg_type: str, epoch: int
) -> torch.Tensor:
    """Cosine curriculum: S-curve severity ramp."""
    sev_val = get_curriculum_severity(deg_type, epoch, EPOCHS, schedule="cosine")
    out = []
    for img in images:
        out.append(apply_degradation(img, deg_type, sev_val))
    return torch.stack(out)


def augment_batch_curriculum_clean50(
    images: torch.Tensor, deg_type: str, epoch: int
) -> torch.Tensor:
    """Curriculum with 50% clean mixing: half the batch stays clean."""
    sev_val = get_curriculum_severity(deg_type, epoch, EPOCHS)
    out = []
    for img in images:
        if random.random() < 0.5:
            out.append(apply_degradation(img, deg_type, sev_val))
        else:
            out.append(img)
    return torch.stack(out)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    strategy: str,
    deg_type: str,
    epoch: int,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        # Apply degradation augmentation (on CPU, before normalization)
        if strategy == "single":
            images = augment_batch_single(images, deg_type)
        elif strategy == "mixed":
            images = augment_batch_mixed(images)
        elif strategy == "curriculum":
            images = augment_batch_curriculum(images, deg_type, epoch)
        elif strategy == "curriculum_capped":
            images = augment_batch_curriculum_capped(images, deg_type, epoch)
        elif strategy == "curriculum_cosine":
            images = augment_batch_curriculum_cosine(images, deg_type, epoch)
        elif strategy == "curriculum_clean50":
            images = augment_batch_curriculum_clean50(images, deg_type, epoch)
        elif strategy in ("augmix", "augmix_nojsd"):
            pass  # AugMix handled below
        # baseline: no augmentation

        labels = labels.to(device)

        if strategy == "augmix":
            # AugMix: 3 forward passes + JSD consistency loss
            images_clean = normalize_batch(images.to(device), device)
            images_aug1 = normalize_batch(augmix_batch(images).to(device), device)
            images_aug2 = normalize_batch(augmix_batch(images).to(device), device)

            optimizer.zero_grad()
            logits_clean = model(images_clean)
            logits_aug1 = model(images_aug1)
            logits_aug2 = model(images_aug2)

            ce_loss = criterion(logits_clean, labels)
            js_loss = jsd_loss(logits_clean, logits_aug1, logits_aug2)
            loss = ce_loss + 12.0 * js_loss  # lambda=12 from AugMix paper
            loss.backward()
            optimizer.step()
        elif strategy == "augmix_nojsd":
            # AugMix augmentation only, standard CE loss (no JSD)
            images = normalize_batch(augmix_batch(images).to(device), device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            # Standard single forward pass
            images = normalize_batch(images.to(device), device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

    return total_loss / total_samples


def run_experiment(
    strategy: str,
    model_name: str,
    seed: int,
    deg_type: str = "none",
    results_dir: str = "results",
) -> pd.DataFrame:
    """
    Run a single training experiment (200 epochs) and evaluate robustness.
    Returns DataFrame with results.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy} | Model: {model_name} | Seed: {seed} | Deg: {deg_type}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # Build model
    model = get_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)

    # Data
    loader = get_train_loader()

    # Train
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(
            model, loader, optimizer, criterion, device, strategy, deg_type, epoch
        )
        scheduler.step()
        if epoch % 20 == 0 or epoch == 1:
            elapsed = time.time() - start
            print(f"  Epoch {epoch:3d}/{EPOCHS} | Loss: {loss:.4f} | Time: {elapsed:.0f}s")

    train_time = time.time() - start
    print(f"Training complete in {train_time:.0f}s")

    # Save checkpoint
    subdir = "baseline" if strategy == "baseline" else "augmented"
    ckpt_dir = os.path.join(results_dir, subdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    strategies_with_deg = ("single", "curriculum", "curriculum_capped", "curriculum_cosine", "curriculum_clean50")
    if strategy in strategies_with_deg:
        ckpt_name = f"{model_name}_{strategy}_{deg_type}_seed{seed}.pt"
    else:
        ckpt_name = f"{model_name}_{strategy}_seed{seed}.pt"
    torch.save(model.state_dict(), os.path.join(ckpt_dir, ckpt_name))

    # Evaluate robustness
    print("Evaluating robustness...")
    results = full_robustness_eval(model, device, model_name, strategy, seed, deg_type)

    # Save results CSV (append-friendly)
    df = pd.DataFrame(results)
    csv_dir = os.path.join(results_dir, subdir)
    os.makedirs(csv_dir, exist_ok=True)
    if strategy in strategies_with_deg:
        csv_name = f"{model_name}_{strategy}_{deg_type}_robustness.csv"
    else:
        csv_name = f"{model_name}_{strategy}_robustness.csv"
    csv_path = os.path.join(csv_dir, csv_name)

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Run all experiments
# ──────────────────────────────────────────────────────────────────────────────
def run_all(results_dir: str = "results"):
    """Run the full experimental matrix (50 original + 30 extension runs)."""
    experiments = []

    # Baseline: 2 models × 5 seeds = 10 runs
    for model_name in MODELS:
        for seed in SEEDS:
            experiments.append(("baseline", model_name, seed, "none"))

    # Mixed: 2 models × 5 seeds = 10 runs
    for model_name in MODELS:
        for seed in SEEDS:
            experiments.append(("mixed", model_name, seed, "none"))

    # Single-type: 2 deg × 2 models × 5 seeds = 20 runs
    for deg_type in SINGLE_DEG_TYPES:
        for model_name in MODELS:
            for seed in SEEDS:
                experiments.append(("single", model_name, seed, deg_type))

    # Curriculum: 2 deg × 2 models × 5 seeds = 20 runs
    for deg_type in SINGLE_DEG_TYPES:
        for model_name in MODELS:
            for seed in SEEDS:
                experiments.append(("curriculum", model_name, seed, deg_type))

    # === EXTENSIONS ===

    # AugMix: 2 models × 5 seeds = 10 runs
    for model_name in MODELS:
        for seed in SEEDS:
            experiments.append(("augmix", model_name, seed, "none"))

    # AugMix no JSD: 2 models × 5 seeds = 10 runs
    for model_name in MODELS:
        for seed in SEEDS:
            experiments.append(("augmix_nojsd", model_name, seed, "none"))

    # Capped curriculum (blur only): 2 models × 5 seeds = 10 runs
    for model_name in MODELS:
        for seed in SEEDS:
            experiments.append(("curriculum_capped", model_name, seed, "gaussian_blur"))

    # Curriculum with 50% clean (blur only): 2 models × 5 seeds = 10 runs
    for model_name in MODELS:
        for seed in SEEDS:
            experiments.append(("curriculum_clean50", model_name, seed, "gaussian_blur"))

    # Cosine curriculum: 2 deg × 2 models × 5 seeds = 20 runs
    for deg_type in SINGLE_DEG_TYPES:
        for model_name in MODELS:
            for seed in SEEDS:
                experiments.append(("curriculum_cosine", model_name, seed, deg_type))

    print(f"Total experiments: {len(experiments)}")
    for i, (strategy, model_name, seed, deg_type) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}]")
        run_experiment(strategy, model_name, seed, deg_type, results_dir)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Robustness training experiments")
    parser.add_argument("--strategy", choices=[
        "baseline", "single", "mixed", "curriculum",
        "augmix", "augmix_nojsd", "curriculum_capped", "curriculum_cosine", "curriculum_clean50",
    ])
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--deg", choices=list(DEG_GRID.keys()), default="none")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--run-all", action="store_true", help="Run full experimental matrix")
    args = parser.parse_args()

    if args.run_all:
        run_all(args.results_dir)
    else:
        if not args.strategy or not args.model:
            parser.error("--strategy and --model are required unless using --run-all")
        needs_deg = ("single", "curriculum", "curriculum_capped", "curriculum_cosine", "curriculum_clean50")
        if args.strategy in needs_deg and args.deg == "none":
            parser.error(f"--deg is required for {args.strategy} strategy")
        run_experiment(args.strategy, args.model, args.seed, args.deg, args.results_dir)


if __name__ == "__main__":
    main()
