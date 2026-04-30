"""
Robustness evaluation: test a trained model on clean + all 18 degradation conditions.
Returns a list of dicts suitable for building a DataFrame.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import numpy as np

from src.degradations import DEG_GRID, apply_degradation


def _get_test_loader(batch_size: int = 256) -> DataLoader:
    """CIFAR-10 test set, normalized to [0,1] only (no augmentation)."""
    transform = T.ToTensor()  # [0,1] float tensor
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


# CIFAR-10 channel-wise mean and std for normalizing after degradation
CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)


def _normalize(img_tensor: torch.Tensor) -> torch.Tensor:
    """Normalize a [0,1] tensor with CIFAR-10 stats."""
    return (img_tensor - CIFAR10_MEAN.to(img_tensor.device)) / CIFAR10_STD.to(img_tensor.device)


@torch.no_grad()
def evaluate_clean(model: nn.Module, device: torch.device, batch_size: int = 256) -> float:
    """Evaluate on clean CIFAR-10 test set. Returns accuracy in %."""
    model.eval()
    loader = _get_test_loader(batch_size)
    correct = total = 0
    for images, labels in loader:
        images = _normalize(images).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def evaluate_degraded(
    model: nn.Module,
    device: torch.device,
    deg_type: str,
    severity_value: float,
    batch_size: int = 256,
) -> float:
    """Evaluate on degraded CIFAR-10 test set. Returns accuracy in %."""
    model.eval()
    loader = _get_test_loader(batch_size)
    correct = total = 0
    for images, labels in loader:
        # Apply degradation per image (images are [0,1] tensors)
        degraded = []
        for img in images:
            degraded.append(apply_degradation(img, deg_type, severity_value))
        images = torch.stack(degraded)
        images = _normalize(images).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def full_robustness_eval(
    model: nn.Module,
    device: torch.device,
    model_name: str,
    strategy: str,
    seed: int,
    deg_type_trained: str = "none",
) -> list[dict]:
    """
    Run clean + all 18 degradation evaluations.
    Returns list of result dicts.
    """
    results = []

    # Clean accuracy
    clean_acc = evaluate_clean(model, device)
    results.append({
        "model": model_name,
        "strategy": strategy,
        "deg_type_trained": deg_type_trained,
        "seed": seed,
        "eval_degradation": "clean",
        "eval_severity": 0,
        "accuracy": clean_acc,
    })
    print(f"  Clean: {clean_acc:.2f}%")

    # All degradation conditions
    for deg_name, levels in DEG_GRID.items():
        for sev_idx, sev_val in enumerate(levels, 1):
            acc = evaluate_degraded(model, device, deg_name, sev_val)
            results.append({
                "model": model_name,
                "strategy": strategy,
                "deg_type_trained": deg_type_trained,
                "seed": seed,
                "eval_degradation": deg_name,
                "eval_severity": sev_idx,
                "accuracy": acc,
            })
            print(f"  {deg_name} sev{sev_idx} ({sev_val}): {acc:.2f}%")

    return results
