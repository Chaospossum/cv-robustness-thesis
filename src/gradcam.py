"""
Grad-CAM visualization for comparing what different training strategies focus on.

Generates heatmap grids showing model attention under clean and degraded inputs
for baseline, mixed, and curriculum/blur trained models.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.models import get_model
from src.degradations import apply_degradation
from src.evaluate import CIFAR10_MEAN, CIFAR10_STD


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """Grad-CAM: extract and visualize class activation maps."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap for a single image tensor (1, C, H, W).
        Returns heatmap as numpy array (H, W) in [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        # Weight activation maps by gradient (global average pooling of gradients)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upsample to input size
        cam = np.array(
            torch.nn.functional.interpolate(
                torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float(),
                size=(input_tensor.shape[2], input_tensor.shape[3]),
                mode="bilinear",
                align_corners=False,
            ).squeeze().numpy()
        )

        return cam

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def get_target_layer(model, model_name: str):
    """Return the appropriate target layer for Grad-CAM."""
    if model_name == "resnet18":
        return model.layer4[-1]  # last BasicBlock
    elif model_name == "mobilenetv2":
        return model.features[-1]  # last ConvBNReLU
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def overlay_heatmap(img_np, heatmap, alpha=0.4):
    """
    Overlay a heatmap on an image.
    img_np: (H, W, 3) in [0, 1]
    heatmap: (H, W) in [0, 1]
    Returns (H, W, 3) in [0, 1]
    """
    colormap = cm.jet(heatmap)[:, :, :3]  # (H, W, 3)
    overlay = (1 - alpha) * img_np + alpha * colormap
    return np.clip(overlay, 0, 1)


def generate_gradcam_grid(
    checkpoints: dict,
    model_name: str,
    device: torch.device,
    out_path: str,
    n_images: int = 8,
):
    """
    Generate a Grad-CAM comparison grid.

    checkpoints: dict of {strategy_label: checkpoint_path}
    Rows = test images, columns = strategy × condition (clean + blur + noise)
    """
    # Load test images (one per class)
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=T.ToTensor()
    )

    # Pick one image per class
    class_images = {}
    for img, label in testset:
        if label not in class_images and len(class_images) < n_images:
            class_images[label] = img
        if len(class_images) >= n_images:
            break

    images = [class_images[k] for k in sorted(class_images.keys())[:n_images]]
    cifar_classes = ["airplane", "auto", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # Conditions to visualize
    conditions = [
        ("clean", None, None),
        ("gauss_blur s3", "gaussian_blur", 7),
        ("gauss_noise s3", "gaussian_noise", 0.10),
    ]

    strategies = list(checkpoints.keys())
    n_cols = len(strategies) * len(conditions)
    n_rows = n_images

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for strat_idx, (strat_label, ckpt_path) in enumerate(checkpoints.items()):
        # Load model
        model = get_model(model_name).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        target_layer = get_target_layer(model, model_name)
        gradcam = GradCAM(model, target_layer)

        for cond_idx, (cond_name, deg_type, sev_val) in enumerate(conditions):
            col = strat_idx * len(conditions) + cond_idx

            for row, img_tensor in enumerate(images):
                # Apply degradation if needed
                if deg_type is not None:
                    vis_img = apply_degradation(img_tensor, deg_type, sev_val)
                else:
                    vis_img = img_tensor.clone()

                # Normalize for model input
                mean = CIFAR10_MEAN.squeeze()
                std = CIFAR10_STD.squeeze()
                normalized = ((vis_img - mean.view(3, 1, 1)) / std.view(3, 1, 1)).unsqueeze(0).to(device)

                # Get Grad-CAM
                heatmap = gradcam(normalized)

                # Overlay
                img_display = vis_img.permute(1, 2, 0).numpy()
                overlay = overlay_heatmap(img_display, heatmap)

                axes[row, col].imshow(overlay)
                axes[row, col].axis("off")

                # Labels
                if row == 0:
                    axes[row, col].set_title(f"{strat_label}\n{cond_name}", fontsize=8)
                if col == 0:
                    label_idx = sorted(class_images.keys())[row]
                    axes[row, col].set_ylabel(cifar_classes[label_idx], fontsize=8, rotation=0, labelpad=40)

        gradcam.remove_hooks()

    plt.suptitle(f"Grad-CAM: {model_name}", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
