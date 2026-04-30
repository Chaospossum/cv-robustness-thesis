"""
Six image degradation functions for robustness evaluation.
All operate on PIL Images or torch Tensors as specified.
"""

import io
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance
from scipy.ndimage import convolve1d


# ---------------------------------------------------------------------------
# Degradation grid — single source of truth
# ---------------------------------------------------------------------------
DEG_GRID = {
    "gaussian_noise": [0.02, 0.05, 0.10],
    "gaussian_blur":  [3, 5, 7],
    "motion_blur":    [3, 5, 7],
    "jpeg":           [50, 30, 10],
    "contrast":       [0.8, 0.6, 0.4],
    "darkening":      [0.8, 0.6, 0.4],
}

# For curriculum: severity index 0=mild, 2=hard for ALL types.
# For jpeg/contrast/darkening the *value* decreases with severity,
# but index 0 is always mildest.


# ---------------------------------------------------------------------------
# Tensor-level degradations (operate on float tensors in [0,1])
# ---------------------------------------------------------------------------

def apply_gaussian_noise(img_tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add Gaussian noise with std=sigma. img_tensor in [0,1]."""
    noise = torch.randn_like(img_tensor) * sigma
    return torch.clamp(img_tensor + noise, 0.0, 1.0)


# ---------------------------------------------------------------------------
# PIL-level degradations
# ---------------------------------------------------------------------------

def apply_gaussian_blur(img: Image.Image, kernel_size: int) -> Image.Image:
    """Gaussian blur with given kernel size (radius = kernel_size // 2)."""
    radius = kernel_size // 2
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_motion_blur(img: Image.Image, kernel_size: int) -> Image.Image:
    """Horizontal motion blur using a 1D box kernel."""
    img_np = np.array(img).astype(np.float64)
    kernel = np.ones(kernel_size) / kernel_size
    # Apply 1D horizontal convolution to each channel
    blurred = convolve1d(img_np, kernel, axis=1, mode='nearest')
    return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8))


def apply_jpeg_compression(img: Image.Image, quality: int) -> Image.Image:
    """JPEG compression at given quality level."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_contrast(img: Image.Image, factor: float) -> Image.Image:
    """Reduce contrast by given factor (1.0 = original, 0.0 = grey)."""
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def apply_darkening(img: Image.Image, factor: float) -> Image.Image:
    """Darken image by given factor (1.0 = original, 0.0 = black)."""
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

# Map from degradation name to (function, input_type)
_DEGRADATION_FNS = {
    "gaussian_noise": (apply_gaussian_noise, "tensor"),
    "gaussian_blur":  (apply_gaussian_blur,  "pil"),
    "motion_blur":    (apply_motion_blur,    "pil"),
    "jpeg":           (apply_jpeg_compression, "pil"),
    "contrast":       (apply_contrast,       "pil"),
    "darkening":      (apply_darkening,      "pil"),
}


def apply_degradation(img_tensor: torch.Tensor, deg_type: str, severity_value) -> torch.Tensor:
    """
    Apply a degradation to a CHW float tensor in [0,1].
    Returns a CHW float tensor in [0,1].
    """
    fn, input_type = _DEGRADATION_FNS[deg_type]

    if input_type == "tensor":
        return fn(img_tensor, severity_value)
    else:
        # Convert tensor -> PIL -> apply -> tensor
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        pil_img = fn(pil_img, severity_value)
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).permute(2, 0, 1)
