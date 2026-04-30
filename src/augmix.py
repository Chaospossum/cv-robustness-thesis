"""
AugMix implementation (Hendrycks et al., 2020).

Standard AugMix augmentation operations + Jensen-Shannon consistency loss.
Uses PIL-based operations (autocontrast, equalize, posterize, rotate, etc.)
independent of the project's 6 degradation types.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageEnhance


# ---------------------------------------------------------------------------
# AugMix augmentation operations (standard set from the paper)
# ---------------------------------------------------------------------------

def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)

def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)

def posterize(pil_img, level):
    # level 0-3 maps to bits 4-1
    bits = max(1, 4 - level)
    return ImageOps.posterize(pil_img, bits)

def rotate(pil_img, level):
    degrees = (level / 3.0) * 30  # max 30 degrees
    angle = degrees if np.random.random() > 0.5 else -degrees
    return pil_img.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))

def solarize(pil_img, level):
    threshold = 256 - int((level / 3.0) * 128)
    return ImageOps.solarize(pil_img, threshold)

def shear_x(pil_img, level):
    v = (level / 3.0) * 0.3
    v = v if np.random.random() > 0.5 else -v
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, v, 0, 0, 1, 0),
                             resample=Image.BILINEAR, fillcolor=(128, 128, 128))

def shear_y(pil_img, level):
    v = (level / 3.0) * 0.3
    v = v if np.random.random() > 0.5 else -v
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, v, 1, 0),
                             resample=Image.BILINEAR, fillcolor=(128, 128, 128))

def translate_x(pil_img, level):
    pixels = int((level / 3.0) * 10)
    pixels = pixels if np.random.random() > 0.5 else -pixels
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0),
                             resample=Image.BILINEAR, fillcolor=(128, 128, 128))

def translate_y(pil_img, level):
    pixels = int((level / 3.0) * 10)
    pixels = pixels if np.random.random() > 0.5 else -pixels
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels),
                             resample=Image.BILINEAR, fillcolor=(128, 128, 128))

def enhance_contrast(pil_img, level):
    factor = 1.0 + (level / 3.0) * 0.9
    factor = factor if np.random.random() > 0.5 else 2.0 - factor
    return ImageEnhance.Contrast(pil_img).enhance(max(0.1, factor))

def enhance_brightness(pil_img, level):
    factor = 1.0 + (level / 3.0) * 0.9
    factor = factor if np.random.random() > 0.5 else 2.0 - factor
    return ImageEnhance.Brightness(pil_img).enhance(max(0.1, factor))

def enhance_sharpness(pil_img, level):
    factor = 1.0 + (level / 3.0) * 0.9
    factor = factor if np.random.random() > 0.5 else 2.0 - factor
    return ImageEnhance.Sharpness(pil_img).enhance(max(0.1, factor))


AUGMIX_OPS = [
    autocontrast, equalize, posterize, rotate, solarize,
    shear_x, shear_y, translate_x, translate_y,
    enhance_contrast, enhance_brightness, enhance_sharpness,
]


# ---------------------------------------------------------------------------
# Core AugMix algorithm
# ---------------------------------------------------------------------------

def augmix_single_image(
    pil_img: Image.Image,
    severity: int = 3,
    width: int = 3,
    depth: int = -1,
    alpha: float = 1.0,
) -> Image.Image:
    """
    Apply AugMix to a single PIL image.

    severity: max operation severity (1-3)
    width: number of augmentation chains to mix
    depth: depth of each chain (-1 = random 1-3)
    alpha: Dirichlet/Beta parameter for mixing weights
    """
    img_np = np.array(pil_img).astype(np.float32) / 255.0

    # Mixing weights
    ws = np.random.dirichlet([alpha] * width)
    m = np.random.beta(alpha, alpha)

    mix = np.zeros_like(img_np)
    for i in range(width):
        chain_img = pil_img.copy()
        chain_depth = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(chain_depth):
            op = np.random.choice(AUGMIX_OPS)
            chain_img = op(chain_img, severity)
        mix += ws[i] * (np.array(chain_img).astype(np.float32) / 255.0)

    # Interpolate between original and augmented
    result = m * img_np + (1 - m) * mix
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def augmix_batch(images: torch.Tensor, severity: int = 3) -> torch.Tensor:
    """
    Apply AugMix to a batch of CHW float tensors in [0,1].
    Returns batch of CHW float tensors in [0,1].
    """
    out = []
    for img_tensor in images:
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        aug_img = augmix_single_image(pil_img, severity=severity)
        aug_np = np.array(aug_img).astype(np.float32) / 255.0
        out.append(torch.from_numpy(aug_np).permute(2, 0, 1))
    return torch.stack(out)


# ---------------------------------------------------------------------------
# Jensen-Shannon Divergence consistency loss
# ---------------------------------------------------------------------------

def jsd_loss(logits_clean: torch.Tensor, logits_aug1: torch.Tensor, logits_aug2: torch.Tensor) -> torch.Tensor:
    """
    Jensen-Shannon divergence between predictions on clean and two augmented views.
    All inputs: (batch_size, num_classes) raw logits.
    Returns scalar loss.
    """
    p_clean = F.softmax(logits_clean, dim=1)
    p_aug1 = F.softmax(logits_aug1, dim=1)
    p_aug2 = F.softmax(logits_aug2, dim=1)

    # Mean distribution
    p_mean = (p_clean + p_aug1 + p_aug2) / 3.0

    # KL divergences
    kl_clean = F.kl_div(p_mean.log(), p_clean, reduction="batchmean")
    kl_aug1 = F.kl_div(p_mean.log(), p_aug1, reduction="batchmean")
    kl_aug2 = F.kl_div(p_mean.log(), p_aug2, reduction="batchmean")

    return (kl_clean + kl_aug1 + kl_aug2) / 3.0
