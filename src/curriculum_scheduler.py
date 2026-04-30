"""
Curriculum severity scheduler.

Supports three schedule types:
  - linear: ramp from mild to hard over epochs 1-100 (of 200)
  - cosine: S-curve ramp (slow start, fast middle, slow end)
  - capped: linear ramp but capped at the medium severity level

After the ramp period, severity stays at maximum (or capped max).
"""

import math
from src.degradations import DEG_GRID


def get_curriculum_severity(
    deg_type: str,
    epoch: int,
    total_epochs: int = 200,
    schedule: str = "linear",
    max_severity: float = None,
) -> float:
    """
    Return the severity value for a given degradation type and epoch.

    schedule: "linear", "cosine", or "capped"
      - linear: straight line from mild to hard
      - cosine: S-curve (0.5*(1-cos(pi*progress))) from mild to hard
      - capped: linear ramp but max severity is the middle level (DEG_GRID[deg_type][1])

    max_severity: override the hard endpoint. If None, uses DEG_GRID[-1] (or [1] for capped).
    """
    levels = DEG_GRID[deg_type]  # [mild, medium, hard]
    ramp_end = total_epochs // 2  # epoch 100 for 200 total

    # Progress from 0.0 (epoch 1) to 1.0 (epoch ramp_end)
    if epoch <= 1:
        progress = 0.0
    elif epoch >= ramp_end:
        progress = 1.0
    else:
        progress = (epoch - 1) / (ramp_end - 1)

    # Apply schedule transform
    if schedule == "cosine":
        progress = 0.5 * (1 - math.cos(math.pi * progress))
    # linear and capped both use linear progress

    # Determine endpoints
    mild = levels[0]
    if schedule == "capped":
        hard = levels[1] if max_severity is None else max_severity  # cap at medium
    else:
        hard = levels[-1] if max_severity is None else max_severity

    value = mild + progress * (hard - mild)

    # For integer-valued params (blur kernels), round to nearest odd
    if deg_type in ("gaussian_blur", "motion_blur"):
        value = int(round(value))
        if value % 2 == 0:
            value += 1

    return value
