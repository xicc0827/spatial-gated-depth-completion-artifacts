"""Reference implementation of the mask-only Spatial and Image Mean gates."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def spatial_gate(support_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(support_mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("support_mask must be a nonempty 2-D mask")
    distance = distance_transform_edt(~mask)
    scale2 = mask.size / int(mask.sum())
    return (1.0 - np.exp(-(distance ** 2) / scale2)).astype(np.float32)


def image_mean_gate(support_mask: np.ndarray) -> np.ndarray:
    gate = spatial_gate(support_mask)
    return np.full_like(gate, gate.mean())
