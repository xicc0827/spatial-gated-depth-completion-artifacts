"""Depth-completion metrics used by the frozen evaluation."""
from __future__ import annotations

import numpy as np


def evaluate(prediction, target, valid, observed=None):
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(target, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if not mask.any():
        raise ValueError("valid mask is empty")
    error = pred[mask] - gt[mask]
    out = {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "rel": float(np.mean(np.abs(error) / np.maximum(gt[mask], 1e-8))),
    }
    if observed is not None:
        unobserved = mask & ~np.asarray(observed, dtype=bool)
        extra = evaluate(pred, gt, unobserved)
        out.update({f"unobserved_{key}": value for key, value in extra.items()})
    return out
