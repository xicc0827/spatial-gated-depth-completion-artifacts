"""Matched-task teacher/no-op factorial core, plus a capacity-matched residual."""
import hashlib
import random
from contextlib import contextmanager
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ARMS = {"task": (False, False), "task_teacher": (True, False),
        "task_noop": (False, True), "task_teacher_noop": (True, True)}
STAGE, EPS = 2, 1e-5


def lowpass(x):
    if min(x.shape[-2:]) <= 3:
        raise ValueError("Stage map too small for reflect-padded 7x7 blur")
    return F.avg_pool2d(F.pad(x, (3, 3, 3, 3), mode="reflect"), 7, 1)


def decompose(x):
    x = x.float()
    mu = x.mean((-2, -1), keepdim=True)
    sigma = ((x - mu).square().mean((-2, -1), keepdim=True) + EPS).sqrt()
    return mu, sigma, (x - mu) / sigma


def groups(channels):
    return next(g for g in (8, 4, 2, 1) if channels % g == 0)


class Corrector(nn.Module):
    """Historical tensor schema; unfiltered option removes BOTH fixed blurs."""
    def __init__(self, depth_channels, rgb_channels, low_band=True):
        super().__init__()
        self.low_band = low_band
        hidden = max(32, min(128, depth_channels))
        self.in_proj = nn.Conv2d(depth_channels + rgb_channels + 2, hidden, 1)
        self.norm0 = nn.GroupNorm(groups(hidden), hidden)
        self.conv1 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups(hidden), hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups(hidden), hidden)
        self.out = nn.Conv2d(hidden, depth_channels, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, normalized_depth, rgb, sparse):
        z = lowpass(normalized_depth) if self.low_band else normalized_depth
        r = decompose(rgb)[2]
        occ = F.interpolate((sparse > 0).float(), size=z.shape[-2:], mode="area")
        val = F.interpolate(sparse.float() / 10., size=z.shape[-2:], mode="area")
        h = F.gelu(self.norm0(self.in_proj(torch.cat((z, r, occ, val), 1))))
        h = h + F.gelu(self.norm1(self.conv1(h)))
        h = h + F.gelu(self.norm2(self.conv2(h)))
        delta = self.out(h)
        return lowpass(delta) if self.low_band else delta


def state_digest(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fresh_corrector(seed, depth_channels=256, rgb_channels=256, low_band=True):
    # Do not consume the model/training RNG while choosing a matched initialization.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        result = Corrector(depth_channels, rgb_channels, low_band)
    return result


def predict(corrector, bundle, sparse):
    rgb, depth = bundle[STAGE]
    return corrector(decompose(depth)[2].detach(), rgb.detach(), sparse)


def reinject(bundle, delta):
    result = list(bundle)
    rgb, depth = bundle[STAGE]
    sigma = decompose(depth)[1].detach()
    result[STAGE] = (rgb, depth.float() + sigma * delta)
    return result


def valid_gt(gt):
    return torch.isfinite(gt) & (gt > 1e-4) & (gt <= 10.)


def masked_l1(pred, gt):
    valid = valid_gt(gt)
    if not valid.any():
        raise ValueError("No valid task pixels")
    if not torch.isfinite(pred[valid]).all():
        raise ValueError("Nonfinite prediction on valid task pixels")
    return (pred[valid].float() - gt[valid].float()).abs().mean()


def geometry(pred, gt):
    valid = valid_gt(gt)
    lp = pred.float().clamp_min(1e-4).log()
    lg = torch.where(valid, gt, torch.ones_like(gt)).float().clamp_min(1e-4).log()
    scales = []
    for r in (1, 2, 4):
        pieces = []
        for dim in (-1, -2):
            a, b = [slice(None)] * 4, [slice(None)] * 4
            a[dim], b[dim] = slice(r, None), slice(None, -r)
            a, b = tuple(a), tuple(b)
            mask = F.avg_pool2d((valid[a] & valid[b]).float(), r, r) >= 1 - 1e-6
            pd = F.avg_pool2d(lp[a] - lp[b], r, r)
            gd = F.avg_pool2d(lg[a] - lg[b], r, r)
            if mask.any():
                pieces.append(F.smooth_l1_loss(pd[mask], gd[mask], beta=.01))
        if not pieces:
            raise ValueError(f"No geometry equations at scale {r}")
        scales.append(torch.stack(pieces).mean())
    return torch.stack(scales).mean()


def loss_components(model, corrector, uniform, patterned, su, sp, gt):
    """Every arm encodes and task-supervises both views; teacher is detached."""
    du, dp = predict(corrector, uniform, su), predict(corrector, patterned, sp)
    pu = model.decoder(reinject(uniform, du))["depth"]
    pp = model.decoder(reinject(patterned, dp))["depth"]
    target = (lowpass(decompose(uniform[STAGE][1])[2]) -
              lowpass(decompose(patterned[STAGE][1])[2])).detach()
    return dict(depth=.5 * (masked_l1(pp, gt) + masked_l1(pu, gt)),
                geom=.5 * (geometry(pp, gt) + geometry(pu, gt)),
                teacher=.5 * F.smooth_l1_loss(dp, target, beta=1.),
                noop=.5 * F.smooth_l1_loss(du, torch.zeros_like(du), beta=1.),
                target=target, pred_uniform=pu, pred_pattern=pp,
                delta_uniform=du, delta_pattern=dp)


def objective(components, arm, weights):
    teacher, noop = ARMS[arm] if arm in ARMS else (False, False)
    if arm not in ARMS and arm != "generic_residual":
        raise ValueError(arm)
    loss = components["depth"] + weights["geom"] * components["geom"]
    if teacher:
        loss = loss + weights["feature"] * components["teacher"]
    if noop:
        loss = loss + weights["feature"] * components["noop"]
    return loss


def grad_norm(loss, parameters):
    gradients = torch.autograd.grad(loss, tuple(parameters), retain_graph=True,
                                    allow_unused=True)
    return float(sum(g.detach().float().square().sum() for g in gradients
                     if g is not None).sqrt())


def calibrate(model, corrector, encoded_batches):
    """Same train-only calibration at the shared zero-output start; no updates."""
    norms = {key: [] for key in ("depth", "geom", "teacher", "noop")}
    digest_before = state_digest(corrector.state_dict())
    for uniform, patterned, su, sp, gt in encoded_batches:
        c = loss_components(model, corrector, uniform, patterned, su, sp, gt)
        for key in norms:
            norms[key].append(grad_norm(c[key], corrector.parameters()))
    means = {key: float(np.mean(values)) for key, values in norms.items()}
    if any(not np.isfinite(means[k]) or means[k] <= 1e-12
           for k in ("depth", "geom", "teacher")):
        raise ValueError(f"Degenerate calibration gradients: {means}")
    if means["noop"] != 0.:
        raise ValueError("No-op gradient must be zero at zero-output start")
    assert state_digest(corrector.state_dict()) == digest_before
    return dict(geom=.5 * means["depth"] / means["geom"],
                feature=.5 * means["depth"] / means["teacher"],
                gradient_norms=norms, mean_gradient_norms=means,
                initialization_sha256=digest_before,
                policy="Geometry and teacher components each target 0.5 times depth gradient. No-op has zero initial gradient; share teacher feature coefficient instead of division by zero. Same weights for all arms.")


@contextmanager
def isolated_augmentation(seed):
    """Paired dataset augmentations must be independent of each arm's RNG usage."""
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        random.seed(seed); np.random.seed(seed % 2 ** 32)
        # Set CPU only, without resetting GPU dropout/RNG state.
        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        yield
    finally:
        random.setstate(py); np.random.set_state(np_state); torch.set_rng_state(cpu)


def encode_frozen(model, rgb, sparse):
    device = rgb.device.type
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16,
                                         enabled=device == "cuda"):
        multi = model.fusion(model.rgb_stem(rgb), model.depth_stem(sparse), None, None)
    if len(multi) != 4:
        raise ValueError("Expected four feature stages")
    return [(r.float().detach(), d.float().detach()) for r, d in multi]
