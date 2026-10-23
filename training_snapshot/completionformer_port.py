"""Audited inference port; original MIT-licensed source remains unmodified.

Loads upstream commit 2744eddee9b57595dc3064f7d342569736a6803b from disk.
Changes: torchvision DCN backend, skip backbone initialization checkpoints
before strict full-model loading, remove unused mmcv/mmseg initializer imports.
Not a faithful official training recipe. Inputs require ImageNet normalization.
"""
import hashlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
import torch
from completionformer_dcn_adapter import ModulatedDeformConvFunction

HASHES = {
    "pvt": "6645cf95ad45c8bb739fc90b939562cb703e4d90c3debec2783a2017fca96f77",
    "backbone": "2c451f9c7a30ee120fc9e684f9fcf2407be5ab56e4f12c5897303eb61dda0e09",
    "completionformer": "ad08481fbb0b6e23d07781ce971db30d8c56aef2773684a39f120c0a41351672",
    "nlspn_module": "14c300f56794315e7262648724317b2366ca9400d07d34ba7eb8c4f7a11d8c34",
    "resnet_cbam": "122974fdb6dc83476182dee63c9ff8c14765a068026717f742bc91fd2188952c",
}


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Upstream patch anchor is not unique: {old!r}")
    return source.replace(old, new, 1)


def build_uninitialized(source_root, *, prop_time=6):
    """Infrastructure only. Never report accuracy without load_pretrained()."""
    root = Path(source_root) / "src" / "model"
    texts = {}
    for name, expected in HASHES.items():
        raw = (root / f"{name}.py").read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Upstream file fingerprint mismatch: {name}")
        texts[name] = raw.decode("utf-8").replace("\r\n", "\n")
    package = "audited_completionformer"
    if package in sys.modules:
        raise RuntimeError("Build once per process to avoid module namespace ambiguity")
    pkg = types.ModuleType(package)
    pkg.__path__ = [str(root)]
    sys.modules[package] = pkg
    dcn = types.ModuleType(package + ".modulated_deform_conv_func")
    dcn.ModulatedDeformConvFunction = ModulatedDeformConvFunction
    sys.modules[dcn.__name__] = dcn
    for name in ("resnet_cbam", "pvt", "backbone", "nlspn_module", "completionformer"):
        source = texts[name]
        if name == "pvt":
            source = replace_once(source, "from mmseg.utils import get_root_logger\n", "")
            source = replace_once(source, "from mmcv.runner import load_checkpoint\n", "")
            source = replace_once(source, "net = get_resnet34(pretrained=True)",
                                  "net = get_resnet34(pretrained=False)")
            source = replace_once(source,
                '            logger = get_root_logger()\n'
                "            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)\n"
                '            print("===pretrained weight loaded===")',
                '            raise RuntimeError("Port requires strict full-model weights, not partial initialization")')
        if name == "backbone":
            source = replace_once(source, "pretrained='./pretrained/pvt.pth'", "pretrained=None")
        module = types.ModuleType(package + "." + name)
        module.__file__ = str(root / f"{name}.py")
        module.__package__ = package
        sys.modules[module.__name__] = module
        exec(compile(source, module.__file__ + "[audited-port]", "exec"), module.__dict__)
    args = SimpleNamespace(prop_time=prop_time, prop_kernel=3, preserve_input=False,
                           affinity="TGASS", affinity_gamma=.5, conf_prop=True, legacy=False)
    model = sys.modules[package + ".completionformer"].CompletionFormer(args)
    model._strict_pretrained_loaded = False
    return model


def load_pretrained(model, checkpoint):
    # Do not allow arbitrary pickle execution from downloaded checkpoints.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("net", payload.get("state_dict", payload.get("model", payload)))
    if not isinstance(state, dict) or not state:
        raise ValueError("No recognized model state dictionary")
    if all(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    result = model.load_state_dict(state, strict=True)
    model._strict_pretrained_loaded = True
    return result


@torch.inference_mode()
def predict(model, rgb_unit, sparse_depth):
    if not model._strict_pretrained_loaded:
        raise RuntimeError("Refusing inference with unverified/random weights")
    model.eval()
    mean = rgb_unit.new_tensor([.485, .456, .406])[None, :, None, None]
    std = rgb_unit.new_tensor([.229, .224, .225])[None, :, None, None]
    return model({"rgb": (rgb_unit - mean) / std, "dep": sparse_depth})["pred"]
