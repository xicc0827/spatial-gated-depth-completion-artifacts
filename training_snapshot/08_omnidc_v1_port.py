"""Inference-only compatibility adapter for pinned OMNI-DC v1.0.

No changes to original source files. Skip initialization weights before strict
full-model loading; no DepthAnythingV2 import for the v1.0 architecture. Numerical
solver, backbone forward and propagation are the original upstream code.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import types
import torch
import torch.nn.functional as F

COMMIT = "a5ecd35d087ed5e5fcc2be93989d1db3fde194fd"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Unexpected source anchor: {old}")
    return source.replace(old, new, 1)


def build_uninitialized(root):
    root = Path(root)
    command = ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root)]
    if subprocess.check_output(command + ["rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise ValueError("Upstream commit mismatch")
    if subprocess.check_output(command + ["status", "--porcelain"], text=True).strip():
        raise ValueError("Upstream source is not clean")
    flags = ["--max_depth", "300", "--data_normalize_median", "1", "--num_resolution", "3",
             "--multi_resolution_learnable_gradients_weights", "uniform", "--GRU_iters", "1",
             "--optim_layer_input_clamp", "1.0", "--depth_activation_format", "exp",
             "--whiten_sparse_depths", "1", "--gru_internal_whiten_method", "median",
             "--backbone_mode", "rgbd", "--pred_confidence_input", "1", "--load_dav2", "0",
             # Saved epoch72 checkpoint has the Laplace output-confidence head.
             # This constructs that head for strict loading; no loss is run here.
             "--loss", "0.5*SeqLaplace+1.0*SeqGradL1+1.0*ConfInput+1.0*SeqL1+2.0*GradMatchingScale"]
    config_path = root / "src/config.py"
    text = replace_once(config_path.read_text(encoding="utf-8"),
                        "args = parser.parse_args()", "args = parser.parse_args(port_flags)")
    namespace = {"__file__": str(config_path), "port_flags": flags}
    exec(compile(text, str(config_path) + "[port]", "exec"), namespace)
    args = namespace["args"]
    if args.load_dav2 or args.spn_type != "dyspn":
        raise ValueError("Expected v1.0 without foundation model, with DySPN")
    model_dir = root / "src/model"
    sys.path[:0] = [str(model_dir), str(root / "src")]
    for name in ("pvt", "ognidc"):
        if name in sys.modules:
            raise RuntimeError("Use a fresh process for this adapter")
        path = model_dir / f"{name}.py"
        source = path.read_text(encoding="utf-8")
        if name == "pvt":
            source = replace_once(source, "from mmseg.utils import get_root_logger", "")
            source = replace_once(source, "from mmcv.runner import load_checkpoint", "")
            source = replace_once(source, "net = get_resnet34(pretrained=True)",
                                  "net = get_resnet34(pretrained=False)")
            source = replace_once(source, "self.init_weights(pretrained)", "self.init_weights(None)")
        else:
            source = replace_once(source,
                "from depth_models.depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2", "")
        module = types.ModuleType(name)
        module.__file__ = str(path)
        sys.modules[name] = module
        exec(compile(source, str(path) + "[port]", "exec"), module.__dict__)
    model = sys.modules["ognidc"].OGNIDC(args)
    model._strict_pretrained_loaded = False
    model._port_source_hashes = {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
                                for f in (root / "src/model").rglob("*.py")}
    return model


def load_pretrained(model, path):
    # Namespace is a data container, not arbitrary pickle execution permission.
    with torch.serialization.safe_globals([argparse.Namespace]):
        state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("net", state)
    if all(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    result = model.load_state_dict(state, strict=True)
    model._strict_pretrained_loaded = True
    return result


@torch.inference_mode()
def predict(model, rgb, sparse):
    if not model._strict_pretrained_loaded:
        raise RuntimeError("Refusing random-weight evaluation")
    if not torch.all((sparse > 0).flatten(1).any(1)):
        raise ValueError("OMNI-DC requires observed depths for median normalization")
    model.eval()
    rgb = (rgb - rgb.new_tensor([.485, .456, .406])[None, :, None, None]) / rgb.new_tensor([.229, .224, .225])[None, :, None, None]
    h, w = rgb.shape[-2:]
    divisor = 4 * 2 ** (model.args.num_resolution - 1)
    padding = (0, (-w) % divisor, 0, (-h) % divisor)
    rgb, sparse = F.pad(rgb, padding), F.pad(sparse, padding)
    # Identity K is the official demo convention; inspect configured gradient
    # parameterization before using this port for a different configuration.
    sample = {"rgb": rgb, "dep": sparse, "K": torch.eye(3, device=rgb.device)[None].repeat(rgb.shape[0], 1, 1), "pattern": 0}
    return model(sample)["pred"][..., :h, :w]
