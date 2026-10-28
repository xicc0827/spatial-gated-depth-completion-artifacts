"""Replay one frozen CompletionFormer condition using explicitly supplied inputs.

No private execution-stack modules are imported. This is an inference entry
point, not a dataset downloader, raw-data converter, or training recipe.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
LAWS = ("uniform", "left", "right", "top", "bottom", "center_hole",
        "diag_main_half", "diag_anti_half", "center_vertical_band",
        "center_disk", "two_corners")
METRICS = ("rmse", "mae", "rel", "unobserved_rmse", "unobserved_mae", "unobserved_rel")
TOLERANCE = 1e-5


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_checkpoint(path, method, seed):
    entries = [r for r in read_csv(ROOT / "manifests/checkpoint_hashes.csv")
               if r["method"] == method and r["seed"] == str(seed)]
    require(len(entries) == 1, "Checkpoint identity is missing or ambiguous")
    actual = sha256(path)
    require(actual == entries[0]["sha256"], f"Wrong {method} checkpoint for seed {seed}")
    return actual


def compare_rows(scores, reference, coordinate, pixels, mapping):
    differences = {}
    for output_name, rows in mapping.items():
        require(len(rows) == 1, f"Missing or duplicate reference for {output_name}")
        row = rows[0]
        require(row["coordinate_hash"] == coordinate, "Support-coordinate mismatch")
        require(int(row["score_pixels"]) == pixels and int(row["input_points"]) == 500,
                "Scoring population mismatch")
        differences[output_name] = {key: abs(scores[output_name][key] - float(row[key]))
                                    for key in METRICS}
    maximum = max(value for row in differences.values() for value in row.values())
    return {"reference": reference, "maximum_absolute_metric_difference": maximum,
            "tolerance": TOLERANCE, "passed": maximum < TOLERANCE,
            "differences": differences}


def run(args):
    import numpy as np
    import torch
    import torch.nn.functional as functional
    import torchvision
    import timm
    from gates import spatial_gate
    from metrics import evaluate

    torch.set_num_threads(2)
    require(args.device != "cuda" or torch.cuda.is_available(), "CUDA is unavailable")
    manifest = (ROOT / "training_snapshot/11_diode_converted_manifest.json"
                if args.dataset == "diode" else
                ROOT / "resources/or3/protocol/evaluation_manifest.json")
    metadata = json.loads(manifest.read_text(encoding="utf-8-sig"))["rows"]
    matches = [row for row in metadata if int(row["index"]) == args.index and row["primary"]]
    require(len(matches) == 1, "Index is not a retained primary image")
    require(sha256(args.input) == matches[0]["sha256"], "Converted NPZ fingerprint mismatch")
    with np.load(args.input, allow_pickle=False) as payload:
        rgb, depth, valid, supports = (payload[k].copy() for k in
                                      ("rgb", "depth", "valid", "support_indices"))
    require(rgb.shape == (228, 304, 3) and rgb.dtype == np.uint8, "Invalid RGB array")
    require(depth.shape == valid.shape == (228, 304) and depth.dtype == np.float32,
            "Invalid depth/valid arrays")
    require(valid.dtype == np.bool_ and np.array_equal(valid, np.isfinite(depth) &
                                                    (depth > .1) & (depth <= 10)),
            "Valid mask differs from the frozen scoring rule")
    require(supports.shape == (5, len(LAWS), 500) and supports.dtype.kind in "iu",
            "Invalid support-index array")
    ids = supports[args.repetition, LAWS.index(args.law)]
    require(np.unique(ids).size == 500 and (ids >= 0).all() and (ids < depth.size).all(),
            "Support must contain 500 unique in-bounds indices")
    require(valid.reshape(-1)[ids].all(), "Support contains invalid reference depths")
    coordinate = hashlib.sha256(ids.tobytes()).hexdigest()
    observed = np.zeros_like(valid)
    observed.reshape(-1)[ids] = True
    sparse = np.zeros_like(depth)
    sparse.reshape(-1)[ids] = depth.reshape(-1)[ids]

    parent_hash = check_checkpoint(args.parent_checkpoint, "parent", "parent")
    spatial_hash = check_checkpoint(args.spatial_checkpoint, "spatial", args.seed)
    mean_hash = (check_checkpoint(args.image_mean_checkpoint, "image_mean", args.seed)
                 if args.image_mean_checkpoint else None)
    snapshot = ROOT / "training_snapshot"
    sys.path.insert(0, str(snapshot))
    load_module("factorial_core", snapshot / "03_factorial_core.py")
    core = load_module("completionformer_factorial_core", snapshot / "04_completionformer_factorial_core.py")
    from completionformer_port import build_uninitialized, load_pretrained

    base = build_uninitialized(args.upstream_root, prop_time=6)
    load_pretrained(base, args.parent_checkpoint)
    base = base.to(args.device).eval().requires_grad_(False)

    def corrector(path):
        # File identity is checked above, and arbitrary pickle loading is disabled.
        state = torch.load(path, map_location="cpu", weights_only=True)
        require(state["applied"] == 2000 and state["parent_sha256"] == parent_hash,
                "Adapter training endpoint or parent identity mismatch")
        require(int(state["seed"]) == args.seed, "Adapter seed mismatch")
        model = core.fresh(args.seed, low_band=True)
        model.load_state_dict(state["corrector"], strict=True)
        require(sum(p.numel() for p in model.parameters()) == 24513, "Parameter-count mismatch")
        return model.to(args.device).eval().requires_grad_(False)

    spatial_model = corrector(args.spatial_checkpoint)
    mean_model = corrector(args.image_mean_checkpoint) if args.image_mean_checkpoint else None
    scores = {}
    with torch.inference_mode():
        image = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(args.device) / 255.
        dep = torch.from_numpy(sparse)[None, None].to(args.device)
        view = core.encode(base, image, dep)
        gate = torch.from_numpy(spatial_gate(observed))[None, None].to(args.device)
        if tuple(gate.shape[-2:]) != tuple(view["init"].shape[-2:]):
            gate = functional.interpolate(gate, size=view["init"].shape[-2:], mode="area")
        # Match the frozen implementation: mean after resizing, using torch reduction.
        mean_gate = gate.mean((-2, -1), keepdim=True).expand_as(gate)
        raw = core.predict_delta(spatial_model, view)
        initial = {"parent": view["init"],
                   "spatial": core.corrected(view, raw * gate),
                   "image_mean_intervention": core.corrected(view, raw * mean_gate)}
        if mean_model is not None:
            initial["image_mean"] = core.corrected(view, core.predict_delta(mean_model, view) * mean_gate)
        for name, value in initial.items():
            prediction = core.propagate(base, view, value)[0, 0].float().cpu().numpy()
            require(np.isfinite(prediction[valid]).all(), "Nonfinite prediction on valid pixels")
            scores[name] = evaluate(prediction, depth, valid, observed)

    report = {"dataset": args.dataset, "index": args.index, "image": matches[0]["image"],
              "law": args.law, "repetition": args.repetition, "seed": args.seed,
              "coordinate_hash": coordinate, "score_pixels": int(valid.sum()), "input_points": 500,
              "checkpoint_sha256": {"parent": parent_hash, "spatial": spatial_hash,
                                    "image_mean": mean_hash},
              "environment": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                              "timm": timm.__version__, "device": args.device},
              "gate_mean": float(gate.mean()), "gate_std": float(gate.std(unbiased=False)),
              "metrics": scores, "replay_checks": []}
    if args.law in LAWS[3:]:
        rows = read_csv(ROOT / "resources/or4/per_image" / args.dataset / f"{args.index:04d}.csv")
        rows = [r for r in rows if r["law"] == args.law and
                int(r["repetition"]) == args.repetition and int(r["seed"]) == args.seed]
        require(all(r["checkpoint_sha256"] == spatial_hash for r in rows), "OR4 checkpoint mismatch")
        mapping = {name: [r for r in rows if r["inference_mode"] == name]
                   for name in ("spatial", "image_mean_intervention")}
        report["replay_checks"].append(compare_rows(scores, "OR4", coordinate, int(valid.sum()), mapping))
    if args.original_records:
        rows = [r for r in read_csv(args.original_records) if int(r["index"]) == args.index
                and r["law"] == args.law and int(r["repetition"]) == args.repetition]
        names = {"parent": "parent", "spatial": f"spatial_seed{args.seed}"}
        if mean_model is not None:
            names["image_mean"] = f"image_mean_seed{args.seed}"
        mapping = {name: [r for r in rows if r["model"] == ref] for name, ref in names.items()}
        report["replay_checks"].append(compare_rows(scores, "original condition records", coordinate,
                                                  int(valid.sum()), mapping))
    report["status"] = ("PASS" if report["replay_checks"] else "INFERRED_NO_REFERENCE_CHECK")
    if any(not check["passed"] for check in report["replay_checks"]):
        report["status"] = "REPLAY_MISMATCH"
    output = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if report["status"] != "REPLAY_MISMATCH" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("diode", "arkitscenes"), required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Frozen converted NPZ, acquired locally")
    parser.add_argument("--upstream-root", type=Path, required=True, help="CompletionFormer checkout")
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--spatial-checkpoint", type=Path, required=True)
    parser.add_argument("--image-mean-checkpoint", type=Path, help="Separately trained Image Mean adapter")
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--law", choices=LAWS, default="top")
    parser.add_argument("--repetition", type=int, choices=range(5), default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--original-records", type=Path, help="Optional original per-image score CSV")
    parser.add_argument("--output", type=Path, help="JSON metrics/replay report; no image pixels are exported")
    sys.exit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
