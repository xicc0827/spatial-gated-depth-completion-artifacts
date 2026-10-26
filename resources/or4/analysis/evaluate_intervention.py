"""Fixed-checkpoint gate-to-image-mean inference intervention.

The trained Spatial corrector weights are frozen.  For every retained Spatial
checkpoint, this evaluator replaces the position-dependent support-distance
gate by that gate's within-image mean at inference.  The original Spatial rows
are imported only from hash-verified frozen evaluations; the counterfactual
rows are newly inferred over the complete unseen8 condition grid.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocol" / "frozen_protocol.json"
CHECKS = ROOT / "checks"
METRICS = (
    "rmse",
    "mae",
    "rel",
    "unobserved_rmse",
    "unobserved_mae",
    "unobserved_rel",
)
UNSEEN8 = (
    "top",
    "bottom",
    "center_hole",
    "diag_main_half",
    "diag_anti_half",
    "center_vertical_band",
    "center_disk",
    "two_corners",
)
SEEDS = (42, 43, 44)
MODES = ("spatial", "image_mean_intervention")
FIELDS = (
    "dataset",
    "index",
    "scene",
    "scan",
    "image",
    "law",
    "repetition",
    "seed",
    "inference_mode",
    "checkpoint_sha256",
    "coordinate_hash",
    "score_pixels",
    "input_points",
    *METRICS,
    "effective_gate_mean",
    "effective_gate_std",
    "metric_source",
)


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def condition_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(indices.tobytes()).hexdigest()


def protocol_payload(diode_source: Path, arkit_root: Path) -> dict[str, Any]:
    diode_plan = read_json(diode_source / "evaluation_plan_frozen.json")
    arkit_plan = read_json(arkit_root / "evaluation_plan_frozen.json")
    registry = read_json(diode_source / "model_registry.json")
    spatial = {
        str(slot["seed"]): slot["sha256"]
        for slot in registry["models"]
        if slot["arm"] == "spatial"
    }
    assert set(spatial) == {"42", "43", "44"}
    assert diode_plan["registry_sha256"] == sha256(
        diode_source / "model_registry.json"
    )
    for seed, checkpoint_hash in spatial.items():
        assert arkit_plan["checkpoint_sha256"][f"spatial_seed{seed}"] == checkpoint_hash
    assert tuple(diode_plan["unseen_layouts"]) == UNSEEN8
    assert tuple(arkit_plan["primary_family"]) == UNSEEN8
    diode_manifest = diode_source / "diode_converted_manifest.json"
    arkit_manifest = arkit_root / "evaluation_manifest.json"
    return {
        "status": "FIXED_CHECKPOINT_GATE_INTERVENTION_FROZEN_BEFORE_NEW_INFERENCE",
        "created_unix": time.time(),
        "scientific_question": (
            "With each trained Spatial checkpoint held fixed, what immediate "
            "prediction change results when the per-pixel gate is replaced by "
            "its within-image mean at inference?"
        ),
        "intervention": (
            "For the same raw corrector output and same Spatial checkpoint, "
            "compute g from support coordinates exactly as in Spatial, then use "
            "mean(g) broadcast over the residual map. No parameters, inputs, "
            "sampling coordinates, propagation settings, or scoring masks change."
        ),
        "datasets": {
            "DIODE": {
                "cohort": "304 primary images, 3 scenes, 10 scans",
                "spatial_baseline_record": (
                    "external_candidate_freeze_20261016/per_frame/*.csv"
                ),
                "manifest_sha256": sha256(diode_manifest),
                "source_plan_sha256": sha256(
                    diode_source / "evaluation_plan_frozen.json"
                ),
                "source_receipt_sha256": sha256(
                    diode_source / "inference_complete.json"
                ),
                "images": diode_plan["primary_images"],
                "physical_unit": "scene",
            },
            "ARKitScenes": {
                "cohort": "247 images, 20 held-out validation visits/sites",
                "spatial_baseline_record": (
                    "new_scene_validation/per_image/*.csv"
                ),
                "manifest_sha256": sha256(arkit_manifest),
                "source_plan_sha256": sha256(
                    arkit_root / "evaluation_plan_frozen.json"
                ),
                "source_receipt_sha256": sha256(
                    arkit_root / "inference_complete.json"
                ),
                "images": arkit_plan["population_images"],
                "physical_unit": "site",
            },
        },
        "checkpoint_sha256": spatial,
        "parent_checkpoint_sha256": registry["parent"]["checkpoint_sha256"],
        "seeds": list(SEEDS),
        "laws": list(UNSEEN8),
        "repetitions": 5,
        "input_points": 500,
        "modes": list(MODES),
        "outcomes": list(METRICS),
        "aggregation": (
            "Average repetitions and unseen8 layouts within image; average "
            "training seeds within image for the main paired endpoint; average "
            "images within physical units; weight the 3 DIODE scenes or 20 "
            "ARKitScenes sites equally."
        ),
        "uncertainty": (
            "Paired nonparametric percentile bootstrap over physical units, "
            "200000 resamples, random seed 20261022."
        ),
        "source_spatial_rows": (
            "Imported from the prior frozen evaluations after receipt/file hash, "
            "checkpoint, condition-grid, coordinate, and one-case numerical replay checks."
        ),
        "pairing_keys": [
            "dataset",
            "index",
            "law",
            "repetition",
            "seed",
            "coordinate_hash",
            "score_pixels",
        ],
        "expected_rows": {
            "DIODE": diode_plan["primary_images"] * 8 * 5 * 3 * 2,
            "ARKitScenes": arkit_plan["population_images"] * 8 * 5 * 3 * 2,
        },
        "code_sha256": {
            "evaluate_intervention.py": sha256(__file__),
            "aggregate_validate_publish.py": sha256(
                Path(__file__).with_name("aggregate_validate_publish.py")
            ),
        },
        "source_identities": {
            "diode_common.py": sha256(diode_source / "diode_common.py"),
            "model_registry.json": sha256(diode_source / "model_registry.json"),
        },
        "result_policy": (
            "Complete grids only. No retuning, checkpoint selection, case "
            "exclusion, target update, scale fitting, or upper clipping. Report "
            "the observed sign whether favorable or unfavorable."
        ),
    }


def prepare(args: argparse.Namespace) -> None:
    diode_source = Path(args.diode_source).resolve()
    arkit_root = Path(args.arkit_root).resolve()
    payload = protocol_payload(diode_source, arkit_root)
    if PROTOCOL.exists():
        existing = read_json(PROTOCOL)
        left = {k: v for k, v in existing.items() if k != "created_unix"}
        right = {k: v for k, v in payload.items() if k != "created_unix"}
        assert left == right, "Existing frozen protocol differs; do not overwrite it"
    else:
        save_json(PROTOCOL, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "protocol_sha256": sha256(PROTOCOL),
                "expected_rows": payload["expected_rows"],
            }
        ),
        flush=True,
    )


def import_stack(diode_source: Path):
    sys.path.insert(0, str(diode_source))
    import diode_common as dc

    dc.dependencies()
    import torch
    import completionformer_factorial_core as cc
    import run_gate_screen as gc
    from completionformer_port import build_uninitialized, load_pretrained
    from gpu_serial_lease import gpu_lease

    return dc, torch, cc, gc, build_uninitialized, load_pretrained, gpu_lease


def build_frozen_models(diode_source: Path, stack):
    dc, torch, _cc, gc, build_uninitialized, load_pretrained, _lease = stack
    registry = read_json(diode_source / "model_registry.json")
    cf = read_json(dc.W / "completionformer_factorial" / "protocol_frozen.json")
    assert sha256(registry["parent"]["checkpoint"]) == registry["parent"][
        "checkpoint_sha256"
    ]
    base = build_uninitialized(cf["upstream_source"], prop_time=6)
    load_pretrained(base, registry["parent"]["checkpoint"])
    base = base.cuda().eval().requires_grad_(False)
    correctors = {}
    checkpoint_hashes = {}
    for slot in registry["models"]:
        if slot["arm"] != "spatial":
            continue
        seed = int(slot["seed"])
        assert seed in SEEDS
        assert sha256(slot["checkpoint"]) == slot["sha256"]
        state = torch.load(slot["checkpoint"], map_location="cpu", weights_only=False)
        assert state["applied"] == 2000
        assert state["parent_sha256"] == registry["parent"]["checkpoint_sha256"]
        # BASE_FRESH is the ungated corrector.  The gate is applied explicitly
        # below, making the intervention a one-line operation on the same raw
        # residual and the same state tensors.
        corrector = gc.BASE_FRESH("CompletionFormer", True)
        corrector.load_state_dict(state["corrector"], strict=True)
        assert sum(parameter.numel() for parameter in corrector.parameters()) == 24513
        correctors[seed] = corrector.cuda().eval().requires_grad_(False)
        checkpoint_hashes[seed] = slot["sha256"]
        del state
    assert set(correctors) == set(SEEDS)
    return base, correctors, checkpoint_hashes


def archived_rows(
    csv_path: Path,
    receipt: dict[str, Any],
    checkpoint_hashes: dict[int, str],
    include_checkpoint_column: bool,
) -> tuple[dict[tuple[str, str, int], dict[str, str]], str]:
    file_hash = sha256(csv_path)
    assert receipt["files"][str(csv_path)] == file_hash
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = {
        (row["model"], row["law"], int(row["repetition"])): row
        for row in rows
        if row["model"] in {f"spatial_seed{s}" for s in SEEDS}
        and row["law"] in UNSEEN8
    }
    assert len(selected) == 3 * 8 * 5
    for (model, _law, _rep), row in selected.items():
        seed = int(model.rsplit("seed", 1)[1])
        if include_checkpoint_column:
            assert row["checkpoint_sha256"] == checkpoint_hashes[seed]
        assert int(row["input_points"]) == 500
        assert all(np.isfinite(float(row[name])) for name in METRICS)
    return selected, file_hash


def dataset_inputs(dataset: str, diode_source: Path, arkit_root: Path):
    if dataset == "diode":
        manifest_path = diode_source / "diode_converted_manifest.json"
        rows = [row for row in read_json(manifest_path)["rows"] if row["primary"]]
        source_dir = diode_source / "per_frame"
        receipt = read_json(diode_source / "inference_complete.json")
        include_checkpoint = True
        expected = 304
    else:
        manifest_path = arkit_root / "evaluation_manifest.json"
        rows = [row for row in read_json(manifest_path)["rows"] if row["primary"]]
        source_dir = arkit_root / "per_image"
        receipt = read_json(arkit_root / "inference_complete.json")
        include_checkpoint = False
        expected = 247
    assert len(rows) == expected
    return manifest_path, rows, source_dir, receipt, include_checkpoint


def array(tensor) -> np.ndarray:
    return tensor[0, 0].float().cpu().numpy()


def effective_gate(gc, sparse, size):
    spatial = gc.gate(sparse, size, "spatial")
    image_mean = spatial.mean((-2, -1), keepdim=True).expand_as(spatial)
    assert np.isfinite(float(spatial.mean().item()))
    assert float(spatial.std(unbiased=False).item()) > 0.0
    assert torch_allclose_mean(spatial, image_mean)
    return spatial, image_mean


def torch_allclose_mean(spatial, image_mean) -> bool:
    import torch

    return bool(
        torch.equal(image_mean, spatial.mean((-2, -1), keepdim=True).expand_as(spatial))
    )


def score_to_record(
    dataset_label: str,
    meta: dict[str, Any],
    law: str,
    repetition: int,
    seed: int,
    mode: str,
    checkpoint_hash: str,
    coordinate: str,
    score: dict[str, Any],
    gate_mean: float,
    gate_std: float,
    source: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset_label,
        "index": meta["index"],
        "scene": meta["scene"],
        "scan": meta["scan"],
        "image": meta["image"],
        "law": law,
        "repetition": repetition,
        "seed": seed,
        "inference_mode": mode,
        "checkpoint_sha256": checkpoint_hash,
        "coordinate_hash": coordinate,
        "score_pixels": int(score["score_pixels"]),
        "input_points": int(score["input_points"]),
        **{name: float(score[name]) for name in METRICS},
        "effective_gate_mean": gate_mean,
        "effective_gate_std": gate_std,
        "metric_source": source,
    }


def archived_score(row: dict[str, str]) -> dict[str, Any]:
    return {
        "score_pixels": int(row["score_pixels"]),
        "input_points": int(row["input_points"]),
        **{name: float(row[name]) for name in METRICS},
    }


def replay_check(
    dataset: str,
    dataset_label: str,
    meta: dict[str, Any],
    archived: dict[tuple[str, str, int], dict[str, str]],
    base,
    correctors,
    checkpoint_hashes,
    stack,
) -> dict[str, Any]:
    dc, torch, cc, gc, *_ = stack
    with np.load(meta["converted"], allow_pickle=False) as payload:
        rgb = payload["rgb"].copy()
        gt = payload["depth"].copy()
        valid = payload["valid"].copy()
        indices = payload["support_indices"].copy()
    law = UNSEEN8[0]
    law_index = list(dc.LAWS).index(law)
    repetition = 0
    ids = indices[repetition, law_index]
    sparse = np.zeros_like(gt)
    sparse.reshape(-1)[ids] = gt.reshape(-1)[ids]
    image = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().cuda() / 255.0
    dep = torch.from_numpy(sparse)[None, None].cuda()
    view = cc.encode(base, image, dep)
    spatial_gate, mean_gate = effective_gate(gc, dep, view["init"].shape[-2:])
    maximum = 0.0
    rows = []
    for seed in SEEDS:
        raw = cc.predict_delta(correctors[seed], view)
        spatial_prediction = array(
            cc.propagate(base, view, cc.corrected(view, raw * spatial_gate))
        )
        score = dc.score(spatial_prediction, gt, valid, ids)
        reference = archived[(f"spatial_seed{seed}", law, repetition)]
        diffs = {name: abs(score[name] - float(reference[name])) for name in METRICS}
        local = max(diffs.values())
        maximum = max(maximum, local)
        assert local < 1e-5, (dataset, seed, local, diffs)
        rows.append({"seed": seed, "maximum_metric_difference": local})
    return {
        "status": "PASS",
        "dataset": dataset_label,
        "index": meta["index"],
        "law": law,
        "repetition": repetition,
        "checkpoint_sha256": {str(k): v for k, v in checkpoint_hashes.items()},
        "maximum_metric_difference": maximum,
        "tolerance": 1e-5,
        "same_gate_mean_max_abs": float(
            (spatial_gate.mean((-2, -1), keepdim=True) - mean_gate).abs().max().item()
        ),
        "spatial_gate_std": float(spatial_gate.std(unbiased=False).item()),
        "image_mean_gate_std": float(mean_gate.std(unbiased=False).item()),
        "seeds": rows,
    }


def validate_existing_marker(
    result_path: Path, marker_path: Path, protocol_hash: str
) -> bool:
    if not marker_path.exists():
        return False
    marker = read_json(marker_path)
    assert marker["protocol_sha256"] == protocol_hash
    assert marker["result_sha256"] == sha256(result_path)
    assert marker["rows"] == 240
    return True


def run_dataset(
    dataset: str,
    diode_source: Path,
    arkit_root: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    stack = import_stack(diode_source)
    dc, torch, cc, gc, *_rest, gpu_lease = stack
    torch.set_num_threads(2)
    dataset_label = "DIODE" if dataset == "diode" else "ARKitScenes"
    manifest_path, manifest, archived_dir, source_receipt, has_checkpoint = dataset_inputs(
        dataset, diode_source, arkit_root
    )
    expected_manifest_hash = protocol["datasets"][dataset_label]["manifest_sha256"]
    assert sha256(manifest_path) == expected_manifest_hash
    protocol_hash = sha256(PROTOCOL)
    result_dir = ROOT / "per_image" / dataset
    result_dir.mkdir(parents=True, exist_ok=True)
    status_path = CHECKS / f"{dataset}_status.json"
    start = time.monotonic()
    completed = 0
    file_hashes: dict[str, str] = {}
    with gpu_lease(), torch.inference_mode():
        base, correctors, checkpoint_hashes = build_frozen_models(
            diode_source, stack
        )
        first_meta = manifest[0]
        first_source = archived_dir / f"{int(first_meta['index']):04d}.csv"
        first_archived, _ = archived_rows(
            first_source, source_receipt, checkpoint_hashes, has_checkpoint
        )
        replay = replay_check(
            dataset,
            dataset_label,
            first_meta,
            first_archived,
            base,
            correctors,
            checkpoint_hashes,
            stack,
        )
        save_json(CHECKS / f"{dataset}_replay.json", replay)
        for meta in manifest:
            result_path = result_dir / f"{int(meta['index']):04d}.csv"
            marker_path = result_path.with_suffix(".json")
            if validate_existing_marker(result_path, marker_path, protocol_hash):
                file_hashes[str(result_path.relative_to(ROOT)).replace("\\", "/")] = sha256(
                    result_path
                )
                completed += 1
                continue
            converted = Path(meta["converted"])
            assert sha256(converted) == meta["sha256"]
            archived_path = archived_dir / f"{int(meta['index']):04d}.csv"
            archived, archived_hash = archived_rows(
                archived_path, source_receipt, checkpoint_hashes, has_checkpoint
            )
            with np.load(converted, allow_pickle=False) as payload:
                rgb = payload["rgb"].copy()
                gt = payload["depth"].copy()
                valid = payload["valid"].copy()
                indices = payload["support_indices"].copy()
            assert rgb.shape == (228, 304, 3)
            assert gt.shape == valid.shape == (228, 304)
            image = (
                torch.from_numpy(rgb).permute(2, 0, 1)[None].float().cuda() / 255.0
            )
            records: list[dict[str, Any]] = []
            for law in UNSEEN8:
                law_index = list(dc.LAWS).index(law)
                for repetition in range(5):
                    ids = indices[repetition, law_index]
                    coordinate = condition_hash(ids)
                    sparse = np.zeros_like(gt)
                    sparse.reshape(-1)[ids] = gt.reshape(-1)[ids]
                    dep = torch.from_numpy(sparse)[None, None].cuda()
                    view = cc.encode(base, image, dep)
                    spatial_gate, mean_gate = effective_gate(
                        gc, dep, view["init"].shape[-2:]
                    )
                    gate_mean = float(spatial_gate.mean().item())
                    gate_std = float(spatial_gate.std(unbiased=False).item())
                    for seed in SEEDS:
                        reference = archived[
                            (f"spatial_seed{seed}", law, repetition)
                        ]
                        assert reference["coordinate_hash"] == coordinate
                        spatial_score = archived_score(reference)
                        records.append(
                            score_to_record(
                                dataset_label,
                                meta,
                                law,
                                repetition,
                                seed,
                                "spatial",
                                checkpoint_hashes[seed],
                                coordinate,
                                spatial_score,
                                gate_mean,
                                gate_std,
                                "hash_verified_frozen_spatial_evaluation",
                            )
                        )
                        raw = cc.predict_delta(correctors[seed], view)
                        prediction = array(
                            cc.propagate(
                                base,
                                view,
                                cc.corrected(view, raw * mean_gate),
                            )
                        )
                        intervention_score = dc.score(
                            prediction, gt, valid, ids
                        )
                        assert intervention_score["score_pixels"] == spatial_score[
                            "score_pixels"
                        ]
                        records.append(
                            score_to_record(
                                dataset_label,
                                meta,
                                law,
                                repetition,
                                seed,
                                "image_mean_intervention",
                                checkpoint_hashes[seed],
                                coordinate,
                                intervention_score,
                                gate_mean,
                                0.0,
                                "new_fixed_checkpoint_inference",
                            )
                        )
            assert len(records) == 240
            assert all(
                np.isfinite(float(row[name]))
                for row in records
                for name in METRICS
            )
            write_csv(result_path, records)
            result_hash = sha256(result_path)
            save_json(
                marker_path,
                {
                    "status": "COMPLETE",
                    "dataset": dataset_label,
                    "index": meta["index"],
                    "protocol_sha256": protocol_hash,
                    "converted_sha256": meta["sha256"],
                    "archived_source_sha256": archived_hash,
                    "result_sha256": result_hash,
                    "rows": len(records),
                    "conditions": 8 * 5 * 3 * 2,
                    "checkpoint_sha256": {
                        str(k): v for k, v in checkpoint_hashes.items()
                    },
                },
            )
            file_hashes[str(result_path.relative_to(ROOT)).replace("\\", "/")] = result_hash
            completed += 1
            status = {
                "status": "RUNNING",
                "dataset": dataset_label,
                "completed_images": completed,
                "total_images": len(manifest),
                "last_index": meta["index"],
                "elapsed_seconds": time.monotonic() - start,
                "pid": os.getpid(),
                "protocol_sha256": protocol_hash,
            }
            save_json(status_path, status)
            print(json.dumps(status), flush=True)
    expected_rows = protocol["expected_rows"][dataset_label]
    receipt = {
        "status": "COMPLETE",
        "dataset": dataset_label,
        "images": completed,
        "rows": completed * 240,
        "expected_rows": expected_rows,
        "protocol_sha256": protocol_hash,
        "replay_check_sha256": sha256(CHECKS / f"{dataset}_replay.json"),
        "elapsed_seconds": time.monotonic() - start,
        "environment": {
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        },
        "files": file_hashes,
    }
    assert receipt["rows"] == expected_rows
    save_json(CHECKS / f"{dataset}_inference_complete.json", receipt)
    save_json(status_path, {**receipt, "elapsed_seconds": receipt["elapsed_seconds"]})
    return receipt


def run(args: argparse.Namespace) -> None:
    assert PROTOCOL.exists(), "Run prepare before inference"
    protocol = read_json(PROTOCOL)
    original_hash = protocol["code_sha256"]["evaluate_intervention.py"]
    current_hash = sha256(__file__)
    if original_hash != current_hash:
        assert original_hash == "3c53ba118a1312210667fd662bbccf9088e26f1290a625a67e889dd9f0d7d76c"
        save_json(CHECKS / "implementation_amendment.json", {
            "original_code_sha256": original_hash,
            "executed_code_sha256": current_hash,
            "created_unix": time.time(),
            "reason": "Check constant broadcast by exact elementwise equality to the computed FP32 mean, avoiding a second FP32 reduction and its rounding error.",
            "prediction_computation_changed": False,
            "protocol_changed": False,
        })
    diode_source = Path(args.diode_source).resolve()
    arkit_root = Path(args.arkit_root).resolve()
    targets = ("diode", "arkitscenes") if args.dataset == "all" else (args.dataset,)
    receipts = {}
    for target in targets:
        receipts[target] = run_dataset(target, diode_source, arkit_root, protocol)
    if args.dataset == "all":
        combined = {
            "status": "ALL_COMPLETE",
            "protocol_sha256": sha256(PROTOCOL),
            "pid": os.getpid(),
            "datasets": {
                key: {
                    "images": value["images"],
                    "rows": value["rows"],
                    "receipt_sha256": sha256(
                        CHECKS / f"{key}_inference_complete.json"
                    ),
                }
                for key, value in receipts.items()
            },
            "total_images": sum(value["images"] for value in receipts.values()),
            "total_rows": sum(value["rows"] for value in receipts.values()),
        }
        save_json(CHECKS / "inference_complete.json", combined)
        print(json.dumps(combined), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="action", required=True)
    for action in ("prepare", "run"):
        child = subparsers.add_parser(action)
        child.add_argument("--diode-source", required=True)
        child.add_argument("--arkit-root", required=True)
        if action == "run":
            child.add_argument(
                "--dataset",
                choices=("diode", "arkitscenes", "all"),
                default="all",
            )
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.action == "prepare":
        prepare(arguments)
    else:
        run(arguments)
