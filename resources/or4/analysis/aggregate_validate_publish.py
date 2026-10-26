"""Validate, aggregate, and publish the fixed-checkpoint intervention."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocol" / "frozen_protocol.json"
CHECKS = ROOT / "checks"
SUMMARY = ROOT / "summary"
METRICS = (
    "rmse",
    "mae",
    "rel",
    "unobserved_rmse",
    "unobserved_mae",
    "unobserved_rel",
)
DATASETS = {
    "diode": {
        "label": "DIODE",
        "images": 304,
        "physical_column": "scene",
        "physical_label": "scene",
        "physical_units": 3,
    },
    "arkitscenes": {
        "label": "ARKitScenes",
        "images": 247,
        "physical_column": "scene",
        "physical_label": "site",
        "physical_units": 20,
    },
}
MODES = ("spatial", "image_mean_intervention")
SEEDS = (42, 43, 44)
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
BOOTSTRAP_REPS = 200_000
BOOTSTRAP_SEED = 20_261_022


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


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def verify_and_load(dataset: str, protocol: dict[str, Any]):
    spec = DATASETS[dataset]
    receipt_path = CHECKS / f"{dataset}_inference_complete.json"
    receipt = read_json(receipt_path)
    assert receipt["status"] == "COMPLETE"
    assert receipt["protocol_sha256"] == sha256(PROTOCOL)
    assert receipt["images"] == spec["images"]
    result_dir = ROOT / "per_image" / dataset
    frames = []
    marker_count = 0
    for csv_path in sorted(result_dir.glob("*.csv")):
        marker_path = csv_path.with_suffix(".json")
        assert marker_path.exists()
        marker = read_json(marker_path)
        relative = str(csv_path.relative_to(ROOT)).replace("\\", "/")
        actual = sha256(csv_path)
        assert receipt["files"][relative] == actual == marker["result_sha256"]
        assert marker["protocol_sha256"] == sha256(PROTOCOL)
        assert marker["rows"] == 240
        frame = pd.read_csv(csv_path)
        assert len(frame) == 240
        frames.append(frame)
        marker_count += 1
    assert marker_count == spec["images"]
    data = pd.concat(frames, ignore_index=True)
    expected_rows = protocol["expected_rows"][spec["label"]]
    assert len(data) == receipt["rows"] == expected_rows
    assert set(data.dataset) == {spec["label"]}
    assert set(data.inference_mode) == set(MODES)
    assert set(data.seed) == set(SEEDS)
    assert set(data.law) == set(UNSEEN8)
    assert set(data.repetition) == set(range(5))
    assert data.groupby("index").size().eq(240).all()
    keys = ["index", "law", "repetition", "seed", "inference_mode"]
    assert not data.duplicated(keys).any()
    pairing = ["index", "law", "repetition", "seed"]
    assert data.groupby(pairing).inference_mode.nunique().eq(2).all()
    assert data.groupby(pairing).checkpoint_sha256.nunique().eq(1).all()
    assert data.groupby(pairing).coordinate_hash.nunique().eq(1).all()
    assert data.groupby(pairing).score_pixels.nunique().eq(1).all()
    assert data.groupby(pairing).input_points.nunique().eq(1).all()
    assert (data.input_points == 500).all()
    expected_hashes = protocol["checkpoint_sha256"]
    for seed in SEEDS:
        assert set(data.loc[data.seed == seed, "checkpoint_sha256"]) == {
            expected_hashes[str(seed)]
        }
    values = data[list(METRICS)].to_numpy(dtype=np.float64)
    assert np.isfinite(values).all() and (values >= 0).all()
    assert (
        data.loc[
            data.inference_mode == "spatial", "metric_source"
        ].eq("hash_verified_frozen_spatial_evaluation")
    ).all()
    assert (
        data.loc[
            data.inference_mode == "image_mean_intervention", "metric_source"
        ].eq("new_fixed_checkpoint_inference")
    ).all()
    assert (
        data.loc[
            data.inference_mode == "image_mean_intervention", "effective_gate_std"
        ].abs()
        < 1e-12
    ).all()
    assert (
        data.loc[data.inference_mode == "spatial", "effective_gate_std"] > 0
    ).all()
    gate_pair = data.pivot(
        index=pairing, columns="inference_mode", values="effective_gate_mean"
    )
    gate_mean_max = float(
        (
            gate_pair["spatial"] - gate_pair["image_mean_intervention"]
        ).abs().max()
    )
    assert gate_mean_max < 1e-7
    replay = read_json(CHECKS / f"{dataset}_replay.json")
    assert replay["status"] == "PASS"
    assert replay["maximum_metric_difference"] < replay["tolerance"]
    return data, receipt, replay, gate_mean_max


def per_image_table(data: pd.DataFrame) -> pd.DataFrame:
    columns = ["dataset", "index", "scene", "scan", "image", "seed", "inference_mode"]
    result = data.groupby(columns, as_index=False)[list(METRICS)].mean()
    assert result.groupby(["dataset", "index"]).size().eq(6).all()
    return result


def unit_table(per_image: pd.DataFrame, dataset: str) -> pd.DataFrame:
    spec = DATASETS[dataset]
    source = per_image[per_image.dataset == spec["label"]]
    # Main endpoint: seed average inside each image, then image average inside
    # the physical unit.  Modes remain paired throughout.
    image_seed_mean = source.groupby(
        ["dataset", "index", "scene", "scan", "image", "inference_mode"],
        as_index=False,
    )[list(METRICS)].mean()
    units = image_seed_mean.groupby(
        ["dataset", spec["physical_column"], "inference_mode"], as_index=False
    )[list(METRICS)].mean()
    assert units[spec["physical_column"]].nunique() == spec["physical_units"]
    return units


def bootstrap_pair(spatial: np.ndarray, intervention: np.ndarray):
    assert len(spatial) == len(intervention)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0, len(spatial), size=(BOOTSTRAP_REPS, len(spatial)), endpoint=False
    )
    sampled_spatial = spatial[indices].mean(axis=1)
    sampled_intervention = intervention[indices].mean(axis=1)
    delta = sampled_spatial - sampled_intervention
    change = 100.0 * (sampled_spatial / sampled_intervention - 1.0)
    return (
        float(np.quantile(delta, 0.025)),
        float(np.quantile(delta, 0.975)),
        float(np.quantile(change, 0.025)),
        float(np.quantile(change, 0.975)),
    )


def paired_effects(per_image: pd.DataFrame):
    effect_rows = []
    unit_rows = []
    for dataset_index, (dataset, spec) in enumerate(DATASETS.items()):
        units = unit_table(per_image, dataset)
        unit_column = spec["physical_column"]
        for metric_index, metric in enumerate(METRICS):
            wide = units.pivot(
                index=unit_column, columns="inference_mode", values=metric
            ).sort_index()
            assert len(wide) == spec["physical_units"] and not wide.isna().any().any()
            spatial = wide["spatial"].to_numpy(dtype=np.float64)
            intervention = wide["image_mean_intervention"].to_numpy(
                dtype=np.float64
            )
            delta = spatial - intervention
            ci = bootstrap_pair(spatial, intervention)
            effect_rows.append(
                {
                    "dataset": spec["label"],
                    "endpoint": metric,
                    "unit": spec["physical_label"],
                    "n_units": len(wide),
                    "spatial_mean": float(spatial.mean()),
                    "image_mean_intervention_mean": float(intervention.mean()),
                    "delta_spatial_minus_intervention": float(delta.mean()),
                    "change_percent": float(
                        100.0 * (spatial.mean() / intervention.mean() - 1.0)
                    ),
                    "ci95_delta_low": ci[0],
                    "ci95_delta_high": ci[1],
                    "ci95_change_low_percent": ci[2],
                    "ci95_change_high_percent": ci[3],
                    "bootstrap_reps": BOOTSTRAP_REPS,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "benefiting_units": int((delta < 0).sum()),
                    "worsening_units": int((delta > 0).sum()),
                    "tied_units": int((delta == 0).sum()),
                }
            )
            for unit, row in wide.iterrows():
                unit_rows.append(
                    {
                        "dataset": spec["label"],
                        "endpoint": metric,
                        "unit_type": spec["physical_label"],
                        "unit_id": unit,
                        "spatial": float(row["spatial"]),
                        "image_mean_intervention": float(
                            row["image_mean_intervention"]
                        ),
                        "delta_spatial_minus_intervention": float(
                            row["spatial"] - row["image_mean_intervention"]
                        ),
                    }
                )
    return pd.DataFrame(effect_rows), pd.DataFrame(unit_rows)


def seed_effects(per_image: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, spec in DATASETS.items():
        source = per_image[per_image.dataset == spec["label"]]
        unit_column = spec["physical_column"]
        units = source.groupby(
            ["dataset", unit_column, "seed", "inference_mode"], as_index=False
        )[list(METRICS)].mean()
        for seed in SEEDS:
            subset = units[units.seed == seed]
            for metric in METRICS:
                wide = subset.pivot(
                    index=unit_column, columns="inference_mode", values=metric
                ).sort_index()
                spatial = wide["spatial"].to_numpy(dtype=np.float64)
                intervention = wide["image_mean_intervention"].to_numpy(
                    dtype=np.float64
                )
                delta = spatial - intervention
                rows.append(
                    {
                        "dataset": spec["label"],
                        "seed": seed,
                        "endpoint": metric,
                        "unit": spec["physical_label"],
                        "n_units": len(wide),
                        "spatial_mean": float(spatial.mean()),
                        "image_mean_intervention_mean": float(
                            intervention.mean()
                        ),
                        "delta_spatial_minus_intervention": float(delta.mean()),
                        "change_percent": float(
                            100.0 * (spatial.mean() / intervention.mean() - 1.0)
                        ),
                        "benefiting_units": int((delta < 0).sum()),
                        "worsening_units": int((delta > 0).sum()),
                        "tied_units": int((delta == 0).sum()),
                        "sample_sd_unit_delta": float(delta.std(ddof=1)),
                    }
                )
    return pd.DataFrame(rows)


def coverage(per_image: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, spec in DATASETS.items():
        source = per_image[per_image.dataset == spec["label"]]
        # Average seeds before coverage counts, as in the main endpoint.
        pooled = source.groupby(
            ["dataset", "index", "scene", "scan", "image", "inference_mode"],
            as_index=False,
        )["rmse"].mean()
        levels = [("image", ["index"])]
        if dataset == "diode":
            levels += [("scan", ["scan"]), ("scene", ["scene"])]
        else:
            levels += [("scan", ["scan"]), ("site", ["scene"])]
        for label, group_columns in levels:
            grouped = pooled.groupby(group_columns + ["inference_mode"])[
                "rmse"
            ].mean().unstack("inference_mode")
            delta = grouped["spatial"] - grouped["image_mean_intervention"]
            rows.append(
                {
                    "dataset": spec["label"],
                    "endpoint": "rmse",
                    "level": label,
                    "n_total": len(delta),
                    "n_spatial_better": int((delta < 0).sum()),
                    "n_intervention_better": int((delta > 0).sum()),
                    "n_tied": int((delta == 0).sum()),
                    "spatial_better_percent": float(100 * (delta < 0).mean()),
                    "mean_delta_spatial_minus_intervention": float(delta.mean()),
                }
            )
    return pd.DataFrame(rows)


def readme_text(effects: pd.DataFrame, validation: dict[str, Any]) -> str:
    primary = effects[effects.endpoint == "rmse"].set_index("dataset")
    lines = [
        "# OR4: fixed-checkpoint gate intervention",
        "",
        "This resource reports the complete inference-only intervention in which",
        "each trained Spatial checkpoint is held fixed and its position-dependent",
        "gate is replaced by the within-image mean of that same gate. The separately",
        "trained Image Mean branch is not used in this comparison.",
        "",
        "## Scope and pairing",
        "",
        "- DIODE: 304 images, all eight unseen layouts, five paired support draws,",
        "  three Spatial checkpoints, and three equally weighted physical scenes.",
        "- ARKitScenes: 247 images from 20 held-out sites, with the same eight",
        "  unseen layouts, five paired draws, and three checkpoints.",
        "- Pairing keys are dataset, image index, layout, repetition, training seed,",
        "  support-coordinate hash, and scoring mask/valid-pixel count.",
        "- Negative `delta_spatial_minus_intervention` means that retaining the",
        "  spatial variation of the gate produced lower error.",
        "",
        "## Primary RMSE result",
        "",
        "| Dataset | Spatial | Fixed-checkpoint mean-gate | Delta | Change | 95% CI of delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in ("DIODE", "ARKitScenes"):
        row = primary.loc[label]
        lines.append(
            f"| {label} | {row.spatial_mean:.6f} | "
            f"{row.image_mean_intervention_mean:.6f} | "
            f"{row.delta_spatial_minus_intervention:+.6f} | "
            f"{row.change_percent:+.2f}% | "
            f"[{row.ci95_delta_low:+.6f}, {row.ci95_delta_high:+.6f}] |"
        )
    lines += [
        "",
        "The table above is descriptive of the frozen cohorts. DIODE intervals",
        "resample three scenes and therefore mainly expose composition sensitivity.",
        "All endpoints, seeds, individual physical units, and coverage counts are",
        "retained under `summary/`.",
        "",
        "## Files",
        "",
        "- `protocol/frozen_protocol.json`: intervention, identities, grids, and",
        "  aggregation rules fixed before the new counterfactual inference.",
        "- `per_image/{diode,arkitscenes}/`: complete paired condition records and",
        "  per-file receipts.",
        "- `summary/paired_effects.csv`: physical-unit-balanced effects and CIs.",
        "- `summary/coverage.csv`: image/scan/scene/site directional coverage.",
        "- `summary/seed_effects.csv`: all three checkpoint-specific effects.",
        "- `checks/validation.json`: identity, grid, pairing, replay, finite-value,",
        "  and aggregation checks.",
        "",
        "## Reaggregation",
        "",
        "Run `python analysis/aggregate_validate_publish.py aggregate` from a copy",
        "of this OR4 directory after adapting `ROOT` or placing the script back in",
        "the experiment layout. Model-level reruns additionally require the private",
        "Level 3 checkpoints, frozen converted inputs, and original source stack.",
        "",
        f"Validation status: `{validation['status']}`; total paired rows: "
        f"{validation['total_rows']}; protocol SHA-256: `{validation['protocol_sha256']}`.",
        "",
    ]
    return "\n".join(lines)


def aggregate() -> dict[str, Any]:
    protocol = read_json(PROTOCOL)
    assert protocol["code_sha256"]["aggregate_validate_publish.py"] == sha256(
        __file__
    )
    all_frames = []
    receipts = {}
    replay_checks = {}
    gate_differences = {}
    for dataset in DATASETS:
        data, receipt, replay, gate_mean_max = verify_and_load(dataset, protocol)
        all_frames.append(data)
        receipts[dataset] = receipt
        replay_checks[dataset] = replay
        gate_differences[dataset] = gate_mean_max
    data = pd.concat(all_frames, ignore_index=True)
    per_image = per_image_table(data)
    effects, unit_effects = paired_effects(per_image)
    seeds = seed_effects(per_image)
    cover = coverage(per_image)
    write_frame(SUMMARY / "per_image_unseen8.csv", per_image)
    write_frame(SUMMARY / "paired_effects.csv", effects)
    write_frame(SUMMARY / "physical_unit_effects.csv", unit_effects)
    write_frame(SUMMARY / "seed_effects.csv", seeds)
    write_frame(SUMMARY / "coverage.csv", cover)

    # Independent direct aggregation of the two RMSE primary endpoints.
    direct = {}
    for dataset, spec in DATASETS.items():
        source = data[data.dataset == spec["label"]]
        direct_values = source.groupby(
            [spec["physical_column"], "inference_mode"]
        ).rmse.mean().groupby("inference_mode").mean()
        row = effects[
            (effects.dataset == spec["label"]) & (effects.endpoint == "rmse")
        ].iloc[0]
        difference = max(
            abs(float(direct_values["spatial"]) - row.spatial_mean),
            abs(
                float(direct_values["image_mean_intervention"])
                - row.image_mean_intervention_mean
            ),
        )
        assert difference < 1e-12
        direct[dataset] = difference
    validation = {
        "status": "PASS",
        "protocol_sha256": sha256(PROTOCOL),
        "total_rows": int(len(data)),
        "expected_total_rows": int(
            sum(protocol["expected_rows"].values())
        ),
        "images": {dataset: spec["images"] for dataset, spec in DATASETS.items()},
        "complete_condition_grids": True,
        "all_file_and_receipt_hashes_verified": True,
        "same_spatial_checkpoint_in_both_modes": True,
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "pairing_keys": [
            "dataset",
            "index",
            "law",
            "repetition",
            "seed",
            "coordinate_hash",
            "score_pixels",
        ],
        "coordinate_and_scoring_mask_pairing_verified": True,
        "finite_nonnegative_metrics_verified": True,
        "effective_gate_means_match_max_abs": gate_differences,
        "spatial_replay": {
            dataset: {
                "status": replay["status"],
                "maximum_metric_difference": replay[
                    "maximum_metric_difference"
                ],
                "tolerance": replay["tolerance"],
                "check_sha256": sha256(CHECKS / f"{dataset}_replay.json"),
            }
            for dataset, replay in replay_checks.items()
        },
        "source_spatial_records": {
            "DIODE": {
                "record": "external_candidate_freeze_20261016/per_frame/*.csv",
                "receipt_sha256": protocol["datasets"]["DIODE"][
                    "source_receipt_sha256"
                ],
                "plan_sha256": protocol["datasets"]["DIODE"][
                    "source_plan_sha256"
                ],
            },
            "ARKitScenes": {
                "record": "new_scene_validation/per_image/*.csv",
                "receipt_sha256": protocol["datasets"]["ARKitScenes"][
                    "source_receipt_sha256"
                ],
                "plan_sha256": protocol["datasets"]["ARKitScenes"][
                    "source_plan_sha256"
                ],
            },
        },
        "aggregation_independent_recalculation_max_abs": direct,
        "bootstrap": {
            "method": "paired nonparametric percentile over physical units",
            "repetitions": BOOTSTRAP_REPS,
            "base_seed": BOOTSTRAP_SEED,
        },
        "result_sign_policy": protocol["result_policy"],
    }
    assert validation["total_rows"] == validation["expected_total_rows"]
    save_json(CHECKS / "validation.json", validation)
    (ROOT / "README.md").write_text(
        readme_text(effects, validation), encoding="utf-8"
    )
    print(json.dumps(validation, indent=2), flush=True)
    print(
        effects[effects.endpoint == "rmse"].to_string(index=False), flush=True
    )
    return validation


def publish(repository: Path) -> None:
    validation = read_json(CHECKS / "validation.json")
    assert validation["status"] == "PASS"
    destination = repository / "resources" / "or4"
    destination.mkdir(parents=True, exist_ok=True)
    mappings = {
        ROOT / "README.md": destination / "README.md",
        ROOT / "source": destination / "analysis",
        ROOT / "protocol": destination / "protocol",
        ROOT / "per_image": destination / "per_image",
        ROOT / "summary": destination / "summary",
        ROOT / "checks": destination / "checks",
    }
    for source, target in mappings.items():
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    manifest_path = repository / "manifests" / "or4_sha256.txt"
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    lines = [
        f"{sha256(path)}  {path.relative_to(destination).as_posix()}" for path in files
    ]
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    for line in lines:
        expected, relative = line.split("  ", 1)
        assert sha256(destination / relative) == expected
    save_json(
        CHECKS / "publish_receipt.json",
        {
            "status": "PASS",
            "files": len(files),
            "manifest_sha256": sha256(manifest_path),
            "validation_sha256": sha256(destination / "checks" / "validation.json"),
        },
    )
    # Include the final publish receipt itself, then refresh the manifest once.
    shutil.copy2(
        CHECKS / "publish_receipt.json", destination / "checks" / "publish_receipt.json"
    )
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    lines = [
        f"{sha256(path)}  {path.relative_to(destination).as_posix()}" for path in files
    ]
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(
        json.dumps(
            {
                "status": "PUBLISHED",
                "destination": str(destination),
                "files": len(files),
                "manifest": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("aggregate", "publish", "all"))
    parser.add_argument("--repository")
    args = parser.parse_args()
    if args.action in ("aggregate", "all"):
        aggregate()
    if args.action in ("publish", "all"):
        assert args.repository, "--repository is required for publish"
        publish(Path(args.repository).resolve())


if __name__ == "__main__":
    main()
