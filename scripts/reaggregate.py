"""Reproduce the principal DIODE and ARKitScenes aggregate RMSE results."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

UNSEEN8 = [
    "top", "bottom", "center_hole", "diag_main_half", "diag_anti_half",
    "center_vertical_band", "center_disk", "two_corners",
]


def diode(root: Path) -> pd.DataFrame:
    files = sorted((root / "resources/or2/mechanism/per_image").glob("*.csv"))
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    data = data[(data.region == "all") & data.law.isin(UNSEEN8)]
    image_mean = data[data.control == "image_mean"].copy()
    image_mean["spatial"] = np.sqrt(image_mean.spatial_sse / image_mean.valid_pixels)
    image_mean["image_mean"] = np.sqrt(image_mean.control_sse / image_mean.valid_pixels)
    parent = data[data.control == "parent"].copy()
    parent["parent"] = np.sqrt(parent.control_sse / parent.valid_pixels)
    rows = []
    for method, frame in [("spatial", image_mean), ("image_mean", image_mean), ("parent", parent)]:
        by_scene = (
            frame.groupby(["scene", "index", "image", "seed"])[method].mean()
            .groupby(["scene", "seed"]).mean().groupby("scene").mean()
        )
        rows.extend(
            {"dataset": "DIODE", "unit": scene, "method": method, "rmse": value}
            for scene, value in by_scene.items()
        )
    return pd.DataFrame(rows)


def arkitscenes(root: Path) -> pd.DataFrame:
    files = sorted((root / "resources/or3/per_image").glob("[0-9][0-9][0-9][0-9].csv"))
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    data = data[data.law.isin(UNSEEN8)].copy()
    data["method"] = data.model.str.replace(r"_seed(?:42|43|44)$", "", regex=True)
    data["seed"] = data.model.str.extract(r"_seed(\d+)$")[0].fillna("parent")
    by_venue = (
        data.groupby(["scene", "index", "image", "method", "seed"]).rmse.mean()
        .groupby(["scene", "method", "seed"]).mean()
        .groupby(["scene", "method"]).mean().reset_index()
    )
    by_venue.insert(0, "dataset", "ARKitScenes")
    return by_venue.rename(columns={"scene": "unit"})[["dataset", "unit", "method", "rmse"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    units = pd.concat([diode(args.root), arkitscenes(args.root)], ignore_index=True)
    summary = units.groupby(["dataset", "method"]).rmse.mean().reset_index()
    output = args.out or args.root / "derived/principal_rmse_reaggregated.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
