from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "derived" / "figures"
STATS_DIR = Path(__file__).resolve().parent
SEED = 20261022
REPS = 200_000

COLORS = {
    "ink": "#222222",
    "muted": "#666666",
    "light": "#D9D9D9",
    "spatial": "#0072B2",
    "image_mean": "#E69F00",
    "green": "#009E73",
    "near": "#E69F00",
    "middle": "#CC79A7",
    "far": "#0072B2",
    "input": "#888888",
    "net": "#111111",
}

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "SimSun", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": True,
})


def read_csv(rel: str) -> list[dict[str, str]]:
    with (ROOT / "resources" / rel).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def performance_stats() -> tuple[list[dict], dict]:
    rng = np.random.default_rng(SEED)
    records: list[dict] = []

    diode_rows = read_csv("or1/tables/paired_units.csv")
    diode_scene = [
        r for r in diode_rows
        if r["family"] == "unseen8" and r["unit"] == "scene"
        and r["control"] in {"parent", "image_mean"}
    ]
    diode_scan = [
        r for r in diode_rows
        if r["family"] == "unseen8" and r["unit"] == "scan"
        and r["control"] in {"parent", "image_mean"}
    ]
    scenes = sorted({r["id"] for r in diode_scene})
    diode_plot = {}
    diode_idx = rng.integers(0, len(scenes), size=(REPS, len(scenes)))
    for control in ["parent", "image_mean"]:
        lookup = {r["id"]: r for r in diode_scene if r["control"] == control}
        spatial = np.array([float(lookup[u]["spatial_rmse_m"]) for u in scenes])
        baseline = np.array([float(lookup[u]["control_rmse_m"]) for u in scenes])
        boot_spatial = spatial[diode_idx].mean(axis=1)
        boot_control = baseline[diode_idx].mean(axis=1)
        boot_delta = boot_spatial - boot_control
        boot_change = 100.0 * (boot_spatial / boot_control - 1.0)
        estimate_delta = float(spatial.mean() - baseline.mean())
        estimate_change = float(100.0 * (spatial.mean() / baseline.mean() - 1.0))
        ci_delta = percentile_ci(boot_delta)
        ci_change = percentile_ci(boot_change)
        points = np.array([
            float(r["delta_m"]) for r in diode_scan if r["control"] == control
        ])
        diode_plot[control] = {
            "points": points,
            "estimate": estimate_delta,
            "ci": ci_delta,
            "benefit": int(np.sum(points < 0)),
            "n_points": int(points.size),
        }
        records.append({
            "analysis": "performance",
            "dataset": "DIODE",
            "comparison": f"Spatial - {'Parent' if control == 'parent' else 'Image Mean'}",
            "unit": "scene",
            "n_units": len(scenes),
            "estimate_delta_rmse_m": estimate_delta,
            "ci95_delta_low_m": ci_delta[0],
            "ci95_delta_high_m": ci_delta[1],
            "estimate_change_percent": estimate_change,
            "ci95_change_low_percent": ci_change[0],
            "ci95_change_high_percent": ci_change[1],
            "bootstrap_reps": REPS,
            "bootstrap_seed": SEED,
        })

    arkit_rows = [
        r for r in read_csv("or3/summary/venue_metrics.csv")
        if r["family"] == "unseen8" and r["method"] in {"parent", "spatial", "image_mean"}
    ]
    venues = sorted({r["scene"] for r in arkit_rows})
    arkit_lookup = {(r["scene"], r["method"]): float(r["rmse"]) for r in arkit_rows}
    arkit_idx = rng.integers(0, len(venues), size=(REPS, len(venues)))
    spatial = np.array([arkit_lookup[(u, "spatial")] for u in venues])
    arkit_plot = {}
    for control in ["parent", "image_mean"]:
        baseline = np.array([arkit_lookup[(u, control)] for u in venues])
        boot_spatial = spatial[arkit_idx].mean(axis=1)
        boot_control = baseline[arkit_idx].mean(axis=1)
        boot_delta = boot_spatial - boot_control
        boot_change = 100.0 * (boot_spatial / boot_control - 1.0)
        estimate_delta = float(spatial.mean() - baseline.mean())
        estimate_change = float(100.0 * (spatial.mean() / baseline.mean() - 1.0))
        ci_delta = percentile_ci(boot_delta)
        ci_change = percentile_ci(boot_change)
        points = spatial - baseline
        arkit_plot[control] = {
            "points": points,
            "estimate": estimate_delta,
            "ci": ci_delta,
            "benefit": int(np.sum(points < 0)),
            "n_points": int(points.size),
        }
        records.append({
            "analysis": "performance",
            "dataset": "ARKitScenes",
            "comparison": f"Spatial - {'Parent' if control == 'parent' else 'Image Mean'}",
            "unit": "venue",
            "n_units": len(venues),
            "estimate_delta_rmse_m": estimate_delta,
            "ci95_delta_low_m": ci_delta[0],
            "ci95_delta_high_m": ci_delta[1],
            "estimate_change_percent": estimate_change,
            "ci95_change_low_percent": ci_change[0],
            "ci95_change_high_percent": ci_change[1],
            "bootstrap_reps": REPS,
            "bootstrap_seed": SEED,
        })
    return records, {"DIODE": diode_plot, "ARKitScenes": arkit_plot}


def regional_stats() -> tuple[list[dict], dict]:
    rng = np.random.default_rng(SEED)
    records: list[dict] = []
    plot_data: dict[str, dict] = {}
    regions = ["all", "far", "middle", "near_unobserved", "input"]
    sources = {
        "DIODE": ("or2/summary/scene_regions.csv", "scene"),
        "ARKitScenes": ("or3/summary/venue_regions.csv", "scene"),
    }
    for dataset, (rel, unit_key) in sources.items():
        rows = [r for r in read_csv(rel) if r["control"] == "image_mean"]
        units = sorted({r[unit_key] for r in rows})
        lookup = {(r[unit_key], r["region"]): r for r in rows}
        numerators = np.array([
            [float(lookup[(u, reg)]["delta_mse"]) for reg in regions]
            for u in units
        ])
        denominators = np.array([
            float(lookup[(u, "all")]["whole_control_mse"]) for u in units
        ])
        unit_points = np.array([
            [float(lookup[(u, reg)]["contribution_percent"]) for reg in regions]
            for u in units
        ])
        idx = rng.integers(0, len(units), size=(REPS, len(units)))
        boot_num = numerators[idx].mean(axis=1)
        boot_den = denominators[idx].mean(axis=1)
        boot = 100.0 * boot_num / boot_den[:, None]
        estimates = 100.0 * numerators.mean(axis=0) / denominators.mean()
        ds_plot = {}
        for j, reg in enumerate(regions):
            ci = percentile_ci(boot[:, j])
            ds_plot[reg] = {
                "estimate": float(estimates[j]),
                "ci": ci,
                "points": unit_points[:, j],
            }
            records.append({
                "analysis": "regional_contribution",
                "dataset": dataset,
                "comparison": "Spatial - Image Mean",
                "region": reg,
                "unit": "scene" if dataset == "DIODE" else "venue",
                "n_units": len(units),
                "estimate_percent_of_whole_image_mean_mse": float(estimates[j]),
                "ci95_low_percent": ci[0],
                "ci95_high_percent": ci[1],
                "bootstrap_reps": REPS,
                "bootstrap_seed": SEED,
            })
        plot_data[dataset] = ds_plot
    return records, plot_data


def save_all(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def add_box(ax, xy, width, height, text, *, fc, ec, fontsize=8.2, weight="normal"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=fc, edgecolor=ec, linewidth=1.0,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            fontsize=fontsize, color=COLORS["ink"], weight=weight, linespacing=1.12)
    return patch


def arrow(ax, start, end, *, color=None, style="-"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=10,
        linewidth=1.05, linestyle=style,
        color=color or COLORS["ink"], shrinkA=2, shrinkB=2,
        connectionstyle="arc3,rad=0.0",
    ))


def draw_method_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.05, 3.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(ax, (0.015, 0.49), 0.14, 0.25, "RGB + sparse depth\nmask; 500 points", fc="#F2F2F2", ec="#777777")
    add_box(ax, (0.205, 0.51), 0.145, 0.21, "Frozen parent\nfeature extractor", fc="#E9EEF3", ec="#596A7A", weight="bold")
    add_box(ax, (0.395, 0.55), 0.135, 0.17, "Initial dense depth\n$D^{init}$", fc="#E9EEF3", ec="#596A7A")
    arrow(ax, (0.155, 0.615), (0.205, 0.615))
    arrow(ax, (0.35, 0.615), (0.395, 0.635))

    add_box(ax, (0.395, 0.17), 0.16, 0.22, "Lightweight corrector\n7-channel input\n24,513 parameters", fc="#E8F4FB", ec=COLORS["spatial"], weight="bold")
    arrow(ax, (0.155, 0.54), (0.395, 0.29), color="#777777", style="--")
    arrow(ax, (0.462, 0.55), (0.462, 0.39), color="#596A7A")
    add_box(ax, (0.59, 0.18), 0.12, 0.18, "Filtered residual\n" + r"$L(\delta)$", fc="#E8F4FB", ec=COLORS["spatial"])
    arrow(ax, (0.555, 0.28), (0.59, 0.27), color=COLORS["spatial"])

    add_box(ax, (0.59, 0.56), 0.12, 0.17, "Support-distance gate\n$g(p)$; mean $\\bar{g}$", fc="#FDF1E3", ec=COLORS["image_mean"], fontsize=7.1)

    branch = FancyBboxPatch(
        (0.745, 0.20), 0.16, 0.55,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor="#FAFAFA", edgecolor="#777777", linewidth=1.0,
    )
    ax.add_patch(branch)
    ax.text(0.825, 0.705, "Spatial", ha="center", va="center", fontsize=8.5,
            weight="bold", color=COLORS["spatial"])
    ax.text(0.825, 0.62, r"$D^{init}+\sigma g(p)L(\delta)$", ha="center", va="center", fontsize=7.4)
    ax.plot([0.762, 0.888], [0.535, 0.535], color="#D9D9D9", lw=0.8)
    ax.text(0.825, 0.485, "Image Mean", ha="center", va="center", fontsize=8.5,
            weight="bold", color=COLORS["image_mean"])
    ax.text(0.825, 0.40, r"$D^{init}+\sigma\bar{g}L(\delta)$", ha="center", va="center", fontsize=7.4)
    ax.plot([0.762, 0.888], [0.315, 0.315], color="#D9D9D9", lw=0.8)
    ax.text(0.825, 0.265, "Parent", ha="center", va="center", fontsize=8.5,
            weight="bold", color=COLORS["muted"])
    ax.text(0.825, 0.225, r"$D^{init}$", ha="center", va="center", fontsize=8.0)

    arrow(ax, (0.71, 0.64), (0.745, 0.62), color=COLORS["image_mean"])
    arrow(ax, (0.71, 0.27), (0.745, 0.43), color=COLORS["spatial"])

    add_box(ax, (0.915, 0.30), 0.08, 0.36, "Frozen\npropagation\n+ output\nhead", fc="#E9EEF3", ec="#596A7A", fontsize=6.9)
    arrow(ax, (0.905, 0.66), (0.915, 0.59), color=COLORS["spatial"])
    arrow(ax, (0.905, 0.47), (0.915, 0.48), color=COLORS["image_mean"])
    arrow(ax, (0.905, 0.255), (0.915, 0.37), color="#777777")

    ax.text(0.015, 0.91, "Shared input and frozen parent", fontsize=8.4, weight="bold", color=COLORS["muted"])
    ax.text(0.395, 0.91, "Shared corrector architecture and training protocol", fontsize=8.4,
            weight="bold", color=COLORS["muted"])
    save_all(fig, "图1_空间门控适配框架")


def draw_performance_figure(plot_data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.65), gridspec_kw={"wspace": 0.33})
    panel_specs = [
        ("DIODE", axes[0], (-0.76, 0.10), "3 scenes; 10 scans"),
        ("ARKitScenes", axes[1], (-0.115, 0.065), "20 venues"),
    ]
    rng = np.random.default_rng(111)
    for panel, (dataset, ax, xlim, subtitle) in enumerate(panel_specs):
        ax.axvline(0, color="#333333", lw=0.9, zorder=0)
        for y, control, color, marker in [
            (1.0, "parent", COLORS["muted"], "o"),
            (0.0, "image_mean", COLORS["spatial"], "s"),
        ]:
            item = plot_data[dataset][control]
            points = np.asarray(item["points"])
            jitter = rng.uniform(-0.105, 0.105, size=points.size)
            ax.scatter(points, y + jitter, s=18, facecolor="white", edgecolor=color,
                       linewidth=0.8, marker=marker, alpha=0.82, zorder=2)
            lo, hi = item["ci"]
            ax.errorbar(item["estimate"], y, xerr=[[item["estimate"] - lo], [hi - item["estimate"]]],
                        fmt="D", color=color, markerfacecolor=color, markeredgecolor="white",
                        markeredgewidth=0.55, markersize=6.3, elinewidth=2.0, capsize=3.0,
                        zorder=4)
            label = f"lower in {item['benefit']}/{item['n_points']}"
            ax.text(0.985, y + 0.22, label, transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=7.6, color=color)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.52, 1.52)
        ax.set_yticks([1, 0], ["vs Parent", "vs Image Mean"])
        ax.set_xlabel("Paired ΔRMSE (m)  ← lower for Spatial")
        ax.set_title(f"{dataset}\n{subtitle}", loc="left", pad=7, weight="bold")
        ax.text(-0.12, 1.12, f"({chr(97 + panel)})", transform=ax.transAxes,
                fontsize=9.5, weight="bold", va="top")
        ax.grid(axis="x", color="#E6E6E6", lw=0.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

    legend = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="white",
               markeredgecolor=COLORS["muted"], markersize=5.0, label="scan / venue"),
        Line2D([0], [0], marker="D", color=COLORS["ink"], markersize=5.3,
               linewidth=1.8, label="equal-unit estimate (95% CI)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.015), handlelength=2.2)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.83, bottom=0.22)
    save_all(fig, "图2_跨队列配对效应")


def draw_region_figure(plot_data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.45), sharex=True,
                             gridspec_kw={"wspace": 0.24})
    display = [
        ("all", "Net", COLORS["net"], "D"),
        ("far", "Far", COLORS["far"], "o"),
        ("middle", "Middle", COLORS["middle"], "s"),
        ("near_unobserved", "Near", COLORS["near"], "^"),
        ("input", "Input", COLORS["input"], "x"),
    ]
    rng = np.random.default_rng(222)
    for panel, (dataset, ax, subtitle) in enumerate([
        ("DIODE", axes[0], "3 scenes"),
        ("ARKitScenes", axes[1], "20 venues"),
    ]):
        ax.axvline(0, color="#333333", lw=0.9, zorder=0)
        for y, (reg, label, color, marker) in zip(range(4, -1, -1), display):
            item = plot_data[dataset][reg]
            points = np.asarray(item["points"])
            jitter = rng.uniform(-0.09, 0.09, size=points.size)
            if marker == "x":
                ax.scatter(points, y + jitter, s=20, color=color,
                           linewidth=0.8, marker=marker, alpha=0.55, zorder=2)
            else:
                ax.scatter(points, y + jitter, s=15, facecolor="white", edgecolor=color,
                           linewidth=0.7, marker=marker, alpha=0.55, zorder=2)
            lo, hi = item["ci"]
            ax.errorbar(item["estimate"], y,
                        xerr=[[item["estimate"] - lo], [hi - item["estimate"]]],
                        fmt=marker, color=color, markerfacecolor=color if marker != "x" else "none",
                        markeredgecolor=color, markersize=6.0, elinewidth=2.0, capsize=3.0,
                        zorder=4)
        ax.set_xlim(-27, 8)
        ax.set_ylim(-0.55, 4.55)
        ax.set_yticks(range(4, -1, -1), [x[1] for x in display])
        ax.set_xlabel("Contribution to whole-image MSE difference (%)\n← lower for Spatial")
        ax.set_title(f"{dataset}\n{subtitle}", loc="left", pad=7, weight="bold")
        ax.text(-0.12, 1.12, f"({chr(97 + panel)})", transform=ax.transAxes,
                fontsize=9.5, weight="bold", va="top")
        ax.grid(axis="x", color="#E6E6E6", lw=0.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[1].tick_params(axis="y", labelleft=True)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.82, bottom=0.23)
    save_all(fig, "图3_区域误差贡献")


def write_statistics(performance: list[dict], regions: list[dict]) -> None:
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    combined = performance + regions
    keys = sorted({key for row in combined for key in row})
    with (STATS_DIR / "core_figure_statistics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(combined)
    metadata = {
        "bootstrap": {
            "type": "paired nonparametric percentile bootstrap",
            "repetitions": REPS,
            "seed": SEED,
            "DIODE_unit": "scene (n=3)",
            "ARKitScenes_unit": "venue (n=20)",
            "training_seeds": "averaged within each physical unit before resampling",
            "interpretation": "cohort-composition uncertainty conditional on frozen evaluated models",
        },
        "source_files": [
            "or1/tables/paired_units.csv",
            "or2/summary/scene_regions.csv",
            "or3/summary/venue_metrics.csv",
            "or3/summary/venue_regions.csv",
        ],
    }
    (STATS_DIR / "core_figure_ci_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    performance, perf_plot = performance_stats()
    regions, region_plot = regional_stats()
    draw_method_figure()
    draw_performance_figure(perf_plot)
    draw_region_figure(region_plot)
    write_statistics(performance, regions)
    print(OUT / "图1_空间门控适配框架.png")
    print(OUT / "图2_跨队列配对效应.png")
    print(OUT / "图3_区域误差贡献.png")


if __name__ == "__main__":
    main()
