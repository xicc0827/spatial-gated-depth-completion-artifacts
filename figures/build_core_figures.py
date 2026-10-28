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
STATS_DIR = OUT
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
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
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


def add_box(ax, xy, width, height, text, *, fc, ec, fontsize=9.5, weight="normal"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=fc, edgecolor=ec, linewidth=1.0,
        clip_on=False,
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
    fig = plt.figure(figsize=(7.05, 4.50))
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.005, 0.975, "Shared computation and correction interface", fontsize=10.5,
            va="top", weight="bold")
    boxes = [
        (0.005, 0.175, "RGB, sparse depth\nand support mask\n500 points"),
        (0.225, 0.170, "Frozen parent\nfeature extractor"),
        (0.440, 0.125, "Initial depth\n$D_i^{init}$"),
        (0.610, 0.145, "Add correction\n(see modes below)"),
        (0.800, 0.190, "Frozen propagation\nand output head"),
    ]
    for x, w, label in boxes:
        add_box(ax, (x, 0.760), w, 0.145, label, fc="#E9EEF3", ec="#596A7A")
    for left, right in zip(boxes[:-1], boxes[1:]):
        arrow(ax, (left[0] + left[1] + 0.01, 0.832), (right[0] - 0.01, 0.832))
    add_box(ax, (0.285, 0.565), 0.280, 0.130,
            "7-channel corrector + mean filter\n24,513 trainable parameters\n" + r"$L(\delta_i^{S})$ or $L(\delta_i^{M})$",
            fc="#E8F4FB", ec=COLORS["spatial"])
    add_box(ax, (0.625, 0.565), 0.285, 0.130,
            "Support-distance gate\n" + r"$g_i(p)$ or image mean $\bar g_i$",
            fc="#FDF1E3", ec=COLORS["image_mean"])
    arrow(ax, (0.500, 0.760), (0.500, 0.695), color="#596A7A")
    arrow(ax, (0.565, 0.655), (0.645, 0.760), color=COLORS["spatial"])
    arrow(ax, (0.735, 0.695), (0.715, 0.760), color=COLORS["image_mean"])
    ax.text(0.005, 0.645, "RGB, sparse depth,\nmask and initial depth\nform corrector input", va="center", fontsize=9.5)

    ax.text(0.005, 0.495, "Execution mode", weight="bold", fontsize=10.0)
    ax.text(0.390, 0.495, "Correction added to initial depth", weight="bold", fontsize=10.0)
    ax.text(0.760, 0.495, "Corrector weights", weight="bold", fontsize=10.0)
    rows = [
        (0.435, "Parent", r"$0$", "None", COLORS["muted"]),
        (0.365, "Spatial", r"$\sigma_i g_i\odot L(\delta_i^{S})$", r"$\theta_S$", COLORS["spatial"]),
        (0.295, "Image Mean (separately trained)", r"$\sigma_i\bar g_i L(\delta_i^{M})$", r"$\theta_M$", COLORS["image_mean"]),
        (0.225, "Fixed-checkpoint mean gate", r"$\sigma_i\bar g_i L(\delta_i^{S})$", r"Same $\theta_S$ as Spatial", COLORS["green"]),
    ]
    ax.plot([0.005, 0.99], [0.475, 0.475], color="#777777", lw=0.8)
    for y, label, correction, weights, color in rows:
        ax.text(0.005, y, label, va="center", color=color, fontsize=9.5)
        ax.text(0.465, y, correction, va="center", fontsize=10.0)
        ax.text(0.760, y, weights, va="center", fontsize=9.5)
    ax.plot([0.005, 0.99], [0.185, 0.185], color="#777777", lw=0.8)
    ax.text(0.005, 0.130, "Spatial vs Parent: whole-adapter effect", fontsize=9.5)
    ax.text(0.005, 0.080, "Spatial vs Image Mean: complete training-and-gating design", fontsize=9.5)
    ax.text(0.005, 0.030, "Spatial vs fixed-checkpoint mean gate: immediate gate dependence; weights and residual held fixed", fontsize=9.5)
    save_all(fig, "图1_空间门控适配框架")


def draw_performance_figure(plot_data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.65), gridspec_kw={"wspace": 0.33})
    panel_specs = [
        ("DIODE", axes[0], (-0.76, 0.10), "3 scenes; 10 scans"),
        ("ARKitScenes", axes[1], (-0.115, 0.065), "20 sites"),
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
                    fontsize=9.5, color=color)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.52, 1.52)
        ax.set_yticks([1, 0], ["vs Parent", "vs Image Mean"])
        ax.set_xlabel("Paired ΔRMSE (m)\n← lower for Spatial")
        ax.set_title(f"{dataset}\n{subtitle}", loc="left", pad=7, weight="bold")
        ax.text(-0.12, 1.12, f"({chr(97 + panel)})", transform=ax.transAxes,
                fontsize=9.5, weight="bold", va="top")
        ax.grid(axis="x", color="#E6E6E6", lw=0.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

    legend = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="white",
               markeredgecolor=COLORS["muted"], markersize=5.0, label="scan / site"),
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
        ("ARKitScenes", axes[1], "20 sites"),
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
