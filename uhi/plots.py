"""Plots for the multi-scene analysis. Small PNGs, meant to be committed."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CLASS_ORDER = ["small", "medium", "large"]
CLASS_COLOR = {"small": "#2a78d6", "medium": "#eb6834", "large": "#1baf7a"}
CLASS_LABEL = {"small": "small (1–3 ha)", "medium": "medium (3–30 ha)", "large": "large (≥ 30 ha)"}
SEALED_COLOR = "#52514e"
Y_LABEL = "Surface temperature vs sealed sites (°C)"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "font.size": 9,
        "legend.frameon": False,
    }
)


def _zero_line(ax) -> None:
    ax.axhline(0, color=MUTED, linewidth=1)


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=110)
    plt.close(fig)


def cooling_by_size(parks: pd.DataFrame, n_scenes: int, path: Path) -> None:
    """Left: site medians per size class. Right: the same against park area.

    Parks away from water are the headline (filled dots, class medians);
    waterside parks are shown as open diamonds.
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 4.2), sharey=True, constrained_layout=True, width_ratios=[1, 1.4]
    )
    rng = np.random.default_rng(0)
    near = parks["near_water"].astype(bool)
    dry, wet = parks[~near], parks[near]
    for x, cls in enumerate(CLASS_ORDER):
        d = dry[dry["size_class"] == cls]
        w = wet[wet["size_class"] == cls]
        y = d["delta_median_c"].to_numpy()
        ax1.scatter(
            x + rng.uniform(-0.18, 0.18, len(y)), y, s=22, color=CLASS_COLOR[cls],
            alpha=0.75, edgecolor=SURFACE, linewidth=0.6,
        )
        ax1.scatter(
            x + rng.uniform(-0.18, 0.18, len(w)), w["delta_median_c"], s=24, marker="D",
            facecolor="none", edgecolor=CLASS_COLOR[cls], linewidth=1.1,
        )
        if len(y):
            q25, med, q75 = np.percentile(y, [25, 50, 75])
            ax1.plot([x - 0.3, x + 0.3], [med, med], color=INK, linewidth=2)
            ax1.plot([x + 0.36, x + 0.36], [q25, q75], color=INK, linewidth=1)
            ax1.annotate(f"{med:+.1f}", (x + 0.4, med), va="center", fontsize=9, color=INK)
        ax2.scatter(
            d["area_ha"], y, s=26, color=CLASS_COLOR[cls], alpha=0.8,
            edgecolor=SURFACE, linewidth=0.6, label=CLASS_LABEL[cls],
        )
        ax2.scatter(
            w["area_ha"], w["delta_median_c"], s=28, marker="D", facecolor="none",
            edgecolor=CLASS_COLOR[cls], linewidth=1.1,
        )
    ax2.scatter([], [], s=28, marker="D", facecolor="none", edgecolor=MUTED, linewidth=1.1,
                label="within 100 m of water")
    ax1.set_xticks(
        range(3),
        [f"{c}\nn = {(dry['size_class'] == c).sum()} + {(wet['size_class'] == c).sum()}"
         for c in CLASS_ORDER],
    )
    ax1.set_xlim(-0.5, 2.75)
    ax1.set_ylabel(Y_LABEL)
    ax1.set_title("By size class")
    ax1.grid(axis="x", visible=False)
    ax2.set_xscale("log")
    ax2.set_xlabel("Park area (ha, log scale)")
    ax2.set_title("Against park area")
    for name in ("Prater", "Augarten", "Stadtpark", "Türkenschanzpark"):
        row = parks[parks["site"] == name]
        if len(row):
            ax2.annotate(
                name, (row["area_ha"].iloc[0], row["delta_median_c"].iloc[0]),
                xytext=(5, 5), textcoords="offset points", fontsize=8, color=MUTED,
            )
    ax2.legend(loc="upper right", ncols=4, columnspacing=0.8, handletextpad=0.2, fontsize=8)
    ax2.set_ylim(top=1.6)
    for ax in (ax1, ax2):
        _zero_line(ax)
    fig.suptitle(
        f"Each mark is one park: its median over up to {n_scenes} summer Landsat scenes, "
        "2013–2026. Bar and line = median and quartiles of the parks away from water "
        "(n = away + near water).",
        x=0.01, ha="left", fontsize=8, color=MUTED,
    )
    _save(fig, path)


def cooling_vs_ndvi(sites: pd.DataFrame, fit: tuple[float, float], r: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.4), constrained_layout=True)
    parks = sites[sites["kind"] == "park"]
    dry = parks[~parks["near_water"]]
    for cls in CLASS_ORDER:
        d = dry[dry["size_class"] == cls]
        ax.scatter(
            d["ndvi_s2"], d["delta_median_c"], s=26, color=CLASS_COLOR[cls], alpha=0.85,
            edgecolor=SURFACE, linewidth=0.6, label=CLASS_LABEL[cls],
        )
    wet = parks[parks["near_water"]]
    ax.scatter(
        wet["ndvi_s2"], wet["delta_median_c"], s=30, marker="D", facecolor="none",
        edgecolor=[CLASS_COLOR[c] for c in wet["size_class"]], linewidth=1.3,
    )
    ax.scatter([], [], s=30, marker="D", facecolor="none", edgecolor=MUTED, linewidth=1.3,
               label="park within 100 m of water")
    sealed = sites[sites["kind"] == "sealed"]
    ax.scatter(
        sealed["ndvi_s2"], sealed["delta_median_c"], s=40, marker="s", color=SEALED_COLOR,
        edgecolor=SURFACE, linewidth=0.6, label="sealed sites",
    )
    for row in sealed.itertuples():
        ax.annotate(
            row.site, (row.ndvi_s2, row.delta_median_c), xytext=(6, -3),
            textcoords="offset points", fontsize=8, color=MUTED,
        )
    xs = np.array([dry["ndvi_s2"].min(), dry["ndvi_s2"].max()])
    ax.plot(xs, fit[0] * xs + fit[1], color=INK, linewidth=1.5, linestyle="--")
    ax.annotate(
        f"parks away from water:\n{fit[0] / 10:+.1f} °C per 0.1 NDVI, r = {r:.2f}",
        (xs[0], fit[0] * xs[0] + fit[1]), xytext=(0, 8), textcoords="offset points",
        ha="left", fontsize=9, color=INK,
    )
    _zero_line(ax)
    ax.set_xlabel("Mean NDVI of the site (Sentinel-2, 10 m, summer median)")
    ax.set_ylabel(Y_LABEL)
    ax.set_title("Park cooling against greenness")
    ax.legend(loc="lower left")
    _save(fig, path)


def cooling_per_year(years: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    for cls in CLASS_ORDER:
        d = years[years["size_class"] == cls].sort_values("year")
        ax.plot(
            d["year"], d["delta_median_c"], color=CLASS_COLOR[cls], linewidth=2, marker="o",
            markersize=5, markeredgecolor=SURFACE, label=CLASS_LABEL[cls],
        )
    counts = years.drop_duplicates("year").sort_values("year")
    low = years["delta_median_c"].min()
    for row in counts.itertuples():
        ax.annotate(
            str(row.scenes), (row.year, low - 0.9), ha="center", fontsize=8, color=MUTED
        )
    ax.annotate(
        "scenes:", (counts["year"].min() - 0.6, low - 0.9), ha="right", fontsize=8, color=MUTED
    )
    ax.set_ylim(low - 1.4, 1.4)
    ax.set_xlim(counts["year"].min() - 1.6, counts["year"].max() + 0.5)
    ax.set_xticks(counts["year"])
    _zero_line(ax)
    ax.set_ylabel(Y_LABEL)
    ax.set_title("Median park cooling per summer, parks away from water")
    ax.legend(loc="upper right", ncols=3)
    _save(fig, path)
