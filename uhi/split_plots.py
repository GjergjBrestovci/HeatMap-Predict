"""Plots for the warming split. Small PNGs, meant to be committed."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import INK, MUTED, SURFACE, _save, _zero_line

URBAN, RURAL, GRID_SERIES = "#eb6834", "#1baf7a", "#2a78d6"
SENSOR_COLOR = {"TM": "#2a78d6", "ETM+": "#8a5cd0", "OLI/TIRS": "#eb6834"}
SENSOR_LABEL = {"TM": "Landsat 4/5", "ETM+": "Landsat 7", "OLI/TIRS": "Landsat 8/9"}


def _trend_line(ax, years, values, color) -> None:
    keep = np.isfinite(values)
    if keep.sum() >= 3:
        slope, intercept = np.polyfit(np.asarray(years)[keep], np.asarray(values)[keep], 1)
        x = np.array([np.min(years), np.max(years)])
        ax.plot(x, slope * x + intercept, color=color, linewidth=1, linestyle="--")


def air_trends(wide: pd.DataFrame, stats: dict, path: Path) -> None:
    """Top: summer air temperature, urban and rural stations. Bottom: urban minus rural."""
    urban = wide[stats["urban_stations"]].mean(axis=1, skipna=False)
    rural = wide[stats["rural_stations"]].mean(axis=1, skipna=False)
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.5, 6.4), sharex=True, constrained_layout=True, height_ratios=[3, 2]
    )
    for series, color, label in (
        (urban, URBAN, "urban stations: " + ", ".join(stats["urban_stations"])),
        (rural, RURAL, f"{len(stats['rural_stations'])} rural stations around the city"),
    ):
        top.plot(wide.index, series, color=color, linewidth=1.6, marker="o", markersize=3, label=label)
        _trend_line(top, wide.index.to_numpy(), series.to_numpy(), color)
    top.set_ylabel("June–August mean air temperature (°C)")
    top.set_title("Summer air temperature at urban and rural stations")
    top.legend(loc="upper left")

    diff = urban - rural
    bottom.plot(wide.index, diff, color=INK, linewidth=1.4, marker="o", markersize=3,
                label="urban minus rural stations")
    _trend_line(bottom, wide.index.to_numpy(), diff.to_numpy(), INK)
    for name, color in zip(stats["urban_stations"], (URBAN, "#b4532b")):
        bottom.plot(wide.index, wide[name] - rural, color=color, linewidth=0.9, alpha=0.8,
                    label=f"{name} minus rural")
    bottom.set_ylabel("Difference (°C)")
    bottom.set_title("Urban minus rural")
    bottom.legend(loc="upper left", ncols=2, fontsize=8)
    _save(fig, path)


def land_use(table: pd.DataFrame, path: Path) -> None:
    """Sealed and green land per land-use edition, with GHSL built-up surface for comparison."""
    fig, ax = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    d = table.dropna(subset=["built_ha"])
    ax.plot(d["year"], d["built_ha"], color=URBAN, marker="o", markersize=4,
            label="sealed classes (land-use map)")
    ax.plot(d["year"], d["green_ha"], color=RURAL, marker="o", markersize=4,
            label="green classes (land-use map)")
    ax.plot(d["year"], d["unmapped_ha"], color=MUTED, marker="o", markersize=3, linewidth=1,
            label="not mapped (streets before 2007)")
    g = table.dropna(subset=["ghsl_built_ha"])
    ax.plot(g["year"], g["ghsl_built_ha"], color=INK, marker="s", markersize=4, linestyle=":",
            label="built-up surface (GHSL, roofs only)")
    ax.set_ylabel("Area inside the city (ha)")
    ax.set_title("Land use in Vienna by map edition")
    ax.legend(loc="center left", fontsize=8)
    _save(fig, path)


def surface_gap(gaps: pd.DataFrame, by_year: pd.DataFrame, path: Path) -> None:
    """Stable sealed minus stable green surface temperature: every scene and the yearly median."""
    col = "built_minus_green_c"
    fig, ax = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    for sensor, color in SENSOR_COLOR.items():
        d = gaps[gaps["sensor"] == sensor]
        if len(d):
            ax.scatter(d["year"], d[col], s=14, color=color, alpha=0.55, edgecolor=SURFACE,
                       linewidth=0.4, label=SENSOR_LABEL[sensor])
    ax.plot(by_year["year"], by_year[col], color=INK, linewidth=1.2, label="median per summer")
    ax.set_ylabel("Sealed minus green, same scene (°C)")
    ax.set_title("Surface: land that stayed sealed against land that stayed green")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=4, fontsize=8)
    _save(fig, path)


def converted_land(event: pd.DataFrame, cohorts: pd.DataFrame, reference: str, bin_years: int,
                   path: Path) -> None:
    """Left: converted land against time since conversion. Right: effect per cohort."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 4.4), constrained_layout=True)
    if len(event):
        colors = [RURAL if b < 0 else URBAN for b in event["bin"]]
        left.bar(event["bin"], event["shift_c"], color=colors, width=0.8)
        labels = [
            f"{-t - 1}–{-f - 1}" if f < 0 else f"{f}–{t}"
            for f, t in zip(event["from_year"], event["to_year"])
        ]
        left.set_xticks(event["bin"], labels, rotation=45, ha="right")
    _zero_line(left)
    left.set_xlabel("Years before the last green map (green)  |  after the first sealed map (orange)")
    left.set_ylabel("Converted minus stable green, shift from before (°C)")
    left.set_title("Converted land by time since conversion")

    d = cohorts[(cohorts["reference"] == reference) & cohorts["effect_c"].notna()]
    d = d.sort_values("first_built_year")
    for flag, color, label in ((False, URBAN, "cohort"), (True, MUTED, "across the scheme change")):
        m = d[d["scheme_change"] == flag]
        if len(m):
            right.scatter(m["first_built_year"], m["effect_c"], s=np.sqrt(m["pixels"]) * 3,
                          color=color, alpha=0.75, edgecolor=SURFACE, linewidth=0.6, label=label)
    _zero_line(right)
    right.set_xlabel("First map edition that shows the area sealed")
    right.set_ylabel("After minus before (°C)")
    right.set_title("Per conversion period (dot size: area)")
    if len(d):
        right.legend(loc="best", fontsize=8)
    _save(fig, path)
