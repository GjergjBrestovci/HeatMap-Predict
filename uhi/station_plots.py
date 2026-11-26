"""Plots for the station comparison. Small PNGs, meant to be committed."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import INK, MUTED, SURFACE, _save

STATION_COLOR = {"urban": "#eb6834", "green": "#1baf7a"}
STATION_LABEL = {"urban": "urban station", "green": "green station"}


def _scatter_by_class(ax, df: pd.DataFrame, x: str, y: str) -> None:
    for cls in ("urban", "green"):
        d = df[df["station_class"] == cls]
        ax.scatter(d[x], d[y], s=20, color=STATION_COLOR[cls], alpha=0.7,
                   edgecolor=SURFACE, linewidth=0.5, label=STATION_LABEL[cls])


def surface_vs_air(table: pd.DataFrame, path: Path) -> None:
    """Every station and scene: surface temperature around the station against air temperature."""
    d = table.dropna(subset=["lst_ring_c", "air_c"])
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    _scatter_by_class(ax, d, "air_c", "lst_ring_c")
    lo = float(min(d["air_c"].min(), d["lst_ring_c"].min())) - 1
    hi = float(max(d["air_c"].max(), d["lst_ring_c"].max())) + 1
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1)
    ax.annotate("surface = air", (hi, hi), xytext=(-4, -12), textcoords="offset points",
                ha="right", fontsize=8, color=MUTED)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Air temperature at the station, 2 m (°C)")
    ax.set_ylabel("Landsat surface temperature within 300 m (°C)")
    ax.set_title("The surface is much hotter than the air")
    ax.legend(loc="upper left")
    _save(fig, path)


def gap_per_station(summary: pd.DataFrame, path: Path) -> None:
    """Surface minus air per station, ordered by how green the surroundings are."""
    d = summary.sort_values("green_share").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(d) + 1.6), constrained_layout=True)
    y = np.arange(len(d))
    for cls in ("urban", "green"):
        m = (d["station_class"] == cls).to_numpy()
        ax.hlines(y[m], d.loc[m, "gap_q25_c"], d.loc[m, "gap_q75_c"], color=STATION_COLOR[cls],
                  linewidth=2)
        ax.scatter(d.loc[m, "gap_median_c"], y[m], s=46, color=STATION_COLOR[cls],
                   edgecolor=SURFACE, linewidth=0.8, zorder=3, label=STATION_LABEL[cls])
    for i, row in d.iterrows():
        ax.annotate(f"{row['gap_median_c']:+.1f}", (row["gap_q75_c"], i), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [f"{r.station} ({r.green_share:.0%} green)" for r in d.itertuples()])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Surface temperature within 300 m minus air temperature (°C)")
    ax.set_title("Surface–air gap per station (dot = median, line = quartiles)")
    ax.legend(loc="lower right")
    _save(fig, path)


def between_places(anomalies: pd.DataFrame, relation: dict, path: Path) -> None:
    """Differences between places: air anomaly against surface anomaly, same scene."""
    fig, ax = plt.subplots(figsize=(6.6, 4.6), constrained_layout=True)
    _scatter_by_class(ax, anomalies, "lst_anom_c", "air_anom_c")
    slope = relation.get("air_per_lst", np.nan)
    if len(anomalies) and np.isfinite(slope):
        xs = np.array([anomalies["lst_anom_c"].min(), anomalies["lst_anom_c"].max()])
        ax.plot(xs, slope * xs, color=INK, linewidth=1.5, linestyle="--")
        ax.annotate(
            f"{slope:+.2f} °C of air per °C of surface, r = {relation['r']:.2f}",
            (xs[1], slope * xs[1]), xytext=(-4, 10), textcoords="offset points",
            ha="right", fontsize=9, color=INK,
        )
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xlabel("Surface temperature: station minus mean of these stations, same scene (°C)")
    ax.set_ylabel("Air temperature: same difference (°C)")
    ax.set_title("Differences between stations below 300 m: air against surface")
    ax.legend(loc="upper left")
    _save(fig, path)
