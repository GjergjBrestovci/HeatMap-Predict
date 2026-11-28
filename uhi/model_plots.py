"""Plots for the model and the scenario tool. Small PNGs, meant to be committed."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import INK, MUTED, SURFACE, _save, _zero_line

COOL_WARM = "RdBu_r"
ACCENT, SECOND = "#eb6834", "#2a78d6"


def _map(ax, values, limit, title, cmap=COOL_WARM):
    image = ax.imshow(values, cmap=cmap, vmin=-limit, vmax=limit, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    return image


def maps(target: np.ndarray, predicted: np.ndarray, usable: np.ndarray, path: Path) -> None:
    """Observed anomaly, held-out prediction and their difference, side by side."""
    obs = np.where(usable, target, np.nan)
    pred = np.where(usable, predicted, np.nan)
    limit = float(np.nanpercentile(np.abs(obs), 98)) if np.isfinite(obs).any() else 1.0
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), constrained_layout=True)
    image = _map(axes[0], obs, limit, "Observed: surface minus city mean")
    _map(axes[1], pred, limit, "Predicted for blocks not trained on")
    _map(axes[2], pred - obs, limit, "Predicted minus observed")
    fig.colorbar(image, ax=axes, shrink=0.8, label="°C")
    _save(fig, path)


def importance(grouped: pd.DataFrame, path: Path) -> None:
    d = grouped.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.36 * len(d) + 1.3), constrained_layout=True)
    ax.barh(d["quantity"], d["mae_increase_c"], color=SECOND)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Rise of the held-out error when the quantity is shuffled (°C)")
    ax.set_title("What the model relies on (cell, 150 m and 500 m together)")
    _save(fig, path)


def dependence(table: pd.DataFrame, path: Path) -> None:
    names = list(dict.fromkeys(table["feature"]))
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.4), constrained_layout=True, sharey=True)
    for ax, name in zip(axes.ravel(), names):
        d = table[table["feature"] == name]
        ax.plot(d["value"], d["prediction_c"], color=ACCENT, linewidth=1.6)
        _zero_line(ax)
        ax.set_title(name, fontsize=9)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    fig.supylabel("Predicted anomaly (°C)", fontsize=9, color=MUTED)
    fig.suptitle("Prediction against one feature, the others as they are", fontsize=11,
                 fontweight="bold", x=0.01, ha="left")
    _save(fig, path)


def _scatter(ax, x, y, title, xlabel, ylabel):
    ax.scatter(x, y, s=18, color=SECOND, alpha=0.7, edgecolor=SURFACE, linewidth=0.4)
    if len(x):
        lo, hi = float(min(x.min(), y.min())) - 0.5, float(max(x.max(), y.max())) + 0.5
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def checks(parks: pd.DataFrame, conversions: pd.DataFrame, path: Path) -> None:
    """Left: parks, observed against held-out prediction. Right: real conversions."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 4.4), constrained_layout=True)
    _scatter(left, parks.get("observed_c", pd.Series(dtype=float)),
             parks.get("predicted_c", pd.Series(dtype=float)),
             "Parks the model did not train on", "Observed anomaly (°C)", "Predicted (°C)")
    _scatter(right, conversions.get("observed_change_c", pd.Series(dtype=float)),
             conversions.get("predicted_change_c", pd.Series(dtype=float)),
             "Cells built over since 2014", "Observed change (°C)", "Predicted change (°C)")
    _save(fig, path)


def scenario_map(cells: pd.DataFrame, edits, name: str, path: Path) -> None:
    """Predicted change per cell around the edited polygons."""
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    if len(cells):
        limit = max(float(cells["change_c"].abs().max()), 0.1)
        points = ax.scatter(cells["x"], cells["y"], c=cells["change_c"], cmap=COOL_WARM, vmin=-limit,
                            vmax=limit, marker="s", s=60)
        fig.colorbar(points, ax=ax, shrink=0.8, label="Predicted change (°C)")
    edits.boundary.plot(ax=ax, color=INK, linewidth=1.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(name.replace("_", " "))
    _save(fig, path)
