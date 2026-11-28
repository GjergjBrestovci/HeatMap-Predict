"""Scenario tool: what would the surface temperature do if a polygon got a new cover?

Usage (from the repository root, after the model step has run):
    python -m uhi.scenario scenarios/my_idea.geojson [--out results/scenarios]

The GeoJSON holds one or more polygons (longitude/latitude). Each needs a
`cover` property: lawn, trees, car_park, paved, building, housing or water;
`building` and `housing` also take `height_m`. `housing` means "built over as
Vienna has built on green land since 2014": the mix of surfaces, the greenness
and the building height are taken from those places.
All polygons of one file are applied together.

The answer is the predicted change of the late-morning summer surface
temperature per 90 m cell, with a band. It is not air temperature.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from affine import Affine

from . import data, layout, model_plots
from . import features as ft
from . import model_stats as ms

DEFAULT_HEIGHT_M = 18.0
MIN_CHANGE = 1e-4  # feature change below this counts as untouched


def load_bundle(cache: Path) -> tuple[dict, dict]:
    """The fitted models and the layout layers written by `python -m uhi.model`."""
    folder = cache / "model"
    if not (folder / "bundle.joblib").exists():
        raise SystemExit(f"No model in {folder}. Run the model step first (or download its artifact there).")
    bundle = joblib.load(folder / "bundle.joblib")
    with np.load(folder / "layers.npz") as z:
        layers = {k: z[k] for k in z.files}
    return bundle, layers


def bundle_grid(bundle: dict) -> data.Grid:
    return data.Grid(Affine(*bundle["grid30"]["transform"]), bundle["grid30"]["width"], bundle["grid30"]["height"])


def predict_change(bundle: dict, layers: dict, edited: dict, grid30: data.Grid) -> pd.DataFrame:
    """Per 90 m cell whose features changed: prediction before, change, band, warnings."""
    names = bundle["features"]
    before = ft.matrix(ft.build_features(layers), names)
    after = ft.matrix(ft.build_features(edited), names)
    touched = np.flatnonzero(np.nanmax(np.abs(after - before), axis=1) > MIN_CHANGE)
    cells = layout.cell_grid(grid30)
    rows, cols = np.unravel_index(touched, cells.shape)
    t = cells.transform
    pred_before = ms.ensemble_predict(bundle["models"], before[touched])
    pred_after = ms.ensemble_predict(bundle["models"], after[touched])
    change, low, high = ms.change_band(pred_before, pred_after, bundle["held_out_mae"])
    return pd.DataFrame(
        {
            "row": rows, "col": cols,
            "x": t.c + (cols + 0.5) * t.a, "y": t.f + (rows + 0.5) * t.e,
            "before_c": pred_before.mean(axis=0),
            "change_c": change, "low_c": low, "high_c": high,
            "features_outside_training": ms.outside_range(after[touched], bundle["limits"]),
        }
    )


def run(bundle: dict, layers: dict, edits: gpd.GeoDataFrame, grid30: data.Grid) -> tuple[pd.DataFrame, dict]:
    """Apply all polygons of `edits` (columns cover, height_m; in `data.CRS`) and predict."""
    grid10 = layout.fine_grid(grid30)
    cells_grid = layout.cell_grid(grid30)
    edited = layers
    for row in edits.itertuples():
        height = getattr(row, "height_m", None)
        height = None if height is None or not np.isfinite(height) else float(height)
        recipe = bundle.get("housing") or None
        if height is None and not (row.cover == "housing" and recipe):
            height = DEFAULT_HEIGHT_M
        edited = ft.apply_edit(edited, layout.polygon_mask(row.geometry, grid10), row.cover,
                               bundle["ndvi_of"], height, recipe)
    cells = predict_change(bundle, layers, edited, grid30)
    shape = edits.geometry.union_all()
    inside = layout.polygon_mask(shape, cells_grid)
    cells["inside"] = inside[cells["row"], cells["col"]] if len(cells) else []
    summary = {"area_ha": float(shape.area / 1e4), "cells_changed": int(len(cells))}
    for label, part in (("inside", cells[cells["inside"]]), ("around", cells[~cells["inside"]])):
        if len(part):
            summary[label] = {
                "cells": int(len(part)),
                "mean_change_c": float(part["change_c"].mean()),
                "mean_low_c": float(part["low_c"].mean()),
                "mean_high_c": float(part["high_c"].mean()),
                "strongest_change_c": float(part.loc[part["change_c"].abs().idxmax(), "change_c"]),
            }
    summary["cells_outside_training_range"] = int((cells["features_outside_training"] > 0).sum())
    if not summary.get("inside"):
        summary["note"] = "The polygon holds no 90 m cell centre; its effect is spread over neighbours."
    return cells, summary


def read_edits(path: Path) -> gpd.GeoDataFrame:
    edits = gpd.read_file(path)
    if "cover" not in edits.columns:
        raise SystemExit(f"{path}: every polygon needs a 'cover' property")
    if edits.crs is None:
        edits = edits.set_crs("EPSG:4326")
    if "height_m" not in edits.columns:
        edits["height_m"] = np.nan
    return edits.to_crs(data.CRS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path, help="GeoJSON files, one scenario each")
    parser.add_argument("--cache", default=Path("data/cache"), type=Path)
    parser.add_argument("--out", default=Path("results/scenarios"), type=Path)
    args = parser.parse_args()
    bundle, layers = load_bundle(args.cache)
    grid30 = bundle_grid(bundle)
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.files:
        edits = read_edits(path)
        cells, summary = run(bundle, layers, edits, grid30)
        name = path.stem
        cells.round(3).to_csv(args.out / f"{name}_cells.csv", index=False)
        (args.out / f"{name}.json").write_text(json.dumps(summary, indent=2) + "\n")
        model_plots.scenario_map(cells, edits, name, args.out / f"{name}.png")
        print(f"{name}: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
