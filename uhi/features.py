"""Layout features for the model: pure array logic, no network and no geo libraries.

Three aligned grids are used. Land cover lives at 10 m, Landsat delivers at 30 m
(3 x 3 land-cover pixels), and the model works on 90 m cells (3 x 3 Landsat
pixels), close to the 100 m at which the thermal sensor really measures.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

# Land cover at 10 m, from the Flächenmehrzweckkarte.
NONE, BUILDING, ROAD, PAVED, RAIL, GRASS, FOREST, FIELD, WATER, OTHER = range(10)
COVER_NAMES = ("none", "building", "road", "paved", "rail", "grass", "forest", "field", "water", "other")
# Fractions used as features; "none" and "other" together are `unmapped`.
FRACTION_NAMES = ("building", "road", "paved", "rail", "grass", "forest", "field", "water", "unmapped")
RADII_M = (150, 500)
FINE = 3  # 10 m pixels per 30 m pixel, and 30 m pixels per 90 m cell
WATER_DIST_CAP_M = 2000.0
MIN_CELL_VALID = 6  # of the 9 Landsat pixels of a cell


def block_mean(a: np.ndarray, n: int = FINE, min_valid: int = 1) -> np.ndarray:
    """Mean over n x n blocks, ignoring NaN; NaN where fewer than `min_valid` values.

    Rows and columns that do not fill a block are cut off.
    """
    h, w = (a.shape[0] // n) * n, (a.shape[1] // n) * n
    blocks = a[:h, :w].reshape(h // n, n, w // n, n).astype("float32")
    valid = np.isfinite(blocks)
    count = valid.sum(axis=(1, 3))
    total = np.where(valid, blocks, 0).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count >= min_valid, total / count, np.nan).astype("float32")


def disc(radius_px: float) -> np.ndarray:
    r = int(np.floor(radius_px))
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y <= radius_px * radius_px).astype("float32")


def disc_mean(a: np.ndarray, radius_px: float) -> np.ndarray:
    """Mean of the finite values within `radius_px` of every pixel (the edge has fewer)."""
    kernel = disc(radius_px)
    valid = np.isfinite(a)
    total = signal.fftconvolve(np.where(valid, a, 0).astype("float64"), kernel, mode="same")
    count = signal.fftconvolve(valid.astype("float64"), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0.5, total / count, np.nan).astype("float32")


def cell_centres(a30: np.ndarray) -> np.ndarray:
    """The value at the centre Landsat pixel of every 90 m cell."""
    h, w = (a30.shape[0] // FINE) * FINE, (a30.shape[1] // FINE) * FINE
    return a30[1:h:FINE, 1:w:FINE]


def cover_fractions(cover10: np.ndarray) -> dict[str, np.ndarray]:
    """Share of each cover per 30 m pixel."""
    out = {}
    for name in FRACTION_NAMES:
        if name == "unmapped":
            mask = (cover10 == NONE) | (cover10 == OTHER)
        else:
            mask = cover10 == COVER_NAMES.index(name)
        out[name] = block_mean(mask.astype("float32"))
    return out


def multi_scale(name: str, a30: np.ndarray, pixel_m: float = 30.0, radii=RADII_M) -> dict[str, np.ndarray]:
    """A 30 m map as cell features: the cell itself and the mean within each radius of its centre."""
    out = {name: block_mean(a30)}
    for radius in radii:
        out[f"{name}_{radius}m"] = cell_centres(disc_mean(a30, radius / pixel_m))
    return out


def matrix(features: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    """Feature maps as a table: one row per cell (row-major), one column per name."""
    return np.column_stack([features[n].ravel() for n in names]).astype("float32")


def build_features(layers: dict[str, np.ndarray], pixel_m: float = 30.0) -> dict[str, np.ndarray]:
    """Feature maps on the 90 m cell grid.

    `layers` holds, at 10 m: `cover` (codes above), `height` (building height in
    metres, 0 elsewhere), `canopy` (share of the pixel under mapped tree crowns)
    and `ndvi`; at 30 m: `elevation`.

    Every quantity is given for the cell itself and as the mean within 150 m and
    500 m of its centre. There are no coordinates among the features.
    """
    base = cover_fractions(layers["cover"])
    base["height"] = block_mean(layers["height"].astype("float32"))  # mean over all ground
    base["canopy"] = block_mean(layers["canopy"].astype("float32"))
    base["ndvi"] = block_mean(layers["ndvi"].astype("float32"))
    out = {}
    for name, a30 in base.items():
        out.update(multi_scale(name, a30, pixel_m, (150,) if name == "canopy" else RADII_M))
    out["elevation"] = block_mean(layers["elevation"].astype("float32"))
    water = base["water"] >= 0.5
    if water.any():
        dist = ndimage.distance_transform_edt(~water) * pixel_m
    else:
        dist = np.full(water.shape, WATER_DIST_CAP_M)
    out["water_dist_m"] = cell_centres(np.minimum(dist, WATER_DIST_CAP_M).astype("float32"))
    return out


def composite(anomalies: list[np.ndarray], min_scenes: int) -> np.ndarray:
    """Median over scenes per pixel; NaN where fewer than `min_scenes` are valid."""
    stack = np.stack(anomalies)
    count = np.isfinite(stack).sum(axis=0)
    with np.errstate(all="ignore"):
        med = np.nanmedian(stack, axis=0)
    return np.where(count >= min_scenes, med, np.nan).astype("float32")


def scene_anomaly(lst: np.ndarray, clear: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Surface temperature minus the mean over the clear part of `reference` (the city)."""
    valid = clear & np.isfinite(lst)
    ref = valid & reference
    if not ref.any():
        return np.full(lst.shape, np.nan, dtype="float32")
    return np.where(valid, lst - lst[ref].mean(), np.nan).astype("float32")


def to_cells(a30: np.ndarray) -> np.ndarray:
    """A 30 m map as 90 m cells; a cell needs `MIN_CELL_VALID` of its 9 pixels."""
    return block_mean(a30, FINE, MIN_CELL_VALID)


# ------------------------------------------------------------------ scenarios

# What a user can turn a polygon into: (cover code, canopy share, key of the typical NDVI).
EDIT_COVERS = {
    "lawn": (GRASS, 0.0, "lawn"),
    "trees": (GRASS, 0.7, "trees"),
    "car_park": (ROAD, 0.0, "road"),
    "paved": (PAVED, 0.0, "paved"),
    "building": (BUILDING, 0.0, "building"),
    "water": (WATER, 0.0, "water"),
}
HOUSING_PERIOD_PX, HOUSING_BUILDING_PX, HOUSING_PAVED_PX = 5, 2, 1  # per 50 m: 20 m house, 10 m path


def typical_ndvi(layers: dict[str, np.ndarray]) -> dict[str, float]:
    """Median NDVI of each cover in the city today, used for edited pixels."""
    cover, canopy, ndvi = layers["cover"], layers["canopy"], layers["ndvi"]

    def med(mask):
        values = ndvi[mask & np.isfinite(ndvi)]
        return float(np.median(values)) if len(values) else float("nan")

    out = {name: med(cover == COVER_NAMES.index(name)) for name in ("building", "road", "paved", "water")}
    out["lawn"] = med((cover == GRASS) & (canopy == 0))
    out["trees"] = med((cover == GRASS) & (canopy >= 0.5))
    if not np.isfinite(out["trees"]):
        out["trees"] = med(cover == FOREST)
    return out


def development_recipe(layers: dict[str, np.ndarray], cells: np.ndarray) -> dict:
    """What recently built-over land looks like: cover shares, NDVI and building height.

    `cells` marks 90 m cells. The result is the recipe for the `housing` edit, so
    that "build here" means "as Vienna has actually built", not an invented layout.
    """
    fine = np.kron(cells, np.ones((FINE * FINE, FINE * FINE), dtype=bool))
    h, w = layers["cover"].shape
    mask = np.zeros((h, w), dtype=bool)
    mask[: fine.shape[0], : fine.shape[1]] = fine[:h, :w]
    if not mask.any():
        return {}
    cover = layers["cover"][mask]
    counts = np.bincount(cover, minlength=len(COVER_NAMES))
    built = layers["height"][mask][cover == BUILDING]
    return {
        "shares": {name: float(counts[i] / counts.sum()) for i, name in enumerate(COVER_NAMES) if counts[i]},
        "ndvi": float(np.nanmean(layers["ndvi"][mask])),
        "height_m": float(built.mean()) if len(built) else 0.0,
        "cells": int(cells.sum()),
    }


def apply_edit(layers: dict[str, np.ndarray], mask10: np.ndarray, cover: str,
               ndvi_of: dict[str, float], height_m: float | None = None,
               recipe: dict | None = None) -> dict[str, np.ndarray]:
    """A copy of `layers` with the 10 m pixels in `mask10` turned into `cover`.

    `housing` with a `recipe` (see `development_recipe`) scatters the recipe's
    covers over the polygon in its proportions and gives every pixel its NDVI.
    Without a recipe it lays out rows: per 50 m, 20 m of building, 10 m of
    paving and 20 m of lawn.
    """
    out = {k: v.copy() for k, v in layers.items()}
    if cover == "housing" and recipe:
        names = list(recipe["shares"])
        shares = np.array([recipe["shares"][n] for n in names])
        draw = np.random.default_rng(0).choice(len(names), size=int(mask10.sum()), p=shares / shares.sum())
        codes = np.array([COVER_NAMES.index(n) for n in names], dtype="uint8")[draw]
        height = recipe["height_m"] if height_m is None else height_m
        out["cover"][mask10] = codes
        out["canopy"][mask10] = 0.0
        out["height"][mask10] = np.where(codes == BUILDING, height, 0.0)
        out["ndvi"][mask10] = recipe["ndvi"]
        return out
    height_m = 0.0 if height_m is None else height_m
    if cover == "housing":
        row = np.arange(mask10.shape[0]) % HOUSING_PERIOD_PX
        parts = (
            ("building", row < HOUSING_BUILDING_PX),
            ("paved", (row >= HOUSING_BUILDING_PX) & (row < HOUSING_BUILDING_PX + HOUSING_PAVED_PX)),
            ("lawn", row >= HOUSING_BUILDING_PX + HOUSING_PAVED_PX),
        )
        for name, rows in parts:
            out = apply_edit(out, mask10 & rows[:, None], name, ndvi_of, height_m)
        return out
    if cover not in EDIT_COVERS:
        raise SystemExit(f"Unknown cover '{cover}'. Use one of {sorted(EDIT_COVERS) + ['housing']}")
    code, canopy, ndvi_key = EDIT_COVERS[cover]
    out["cover"][mask10] = code
    out["canopy"][mask10] = canopy
    out["height"][mask10] = height_m if code == BUILDING else 0.0
    out["ndvi"][mask10] = ndvi_of[ndvi_key]
    return out
