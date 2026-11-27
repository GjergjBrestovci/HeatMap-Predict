"""The city's layout at 10 m: land cover, building height, tree crowns, NDVI, elevation.

Sources (no account or key needed):
- Stadt Wien WFS: Flächenmehrzweckkarte (FMZK) surfaces and its building model
  with heights, Baumkataster
- Microsoft Planetary Computer: Sentinel-2 (via `data`), Copernicus DEM GLO-30

The FMZK is large (about 2 million polygons). It is fetched in tiles, burnt
into rasters tile by tile, and only the rasters are kept.
"""

from __future__ import annotations

import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pystac_client
import requests
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from scipy import signal

from . import data
from . import features as ft

WFS = "https://data.wien.gv.at/daten/geo"
WFS_LIMIT = 400000  # the server returns at most this many features per request
TILE_M = 4000.0
# FMZK class number (F_KLASSE) -> cover. Classes not listed (street furniture,
# bridge piers, kiosks, pipelines) are left out.
FMZK_COVER = {
    11: ft.BUILDING, 12: ft.BUILDING, 13: ft.BUILDING, 14: ft.BUILDING,
    21: ft.ROAD, 22: ft.ROAD, 26: ft.ROAD, 30: ft.ROAD, 32: ft.ROAD, 33: ft.ROAD, 39: ft.ROAD,
    23: ft.PAVED, 24: ft.PAVED, 25: ft.PAVED, 51: ft.PAVED, 59: ft.PAVED, 63: ft.PAVED, 74: ft.PAVED,
    27: ft.RAIL, 28: ft.RAIL, 29: ft.RAIL,
    53: ft.GRASS, 58: ft.GRASS, 61: ft.GRASS, 69: ft.GRASS,
    52: ft.FOREST,
    54: ft.FIELD,
    41: ft.WATER, 42: ft.WATER, 43: ft.WATER, 49: ft.WATER, 93: ft.WATER,
    60: ft.OTHER,
}
# Later covers are burnt on top of earlier ones where polygons overlap.
BURN_ORDER = (ft.FIELD, ft.FOREST, ft.GRASS, ft.OTHER, ft.PAVED, ft.ROAD, ft.RAIL, ft.WATER, ft.BUILDING)
SURFACE_LAYERS = ("FMZKGRUEN1OGD", "FMZKGRUEN2OGD", "FMZKVERKEHR1OGD", "FMZKVERKEHR2OGD", "FMZKGEWOGD")
BUILDING_LAYER = "FMZKBKMOGD"  # buildings with roof and ground level
MAX_HEIGHT_M = 250.0
DEM_COLLECTION = "cop-dem-glo-30"


def fine_grid(grid30: data.Grid) -> data.Grid:
    """The 10 m grid that splits every pixel of `grid30` into 3 x 3."""
    t = grid30.transform
    return data.Grid(
        Affine(t.a / ft.FINE, 0, t.c, 0, t.e / ft.FINE, t.f),
        grid30.width * ft.FINE, grid30.height * ft.FINE,
    )


def cell_grid(grid30: data.Grid) -> data.Grid:
    """The 90 m grid whose cells hold 3 x 3 pixels of `grid30`."""
    t = grid30.transform
    return data.Grid(
        Affine(t.a * ft.FINE, 0, t.c, 0, t.e * ft.FINE, t.f),
        grid30.width // ft.FINE, grid30.height // ft.FINE,
    )


def tiles(grid: data.Grid, size: float = TILE_M) -> list[tuple[float, float, float, float]]:
    minx, miny, maxx, maxy = grid.bounds
    xs = np.arange(minx, maxx, size)
    ys = np.arange(miny, maxy, size)
    return [(float(x), float(y), float(min(x + size, maxx)), float(min(y + size, maxy)))
            for y in ys for x in xs]


def fetch_tile(layer: str, bounds: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """The features of one FMZK layer that touch `bounds` (in `data.CRS`), in `data.CRS`."""
    west, south, east, north = transform_bounds(data.CRS, "EPSG:4326", *bounds)
    params = {
        "service": "WFS", "version": "1.1.0", "request": "GetFeature",
        "typeName": f"ogdwien:{layer}", "srsName": "EPSG:4326", "outputFormat": "json",
        "bbox": f"{west},{south},{east},{north},EPSG:4326",
    }
    for attempt in range(4):
        try:
            resp = requests.get(WFS, params=params, timeout=600)
            resp.raise_for_status()
            payload = resp.json()
            break
        except (requests.RequestException, json.JSONDecodeError):
            if attempt == 3:
                raise
    features = payload.get("features", [])
    if len(features) >= WFS_LIMIT:
        raise SystemExit(f"{layer}: a tile holds {len(features)} features, the server's limit; use smaller tiles")
    if not features:
        return gpd.GeoDataFrame({"F_KLASSE": []}, geometry=[], crs=data.CRS)
    return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326").to_crs(data.CRS)


def tile_window(grid: data.Grid, bounds: tuple[float, float, float, float]):
    """Row and column slices of `bounds` in `grid`, and the transform of that window."""
    t = grid.transform
    col0 = int(round((bounds[0] - t.c) / t.a))
    col1 = int(round((bounds[2] - t.c) / t.a))
    row0 = int(round((bounds[3] - t.f) / t.e))
    row1 = int(round((bounds[1] - t.f) / t.e))
    window = Affine(t.a, 0, t.c + col0 * t.a, 0, t.e, t.f + row0 * t.e)
    return (slice(row0, row1), slice(col0, col1)), window


def burn_tile(surfaces: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame, grid: data.Grid,
              bounds: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice]]:
    """Cover and building height for the pixels of `grid` inside `bounds`.

    Returns the two small arrays and where they belong in the full grid.
    """
    where, window = tile_window(grid, bounds)
    shape = (where[0].stop - where[0].start, where[1].stop - where[1].start)

    shapes = []
    frames = [surfaces, buildings]
    for cover in BURN_ORDER:
        for frame in frames:
            if frame.empty:
                continue
            codes = frame["F_KLASSE"].map(FMZK_COVER)
            shapes += [(g, cover) for g in frame.geometry[codes == cover] if g is not None and not g.is_empty]
    cover = (
        rasterize(shapes, out_shape=shape, transform=window, fill=ft.NONE, dtype="uint8")
        if shapes else np.zeros(shape, dtype="uint8")
    )

    height = np.zeros(shape, dtype="float32")
    if not buildings.empty and {"O_KOTE", "HOEHE_DGM"} <= set(buildings.columns):
        b = buildings[buildings["F_KLASSE"].map(FMZK_COVER) == ft.BUILDING].copy()
        b["h"] = (b["O_KOTE"].astype(float) - b["HOEHE_DGM"].astype(float)).clip(0, MAX_HEIGHT_M)
        b = b.dropna(subset=["h"]).sort_values("h")  # the tallest part wins where parts overlap
        pairs = [(g, h) for g, h in zip(b.geometry, b["h"]) if g is not None and not g.is_empty]
        if pairs:
            height = rasterize(pairs, out_shape=shape, transform=window, fill=0, dtype="float32")
    height = np.where(cover == ft.BUILDING, height, 0).astype("float32")
    return cover, height, where


def fmzk_layers(cache: Path, grid10: data.Grid, workers: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Cover and building height at 10 m for the whole grid; tiles are cached as rasters."""
    cover = np.zeros(grid10.shape, dtype="uint8")
    height = np.zeros(grid10.shape, dtype="float32")
    todo = tiles(grid10)

    def one(index_bounds):
        index, bounds = index_bounds
        path = cache / "layout" / f"fmzk_{index:03d}.npz"
        if not path.exists():
            surfaces = [fetch_tile(layer, bounds) for layer in SURFACE_LAYERS]
            surfaces = gpd.GeoDataFrame(
                pd.concat([s[["F_KLASSE", "geometry"]] for s in surfaces], ignore_index=True),
                crs=data.CRS,
            )
            c, h, _ = burn_tile(surfaces, fetch_tile(BUILDING_LAYER, bounds), grid10, bounds)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, cover=c, height=h)
            tmp.rename(path)
        with np.load(path) as z:
            return bounds, z["cover"], z["height"]

    with ThreadPoolExecutor(workers) as pool:
        for n, (bounds, c, h) in enumerate(pool.map(one, enumerate(todo)), 1):
            where, _ = tile_window(grid10, bounds)
            cover[where], height[where] = c, h
            if n % 10 == 0 or n == len(todo):
                print(f"  FMZK tiles {n}/{len(todo)}")
    return cover, height


def crown_diameter_m(text) -> float:
    """Crown diameter from the register's class text, e.g. '19-21 m' -> 20; 0 if not given."""
    numbers = [float(n) for n in re.findall(r"\d+(?:[.,]\d+)?", str(text).replace(",", "."))]
    if not numbers:
        return 0.0
    return float(np.mean(numbers[:2]))


def canopy_fraction(x: np.ndarray, y: np.ndarray, diameter_m: np.ndarray, grid: data.Grid) -> np.ndarray:
    """Share of each pixel under mapped tree crowns.

    A crown's area is spread evenly over the pixels within its radius of the
    trunk; overlapping crowns cannot cover more than the whole pixel.
    """
    t = grid.transform
    area = np.zeros(grid.shape, dtype="float32")
    cols = ((x - t.c) / t.a).astype(int)
    rows = ((y - t.f) / t.e).astype(int)
    inside = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width) & (diameter_m > 0)
    pixel_area = abs(t.a * t.e)
    for d in np.unique(np.round(diameter_m[inside])):
        pick = inside & (np.round(diameter_m) == d)
        counts = np.zeros(grid.shape, dtype="float32")
        np.add.at(counts, (rows[pick], cols[pick]), 1.0)
        kernel = ft.disc(max(d / 2 / abs(t.a), 0.5))
        crown = np.pi * (d / 2) ** 2
        area += signal.fftconvolve(counts, kernel / kernel.sum() * crown, mode="same").astype("float32")
    share = np.clip(area / pixel_area, 0, 1)
    return np.where(share < 1e-4, 0, share).astype("float32")  # drop the noise of the transform


def tree_canopy(cache: Path, grid10: data.Grid) -> np.ndarray:
    """Tree crown cover at 10 m from the Baumkataster (trees the city manages)."""
    path = cache / "layout" / "baumkataster.csv"
    if not path.exists():
        params = {
            "service": "WFS", "version": "1.1.0", "request": "GetFeature",
            "typeName": "ogdwien:BAUMKATOGD", "srsName": "EPSG:4326", "outputFormat": "csv",
            "propertyName": "KRONENDURCHMESSER_TXT,SHAPE",
        }
        resp = requests.get(WFS, params=params, timeout=900)
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
    table = pd.read_csv(io.BytesIO(path.read_bytes()))
    missing = {"SHAPE", "KRONENDURCHMESSER_TXT"} - set(table.columns)
    if missing:
        raise SystemExit(f"Baumkataster lacks {sorted(missing)}; has {list(table.columns)}")
    points = gpd.GeoSeries.from_wkt(table["SHAPE"], crs="EPSG:4326").to_crs(data.CRS)
    diameter = np.array([crown_diameter_m(v) for v in table["KRONENDURCHMESSER_TXT"]])
    return canopy_fraction(points.x.to_numpy(), points.y.to_numpy(), diameter, grid10)


def elevation(cache: Path, grid30: data.Grid) -> np.ndarray:
    """Terrain height in metres on `grid30` (Copernicus DEM GLO-30, bilinear)."""
    path = cache / "layout" / f"dem_{grid30.width}x{grid30.height}.npy"
    if path.exists():
        return np.load(path)
    catalog = pystac_client.Client.open(data.STAC_URL)
    bounds = transform_bounds(data.CRS, "EPSG:4326", *grid30.bounds)
    items = list(catalog.search(collections=[DEM_COLLECTION], bbox=bounds).items())
    if not items:
        raise SystemExit("No elevation tile found for Vienna")
    out = np.zeros(grid30.shape, dtype="float32")
    for item in items:
        part = data.read_to_grid(data._href(item, "data"), grid30, Resampling.bilinear)
        out = np.where(out == 0, part, out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out


def build_layers(cache: Path, grid30: data.Grid, s2_years: range, s2_dates: int, refresh: bool = False) -> dict:
    """All layout layers; the finished set is cached in one file for the scenario tool."""
    path = cache / "model" / "layers.npz"
    if path.exists() and not refresh:
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    grid10 = fine_grid(grid30)
    print("Layout: land cover and building heights (FMZK) ...")
    cover, height = fmzk_layers(cache, grid10)
    print("Layout: tree register, Sentinel-2 NDVI, elevation ...")
    ndvi, _ = data.s2_summer_ndvi(grid10, cache, s2_years, s2_dates, refresh=refresh)
    layers = {
        "cover": cover,
        "height": height,
        "canopy": tree_canopy(cache, grid10),
        "ndvi": ndvi.astype("float32"),
        "elevation": elevation(cache, grid30),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **layers)
    return layers


def polygon_mask(geometry, grid: data.Grid) -> np.ndarray:
    """Pixels of `grid` whose centre lies inside `geometry` (in `data.CRS`)."""
    return rasterize([(geometry, 1)], out_shape=grid.shape, transform=grid.transform, fill=0,
                     dtype="uint8").astype(bool)
