"""Land-use history: the Realnutzungskartierung editions and GHSL built-up surface.

Sources (no account or key needed):
- Stadt Wien WFS: 17 editions of the land-use map, 1981-2024
- EU JRC: GHS-BUILT-S R2023A, built-up surface per 100 m cell, 1975-2020

The editions come in two schemes. From 2007/08 on every polygon names its class
on three levels. The editions up to 2005 carry a two-letter code (`NUTCD`) for
which the city publishes no legend; the codes are translated by looking at
where each one lies in the 2007/08 edition (`split_stats.code_crosswalk`).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from rasterio.features import rasterize

from . import data
from . import split_stats as st

# (year, WFS layer). 2007/08 is counted as 2008: scenes up to 2008 may still show the old state.
EDITIONS = (
    (1981, "REALNUT1981OGD"), (1985, "REALNUT1985OGD"), (1991, "REALNUT1991OGD"),
    (1994, "REALNUT1994OGD"), (1997, "REALNUT1997OGD"), (2001, "REALNUT2001OGD"),
    (2003, "REALNUT2003OGD"), (2005, "REALNUT2005OGD"), (2008, "REALNUT200708OGD"),
    (2009, "REALNUT2009OGD"), (2012, "REALNUT2012OGD"), (2014, "REALNUT2014OGD"),
    (2016, "REALNUT2016GOGD"), (2018, "REALNUT2018OGD"), (2020, "REALNUT2020OGD"),
    (2022, "REALNUT2022OGD"), (2024, "REALNUT2024OGD"),
)
LEVEL_COLUMNS = (
    ("NUTZUNG_LEVEL1", "NUTZUNG_LEVEL2", "NUTZUNG_LEVEL3"),
    ("LEV1", "LEV2", "LEV3"),  # 2024
)
CODE_COLUMN = "NUTCD"
CROSSWALK_MIN_SHARE = 0.7

GHSL_EPOCHS = tuple(range(1975, 2021, 5))
GHSL_TILE = "R4_C20"  # Mollweide tile that holds Vienna
GHSL_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/"
    "GHS_BUILT_S_E{epoch}_GLOBE_R2023A_54009_100/V1-0/tiles/"
    "GHS_BUILT_S_E{epoch}_GLOBE_R2023A_54009_100_V1_0_{tile}.zip"
)
GHSL_CELL_M2 = 10000.0  # a cell is 100 m x 100 m; values are built-up square metres


def _burn(shapes, grid: data.Grid) -> np.ndarray:
    shapes = [(g, int(v)) for g, v in shapes if g is not None and not g.is_empty and v]
    if not shapes:
        return np.zeros(grid.shape, dtype="uint8")
    return rasterize(shapes, out_shape=grid.shape, transform=grid.transform, fill=0, dtype="uint8")


def edition_scheme(layer: gpd.GeoDataFrame) -> tuple[str, ...] | None:
    """The three class columns of a named edition, or None for a coded one."""
    for columns in LEVEL_COLUMNS:
        if set(columns) <= set(layer.columns):
            return columns
    if CODE_COLUMN in layer.columns:
        return None
    raise SystemExit(f"Land-use layer has neither class names nor {CODE_COLUMN}: {list(layer.columns)}")


def rasterize_named(layer: gpd.GeoDataFrame, columns: tuple[str, ...], grid: data.Grid) -> np.ndarray:
    """Group per pixel for an edition that names its classes."""
    groups = [st.named_group(a, b, c) for a, b, c in zip(*(layer[c] for c in columns))]
    return _burn(zip(layer.geometry, groups), grid)


def rasterize_coded(layer: gpd.GeoDataFrame, codes: list[str], grid: data.Grid) -> np.ndarray:
    """Index into `codes` (1-based) per pixel for an edition with two-letter codes.

    Polygons without a code are left out: in the early editions they are block
    outlines lying on top of the coded polygons.
    """
    index = {code: i + 1 for i, code in enumerate(codes)}
    values = [index.get(str(c).strip(), 0) if c is not None else 0 for c in layer[CODE_COLUMN]]
    return _burn(zip(layer.geometry, values), grid)


def landuse_stack(cache: Path, grid: data.Grid) -> tuple[np.ndarray, list[int], pd.DataFrame]:
    """Groups per pixel for every edition (oldest first), the edition years and the crosswalk."""
    path = cache / "history" / f"landuse_stack_{grid.width}x{grid.height}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            return z["stack"], [int(y) for y in z["years"]], pd.read_csv(path.with_suffix(".crosswalk.csv"))

    layers = {year: data.wfs(name, cache) for year, name in EDITIONS}
    schemes = {year: edition_scheme(layer) for year, layer in layers.items()}
    coded = [y for y, s in schemes.items() if s is None]
    named = [y for y, s in schemes.items() if s is not None]
    if not named:
        raise SystemExit("No land-use edition names its classes; cannot translate the codes")
    codes = sorted(
        {str(c).strip() for y in coded for c in layers[y][CODE_COLUMN].dropna() if str(c).strip()}
    )
    if len(codes) > 250:
        raise SystemExit(f"{len(codes)} land-use codes do not fit the raster type")

    rasters = {y: rasterize_named(layers[y], schemes[y], grid) for y in named}
    code_rasters = {y: rasterize_coded(layers[y], codes, grid) for y in coded}
    crosswalk = pd.DataFrame(columns=["code", "pixels", "majority_group", "majority_share", "group"])
    if coded:
        # The last coded edition against the first named one: closest in time.
        last_coded = max(coded)
        first_named = min(y for y in named if y > last_coded)
        crosswalk = st.code_crosswalk(
            code_rasters[last_coded], rasters[first_named], len(codes), CROSSWALK_MIN_SHARE
        )
        crosswalk.insert(0, "code", codes)
        lookup = np.zeros(len(codes) + 1, dtype="uint8")
        lookup[1:] = [st.GROUP_NAMES.index(g) for g in crosswalk["group"]]
        for y in coded:
            rasters[y] = lookup[code_rasters[y]]
        crosswalk = crosswalk.drop(columns="code_index")
        crosswalk.insert(1, "compared", f"{last_coded} against {first_named}")

    years = sorted(rasters)
    stack = np.stack([rasters[y] for y in years])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, stack=stack, years=np.array(years))
    crosswalk.to_csv(path.with_suffix(".crosswalk.csv"), index=False)
    return stack, years, crosswalk


def area_by_edition(stack: np.ndarray, years: list[int], inside: np.ndarray, pixel_ha: float) -> pd.DataFrame:
    """Hectares per group and edition inside the city."""
    rows = []
    for year, layer in zip(years, stack):
        counts = np.bincount(layer[inside], minlength=len(st.GROUP_NAMES))
        row = {"year": year}
        row.update({f"{name}_ha": counts[i] * pixel_ha for i, name in enumerate(st.GROUP_NAMES)})
        row["green_ha"] = sum(row[f"{st.GROUP_NAMES[g]}_ha"] for g in st.GREEN_GROUPS)
        rows.append(row)
    return pd.DataFrame(rows)


def ghsl_built_fraction(cache: Path, epoch: int, grid: data.Grid) -> np.ndarray:
    """Built-up share (0-1) of the 100 m GHSL cell under every pixel of `grid`."""
    path = cache / "ghsl" / f"GHS_BUILT_S_E{epoch}_{GHSL_TILE}.zip"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(GHSL_URL.format(epoch=epoch, tile=GHSL_TILE), timeout=600)
        resp.raise_for_status()
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(resp.content)
        tmp.rename(path)
    tif = f"GHS_BUILT_S_E{epoch}_GLOBE_R2023A_54009_100_V1_0_{GHSL_TILE}.tif"
    values = data.read_to_grid(f"/vsizip/{path}/{tif}", grid).astype("float32")
    if values.max() > GHSL_CELL_M2:
        raise SystemExit(f"GHSL {epoch}: value {values.max()} above {GHSL_CELL_M2} m² per cell")
    return values / GHSL_CELL_M2
