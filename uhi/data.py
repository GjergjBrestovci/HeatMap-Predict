"""Data access: Landsat and Sentinel-2 windows, Stadt Wien layers. Everything is cached.

Sources (no account or key needed):
- Microsoft Planetary Computer STAC: `landsat-c2-l2`, `sentinel-2-l2a`
- Stadt Wien open data WFS: green spaces, land use, city boundary
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import pystac
import pystac_client
import rasterio
import requests
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from shapely.geometry import LineString, Point, box

from . import core

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
WFS_URL = (
    "https://data.wien.gv.at/daten/geo?service=WFS&request=GetFeature&version=1.1.0"
    "&srsName=EPSG:4326&outputFormat=json&typeName=ogdwien:{layer}"
)
VIENNA_BBOX = (16.18, 48.11, 16.58, 48.33)  # lon/lat, covers the whole city
CRS = "EPSG:32633"  # UTM 33N, the native CRS of Landsat and Sentinel-2 over Vienna
CENTRE = Point(16.3731, 48.2084)  # Stephansdom

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
}


# --------------------------------------------------------------------------- grids


@dataclass(frozen=True)
class Grid:
    """A north-up raster grid in `CRS`."""

    transform: Affine
    width: int
    height: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        t = self.transform
        return (t.c, t.f + self.height * t.e, t.c + self.width * t.a, t.f)


def make_grid(bounds: tuple[float, float, float, float], res: float, origin: float) -> Grid:
    """Grid covering `bounds` (in `CRS`), cell edges at `origin + k * res`."""
    minx, miny, maxx, maxy = bounds
    x0 = np.floor((minx - origin) / res) * res + origin
    y1 = np.ceil((maxy - origin) / res) * res + origin
    width = int(np.ceil((maxx - x0) / res))
    height = int(np.ceil((y1 - miny) / res))
    return Grid(Affine(res, 0, x0, 0, -res, y1), width, height)


def landsat_grid() -> Grid:
    """30 m grid over Vienna, aligned with Landsat pixels (edges at 15 m + k * 30 m)."""
    return make_grid(transform_bounds("EPSG:4326", CRS, *VIENNA_BBOX), 30.0, 15.0)


def read_to_grid(href: str, grid: Grid, resampling=Resampling.nearest) -> np.ndarray:
    """Read band 1 of a remote raster onto `grid`. Cells outside the raster are 0."""
    with rasterio.Env(**GDAL_ENV), rasterio.open(href) as src, WarpedVRT(
        src,
        crs=CRS,
        transform=grid.transform,
        width=grid.width,
        height=grid.height,
        resampling=resampling,
        src_nodata=None,
        nodata=0,
    ) as vrt:
        return vrt.read(1)


# ------------------------------------------------------------------ Stadt Wien layers


def wfs(layer: str, cache: Path) -> gpd.GeoDataFrame:
    """A Stadt Wien WFS layer in `CRS`, downloaded once."""
    path = cache / "wfs" / f"{layer}.geojson"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(WFS_URL.format(layer=layer), timeout=300)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return gpd.read_file(path).to_crs(CRS)


def _metric(geom):
    return gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(CRS).iloc[0]


# Sealed controls, cut out of the Realnutzungskartierung 2022 land-use map.
WESTBAHN_AOI = box(16.300, 48.1935, 16.3375, 48.2000)  # track field west of Westbahnhof
GUERTEL_LINE = LineString(  # Westgürtel, Westbahnhof -> Nussdorfer Strasse
    [
        (16.3385, 48.1970),
        (16.3385, 48.2080),
        (16.3392, 48.2115),
        (16.3418, 48.2167),
        (16.3440, 48.2212),
        (16.3497, 48.2257),
        (16.3525, 48.2313),
    ]
)
GUERTEL_HALF_WIDTH_M = 40

# Park selection from "Öffentlich zugängliche Grünflächen".
PARK_MIN_HA = 1.0
PARK_MAX_DIST_KM = 6.0
PARK_CLUSTER_GAP_M = 50.0
# River and canal bank strips carry the park flag but are not parks.
PARK_EXCLUDE_PREFIXES = ("Donauinsel", "Donaukanal", "Handelskai", "Neue Donau")
# Large parks whose main polygons are flagged as forest or garden instead of park.
PARK_WHOLE_NAMES = ("Prater", "Schlosspark Schönbrunn")


def sealed_sites(cache: Path) -> gpd.GeoDataFrame:
    landuse = wfs("REALNUT2022OGD", cache)
    rail = landuse[landuse["NUTZUNG_LEVEL3"] == "Bahnhöfe und Bahnanlagen"].union_all()
    streets = landuse[landuse["NUTZUNG_LEVEL2"] == "Straßenraum"].union_all()
    dense = landuse[
        (landuse["NUTZUNG_LEVEL3"] == "dichtes Wohn(misch)gebiet") & (landuse["BEZ"] == "07")
    ].union_all()
    rows = [
        ("Westbahn rail corridor", "railway", rail.intersection(_metric(WESTBAHN_AOI))),
        (
            "Westgürtel (road)",
            "road",
            _metric(GUERTEL_LINE).buffer(GUERTEL_HALF_WIDTH_M).intersection(streets),
        ),
        ("Neubau dense housing", "housing", dense),
    ]
    out = gpd.GeoDataFrame(
        [{"site": n, "kind": "sealed", "size_class": c, "geometry": g} for n, c, g in rows],
        crs=CRS,
    )
    return out


def park_sites(cache: Path) -> gpd.GeoDataFrame:
    """Parks of at least `PARK_MIN_HA` within `PARK_MAX_DIST_KM` of the city centre.

    Polygons with the same name are merged when they lie within
    `PARK_CLUSTER_GAP_M` of each other; the same name elsewhere in the city is
    a different park and gets a numbered suffix.
    """
    green = wfs("OEFFGRUENFLOGD", cache)
    green = green[green["T_LANG"].notna()]
    whole = green["T_LANG"].isin(PARK_WHOLE_NAMES)
    excluded = green["T_LANG"].str.startswith(PARK_EXCLUDE_PREFIXES)
    green = green[whole | ((green["P_PK"] == 1) & ~excluded)]

    rows = []
    for name, group in green.groupby("T_LANG"):
        merged = group.geometry.buffer(PARK_CLUSTER_GAP_M / 2).union_all()
        clusters = list(getattr(merged, "geoms", [merged]))
        parts = []
        for cluster in clusters:
            geom = group[group.intersects(cluster)].geometry.union_all()
            if geom.area / 1e4 >= PARK_MIN_HA:
                parts.append(geom)
        parts.sort(key=lambda g: -g.area)
        for i, geom in enumerate(parts):
            rows.append({"site": name if i == 0 else f"{name} #{i + 1}", "geometry": geom})

    parks = gpd.GeoDataFrame(rows, crs=CRS)
    dist_km = parks.centroid.distance(_metric(CENTRE)) / 1e3
    parks = parks[dist_km <= PARK_MAX_DIST_KM].copy()
    parks["kind"] = "park"
    parks["size_class"] = [core.size_class(a / 1e4) for a in parks.area]
    return parks


def build_sites(cache: Path) -> gpd.GeoDataFrame:
    sites = pd.concat([park_sites(cache), sealed_sites(cache)], ignore_index=True)
    sites = gpd.GeoDataFrame(sites, crs=CRS)
    sites["area_ha"] = (sites.area / 1e4).round(2)
    if sites["site"].duplicated().any():
        raise ValueError("duplicate site names")
    return sites.sort_values(["kind", "area_ha"], ascending=[True, False]).reset_index(drop=True)


def rasterize_sites(sites: gpd.GeoDataFrame, grid: Grid) -> dict[str, np.ndarray]:
    """Flat pixel indices per site: pixels whose centre lies inside the site."""
    return {
        row.site: np.flatnonzero(
            geometry_mask([row.geometry], grid.shape, grid.transform, invert=True)
        )
        for row in sites.itertuples()
    }


# ----------------------------------------------------------------------- STAC search


def _search(cache: Path, name: str, refresh: bool, **kwargs) -> list[pystac.Item]:
    """STAC search with the (unsigned) result cached as JSON."""
    path = cache / f"{name}_items.json"
    if path.exists() and not refresh:
        return [pystac.Item.from_dict(d) for d in json.loads(path.read_text())]
    catalog = pystac_client.Client.open(STAC_URL)
    items = list(catalog.search(**kwargs).items())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([i.to_dict() for i in items]))
    return items


def _href(item: pystac.Item, asset: str) -> str:
    return planetary_computer.sign(item.assets[asset].href)


# --------------------------------------------------------------------------- Landsat

LANDSAT_BANDS = ("lwir11", "red", "nir08", "qa_pixel")


def landsat_items(
    cache: Path, years: range, max_scene_cloud: float, refresh: bool = False
) -> list[pystac.Item]:
    """Landsat 8/9 L2 items over Vienna for June-August of `years`."""
    items = []
    for year in years:
        items += _search(
            cache / "landsat",
            str(year),
            refresh,
            collections=["landsat-c2-l2"],
            bbox=VIENNA_BBOX,
            datetime=f"{year}-06-01/{year}-08-31",
            query={"platform": {"in": ["landsat-8", "landsat-9"]}},
        )
    return [i for i in items if i.properties["eo:cloud_cover"] < max_scene_cloud]


def _landsat_item_arrays(item: pystac.Item, grid: Grid, cache: Path) -> dict[str, np.ndarray]:
    path = cache / "landsat" / f"{item.id}.npz"
    if path.exists():
        with np.load(path) as z:
            return {b: z[b] for b in LANDSAT_BANDS}
    arrays = {b: read_to_grid(_href(item, b), grid) for b in LANDSAT_BANDS}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.rename(path)
    return arrays


def fetch_landsat(items: list[pystac.Item], grid: Grid, cache: Path, workers: int = 8) -> None:
    """Download the Vienna window of every item that is not cached yet."""
    todo = [i for i in items if not (cache / "landsat" / f"{i.id}.npz").exists()]
    if not todo:
        return
    print(f"Downloading {len(todo)} Landsat windows ({len(items) - len(todo)} cached) ...")
    with ThreadPoolExecutor(workers) as pool:
        for n, _ in enumerate(pool.map(lambda i: _landsat_item_arrays(i, grid, cache), todo), 1):
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}")


def landsat_passes(items: list[pystac.Item]) -> dict[str, list[pystac.Item]]:
    """Group items into overpasses: same day, same satellite, same path.

    Two rows of one path are cut from the same overpass and overlap, so they
    must count as one observation.
    """
    passes: dict[str, list[pystac.Item]] = {}
    for item in sorted(items, key=lambda i: i.id):
        p = item.properties
        key = f"{item.datetime:%Y-%m-%d}_{p['platform']}_p{p['landsat:wrs_path']}"
        passes.setdefault(key, []).append(item)
    return dict(sorted(passes.items()))


def load_landsat_pass(items: list[pystac.Item], grid: Grid, cache: Path) -> dict[str, np.ndarray]:
    """Raw bands of one overpass on `grid`; later rows fill where earlier ones have no data."""
    out = _landsat_item_arrays(items[0], grid, cache)
    for item in items[1:]:
        more = _landsat_item_arrays(item, grid, cache)
        fill = (out["qa_pixel"] <= 1) & (more["qa_pixel"] > 1)
        out = {b: np.where(fill, more[b], out[b]) for b in LANDSAT_BANDS}
    return out


# ------------------------------------------------------------------------ Sentinel-2


def s2_summer_ndvi(
    grid: Grid, cache: Path, years: range, max_dates: int, max_scene_cloud: float = 10.0,
    refresh: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Median NDVI at 10 m over the clearest June-August Sentinel-2 dates.

    Returns the NDVI array on `grid` and the dates used.
    """
    tag = f"{years.start}-{years.stop - 1}_n{max_dates}_{grid.width}x{grid.height}"
    path = cache / "sentinel2" / f"ndvi_median_{tag}.npz"
    if path.exists() and not refresh:
        with np.load(path) as z:
            return z["ndvi"], [str(d) for d in z["dates"]]

    wgs_bounds = transform_bounds(CRS, "EPSG:4326", *grid.bounds)
    items = []
    for year in years:
        items += _search(
            cache / "sentinel2",
            str(year),
            refresh,
            collections=["sentinel-2-l2a"],
            bbox=wgs_bounds,
            datetime=f"{year}-06-01/{year}-08-31",
            query={"eo:cloud_cover": {"lt": max_scene_cloud}},
        )
    by_date: dict[str, list[pystac.Item]] = {}
    for item in items:
        by_date.setdefault(f"{item.datetime:%Y-%m-%d}", []).append(item)
    # Clearest dates first; the worst tile of a date decides.
    ranked = sorted(by_date, key=lambda d: max(i.properties["eo:cloud_cover"] for i in by_date[d]))
    dates = sorted(ranked[:max_dates])
    print(f"Sentinel-2: reading {len(dates)} dates at 10 m ...")

    def one_date(date: str) -> np.ndarray:
        result = np.full(grid.shape, np.nan, dtype="float32")
        for item in by_date[date]:
            baseline = item.properties["s2:processing_baseline"]
            red = core.s2_reflectance(read_to_grid(_href(item, "B04"), grid), baseline)
            nir = core.s2_reflectance(read_to_grid(_href(item, "B08"), grid), baseline)
            clear = core.s2_clear(read_to_grid(_href(item, "SCL"), grid))
            value = np.where(clear, core.ndvi(nir, red), np.nan)
            result = np.where(np.isnan(result), value, result).astype("float32")
        return result

    with ThreadPoolExecutor(4) as pool:
        stack = np.stack(list(pool.map(one_date, dates)))
    with np.errstate(all="ignore"):
        median = np.nanmedian(stack, axis=0).astype("float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ndvi=median, dates=np.array(dates))
    return median, dates
