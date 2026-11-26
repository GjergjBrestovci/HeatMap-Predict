"""Proof of concept: land surface temperature of Vienna parks vs sealed areas.

Pulls one Landsat 8/9 Collection 2 Level-2 scene (surface temperature + red/NIR)
from the Microsoft Planetary Computer STAC API, pulls park and land-use polygons
from the Stadt Wien open data WFS, and compares mean LST / NDVI per site.

No account or API key is needed for any of the sources.

Usage:
    python prototype/lst_parks_vs_sealed.py [--date 2026-08-09] [--out prototype/output]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from shapely.geometry import LineString, box, shape

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
WFS_URL = (
    "https://data.wien.gv.at/daten/geo?service=WFS&request=GetFeature&version=1.1.0"
    "&srsName=EPSG:4326&outputFormat=json&typeName=ogdwien:{layer}"
)
VIENNA_BBOX = (16.18, 48.11, 16.58, 48.33)  # lon/lat, covers the whole city
METRIC_CRS = "EPSG:32633"  # UTM 33N, the CRS of the Landsat scenes over Vienna

# Parks: exact names in the "Öffentlich zugängliche Grünflächen" layer (T_LANG).
PARKS = [
    ("Esterhazypark", "park, small"),
    ("Alois-Drasche-Park", "park, small"),
    ("Stadtpark", "park, medium"),
    ("Türkenschanzpark", "park, medium"),
    ("Augarten", "park, medium"),
    ("Prater", "park, large"),
]

# Sealed controls are cut out of the Realnutzungskartierung 2022 land-use map:
# (name, land-use column, value, area of interest in lon/lat)
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
MAP_MARGIN_M = 1500

# Landsat QA_PIXEL bits: fill, dilated cloud, cirrus, cloud, cloud shadow.
QA_BAD_BITS = 0b11111


def find_scene(date: str):
    """Return the least cloudy Landsat 8/9 scene of that day covering all of Vienna."""
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=VIENNA_BBOX,
        datetime=date,
        query={"platform": {"in": ["landsat-8", "landsat-9"]}},
    )
    aoi = box(*VIENNA_BBOX)
    items = [i for i in search.items() if shape(i.geometry).contains(aoi)]
    if not items:
        raise SystemExit(f"No Landsat 8/9 scene fully covering Vienna on {date}.")
    return min(items, key=lambda i: i.properties["eo:cloud_cover"])


def read_band(item, asset: str):
    """Read the Vienna window of one band. Returns (raw array, transform, crs)."""
    with rasterio.open(item.assets[asset].href) as src:
        bounds = transform_bounds("EPSG:4326", src.crs, *VIENNA_BBOX)
        window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        return src.read(1, window=window), src.window_transform(window), src.crs


def load_lst_ndvi(item):
    """Return LST (deg C), NDVI, affine transform and CRS; bad pixels are NaN."""
    lwir, transform, crs = read_band(item, "lwir11")
    red, _, _ = read_band(item, "red")
    nir, _, _ = read_band(item, "nir08")
    qa, _, _ = read_band(item, "qa_pixel")

    bad = (qa & QA_BAD_BITS) != 0
    # Scale factors from the Collection 2 Level-2 product guide.
    lst = np.where(bad | (lwir == 0), np.nan, lwir * 0.00341802 + 149.0 - 273.15)
    red = np.where(bad | (red == 0), np.nan, red * 2.75e-5 - 0.2)
    nir = np.where(bad | (nir == 0), np.nan, nir * 2.75e-5 - 0.2)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / (nir + red)
    return lst, ndvi, transform, crs


def wfs(layer: str) -> gpd.GeoDataFrame:
    return gpd.read_file(WFS_URL.format(layer=layer))


def build_sites() -> gpd.GeoDataFrame:
    """Named parks and sealed control areas as one GeoDataFrame in METRIC_CRS."""
    rows = []

    green = wfs("OEFFGRUENFLOGD").to_crs(METRIC_CRS)
    for name, kind in PARKS:
        geom = green[green["T_LANG"] == name].geometry.union_all()
        if geom.is_empty:
            raise SystemExit(f"Park '{name}' not found in OEFFGRUENFLOGD.")
        rows.append({"site": name, "type": kind, "geometry": geom})

    landuse = wfs("REALNUT2022OGD").to_crs(METRIC_CRS)

    def to_metric(geom):
        return gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(METRIC_CRS).iloc[0]

    rail = landuse[landuse["NUTZUNG_LEVEL3"] == "Bahnhöfe und Bahnanlagen"]
    rail = rail.union_all().intersection(to_metric(WESTBAHN_AOI))
    rows.append({"site": "Westbahn rail corridor", "type": "sealed, railway", "geometry": rail})

    streets = landuse[landuse["NUTZUNG_LEVEL2"] == "Straßenraum"].union_all()
    guertel = to_metric(GUERTEL_LINE).buffer(GUERTEL_HALF_WIDTH_M).intersection(streets)
    rows.append({"site": "Westgürtel (road)", "type": "sealed, road", "geometry": guertel})

    dense = landuse[
        (landuse["NUTZUNG_LEVEL3"] == "dichtes Wohn(misch)gebiet") & (landuse["BEZ"] == "07")
    ]
    rows.append(
        {"site": "Neubau dense housing", "type": "sealed, housing", "geometry": dense.union_all()}
    )

    return gpd.GeoDataFrame(rows, crs=METRIC_CRS)


def zonal_stats(sites, lst, ndvi, transform) -> pd.DataFrame:
    out = []
    for row in sites.itertuples():
        inside = geometry_mask([row.geometry], lst.shape, transform, invert=True)
        valid = inside & np.isfinite(lst)
        out.append(
            {
                "site": row.site,
                "type": row.type,
                "area_ha": round(row.geometry.area / 1e4, 1),
                "pixels": int(valid.sum()),
                "lst_mean_c": round(float(np.nanmean(lst[valid])), 1) if valid.any() else np.nan,
                "lst_p90_c": round(float(np.nanpercentile(lst[valid], 90)), 1)
                if valid.any()
                else np.nan,
                "ndvi_mean": round(float(np.nanmean(ndvi[valid])), 2) if valid.any() else np.nan,
            }
        )
    df = pd.DataFrame(out)
    sealed_mean = df.loc[df["type"].str.startswith("sealed"), "lst_mean_c"].mean()
    df["delta_vs_sealed_c"] = (df["lst_mean_c"] - sealed_mean).round(1)
    return df


def plot_map(lst, ndvi, transform, crs, sites, city, title, path: Path) -> None:
    h, w = lst.shape
    extent = (transform.c, transform.c + w * transform.a, transform.f + h * transform.e, transform.f)
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
    panels = [
        (lst, "inferno", "Land surface temperature (°C)", np.nanpercentile(lst, [2, 98])),
        (ndvi, "YlGn", "NDVI", (0.0, 0.9)),
    ]
    parks = sites[sites["type"].str.startswith("park")]
    sealed = sites[sites["type"].str.startswith("sealed")]
    for ax, (data, cmap, label, (vmin, vmax)) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
        city.boundary.plot(ax=ax, color="white", linewidth=0.8)
        parks.boundary.plot(ax=ax, color="cyan", linewidth=1.2)
        sealed.boundary.plot(ax=ax, color="magenta", linewidth=1.2)
        for row in sites.itertuples():
            p = row.geometry.representative_point()
            ax.annotate(
                row.site,
                (p.x, p.y),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 1, "edgecolor": "none"},
            )
        minx, miny, maxx, maxy = sites.total_bounds  # zoom to the study sites
        ax.set_xlim(minx - MAP_MARGIN_M, maxx + MAP_MARGIN_M)
        ax.set_ylim(miny - MAP_MARGIN_M, maxy + MAP_MARGIN_M)
        ax.set_axis_off()
        ax.set_title(label)
        fig.colorbar(im, ax=ax, shrink=0.7, label=label)
    fig.suptitle(f"{title}  |  cyan = parks, magenta = sealed controls")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", default="2026-08-09", help="scene date, YYYY-MM-DD")
    parser.add_argument("--out", default=Path(__file__).parent / "output", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    item = find_scene(args.date)
    print(f"Scene: {item.id}  ({item.datetime:%Y-%m-%d %H:%M} UTC, "
          f"cloud cover {item.properties['eo:cloud_cover']}%)")
    lst, ndvi, transform, crs = load_lst_ndvi(item)
    if str(crs) != METRIC_CRS:
        raise SystemExit(f"Unexpected scene CRS {crs}, expected {METRIC_CRS}.")

    sites = build_sites()
    city = wfs("LANDESGRENZEOGD").to_crs(METRIC_CRS)
    table = zonal_stats(sites, lst, ndvi, transform)

    in_city = geometry_mask(city.geometry, lst.shape, transform, invert=True)
    ok = in_city & np.isfinite(lst) & np.isfinite(ndvi)
    r = float(np.corrcoef(lst[ok], ndvi[ok])[0, 1])

    table.to_csv(args.out / "site_stats.csv", index=False)
    plot_map(lst, ndvi, transform, crs, sites, city,
             f"Vienna, Landsat {item.datetime:%Y-%m-%d} ({item.id})", args.out / "vienna_lst_ndvi.png")

    print()
    print(table.to_markdown(index=False))
    print()
    print(f"City-wide mean LST: {np.nanmean(lst[ok]):.1f} °C over {ok.sum()} pixels")
    print(f"Pixel-wise correlation LST vs NDVI inside the city boundary: r = {r:.2f}")
    print(f"Wrote {args.out / 'site_stats.csv'} and {args.out / 'vienna_lst_ndvi.png'}")


if __name__ == "__main__":
    main()
