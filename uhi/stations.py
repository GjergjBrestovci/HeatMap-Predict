"""Ground truth: Landsat surface temperature against air temperature at Vienna's weather stations.

Usage (from the repository root):
    python -m uhi.stations [--years 2013-2026] [--out results/stations]

Station data come from the GeoSphere Austria Data Hub (open, no key, CC BY 4.0).
Meant to run in GitHub Actions (see .github/workflows/analysis.yml); everything
downloaded is cached in data/cache.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from rasterio.features import geometry_mask

from . import core, data, station_plots
from . import station_stats as ss

API = "https://dataset.api.hub.geosphere.at/v1/station/historical/klima-v2-10min"
AIR_PARAMETER = "tl"  # air temperature at 2 m, degrees Celsius
STATE = "Wien"
WINDOW_MIN = 40  # minutes of station data fetched on each side of the overpass
RING_M = 300.0
MIN_SCENES_PER_STATION = 10


def station_metadata(cache: Path) -> pd.DataFrame:
    """Vienna stations of the 10-minute dataset: id, name, lon, lat, altitude, validity."""
    path = cache / "stations" / "metadata.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(f"{API}/metadata", timeout=120)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    meta = json.loads(path.read_text())
    names = [p["name"] for p in meta["parameters"]]
    if AIR_PARAMETER not in names:
        raise SystemExit(f"Parameter '{AIR_PARAMETER}' not in the dataset. Available: {names}")
    stations = pd.DataFrame(meta["stations"])
    missing = {"id", "name", "state", "lat", "lon", "altitude", "valid_from", "valid_to"} - set(
        stations.columns
    )
    if missing:
        raise SystemExit(f"Station metadata lacks {sorted(missing)}; has {list(stations.columns)}")
    stations = stations[stations["state"] == STATE].copy()
    stations["id"] = stations["id"].astype(str)
    return stations[["id", "name", "lon", "lat", "altitude", "valid_from", "valid_to"]]


def fetch_air(station_ids: list[str], when: pd.Timestamp, cache: Path) -> pd.DataFrame:
    """10-minute air temperature of all stations around one overpass: time, station, air."""
    path = cache / "stations" / f"{when:%Y%m%dT%H%M}.csv"
    if not path.exists():
        params = {
            "parameters": AIR_PARAMETER,
            "station_ids": ",".join(station_ids),
            "start": f"{when - pd.Timedelta(minutes=WINDOW_MIN):%Y-%m-%dT%H:%M}",
            "end": f"{when + pd.Timedelta(minutes=WINDOW_MIN):%Y-%m-%dT%H:%M}",
            "output_format": "csv",
        }
        resp = requests.get(API, params=params, timeout=120)
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
    return parse_air_csv(path.read_text())


def parse_air_csv(text: str) -> pd.DataFrame:
    """The API's CSV (columns time, station, tl, ...) as time (UTC), station (str), air_c."""
    df = pd.read_csv(io.StringIO(text))
    missing = {"time", "station", AIR_PARAMETER} - set(df.columns)
    if missing:
        raise SystemExit(f"Station CSV lacks {sorted(missing)}; has {list(df.columns)}")
    return pd.DataFrame(
        {
            "time": pd.to_datetime(df["time"], utc=True),
            "station": df["station"].astype(str),
            "air_c": pd.to_numeric(df[AIR_PARAMETER], errors="coerce"),
        }
    )


def green_share(points: gpd.GeoSeries, cache: Path) -> list[float]:
    """Share of green land use ("Grünlandnutzung" without water) within `RING_M` of each point."""
    landuse = data.wfs("REALNUT2022OGD", cache)
    green = landuse[
        (landuse["NUTZUNG_LEVEL1"] == "Grünlandnutzung") & (landuse["NUTZUNG_LEVEL2"] != "Gewässer")
    ].union_all()
    discs = points.buffer(RING_M)
    return [float(d.intersection(green).area / d.area) for d in discs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", default="2013-2026", help="first-last summer")
    parser.add_argument("--cache", default=Path("data/cache"), type=Path)
    parser.add_argument("--out", default=Path("results/stations"), type=Path)
    parser.add_argument("--max-scene-cloud", default=70.0, type=float)
    parser.add_argument("--min-city-clear", default=0.8, type=float)
    parser.add_argument("--min-ring-valid", default=0.8, type=float,
                        help="minimum usable fraction of the pixels around a station")
    parser.add_argument("--refresh", action="store_true", help="repeat the catalogue searches")
    args = parser.parse_args()
    first, last = (int(y) for y in args.years.split("-"))
    args.out.mkdir(parents=True, exist_ok=True)

    # Landsat overpasses, the same ones the park analysis starts from
    grid = data.landsat_grid()
    items = data.landsat_items(args.cache, range(first, last + 1), args.max_scene_cloud, args.refresh)
    data.fetch_landsat(items, grid, args.cache)
    passes = data.landsat_passes(items)
    city_geom = data.wfs("LANDESGRENZEOGD", args.cache).geometry
    city = geometry_mask(city_geom, grid.shape, grid.transform, invert=True)

    # Stations inside the city and on the grid
    stations = station_metadata(args.cache)
    points = gpd.GeoSeries(
        gpd.points_from_xy(stations["lon"], stations["lat"]), crs="EPSG:4326"
    ).to_crs(data.CRS)
    stations["x"], stations["y"] = points.x.to_numpy(), points.y.to_numpy()
    stations["green_share"] = green_share(points, args.cache)
    stations["station_class"] = [ss.station_class(g) for g in stations["green_share"]]
    pixels = {}
    for row in stations.itertuples():
        pixels[f"{row.id}|pixel"] = ss.pixel_index(grid.transform, grid.shape, row.x, row.y)
        pixels[f"{row.id}|ring"] = ss.disc_indices(grid.transform, grid.shape, row.x, row.y, RING_M)

    # Per overpass: surface temperature at each station, air temperature at the overpass time
    rows = []
    for scene, scene_items in passes.items():
        raw = data.load_landsat_pass(scene_items, grid, args.cache)
        clear = core.landsat_clear(raw["qa_pixel"])
        if clear[city].mean() < args.min_city_clear:
            continue
        lst = core.landsat_st_celsius(raw["lwir11"])
        ndvi = core.ndvi(core.landsat_reflectance(raw["nir08"]), core.landsat_reflectance(raw["red"]))
        stats = core.site_stats(lst, ndvi, clear, pixels, args.min_ring_valid).set_index("site")
        when = pd.Timestamp(scene_items[0].datetime)
        air = fetch_air(stations["id"].tolist(), when, args.cache)
        for row in stations.itertuples():
            series = air[air["station"] == row.id]
            ring, pixel = f"{row.id}|ring", f"{row.id}|pixel"
            rows.append(
                {
                    "scene": scene,
                    "overpass_utc": f"{when:%Y-%m-%dT%H:%M}",
                    "station": row.name,
                    "station_id": row.id,
                    "air_c": ss.interp_at(series["time"], series["air_c"], when),
                    "lst_pixel_c": stats["lst_mean_c"].get(pixel, np.nan),
                    "lst_ring_c": stats["lst_mean_c"].get(ring, np.nan),
                    "ndvi_ring": stats["ndvi_landsat"].get(ring, np.nan),
                }
            )
    table = pd.DataFrame(rows)
    table = table.dropna(subset=["air_c"]).dropna(subset=["lst_pixel_c", "lst_ring_c"], how="all")
    table = table.merge(stations[["id", "station_class"]], left_on="station_id", right_on="id")
    table = table.drop(columns="id")
    table = ss.one_series_per_station(table)
    counts = table.groupby("station")["scene"].count()
    keep = counts[counts >= MIN_SCENES_PER_STATION].index
    dropped = sorted(set(stations["name"]) - set(keep))
    table = table[table["station"].isin(keep)].reset_index(drop=True)

    # Summaries, for the 300 m surroundings (main) and for the single pixel
    main_id = table.groupby("station")["station_id"].agg(lambda s: s.value_counts().index[0])
    meta = stations.rename(columns={"name": "station"})[
        ["station", "id", "altitude", "green_share", "station_class"]
    ]
    meta = meta[meta["id"] == meta["station"].map(main_id)]
    summary = meta.merge(ss.station_summary(table, "lst_ring_c"), on="station")
    pixel = ss.station_summary(table, "lst_pixel_c")[["station", "scenes", "gap_median_c"]]
    summary = summary.merge(
        pixel.rename(columns={"scenes": "scenes_pixel", "gap_median_c": "gap_pixel_median_c"}),
        on="station", how="left",
    )
    summary["ndvi_ring"] = summary["station"].map(table.groupby("station")["ndvi_ring"].median())
    summary = summary.sort_values("green_share").reset_index(drop=True)

    anomalies = ss.spatial_anomalies(table, "lst_ring_c")
    both = table.dropna(subset=["lst_ring_c", "air_c"])
    overall_gap = both["lst_ring_c"] - both["air_c"]
    low = table[table["station"].isin(summary.loc[summary["altitude"] < 300, "station"])]
    stats = {
        "overpasses_used": int(table["scene"].nunique()),
        "stations": summary["station"].tolist(),
        "stations_dropped": dropped,
        "station_scene_pairs": len(both),
        "surface_minus_air_median_c": round(float(overall_gap.median()), 2),
        "surface_minus_air_q25_c": round(float(overall_gap.quantile(0.25)), 2),
        "surface_minus_air_q75_c": round(float(overall_gap.quantile(0.75)), 2),
        "r_surface_vs_air_all_pairs": round(float(both["lst_ring_c"].corr(both["air_c"])), 3),
        "between_places_300m": ss.anomaly_relation(anomalies),
        "between_places_300m_below_300m_altitude": ss.anomaly_relation(
            ss.spatial_anomalies(low, "lst_ring_c")
        ),
        "between_places_single_pixel": ss.anomaly_relation(
            ss.spatial_anomalies(table, "lst_pixel_c")
        ),
    }
    classes = ss.class_gap(table, "lst_ring_c")

    table.round(2).to_csv(args.out / "station_scene.csv", index=False)
    summary.round(2).to_csv(args.out / "station_summary.csv", index=False)
    classes.round(2).to_csv(args.out / "urban_vs_green.csv", index=False)
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    station_plots.surface_vs_air(table, args.out / "surface_vs_air.png")
    station_plots.gap_per_station(summary, args.out / "gap_per_station.png")
    station_plots.between_places(
        anomalies.merge(meta[["station", "station_class"]], on="station"),
        stats["between_places_300m"], args.out / "between_places.png",
    )

    print(f"\nOverpasses used: {stats['overpasses_used']}; stations: {len(summary)}; "
          f"dropped: {dropped}")
    print(summary.round(2).to_markdown(index=False))
    print("\nUrban minus green stations, per scene:")
    print(classes.round(2).to_markdown(index=False))
    for key in ("between_places_300m", "between_places_300m_below_300m_altitude",
                "between_places_single_pixel"):
        print(f"\n{key}: {json.dumps(stats[key])}")
    print(f"\nWrote results to {args.out}/")


if __name__ == "__main__":
    main()
