"""Warming split: how much of Vienna's summer warming since 1981 is regional, how much is the city.

Usage (from the repository root):
    python -m uhi.warming_split [--years 1981-2026] [--out results/warming_split]

Three parts:
1. Air: June-August trends, urban stations against rural ones: homogenised
   HISTALP series as the main split, the raw GeoSphere station series and the
   SPARTACUS grid as cross-checks.
2. Land use: which pixels changed from green to sealed between the 17 editions
   of the Realnutzungskartierung; GHSL built-up surface as a cross-check.
3. Surface: Landsat 4/5/7/8/9 summer scenes, always as differences within one
   scene: sealed minus green over time, and converted land before and after.

Meant to run in GitHub Actions (see .github/workflows/analysis.yml): the first
run downloads about 4 GB into data/cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import geometry_mask

from . import climate, core, data, history, split_plots
from . import split_stats as st
from . import station_stats as ss

URBAN_STATIONS = ("Wien Innere Stadt", "Wien Hohe Warte")
REGION_KM = 45.0  # stations and grid points are taken from this far around the centre
RING_KM = (15.0, 40.0)  # rural ring for the gridded series
MAX_ALTITUDE_M = 300.0
MIN_SUMMERS_SHARE = 0.85  # of the summers in the period
# Built-up share within 1 km (GHSL 2020). GHSL counts building footprints only,
# so even the dense inner city stays below one half.
RURAL_MAX_BUILT = 0.10  # for a rural station
RING_MAX_BUILT = 0.05  # for a grid point in the rural ring
CITY_MIN_BUILT = 0.25  # for a grid point in the built-up city
STATION_RADIUS_M = 1000.0
LAST_TM_ETM_YEAR = 2012  # Landsat 4/5/7 up to here, 8/9 afterwards
SENSORS = {"landsat-4": "TM", "landsat-5": "TM", "landsat-7": "ETM+",
           "landsat-8": "OLI/TIRS", "landsat-9": "OLI/TIRS"}
SCHEME_CHANGE = (2005, 2008)  # last edition with codes, first with class names
# Conversions are pooled by period: green up to the first year, sealed from the second.
# The first period that fits a pixel takes it.
CONVERSION_PERIODS = ((1985, 1997), (1997, 2005), (2005, 2016), (2012, 2024))
REFERENCES = ("stable_green", "stable_farmland")


def _round(obj, digits=3):
    """Round floats in nested dicts and lists; NaN becomes None for valid JSON."""
    if isinstance(obj, dict):
        return {k: _round(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, digits) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else round(float(obj), digits)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _to_lat_lon(points_xy: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points_xy:
        return []
    geo = gpd.GeoSeries(gpd.points_from_xy(*zip(*points_xy)), crs=data.CRS).to_crs("EPSG:4326")
    return [(float(p.y), float(p.x)) for p in geo]


def _built_share(built: np.ndarray, grid: data.Grid, x: float, y: float, radius: float) -> float:
    idx = ss.disc_indices(grid.transform, grid.shape, x, y, radius)
    return float(built.ravel()[idx].mean()) if len(idx) else np.nan


# ------------------------------------------------------------------------ air


def air_part(args, first: int, last: int, city_geom) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Summer air temperature per place and year, trends, and the urban/regional split."""
    centre = data._metric(data.CENTRE)
    half = REGION_KM * 1000
    region = data.make_grid(
        (centre.x - half, centre.y - half, centre.x + half, centre.y + half), 100.0, 0.0
    )
    built = history.ghsl_built_fraction(args.cache, 2020, region)
    n_summers = last - first + 1
    min_summers = int(np.ceil(MIN_SUMMERS_SHARE * n_summers))

    # Stations: low-lying, in the region, with a nearly complete record
    meta = climate.station_metadata(args.cache)
    points = gpd.GeoSeries(gpd.points_from_xy(meta["lon"], meta["lat"]), crs="EPSG:4326").to_crs(data.CRS)
    meta = meta.assign(x=points.x.to_numpy(), y=points.y.to_numpy(),
                       dist_km=points.distance(centre).to_numpy() / 1000,
                       in_city=points.within(city_geom).to_numpy())
    near = meta[(meta["dist_km"] <= REGION_KM) & (meta["altitude"] < MAX_ALTITUDE_M)]
    monthly = climate.fetch_station_monthly(near["id"].tolist(), first, last, args.cache)
    summers = {m: st.summer_means(monthly, m, "station") for m in climate.MEASURES}
    stations = climate.pick_series(monthly, near, summers["mean_c"], min_summers)
    stations["built_share_1km"] = [
        _built_share(built, region, r.x, r.y, STATION_RADIUS_M) for r in stations.itertuples()
    ]
    stations["role"] = "not used"
    stations.loc[stations["name"].isin(URBAN_STATIONS), "role"] = "urban"
    rural = ~stations["in_city"] & (stations["built_share_1km"] < RURAL_MAX_BUILT)
    stations.loc[rural & (stations["role"] != "urban"), "role"] = "rural"
    urban_names = [n for n in URBAN_STATIONS if n in set(stations["name"])]
    rural_names = stations.loc[stations["role"] == "rural", "name"].tolist()
    if not urban_names or not rural_names:
        raise SystemExit(f"Need urban and rural stations; found urban {urban_names}, rural {rural_names}")
    id_to_name = dict(zip(stations["id"], stations["name"]))

    # SPARTACUS: a rural ring around the city and the built-up city itself
    ring_xy = []
    for x, y in climate.lattice(region.bounds, args.ring_spacing_km * 1000):
        d = np.hypot(x - centre.x, y - centre.y) / 1000
        if RING_KM[0] <= d <= RING_KM[1] and _built_share(built, region, x, y, STATION_RADIUS_M) < RING_MAX_BUILT:
            ring_xy.append((x, y))
    city_bounds = tuple(city_geom.bounds)
    city_share = {
        (x, y): _built_share(built, region, x, y, STATION_RADIUS_M)
        for x, y in climate.lattice(city_bounds, 2000.0) if city_geom.contains(data.Point(x, y))
    }
    city_xy = [xy for xy, share in city_share.items() if share >= CITY_MIN_BUILT]
    print(f"Grid points: {len(ring_xy)} in the rural ring, {len(city_xy)} of {len(city_share)} "
          f"city points with at least {CITY_MIN_BUILT:.0%} built-up "
          f"(highest {max(city_share.values(), default=np.nan):.0%})")
    grids = {}
    for tag, xy in (("SPARTACUS rural ring", ring_xy), ("SPARTACUS built-up city", city_xy)):
        if not xy:
            raise SystemExit(f"{tag}: no grid points selected")
        raw = climate.fetch_spartacus(_to_lat_lon(xy), first, last, args.cache, tag.split()[1])
        raw["cell"] = raw["lat"].round(4).astype(str) + "," + raw["lon"].round(4).astype(str)
        grids[tag] = raw

    # One table: place x year for each measure
    wide, long_rows = {}, []
    for m in climate.MEASURES:
        s = summers[m][summers[m]["station"].isin(id_to_name)]
        table = s.assign(place=s["station"].map(id_to_name)).pivot(index="year", columns="place", values=m)
        for tag, raw in grids.items():
            cells = st.summer_means(raw, m, "cell")
            complete = cells.groupby("year")[m].agg(["mean", "count"])
            table[tag] = complete.loc[complete["count"] == complete["count"].max(), "mean"]
        wide[m] = table.sort_index()
    kinds = {n: f"station, {r}" for n, r in zip(stations["name"], stations["role"])}
    kinds.update({tag: "grid" for tag in grids})
    for place in wide["mean_c"].columns:
        for year in wide["mean_c"].index:
            row = {"place": place, "kind": kinds[place], "year": int(year)}
            row.update({m: wide[m].loc[year, place] if place in wide[m] else np.nan for m in climate.MEASURES})
            long_rows.append(row)
    air = pd.DataFrame(long_rows).dropna(subset=list(climate.MEASURES), how="all")

    trend_rows = []
    for place in wide["mean_c"].columns:
        for m in climate.MEASURES:
            series = wide[m][place].dropna()
            t = st.linear_trend(series.index, series.to_numpy())
            t.update({"place": place, "kind": kinds[place], "measure": m,
                      "decade_change": st.decade_change(series.index, series.to_numpy())})
            trend_rows.append(t)
    trends = pd.DataFrame(trend_rows)[
        ["place", "kind", "measure", "n", "first_year", "last_year", "slope_per_decade",
         "ci95_per_decade", "change", "change_ci95", "decade_change"]
    ]

    # Homogenised series: the main split, because the raw series contain station moves
    histalp = climate.fetch_histalp(first, last, args.cache)
    histalp = histalp.loc[first:last]
    h_urban = [n for n, v in climate.HISTALP_STATIONS.items() if v[2] == "urban"]
    h_rural = [n for n, v in climate.HISTALP_STATIONS.items() if v[2] == "rural"]
    homogenised = st.split_warming(histalp, h_urban, h_rural)
    homogenised["urban_extra_per_decade_against_each_rural_station"] = {
        r: st.split_warming(histalp, h_urban, [r])["urban_extra"]["slope_per_decade"] for r in h_rural
    }

    ring = "SPARTACUS rural ring"
    splits = {m: st.split_warming(wide[m], urban_names, rural_names) for m in climate.MEASURES}
    per_urban = {
        u: {m: st.split_warming(wide[m], [u], rural_names) for m in climate.MEASURES}
        for u in urban_names
    }
    against_each = {
        r: st.split_warming(wide["mean_c"], urban_names, [r])["urban_extra"]["slope_per_decade"]
        for r in rural_names
    }
    stats = {
        "period": [first, last],
        "urban_stations": urban_names,
        "rural_stations": rural_names,
        "split_homogenised": homogenised,
        "split": splits,
        "split_per_urban_station": per_urban,
        "split_against_grid_ring": {m: st.split_warming(wide[m], urban_names, [ring]) for m in climate.MEASURES},
        "grid_city_against_grid_ring": {
            m: st.split_warming(wide[m], ["SPARTACUS built-up city"], [ring]) for m in climate.MEASURES
        },
        "urban_extra_per_decade_against_each_rural_station": against_each,
        "grid_points": {"rural_ring": len(ring_xy), "built_up_city": len(city_xy)},
        "station_site_changes": {
            r.name: r.site_change_years for r in stations.itertuples() if r.site_change_years
        },
    }
    tables = {
        "air_summer": air,
        "air_trends": trends,
        "stations": stations.drop(columns=["x", "y"]),
        "air_wide_mean": wide["mean_c"],
        "air_homogenised": histalp.rename_axis("year").reset_index(),
    }
    return stats, tables


# -------------------------------------------------------------------- surface


def build_zones(stack: np.ndarray, years: list[int], inside: np.ndarray, gap: np.ndarray,
                args) -> tuple[dict, pd.DataFrame, dict]:
    """Pixel indices per zone: the stable classes and one zone per conversion period.

    Converted pixels are pooled by period (`CONVERSION_PERIODS`): a pixel belongs
    to the first period that holds both its last green and its first sealed
    edition. Flips exactly across the change of scheme form a zone of their own.
    Pixels that never have a temperature in the Landsat product are left out.
    """
    traj = st.trajectories(stack)
    masks = {
        name: st.erode(traj[name] & inside, args.erode_stable)
        for name in ("stable_green", "stable_farmland", "stable_forest", "stable_built")
    }
    zones = {name: np.flatnonzero(mask & ~gap) for name, mask in masks.items()}
    year_of = np.array(years)
    converted = (traj["last_green"] >= 0) & inside
    last_green = np.where(converted, year_of[traj["last_green"]], 0)
    first_built = np.where(converted, year_of[traj["first_built"]], 0)
    scheme = converted & (last_green == SCHEME_CHANGE[0]) & (first_built == SCHEME_CHANGE[1])
    todo = converted & ~scheme
    groups = []
    for start, end in CONVERSION_PERIODS:
        raw = todo & (last_green >= start) & (first_built <= end)
        todo &= ~raw
        groups.append((start, end, raw, False))
    groups.append((*SCHEME_CHANGE, scheme, True))
    rows = []
    for start, end, raw, is_scheme in groups:
        if not raw.any():
            continue
        eroded = st.erode(raw, args.erode_cohort)
        mask = eroded & ~gap
        name = f"conv_{start}_{end}"
        origin = np.bincount(
            stack[traj["last_green"][raw], *np.nonzero(raw)], minlength=len(st.GROUP_NAMES)
        )
        rows.append({
            "cohort": name,
            "last_green_year": start,
            "first_built_year": end,
            "area_ha": raw.sum() * args.pixel_ha,
            "no_temperature_share": float(gap[raw].mean()),
            "pixels": int(mask.sum()),
            "scheme_change": is_scheme,
            "farmland_share": origin[st.FARMLAND] / raw.sum(),
            "forest_share": origin[st.FOREST] / raw.sum(),
        })
        if rows[-1]["pixels"] >= args.min_cohort_pixels:
            zones[name] = np.flatnonzero(mask)
    cohorts = pd.DataFrame(rows)
    masks["converted"] = converted
    masks["converted_outside_periods"] = todo
    return zones, cohorts, masks


def scene_table(passes, grid, cache, zones, inside, args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Mean surface temperature and NDVI per zone for every usable overpass, and a log."""
    lst_rows, ndvi_rows, log = [], [], []
    for scene, items in passes.items():
        raw = data.load_landsat_pass(items, grid, cache)
        qa = raw["qa_pixel"]
        covered = (qa != 0) & ((qa & 1) == 0)
        clear = core.landsat_clear(qa)
        covered_frac = float(covered[inside].mean())
        clear_frac = float(clear[inside].sum() / max(covered[inside].sum(), 1))
        props = items[0].properties
        when = items[0].datetime
        entry = {
            "scene": scene, "date": f"{when:%Y-%m-%d}", "year": when.year,
            "time_utc": f"{when:%H:%M}", "platform": props.get("platform", ""),
            "sensor": SENSORS.get(props.get("platform", ""), "other"),
            "sun_elevation": props.get("view:sun_elevation", np.nan),
            "city_covered_frac": round(covered_frac, 3), "city_clear_frac": round(clear_frac, 3),
            "status": "used",
        }
        log.append(entry)
        if covered_frac < args.min_city_covered:
            entry["status"] = "city not covered"
            continue
        if clear_frac < args.min_city_clear:
            entry["status"] = "city too cloudy"
            continue
        lst = core.landsat_st_celsius(raw["lwir11"])
        ndvi = core.ndvi(core.landsat_reflectance(raw["nir08"]), core.landsat_reflectance(raw["red"]))
        stats = core.site_stats(lst, ndvi, clear, zones, args.min_zone_valid).set_index("site")
        if "stable_green" not in stats.index or "stable_built" not in stats.index:
            entry["status"] = "a stable zone is masked"
            continue
        lst_rows.append({"scene": scene, **stats["lst_mean_c"].to_dict()})
        ndvi_rows.append({"scene": scene, **stats["ndvi_landsat"].to_dict()})
    log = pd.DataFrame(log)
    keys = ["scene", "year", "sensor", "sun_elevation"]
    lst = log[keys].merge(pd.DataFrame(lst_rows, columns=["scene", *zones]), on="scene")
    ndvi = log[keys].merge(pd.DataFrame(ndvi_rows, columns=["scene", *zones]), on="scene")
    return lst, ndvi, log


def surface_part(args, first: int, last: int, city_geom) -> tuple[dict, dict[str, pd.DataFrame]]:
    grid = data.landsat_grid()
    args.pixel_ha = abs(grid.transform.a * grid.transform.e) / 1e4
    inside = geometry_mask([city_geom], grid.shape, grid.transform, invert=True)

    # Land use over all editions
    stack, years, crosswalk = history.landuse_stack(args.cache, grid)
    landuse = history.area_by_edition(stack, years, inside, args.pixel_ha)

    # GHSL: built-up area in the city per epoch, and its change in the zones
    ghsl_rows, ghsl = [], {}
    for epoch in history.GHSL_EPOCHS:
        ghsl[epoch] = history.ghsl_built_fraction(args.cache, epoch, grid)
        ghsl_rows.append({"year": epoch, "ghsl_built_ha": float(ghsl[epoch][inside].sum() * args.pixel_ha)})
    ghsl_table = pd.DataFrame(ghsl_rows)
    # Landsat: thermal sensors change in 2013
    tm_years = range(max(first, 1982), min(last, LAST_TM_ETM_YEAR) + 1)
    oli_years = range(max(first, LAST_TM_ETM_YEAR + 1), last + 1)
    items = data.landsat_items(args.cache, tm_years, args.max_scene_cloud, args.refresh, data.LANDSAT_457)
    items += data.landsat_items(args.cache, oli_years, args.max_scene_cloud, args.refresh)
    data.fetch_landsat(items, grid, args.cache)
    passes = data.landsat_passes(items)
    gap = data.landsat_gap_mask(passes, grid, args.cache)
    zones, cohorts, masks = build_zones(stack, years, inside, gap, args)
    e0, e1 = history.GHSL_EPOCHS[1], history.GHSL_EPOCHS[-1]  # 1980 and 2020
    ghsl_change = {
        name: {
            f"built_share_{e0}": float(ghsl[e0][mask].mean()) if mask.any() else np.nan,
            f"built_share_{e1}": float(ghsl[e1][mask].mean()) if mask.any() else np.nan,
        }
        for name, mask in masks.items() if name in ("stable_green", "stable_built", "converted")
    }
    lst, ndvi, log = scene_table(passes, grid, args.cache, zones, inside, args)
    if lst.empty:
        raise SystemExit("No usable Landsat overpass")

    # Sealed minus green over time
    gaps = lst[["scene", "year", "sensor", "sun_elevation"]].copy()
    for ref in ("stable_green", "stable_farmland", "stable_forest"):
        gaps[f"built_minus_{ref.removeprefix('stable_')}_c"] = lst["stable_built"] - lst[ref]
    gap_cols = [c for c in gaps.columns if c.startswith("built_minus_")]
    by_year = gaps.groupby("year")[gap_cols].median()
    by_year.insert(0, "scenes", gaps.groupby("year")["scene"].count())
    by_year.insert(1, "sensors", gaps.groupby("year")["sensor"].agg(lambda s: "+".join(sorted(set(s)))))
    by_year = by_year.reset_index()
    gap_stats = {}
    for col in gap_cols:
        d = gaps.dropna(subset=[col])
        gap_stats[col] = {
            "trend_of_yearly_medians": st.linear_trend(by_year["year"], by_year[col]),
            "trend_holding_sun_elevation": st.trend_with_covariate(d["year"], d[col], d["sun_elevation"]),
            "median_by_sensor": d.groupby("sensor")[col].median().to_dict(),
            "scenes_by_sensor": d.groupby("sensor")[col].count().to_dict(),
        }

    # Converted land, before and after
    measured = cohorts[cohorts["cohort"].isin(lst.columns)]
    effects = []
    for ref in REFERENCES:
        e = st.cohort_effects(lst, measured, ref, args.min_scenes)
        n = st.cohort_effects(ndvi, measured, ref, args.min_scenes)
        e["ndvi_effect"] = n["effect_c"].to_numpy()
        effects.append(e)
    effects = pd.concat(effects, ignore_index=True)
    cohorts = cohorts.merge(
        effects.drop(columns=["last_green_year", "first_built_year", "pixels"]), on="cohort", how="left"
    )
    headline = cohorts[~cohorts["scheme_change"]]
    pooled = {}
    for ref in REFERENCES:
        h = headline[headline["reference"] == ref]
        a = cohorts[cohorts["reference"] == ref]
        pooled[ref] = {
            "effect": st.pooled_effect(h),
            "effect_same_sensor": st.pooled_effect(h, "effect_same_sensor_c"),
            "effect_including_scheme_change_cohorts": st.pooled_effect(a),
            "ndvi_effect": st.pooled_effect(h, "ndvi_effect"),
        }
    used = headline.loc[headline["effect_c"].notna() & (headline["reference"] == REFERENCES[0])]
    event = st.event_study(lst, used, REFERENCES[0], args.event_bin_years)

    stats = {
        "landuse_editions": years,
        "landuse_codes_unclear": crosswalk.loc[
            crosswalk["majority_share"] < history.CROSSWALK_MIN_SHARE, "code"
        ].tolist() if len(crosswalk) else [],
        "zone_pixels": {k: int(len(v)) for k, v in zones.items() if not k.startswith("conv_")},
        "converted_ha": float(masks["converted"].sum() * args.pixel_ha),
        "converted_ha_without_scheme_change": float(
            headline.drop_duplicates("cohort")["area_ha"].sum()
        ),
        "converted_ha_outside_periods": float(masks["converted_outside_periods"].sum() * args.pixel_ha),
        "converted_share_without_temperature": float(gap[masks["converted"]].mean())
        if masks["converted"].any() else np.nan,
        "city_share_without_temperature": float(gap[inside].mean()),
        "cohorts_found": int(cohorts["cohort"].nunique()),
        "cohorts_measured": int(measured["cohort"].nunique()),
        "ghsl_built_share_in_zones": ghsl_change,
        "landsat_items": len(items),
        "overpasses_considered": len(passes),
        "overpasses_used": len(lst),
        "overpasses_used_by_sensor": lst["sensor"].value_counts().to_dict(),
        "overpasses_skipped": log.loc[log["status"] != "used", "status"].value_counts().to_dict(),
        "first_scene": log.loc[log["status"] == "used", "date"].min(),
        "last_scene": log.loc[log["status"] == "used", "date"].max(),
        "sealed_minus_green": gap_stats,
        "converted_land": pooled,
    }
    scenes = log.merge(lst.drop(columns=["year", "sensor", "sun_elevation"]), on="scene", how="left")
    tables = {
        "landuse_by_edition": landuse.merge(ghsl_table, on="year", how="outer").sort_values("year"),
        "landuse_codes": crosswalk,
        "scenes": scenes,
        "surface_gap_by_year": by_year,
        "surface_gaps": gaps,
        "cohorts": cohorts.sort_values(["reference", "first_built_year", "last_green_year"]),
        "event_study": event,
    }
    return stats, tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", default="1981-2026", help="first-last summer")
    parser.add_argument("--cache", default=Path("data/cache"), type=Path)
    parser.add_argument("--out", default=Path("results/warming_split"), type=Path)
    parser.add_argument("--max-scene-cloud", default=70.0, type=float,
                        help="pre-filter on the cloud cover of the whole Landsat scene (%%)")
    parser.add_argument("--min-city-covered", default=0.6, type=float,
                        help="minimum share of the city with data (Landsat 7 has striped gaps)")
    parser.add_argument("--min-city-clear", default=0.8, type=float,
                        help="minimum clear share of the covered part of the city")
    parser.add_argument("--min-zone-valid", default=0.5, type=float,
                        help="minimum share of a zone's pixels that must be usable")
    parser.add_argument("--erode-stable", default=2, type=int,
                        help="pixels (30 m) dropped at the edge of the stable zones")
    parser.add_argument("--erode-cohort", default=1, type=int,
                        help="pixels (30 m) dropped at the edge of converted areas")
    parser.add_argument("--min-cohort-pixels", default=100, type=int,
                        help="fewest usable pixels for a conversion period to be measured")
    parser.add_argument("--min-scenes", default=5, type=int,
                        help="scenes needed before and after a conversion")
    parser.add_argument("--event-bin-years", default=5, type=int)
    parser.add_argument("--ring-spacing-km", default=4.0, type=float,
                        help="spacing of the grid points sampled in the rural ring")
    parser.add_argument("--refresh", action="store_true", help="repeat the catalogue searches")
    args = parser.parse_args()
    first, last = (int(y) for y in args.years.split("-"))
    args.out.mkdir(parents=True, exist_ok=True)
    city_geom = data.wfs("LANDESGRENZEOGD", args.cache).geometry.union_all()

    air_stats, air = air_part(args, first, last, city_geom)
    surface_stats, surface = surface_part(args, first, last, city_geom)

    split = air_stats["split_homogenised"]
    raw = air_stats["split"]["mean_c"]
    green = surface_stats["converted_land"]["stable_green"]
    headline = {
        "air_source": "HISTALP homogenised series",
        "summers": [split["urban"]["first_year"], split["urban"]["last_year"]],
        "urban_air_warming_c": split["urban"]["change"],
        "regional_air_warming_c": split["regional"]["change"],
        "urban_extra_air_warming_c": split["urban_extra"]["change"],
        "urban_extra_ci95_c": split["urban_extra"]["change_ci95"],
        "raw_stations_summers": [raw["urban"]["first_year"], raw["urban"]["last_year"]],
        "raw_stations_urban_air_warming_c": raw["urban"]["change"],
        "raw_stations_regional_air_warming_c": raw["regional"]["change"],
        "raw_stations_urban_extra_air_warming_c": raw["urban_extra"]["change"],
        "raw_stations_urban_extra_ci95_c": raw["urban_extra"]["change_ci95"],
        "converted_land_surface_warming_c": green["effect"]["weighted_mean_c"],
        "converted_land_surface_warming_same_sensor_c": green["effect_same_sensor"]["weighted_mean_c"],
    }
    stats = _round({"headline": headline, "air": air_stats, "surface": surface_stats})

    for name in ("air_summer", "air_trends", "stations", "air_homogenised"):
        air[name].round(3).to_csv(args.out / f"{name}.csv", index=False)
    for name in ("landuse_by_edition", "landuse_codes", "scenes", "surface_gap_by_year",
                 "cohorts", "event_study"):
        surface[name].round(3).to_csv(args.out / f"{name}.csv", index=False)
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")

    split_plots.air_trends(air["air_homogenised"], air["air_wide_mean"], air_stats,
                            args.out / "air_trends.png")
    split_plots.land_use(surface["landuse_by_edition"], args.out / "land_use.png")
    split_plots.surface_gap(surface["surface_gaps"], surface["surface_gap_by_year"],
                            args.out / "surface_gap.png")
    split_plots.converted_land(surface["event_study"], surface["cohorts"], REFERENCES[0],
                               args.event_bin_years, args.out / "converted_land.png")

    print(f"\nSummer air warming {headline['summers'][0]}-{headline['summers'][1]} "
          f"(trend line, homogenised series):")
    print(f"  urban {split['urban_places']}: {headline['urban_air_warming_c']:+.2f} °C")
    print(f"  rural {split['rural_places']}: {headline['regional_air_warming_c']:+.2f} °C")
    print(f"  urban extra: {headline['urban_extra_air_warming_c']:+.2f} "
          f"± {headline['urban_extra_ci95_c']:.2f} °C")
    print(f"Raw station series {headline['raw_stations_summers']}: "
          f"urban {air_stats['urban_stations']} {raw['urban']['change']:+.2f}, "
          f"rural {air_stats['rural_stations']} {raw['regional']['change']:+.2f}, "
          f"extra {raw['urban_extra']['change']:+.2f} ± {raw['urban_extra']['change_ci95']:.2f} °C")
    print("\nAir trends:")
    print(air["air_trends"][air["air_trends"]["measure"] == "mean_c"].round(2).to_markdown(index=False))
    print("\nStations:")
    print(air["stations"].round(2).to_markdown(index=False))
    print("\nLand use by edition (ha inside the city):")
    print(surface["landuse_by_edition"].round(0).to_markdown(index=False))
    print("\nOld land-use codes:")
    print(surface["landuse_codes"].to_markdown(index=False))
    print(f"\nOverpasses: {surface_stats['overpasses_considered']} considered, "
          f"{surface_stats['overpasses_used']} used {surface_stats['overpasses_used_by_sensor']}; "
          f"skipped {surface_stats['overpasses_skipped']}")
    print("\nSealed minus green, per year:")
    print(surface["surface_gap_by_year"].round(2).to_markdown(index=False))
    print(json.dumps(stats["surface"]["sealed_minus_green"], indent=1))
    print("\nConverted land:")
    print(surface["cohorts"].round(2).to_markdown(index=False))
    print(json.dumps(stats["surface"]["converted_land"], indent=1))
    print(f"\nWrote results to {args.out}/")


if __name__ == "__main__":
    main()
