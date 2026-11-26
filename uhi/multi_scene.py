"""Multi-scene analysis: Vienna parks vs sealed areas over all clear summer Landsat scenes.

Usage (from the repository root):
    python -m uhi.multi_scene [--years 2013-2026] [--out results/multi_scene]

Downloads are cached in data/cache (not committed); a second run needs no network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.features import geometry_mask

from . import core, data, plots

MIN_SCENES_PER_SITE = 10  # sites seen less often are left out of the summaries


def scene_site_table(passes, grid, cache, site_pixels, city, min_city_clear, min_site_valid):
    """Per overpass and site statistics, plus a log of what happened to each overpass."""
    rows, log = [], []
    for scene, items in passes.items():
        raw = data.load_landsat_pass(items, grid, cache)
        clear = core.landsat_clear(raw["qa_pixel"])
        city_clear = float(clear[city].mean())
        entry = {"scene": scene, "city_clear_frac": round(city_clear, 3), "status": "used"}
        log.append(entry)
        if city_clear < min_city_clear:
            entry["status"] = "city too cloudy or not covered"
            continue
        lst = core.landsat_st_celsius(raw["lwir11"])
        ndvi = core.ndvi(
            core.landsat_reflectance(raw["nir08"]), core.landsat_reflectance(raw["red"])
        )
        stats = core.site_stats(lst, ndvi, clear, site_pixels, min_site_valid)
        stats.insert(0, "scene", scene)
        stats.insert(1, "date", scene[:10])
        rows.append(stats)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(log)


def year_summary(table: pd.DataFrame, parks: pd.DataFrame) -> pd.DataFrame:
    """Per summer and size class: median over parks of each park's median that summer."""
    t = table.merge(parks[["site", "size_class"]], on="site")
    t["year"] = t["date"].str[:4].astype(int)
    per_site = t.groupby(["year", "size_class", "site"])["delta_c"].median().reset_index()
    out = (
        per_site.groupby(["year", "size_class"])["delta_c"]
        .agg(delta_median_c="median", sites="count")
        .reset_index()
    )
    scenes = t.groupby("year")["scene"].nunique().rename("scenes")
    return out.merge(scenes, on="year")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", default="2013-2026", help="first-last summer")
    parser.add_argument("--cache", default=Path("data/cache"), type=Path)
    parser.add_argument("--out", default=Path("results/multi_scene"), type=Path)
    parser.add_argument("--max-scene-cloud", default=70.0, type=float,
                        help="pre-filter on the cloud cover of the whole Landsat scene (%%)")
    parser.add_argument("--min-city-clear", default=0.8, type=float,
                        help="minimum clear fraction inside the city boundary")
    parser.add_argument("--min-site-valid", default=0.8, type=float,
                        help="minimum fraction of a site's pixels that must be usable")
    parser.add_argument("--s2-dates", default=12, type=int,
                        help="number of clear Sentinel-2 dates in the NDVI median")
    parser.add_argument("--refresh", action="store_true", help="repeat the catalogue searches")
    args = parser.parse_args()
    first, last = (int(y) for y in args.years.split("-"))
    years = range(first, last + 1)
    args.out.mkdir(parents=True, exist_ok=True)

    # Sites and grids
    sites = data.build_sites(args.cache)
    grid = data.landsat_grid()
    site_pixels = data.rasterize_sites(sites, grid)
    city_geom = data.wfs("LANDESGRENZEOGD", args.cache).geometry
    city = geometry_mask(city_geom, grid.shape, grid.transform, invert=True)
    sealed_names = sites.loc[sites["kind"] == "sealed", "site"].tolist()

    # Landsat: every summer overpass, then filter by what the QA mask says locally
    items = data.landsat_items(args.cache, years, args.max_scene_cloud, args.refresh)
    data.fetch_landsat(items, grid, args.cache)
    passes = data.landsat_passes(items)
    table, log = scene_site_table(
        passes, grid, args.cache, site_pixels, city, args.min_city_clear, args.min_site_valid
    )
    with_ref = core.add_sealed_delta(table, sealed_names)
    no_ref = (log["status"] == "used") & ~log["scene"].isin(with_ref["scene"])
    log.loc[no_ref, "status"] = "a sealed reference site is masked"
    table = with_ref
    n_scenes = table["scene"].nunique()

    # Sentinel-2 NDVI at 10 m over the area that holds the sites
    minx, miny, maxx, maxy = sites.total_bounds
    s2_grid = data.make_grid((minx - 200, miny - 200, maxx + 200, maxy + 200), 10.0, 0.0)
    s2_years = range(max(first, 2022), last + 1)  # one reflectance convention, recent state
    s2_ndvi, s2_dates = data.s2_summer_ndvi(
        s2_grid, args.cache, s2_years, args.s2_dates, refresh=args.refresh
    )
    s2_pixels = data.rasterize_sites(sites, s2_grid)
    flat = s2_ndvi.ravel()
    with np.errstate(all="ignore"):
        sites["ndvi_s2"] = [float(np.nanmean(flat[s2_pixels[s]])) for s in sites["site"]]

    # Summaries
    summary = sites.drop(columns="geometry").merge(core.summarize_sites(table), on="site")
    summary = summary.merge(
        table.groupby("site")
        .agg(lst_median_c=("lst_mean_c", "median"), ndvi_landsat=("ndvi_landsat", "median"),
             pixels_30m=("pixels", "max"))
        .reset_index(),
        on="site",
    )
    rare = summary["scenes"] < MIN_SCENES_PER_SITE
    dropped = sorted(set(sites["site"]) - set(summary.loc[~rare, "site"]))
    summary = summary[~rare].sort_values(["kind", "area_ha"], ascending=[True, False])
    parks = summary[summary["kind"] == "park"]
    table = table[table["site"].isin(summary["site"])]

    # Per size class, for all parks and for parks away from water
    class_tables = []
    for label, subset in (("all parks", parks), ("away from water", parks[~parks["near_water"]])):
        rows = table.merge(subset[["site", "size_class"]], on="site")
        part = core.summarize_groups(subset, "size_class").merge(
            core.summarize_groups_per_scene(rows, "size_class"), on="size_class"
        )
        part.insert(0, "subset", label)
        class_tables.append(part)
    classes = pd.concat(class_tables, ignore_index=True)
    classes["size_class"] = pd.Categorical(classes["size_class"], plots.CLASS_ORDER, ordered=True)
    classes = classes.sort_values(["subset", "size_class"]).reset_index(drop=True)
    years_df = year_summary(table, parks)

    def relation(p: pd.DataFrame) -> dict:
        """How the park medians relate to NDVI and to area."""
        log_area = np.log10(p["area_ha"])
        delta = p["delta_median_c"]
        design = np.column_stack([p["ndvi_s2"], log_area, np.ones(len(p))])
        coef, *_ = np.linalg.lstsq(design, delta.to_numpy(), rcond=None)
        resid = delta - design @ coef
        return {
            "parks": len(p),
            "pearson_r_delta_vs_ndvi_s2": round(float(delta.corr(p["ndvi_s2"])), 2),
            "spearman_r_delta_vs_ndvi_s2": round(float(delta.rank().corr(p["ndvi_s2"].rank())), 2),
            "slope_c_per_0.1_ndvi": round(float(np.polyfit(p["ndvi_s2"], delta, 1)[0]) / 10, 2),
            "pearson_r_delta_vs_log10_area": round(float(delta.corr(log_area)), 2),
            "spearman_r_delta_vs_area": round(float(delta.rank().corr(p["area_ha"].rank())), 2),
            "slope_c_per_10x_area": round(float(np.polyfit(log_area, delta, 1)[0]), 2),
            "pearson_r_ndvi_s2_vs_log10_area": round(float(p["ndvi_s2"].corr(log_area)), 2),
            "joint_fit_c_per_0.1_ndvi": round(float(coef[0]) / 10, 2),
            "joint_fit_c_per_10x_area": round(float(coef[1]), 2),
            "joint_fit_r2": round(float(1 - resid.var() / delta.var()), 2),
        }

    dry = parks[~parks["near_water"]]
    wet = parks[parks["near_water"]]
    fit_dry = np.polyfit(dry["ndvi_s2"], dry["delta_median_c"], 1)
    stats = {
        "landsat_items_considered": len(items),
        "overpasses_considered": len(passes),
        "overpasses_used": int(n_scenes),
        "overpasses_skipped": log.loc[log["status"] != "used", "status"].value_counts().to_dict(),
        "first_scene": table["date"].min(),
        "last_scene": table["date"].max(),
        "parks": len(parks),
        "parks_near_water": len(wet),
        "sites_dropped_too_few_scenes": dropped,
        "sentinel2_dates": s2_dates,
        "near_water_median_delta_c": round(float(wet["delta_median_c"].median()), 2),
        "away_from_water_median_delta_c": round(float(dry["delta_median_c"].median()), 2),
        "relation_all_parks": relation(parks),
        "relation_away_from_water": relation(dry),
    }

    # Outputs
    sealed_ref = table.groupby("scene")["sealed_mean_c"].first().round(2)
    log = log.merge(sealed_ref, on="scene", how="left")
    table.drop(columns=["date", "sealed_mean_c"]).round(
        {"lst_mean_c": 2, "ndvi_landsat": 2, "delta_c": 2, "valid_frac": 2}
    ).to_csv(args.out / "scene_site.csv", index=False)
    summary.round(2).to_csv(args.out / "site_summary.csv", index=False)
    classes.round(2).to_csv(args.out / "class_summary.csv", index=False)
    years_df.round(2).to_csv(args.out / "year_summary.csv", index=False)
    log.to_csv(args.out / "scenes.csv", index=False)
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    plots.cooling_by_size(parks, n_scenes, args.out / "cooling_by_size.png")
    plots.cooling_vs_ndvi(
        summary, fit_dry, stats["relation_away_from_water"]["pearson_r_delta_vs_ndvi_s2"],
        args.out / "cooling_vs_ndvi.png",
    )
    plots.cooling_per_year(years_df, args.out / "cooling_per_year.png")

    print(f"\nOverpasses: {len(passes)} considered, {n_scenes} used "
          f"({stats['first_scene']} to {stats['last_scene']})")
    for reason, n in stats["overpasses_skipped"].items():
        print(f"  skipped, {reason}: {n}")
    print(f"Sites: {len(parks)} parks ({len(wet)} near water) + {len(sealed_names)} sealed; "
          f"{len(dropped)} dropped for fewer than {MIN_SCENES_PER_SITE} scenes")
    print("\nDifference to the sealed sites per park size class (°C):")
    print(classes.round(2).to_markdown(index=False))
    print("\nSealed sites:")
    print(summary.loc[summary["kind"] == "sealed",
                      ["site", "scenes", "delta_median_c", "delta_q25_c", "delta_q75_c", "ndvi_s2"]]
          .round(2).to_markdown(index=False))
    for label in ("relation_all_parks", "relation_away_from_water"):
        print(f"\n{label}: {json.dumps(stats[label])}")
    print(f"\nWrote results to {args.out}/")


if __name__ == "__main__":
    main()
