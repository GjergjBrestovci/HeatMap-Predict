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

from . import core, data, plots, summary

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

    # Landsat: every summer overpass, cached as a window over Vienna
    grid = data.landsat_grid()
    items = data.landsat_items(args.cache, years, args.max_scene_cloud, args.refresh)
    data.fetch_landsat(items, grid, args.cache)
    passes = data.landsat_passes(items)
    gap_mask = data.landsat_gap_mask(passes, grid, args.cache)

    # Sites
    sites = data.build_sites(args.cache, grid, gap_mask)
    site_pixels = data.rasterize_sites(sites, grid)
    city_geom = data.wfs("LANDESGRENZEOGD", args.cache).geometry
    city = geometry_mask(city_geom, grid.shape, grid.transform, invert=True)
    sealed = sites[sites["kind"] == "sealed"]
    baselines = {
        data.BASELINE_CITYWIDE: sealed["site"].tolist(),
        data.BASELINE_WEST: sealed.loc[sealed["baseline"] == data.BASELINE_WEST, "site"].tolist(),
    }
    sealed_names = baselines[data.BASELINE_CITYWIDE]

    # Filter by what the QA mask says locally, then compare with the sealed reference
    raw_table, log = scene_site_table(
        passes, grid, args.cache, site_pixels, city, args.min_city_clear, args.min_site_valid
    )
    table = summary.with_baselines(raw_table, baselines)
    no_ref = (log["status"] == "used") & ~log["scene"].isin(table["scene"])
    log.loc[no_ref, "status"] = "a sealed reference site is masked"
    n_scenes = table["scene"].nunique()
    # The original baseline on all of its own scenes, to compare with the earlier result.
    west_table = summary.with_baselines(
        raw_table, {data.BASELINE_WEST: baselines[data.BASELINE_WEST]}
    )

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
    site_summary = sites.drop(columns="geometry").merge(core.summarize_sites(table), on="site")
    site_summary = site_summary.merge(
        table.groupby("site")
        .agg(delta_west_median_c=("delta_west_c", "median"),
             lst_median_c=("lst_mean_c", "median"), ndvi_landsat=("ndvi_landsat", "median"),
             pixels_30m=("pixels", "max"))
        .reset_index(),
        on="site",
    )
    rare = site_summary["scenes"] < MIN_SCENES_PER_SITE
    dropped = sorted(set(sites["site"]) - set(site_summary.loc[~rare, "site"]))
    site_summary = site_summary[~rare].sort_values(["kind", "area_ha"], ascending=[True, False])
    parks = site_summary[site_summary["kind"] == "park"]
    table = table[table["site"].isin(site_summary["site"])]

    subsets = summary.park_subsets(parks)
    dry, wet = subsets["away from water"], subsets["near water"]
    classes = summary.class_summary(table, parks)
    classes.insert(0, "baseline", data.BASELINE_CITYWIDE)
    classes_west = summary.class_summary(west_table, parks)
    classes_west.insert(0, "baseline", f"{data.BASELINE_WEST} (its own scenes)")
    comparison = summary.baseline_comparison(table, parks, data.BASELINE_WEST)
    years_df = summary.year_summary(table, dry)

    fit_dry = np.polyfit(dry["ndvi_s2"], dry["delta_median_c"], 1)
    headline = classes[classes["subset"] == "away from water"]
    stats = {
        "landsat_items_considered": len(items),
        "overpasses_considered": len(passes),
        "overpasses_used": int(n_scenes),
        "overpasses_used_west_baseline": int(west_table["scene"].nunique()),
        "overpasses_skipped": log.loc[log["status"] != "used", "status"].value_counts().to_dict(),
        "first_scene": table["date"].min(),
        "last_scene": table["date"].max(),
        "sealed_sites": baselines,
        "parks": len(parks),
        "parks_near_water": len(wet),
        "sites_dropped_too_few_scenes": dropped,
        "sentinel2_dates": s2_dates,
        "headline_away_from_water_median_delta_c": {
            str(r.size_class): round(float(r.delta_median_c), 2) for r in headline.itertuples()
        },
        "near_water_median_delta_c": round(float(wet["delta_median_c"].median()), 2),
        "away_from_water_median_delta_c": round(float(dry["delta_median_c"].median()), 2),
        "relation_away_from_water": summary.relation(dry),
        "relation_all_parks": summary.relation(parks),
    }

    # Outputs
    sealed_ref = table.groupby("scene")["sealed_mean_c"].first().round(2)
    log = log.merge(sealed_ref, on="scene", how="left")
    table.drop(columns=["date", "sealed_mean_c"]).round(
        {"lst_mean_c": 2, "ndvi_landsat": 2, "delta_c": 2, "delta_west_c": 2, "valid_frac": 2}
    ).to_csv(args.out / "scene_site.csv", index=False)
    site_summary.round(2).to_csv(args.out / "site_summary.csv", index=False)
    pd.concat([classes, classes_west]).round(2).to_csv(args.out / "class_summary.csv", index=False)
    comparison.round(2).to_csv(args.out / "baseline_comparison.csv", index=False)
    years_df.round(2).to_csv(args.out / "year_summary.csv", index=False)
    log.to_csv(args.out / "scenes.csv", index=False)
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    plots.cooling_by_size(parks, n_scenes, args.out / "cooling_by_size.png")
    plots.cooling_vs_ndvi(
        site_summary, fit_dry, stats["relation_away_from_water"]["pearson_r_delta_vs_ndvi_s2"],
        args.out / "cooling_vs_ndvi.png",
    )
    plots.cooling_per_year(years_df, args.out / "cooling_per_year.png")

    print(f"\nOverpasses: {len(passes)} considered, {n_scenes} used "
          f"({stats['first_scene']} to {stats['last_scene']})")
    for reason, n in stats["overpasses_skipped"].items():
        print(f"  skipped, {reason}: {n}")
    print(f"Sites: {len(parks)} parks ({len(wet)} near water) + {len(sealed_names)} sealed; "
          f"{len(dropped)} dropped for fewer than {MIN_SCENES_PER_SITE} scenes")
    print("\nDifference to the citywide sealed sites per park size class (°C):")
    print(classes.round(2).to_markdown(index=False))
    print("\nDoes the baseline move the headline? Parks away from water, same scenes:")
    print(comparison.round(2).to_markdown(index=False))
    print("\nSealed sites:")
    print(site_summary.loc[site_summary["kind"] == "sealed",
                           ["site", "baseline", "area_ha", "scenes", "delta_median_c",
                            "delta_q25_c", "delta_q75_c", "ndvi_s2"]]
          .round(2).to_markdown(index=False))
    for label in ("relation_away_from_water", "relation_all_parks"):
        print(f"\n{label}: {json.dumps(stats[label])}")
    print(f"\nWrote results to {args.out}/")


if __name__ == "__main__":
    main()
