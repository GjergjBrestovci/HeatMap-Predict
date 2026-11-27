"""Model: predict Vienna's summer surface temperature from the layout of the city.

Usage (from the repository root):
    python -m uhi.model [--years 2013-2026] [--out results/model]

Target: Landsat 8/9 surface temperature minus the city mean of the same scene,
as the median over the clear summer scenes of the last eight years, on 90 m
cells. Features: land cover, building height, tree crowns, NDVI, elevation and
distance to water, for the cell and within 150 m and 500 m. A linear model and
gradient boosting are scored on 2 km blocks the model has not seen.

Meant to run in GitHub Actions (see .github/workflows/analysis.yml): the first
run downloads the Flächenmehrzweckkarte, about 2-3 GB, in tiles.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import requests
from rasterio.features import geometry_mask
from shapely.geometry import box

from . import core, data, history, layout, model_plots, scenario
from . import features as ft
from . import model_stats as ms
from . import split_stats as st

RECENT_YEARS = 8  # the target: the last eight summers
EARLY_YEARS = 4  # the "before" state of the conversion check: the first four
HOT_DAY_C = 30.0  # daily maximum at Hohe Warte that makes a hot day
MIN_HOT_SCENES = 15
DAILY_API = "https://dataset.api.hub.geosphere.at/v1/station/historical/klima-v2-1d"
HOHE_WARTE = "105"
# Conversion check: green in both of the first two editions, sealed in both of the last two.
CHECK_EDITIONS = {2012: "REALNUT2012OGD", 2014: "REALNUT2014OGD", 2022: "REALNUT2022OGD", 2024: "REALNUT2024OGD"}
CHECK_GHSL = (2015, 2020)
HIST_GROUPS = ("built", "farmland", "forest", "green_other", "water")
DEMO_CAR_PARK = "Parkplätze u. Parkhäuser"
DEMO_FIELD = "Acker"
POCKET_PARK_M = 100.0


def _json(obj, digits=3):
    if isinstance(obj, dict):
        return {str(k): _json(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json(v, digits) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else round(float(obj), digits)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def hot_dates(cache: Path, first: int, last: int) -> set[str]:
    """Days on which Hohe Warte reached `HOT_DAY_C`."""
    path = cache / "model" / f"hohe_warte_tlmax_{first}-{last}.csv"
    if not path.exists():
        params = {"parameters": "tlmax", "station_ids": HOHE_WARTE, "start": f"{first}-06-01",
                  "end": f"{last}-08-31", "output_format": "csv"}
        resp = requests.get(DAILY_API, params=params, timeout=300)
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
    df = pd.read_csv(io.BytesIO(path.read_bytes()))
    if not {"time", "tlmax"} <= set(df.columns):
        raise SystemExit(f"Daily station CSV lacks time/tlmax; has {list(df.columns)}")
    hot = df[pd.to_numeric(df["tlmax"], errors="coerce") >= HOT_DAY_C]
    return set(hot["time"].str[:10])


def scene_stacks(passes, grid30, cache, reference, first, last, hot, min_city_clear):
    """Per-scene anomaly and NDVI maps, sorted into the periods that need them."""
    groups = {"recent": [], "early": [], "hot": []}
    ndvi = {"recent": [], "early": []}
    log = []
    for scene, items in passes.items():
        raw = data.load_landsat_pass(items, grid30, cache)
        clear = core.landsat_clear(raw["qa_pixel"])
        clear_frac = float(clear[reference].mean())
        year, date = items[0].datetime.year, f"{items[0].datetime:%Y-%m-%d}"
        used = clear_frac >= min_city_clear
        log.append({"scene": scene, "date": date, "year": year, "city_clear_frac": round(clear_frac, 3),
                    "hot_day": date in hot, "used": used})
        if not used:
            continue
        anomaly = ft.scene_anomaly(core.landsat_st_celsius(raw["lwir11"]), clear, reference)
        scene_ndvi = np.where(
            clear, core.ndvi(core.landsat_reflectance(raw["nir08"]), core.landsat_reflectance(raw["red"])),
            np.nan,
        ).astype("float32")
        if year > last - RECENT_YEARS:
            groups["recent"].append(anomaly)
            ndvi["recent"].append(scene_ndvi)
            if date in hot:
                groups["hot"].append(anomaly)
        if year < first + EARLY_YEARS:
            groups["early"].append(anomaly)
            ndvi["early"].append(scene_ndvi)
    return groups, ndvi, pd.DataFrame(log)


def fit_and_score(features: dict, names: list[str], target: np.ndarray, usable: np.ndarray,
                  folds: np.ndarray, edge: np.ndarray) -> dict:
    """Cross-validate both models on the usable cells; out-of-fold maps for the boosting model."""
    rows = np.flatnonzero((usable & np.isfinite(target)).ravel())
    X, y = ft.matrix(features, names)[rows], target.ravel()[rows]
    fold, near = folds.ravel()[rows], edge.ravel()[rows]
    out = {"rows": rows, "X": X, "y": y, "fold": fold, "edge": near, "report": {}}
    for kind in ("linear", "boosting"):
        pred, models = ms.cross_validate(X, y, fold, kind)
        out["report"][kind] = ms.cv_report(y, pred, near)
        out[f"pred_{kind}"], out[f"models_{kind}"] = pred, models
    oof = np.full(target.shape, np.nan, dtype="float32")
    oof.ravel()[rows] = out["pred_boosting"]
    out["oof_map"] = oof
    return out


def parks_check(cache: Path, cells: data.Grid, target: np.ndarray, oof: np.ndarray, min_cells: int = 3):
    """Do held-out predictions reproduce how much cooler each park is?"""
    rows = []
    for park in data.park_sites(cache).itertuples():
        mask = layout.polygon_mask(park.geometry, cells) & np.isfinite(target) & np.isfinite(oof)
        if mask.sum() >= min_cells:
            rows.append({"park": park.site, "size_class": park.size_class, "cells": int(mask.sum()),
                         "observed_c": float(target[mask].mean()), "predicted_c": float(oof[mask].mean())})
    table = pd.DataFrame(rows, columns=["park", "size_class", "cells", "observed_c", "predicted_c"])
    if len(table) < 3:
        return table, {"parks": len(table)}
    stats = ms.scores(table["observed_c"].to_numpy(), table["predicted_c"].to_numpy())
    stats = {
        "parks": len(table),
        "observed_mean_c": float(table["observed_c"].mean()),
        "predicted_mean_c": float(table["predicted_c"].mean()),
        "r": float(table["observed_c"].corr(table["predicted_c"])),
        "mae_c": stats["mae"],
        "slope_predicted_per_observed": float(np.polyfit(table["observed_c"], table["predicted_c"], 1)[0]),
    }
    return table, stats


def hist_features(groups30: np.ndarray, ghsl30: np.ndarray, ndvi30: np.ndarray, with_ndvi: bool) -> dict:
    """Features that also exist for the past: land-use groups, building footprints, Landsat NDVI."""
    out = {}
    for name in HIST_GROUPS:
        out.update(ft.multi_scale(name, (groups30 == st.GROUP_NAMES.index(name)).astype("float32")))
    out.update(ft.multi_scale("ghsl", ghsl30.astype("float32")))
    if with_ndvi:
        out.update(ft.multi_scale("ndvi_landsat", ndvi30))
    return out


def conversion_check(args, grid30, usable, folds, target_early, target_recent, ndvi_early, ndvi_recent):
    """Train on the early state with features that exist for both dates, predict the recent one.

    Each cell is predicted by the fold model that never saw its block, for both
    dates; the predicted change is the difference. It is compared with the
    observed change on cells that were built over in between.
    """
    editions = {}
    for year, name in CHECK_EDITIONS.items():
        layer = data.wfs(name, args.cache)
        editions[year] = history.rasterize_named(layer, history.edition_scheme(layer), grid30)
    years = sorted(editions)
    green = lambda a: np.isin(a, st.GREEN_GROUPS)
    converted30 = (green(editions[years[0]]) & green(editions[years[1]])
                   & (editions[years[2]] == st.BUILT) & (editions[years[3]] == st.BUILT))
    same30 = editions[years[1]] == editions[years[2]]
    converted = ft.block_mean(converted30.astype("float32")) >= 0.5
    unchanged = ft.block_mean(same30.astype("float32")) == 1.0
    ghsl = {e: history.ghsl_built_fraction(args.cache, e, grid30) for e in CHECK_GHSL}
    observed = target_recent - target_early
    rows, cells_out = {}, None
    for label, with_ndvi in (("with_ndvi", True), ("without_ndvi", False)):
        early = hist_features(editions[years[1]], ghsl[CHECK_GHSL[0]], ndvi_early, with_ndvi)
        recent = hist_features(editions[years[2]], ghsl[CHECK_GHSL[1]], ndvi_recent, with_ndvi)
        names = sorted(early)
        fit = fit_and_score(early, names, target_early, usable, folds, np.zeros_like(usable))
        idx = fit["rows"]
        X_recent = ft.matrix(recent, names)[idx]
        pred_recent = np.full(len(idx), np.nan)
        for k, model in zip(sorted(set(fit["fold"].tolist())), fit["models_boosting"]):
            pick = fit["fold"] == k
            pred_recent[pick] = model.predict(X_recent[pick])
        table = pd.DataFrame({
            "converted": converted.ravel()[idx], "unchanged": unchanged.ravel()[idx],
            "observed_change_c": observed.ravel()[idx],
            "predicted_change_c": pred_recent - fit["pred_boosting"],
        }).dropna()
        conv, same = table[table["converted"]], table[table["unchanged"] & ~table["converted"]]
        entry = {
            "model_r2_on_early_state": fit["report"]["boosting"]["all_test_cells"]["r2"],
            "converted_cells": len(conv),
            "unchanged_cells": len(same),
        }
        for name, part in (("converted", conv), ("unchanged", same)):
            if len(part) >= 2:
                entry[name] = {
                    "observed_mean_c": float(part["observed_change_c"].mean()),
                    "predicted_mean_c": float(part["predicted_change_c"].mean()),
                    "observed_median_c": float(part["observed_change_c"].median()),
                    "predicted_median_c": float(part["predicted_change_c"].median()),
                    "r": float(part["observed_change_c"].corr(part["predicted_change_c"])),
                    "mae_c": float((part["observed_change_c"] - part["predicted_change_c"]).abs().mean()),
                }
        rows[label] = entry
        if with_ndvi:
            cells_out = conv.drop(columns=["converted", "unchanged"]).reset_index(drop=True)
    rows["editions"] = years
    rows["converted_ha"] = float(converted30.sum() * 0.09)
    return rows, cells_out, converted & usable


def demo_scenarios(cache: Path) -> gpd.GeoDataFrame:
    """Example edits chosen by rule from the 2022 land-use map, one scenario per row.

    The largest car park becomes a tree park, or a lawn; a 1 ha square in its
    middle becomes a pocket park; the largest field is built over the way green
    land has been built over since 2014.
    """
    landuse = data.wfs("REALNUT2022OGD", cache)
    level3 = landuse["NUTZUNG_LEVEL3"]
    rows = []
    parks = landuse[level3 == DEMO_CAR_PARK]
    if len(parks):
        car_park = parks.geometry.loc[parks.area.idxmax()]
        c = car_park.representative_point()
        half = POCKET_PARK_M / 2
        pocket = box(c.x - half, c.y - half, c.x + half, c.y + half).intersection(car_park)
        rows += [
            ("car_park_to_tree_park", "trees", np.nan, car_park),
            ("car_park_to_lawn", "lawn", np.nan, car_park),
            ("pocket_park_in_car_park", "trees", np.nan, pocket),
        ]
    fields = landuse[level3 == DEMO_FIELD]
    if len(fields):
        rows.append(("field_to_housing", "housing", np.nan, fields.geometry.loc[fields.area.idxmax()]))
    return gpd.GeoDataFrame(rows, columns=["name", "cover", "height_m", "geometry"], crs=data.CRS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", default="2013-2026", help="first-last summer of Landsat 8/9")
    parser.add_argument("--cache", default=Path("data/cache"), type=Path)
    parser.add_argument("--out", default=Path("results/model"), type=Path)
    parser.add_argument("--max-scene-cloud", default=70.0, type=float)
    parser.add_argument("--min-city-clear", default=0.8, type=float)
    parser.add_argument("--min-scenes", default=10, type=int,
                        help="clear scenes a pixel needs for the composite")
    parser.add_argument("--min-early-scenes", default=4, type=int)
    parser.add_argument("--s2-dates", default=12, type=int)
    parser.add_argument("--hot-days", action="store_true",
                        help="also fit the model on hot days only (daily maximum of 30 °C or more)")
    parser.add_argument("--refresh", action="store_true", help="repeat the catalogue searches")
    args = parser.parse_args()
    first, last = (int(y) for y in args.years.split("-"))
    args.out.mkdir(parents=True, exist_ok=True)

    # Landsat scenes and the target
    grid30 = data.landsat_grid()
    cells = layout.cell_grid(grid30)
    items = data.landsat_items(args.cache, range(first, last + 1), args.max_scene_cloud, args.refresh)
    data.fetch_landsat(items, grid30, args.cache)
    passes = data.landsat_passes(items)
    gap = data.landsat_gap_mask(passes, grid30, args.cache)
    city_geom = data.wfs("LANDESGRENZEOGD", args.cache).geometry.union_all()
    city = geometry_mask([city_geom], grid30.shape, grid30.transform, invert=True)
    hot = hot_dates(args.cache, first, last)
    groups, ndvi_scenes, log = scene_stacks(
        passes, grid30, args.cache, city & ~gap, first, last, hot, args.min_city_clear
    )
    if len(groups["recent"]) < args.min_scenes:
        raise SystemExit(f"Only {len(groups['recent'])} clear scenes in the last {RECENT_YEARS} summers")
    target = ft.to_cells(ft.composite(groups["recent"], args.min_scenes))
    usable = ft.to_cells(np.where(city, 1.0, np.nan)) == 1.0  # most of the cell lies in the city

    # Layout and features
    s2_years = range(max(first, 2022), last + 1)
    layers = layout.build_layers(args.cache, grid30, s2_years, args.s2_dates, args.refresh)
    features = ft.build_features(layers)
    names = sorted(features)
    folds = ms.block_folds(cells.shape)
    edge = ms.near_other_fold(folds)

    # Check 1: blocks the model has not seen
    fit = fit_and_score(features, names, target, usable, folds, edge)
    importance = ms.held_out_importance(fit["models_boosting"], fit["X"], fit["y"], fit["fold"], names)
    grouped = ms.group_importance(importance)
    pd_rows = []
    for name in importance["feature"].head(6):
        column = names.index(name)
        grid = np.nanpercentile(fit["X"][:, column], np.linspace(2, 98, 13))
        curve = ms.partial_dependence(fit["models_boosting"], fit["X"], column, grid)
        pd_rows += [{"feature": name, "value": v, "prediction_c": p} for v, p in zip(grid, curve)]
    dependence = pd.DataFrame(pd_rows)

    # Check 2: parks; check 3: real conversions
    parks, parks_stats = parks_check(args.cache, cells, target, fit["oof_map"])
    target_early = ft.to_cells(ft.composite(groups["early"], args.min_early_scenes)) if groups["early"] else None
    if target_early is not None and np.isfinite(target_early).any():
        conversion_stats, conversion_cells, converted_cells = conversion_check(
            args, grid30, usable, folds, target_early, target,
            ft.composite(ndvi_scenes["early"], 1), ft.composite(ndvi_scenes["recent"], 1),
        )
    else:
        conversion_stats, conversion_cells = {"note": "no early scenes"}, pd.DataFrame()
        converted_cells = np.zeros(cells.shape, dtype=bool)

    # Optional second target: hot days only
    hot_stats = {"hot_day_scenes": len(groups["hot"]), "needed": MIN_HOT_SCENES, "fitted": False}
    if args.hot_days and len(groups["hot"]) >= MIN_HOT_SCENES:
        hot_target = ft.to_cells(ft.composite(groups["hot"], max(5, MIN_HOT_SCENES // 3)))
        hot_fit = fit_and_score(features, names, hot_target, usable, folds, edge)
        both = np.isfinite(hot_target) & np.isfinite(target) & usable
        hot_stats.update({
            "fitted": True,
            "cross_validation": hot_fit["report"],
            "r_with_main_target": float(np.corrcoef(hot_target[both], target[both])[0, 1]),
            "spread_hot_c": float(np.nanstd(hot_target[both])),
            "spread_main_c": float(np.nanstd(target[both])),
        })

    # The model that the scenario tool uses, and the examples
    held_out_mae = fit["report"]["boosting"]["away_from_training_blocks"]["mae"]
    bundle = {
        "models": fit["models_boosting"], "features": names,
        "limits": ms.training_range(fit["X"]), "held_out_mae": held_out_mae,
        "ndvi_of": ft.typical_ndvi(layers),
        "housing": ft.development_recipe(layers, converted_cells),
        "grid30": {"transform": list(grid30.transform)[:6], "width": grid30.width, "height": grid30.height},
    }
    (args.cache / "model").mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.cache / "model" / "bundle.joblib", compress=3)
    demos = demo_scenarios(args.cache)
    demo_rows = {}
    (args.out / "scenarios").mkdir(exist_ok=True)
    for demo in demos.itertuples():
        edits = demos[demos["name"] == demo.name]
        changed, summary = scenario.run(bundle, layers, edits, grid30)
        demo_rows[demo.name] = {"cover": demo.cover, **summary}
        model_plots.scenario_map(changed, edits, demo.name, args.out / "scenarios" / f"{demo.name}.png")
    if len(demos):
        demos.to_crs("EPSG:4326").to_file(args.out / "scenarios" / "examples.geojson", driver="GeoJSON")

    if "field_to_housing" in demo_rows and "with_ndvi" in conversion_stats:
        observed = conversion_stats["with_ndvi"].get("converted", {})
        conversion_stats["full_model_field_built_over"] = {
            "predicted_mean_c": demo_rows["field_to_housing"].get("inside", {}).get("mean_change_c"),
            "observed_on_real_conversions_c": observed.get("observed_mean_c"),
        }
    cover_share = {n: float(np.nanmean(features[n][usable])) for n in ft.FRACTION_NAMES}
    stats = {
        "target": {
            "scenes_recent": len(groups["recent"]), "scenes_early": len(groups["early"]),
            "years_recent": [last - RECENT_YEARS + 1, last],
            "cells_in_city": int(usable.sum()), "cells_with_target": int(len(fit["y"])),
            "spread_c": float(fit["y"].std()),
            "p05_c": float(np.percentile(fit["y"], 5)), "p95_c": float(np.percentile(fit["y"], 95)),
        },
        "features": len(names),
        "cover_share_in_city": cover_share,
        "typical_ndvi": bundle["ndvi_of"],
        "housing_recipe": bundle["housing"],
        "cross_validation": fit["report"],
        "cells_near_another_fold_share": float(fit["edge"].mean()),
        "importance_by_quantity": grouped.set_index("quantity")["mae_increase_c"].to_dict(),
        "check_parks": parks_stats,
        "check_conversions": conversion_stats,
        "hot_days": hot_stats,
        "scenarios": demo_rows,
    }
    stats = _json(stats)

    rows = []
    for kind, report in fit["report"].items():
        for subset, s in report.items():
            rows.append({"model": kind, "test_cells": subset, **s})
    pd.DataFrame(rows).round(3).to_csv(args.out / "cv_scores.csv", index=False)
    importance.round(4).to_csv(args.out / "importance.csv", index=False)
    dependence.round(4).to_csv(args.out / "partial_dependence.csv", index=False)
    parks.round(3).to_csv(args.out / "parks_check.csv", index=False)
    conversion_cells.round(3).to_csv(args.out / "conversion_check.csv", index=False)
    log.to_csv(args.out / "scenes.csv", index=False)
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    model_plots.maps(target, fit["oof_map"], usable, args.out / "maps.png")
    model_plots.importance(grouped, args.out / "importance.png")
    model_plots.dependence(dependence, args.out / "partial_dependence.png")
    model_plots.checks(parks, conversion_cells, args.out / "checks.png")

    print(json.dumps({k: stats[k] for k in ("target", "cross_validation", "check_parks",
                                             "check_conversions", "hot_days")}, indent=1))
    print(grouped.round(3).to_markdown(index=False))
    print(json.dumps(stats["scenarios"], indent=1))
    print(f"\nWrote results to {args.out}/ and the model to {args.cache / 'model'}/")


if __name__ == "__main__":
    main()
