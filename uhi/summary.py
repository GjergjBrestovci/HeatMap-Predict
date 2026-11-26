"""Summaries of the scene/site table. Pure pandas: no network, no geo libraries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import core

CLASS_ORDER = ["small", "medium", "large"]
# Subsets of parks, most important first. The headline is "away from water".
SUBSETS = ["away from water", "near water", "all parks"]


def with_baselines(raw: pd.DataFrame, baselines: dict[str, list[str]]) -> pd.DataFrame:
    """Add the difference to each baseline.

    `baselines` maps a name to its sealed reference sites. The first one is the
    main baseline: it decides which scenes are kept and fills `sealed_mean_c`
    and `delta_c`. Every other baseline adds `delta_<name>_c`, empty in scenes
    where one of its sites is masked.
    """
    names = list(baselines)
    out = core.add_sealed_delta(raw, baselines[names[0]])
    for name in names[1:]:
        other = core.add_sealed_delta(raw, baselines[name])[["scene", "site", "delta_c"]]
        out = out.merge(
            other.rename(columns={"delta_c": f"delta_{name}_c"}), on=["scene", "site"], how="left"
        )
    return out


def park_subsets(parks: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The parks split by the `near_water` flag, in `SUBSETS` order."""
    near = parks["near_water"].astype(bool)
    return {"away from water": parks[~near], "near water": parks[near], "all parks": parks}


def class_summary(table: pd.DataFrame, parks: pd.DataFrame, column: str = "delta_c") -> pd.DataFrame:
    """Per subset and size class: spread between parks and spread between scenes.

    `parks` is the site summary restricted to parks; `column` is the difference
    to summarise.
    """
    rows = table[["scene", "site", column]].dropna().rename(columns={column: "delta_c"})
    parts = []
    for label, subset in park_subsets(parks).items():
        t = rows.merge(subset[["site", "size_class"]], on="site")
        if t.empty:
            continue
        per_site = core.summarize_sites(t).merge(subset[["site", "size_class"]], on="site")
        part = core.summarize_groups(per_site, "size_class").merge(
            core.summarize_groups_per_scene(t, "size_class"), on="size_class"
        )
        part.insert(0, "subset", label)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["subset"] = pd.Categorical(out["subset"], SUBSETS, ordered=True)
    out["size_class"] = pd.Categorical(out["size_class"], CLASS_ORDER, ordered=True)
    return out.sort_values(["subset", "size_class"]).reset_index(drop=True)


def baseline_comparison(
    table: pd.DataFrame, parks: pd.DataFrame, other: str
) -> pd.DataFrame:
    """Does the headline move with the baseline?

    For parks away from water, per size class: the median difference against
    the main baseline and against baseline `other`, both on the scenes that
    have the two, so only the reference changes.
    """
    col = f"delta_{other}_c"
    dry = park_subsets(parks)["away from water"][["site", "size_class"]]
    common = table.dropna(subset=["delta_c", col]).merge(dry, on="site")
    per_site = common.groupby(["size_class", "site"])[["delta_c", col]].median().reset_index()
    out = (
        per_site.groupby("size_class")
        .agg(sites=("site", "count"), delta_median_c=("delta_c", "median"),
             **{f"delta_{other}_median_c": (col, "median")})
        .reset_index()
    )
    out["shift_c"] = out["delta_median_c"] - out[f"delta_{other}_median_c"]
    out["scenes"] = common["scene"].nunique()
    out["size_class"] = pd.Categorical(out["size_class"], CLASS_ORDER, ordered=True)
    return out.sort_values("size_class").reset_index(drop=True)


def year_summary(table: pd.DataFrame, parks: pd.DataFrame) -> pd.DataFrame:
    """Per summer and size class: median over parks of each park's median that summer."""
    t = table.merge(parks[["site", "size_class"]], on="site")
    t["year"] = t["scene"].str[:4].astype(int)
    per_site = t.groupby(["year", "size_class", "site"])["delta_c"].median().reset_index()
    out = (
        per_site.groupby(["year", "size_class"])["delta_c"]
        .agg(delta_median_c="median", sites="count")
        .reset_index()
    )
    scenes = t.groupby("year")["scene"].nunique().rename("scenes")
    return out.merge(scenes, on="year")


def relation(parks: pd.DataFrame) -> dict:
    """How the park medians relate to Sentinel-2 NDVI and to area."""
    log_area = np.log10(parks["area_ha"])
    delta = parks["delta_median_c"]
    ndvi = parks["ndvi_s2"]
    design = np.column_stack([ndvi, log_area, np.ones(len(parks))])
    coef, *_ = np.linalg.lstsq(design, delta.to_numpy(), rcond=None)
    resid = delta - design @ coef
    return {
        "parks": len(parks),
        "pearson_r_delta_vs_ndvi_s2": round(float(delta.corr(ndvi)), 2),
        "spearman_r_delta_vs_ndvi_s2": round(float(delta.rank().corr(ndvi.rank())), 2),
        "slope_c_per_0.1_ndvi": round(float(np.polyfit(ndvi, delta, 1)[0]) / 10, 2),
        "pearson_r_delta_vs_log10_area": round(float(delta.corr(log_area)), 2),
        "spearman_r_delta_vs_area": round(float(delta.rank().corr(parks["area_ha"].rank())), 2),
        "slope_c_per_10x_area": round(float(np.polyfit(log_area, delta, 1)[0]), 2),
        "pearson_r_ndvi_s2_vs_log10_area": round(float(ndvi.corr(log_area)), 2),
        "joint_fit_c_per_0.1_ndvi": round(float(coef[0]) / 10, 2),
        "joint_fit_c_per_10x_area": round(float(coef[1]), 2),
        "joint_fit_r2": round(float(1 - resid.var() / delta.var()), 2),
    }
