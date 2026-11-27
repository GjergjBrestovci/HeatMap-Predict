"""Pure logic for the warming split: land-use trajectories, trends, before/after effects.

No network and no geo libraries, so everything here is tested on small arrays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Land-use groups on the pixel grid. 0 means no polygon covers the pixel.
UNMAPPED, BUILT, FARMLAND, FOREST, GREEN_OTHER, WATER, OTHER = range(7)
GROUP_NAMES = ("unmapped", "built", "farmland", "forest", "green_other", "water", "other")
GREEN_GROUPS = (FARMLAND, FOREST, GREEN_OTHER)

SUMMER_MONTHS = (6, 7, 8)


# ------------------------------------------------------------------- land use


def named_group(level1: str | None, level2: str | None, level3: str | None) -> int:
    """Group of a land-use class in the three-level scheme (editions from 2007/08 on).

    Water is listed under green land use in the source and is split off here.
    Construction sites and gravel pits ("Transformationsfläche ...") are neither
    green nor sealed and go to OTHER.
    """
    level1, level2, level3 = (s or "" for s in (level1, level2, level3))
    if level2 == "Gewässer":
        return WATER
    if level3.startswith("Transformationsfl"):
        return OTHER
    if level1 == "Grünlandnutzung":
        if level2 == "Landwirtschaft":
            return FARMLAND
        if level3 == "Wald":
            return FOREST
        return GREEN_OTHER
    if level1 in ("Baulandnutzung", "Verkehr"):
        return BUILT
    return OTHER


def code_crosswalk(
    codes: np.ndarray, groups: np.ndarray, n_codes: int, min_share: float
) -> pd.DataFrame:
    """Translate the old two-letter codes into groups by where they lie in a named edition.

    `codes` holds the index (1..n_codes) of the old code per pixel, 0 where there
    is none; `groups` holds the group of the same pixel in a later edition that
    names its classes. A code gets the group that covers most of its pixels; if
    that is less than `min_share` the code is unclear and becomes OTHER.
    """
    codes, groups = np.asarray(codes).ravel(), np.asarray(groups).ravel()
    both = (codes > 0) & (groups != UNMAPPED)
    counts = np.zeros((n_codes + 1, len(GROUP_NAMES)), dtype="int64")
    np.add.at(counts, (codes[both], groups[both]), 1)
    rows = []
    for code in range(1, n_codes + 1):
        total = int(counts[code].sum())
        best = int(counts[code].argmax()) if total else OTHER
        share = counts[code, best] / total if total else 0.0
        rows.append(
            {
                "code_index": code,
                "pixels": total,
                "majority_group": GROUP_NAMES[best],
                "majority_share": round(float(share), 3),
                "group": GROUP_NAMES[best if share >= min_share else OTHER],
            }
        )
    return pd.DataFrame(rows)


def trajectories(stack: np.ndarray, min_before: int = 2, min_after: int = 2) -> dict[str, np.ndarray]:
    """Classify every pixel by its land use over all editions.

    `stack` has shape (editions, rows, columns) and holds groups, oldest first.
    Returns boolean maps `stable_green`, `stable_farmland`, `stable_forest`,
    `stable_built`, and for converted pixels the edition indices `last_green`
    and `first_built` (-1 elsewhere).

    A pixel is converted when it is green in every edition up to `last_green`,
    sealed in every edition from `first_built` on, and anything between the two
    is OTHER (a construction site). It must be green in at least `min_before`
    and sealed in at least `min_after` editions, so that a class that flickers
    between two editions is not taken for a conversion.
    """
    stack = np.asarray(stack)
    n = stack.shape[0]
    green = np.isin(stack, GREEN_GROUPS)
    built = stack == BUILT
    other = stack == OTHER

    n_green_lead = np.cumprod(green, axis=0).sum(axis=0)  # green editions from the start
    n_built_tail = np.cumprod(built[::-1], axis=0).sum(axis=0)  # sealed editions to the end
    index = np.arange(n).reshape(-1, 1, 1)
    in_between = (index >= n_green_lead) & (index < n - n_built_tail)
    between = np.where(in_between, other, True).all(axis=0)
    converted = (
        (n_green_lead >= min_before)
        & (n_built_tail >= min_after)
        & (n_green_lead + n_built_tail <= n)
        & between
    )
    return {
        "stable_green": green.all(axis=0),
        "stable_farmland": (stack == FARMLAND).all(axis=0),
        "stable_forest": (stack == FOREST).all(axis=0),
        "stable_built": built.all(axis=0),
        "last_green": np.where(converted, n_green_lead - 1, -1),
        "first_built": np.where(converted, n - n_built_tail, -1),
    }


def erode(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Drop cells within `pixels` of the edge of a mask (8-neighbourhood, edges count as outside)."""
    out = np.asarray(mask, dtype=bool)
    for _ in range(pixels):
        padded = np.pad(out, 1, constant_values=False)
        h, w = out.shape
        result = np.ones_like(out)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                result &= padded[dy : dy + h, dx : dx + w]
        out = result
    return out


# --------------------------------------------------------------------- trends


def summer_means(monthly: pd.DataFrame, value: str, by: str) -> pd.DataFrame:
    """June-August mean per `by` and year from monthly rows (`time`, `by`, `value`).

    A summer counts only when all three months have a value.
    """
    df = monthly[monthly["time"].dt.month.isin(SUMMER_MONTHS)].dropna(subset=[value])
    out = df.groupby([by, df["time"].dt.year.rename("year")])[value].agg(["mean", "count"])
    out = out[out["count"] == len(SUMMER_MONTHS)]["mean"].rename(value)
    return out.reset_index()


def linear_trend(years, values) -> dict:
    """Ordinary least-squares trend: slope per decade with a 95 % interval.

    The interval treats the years as independent and uses 1.96 + 2.4 / df for
    the t quantile (within 0.02 of the exact value from 8 degrees of freedom on).
    """
    x = np.asarray(years, dtype="float64")
    y = np.asarray(values, dtype="float64")
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = len(x)
    if n < 3 or np.ptp(x) == 0:
        return {"n": int(n), "slope_per_decade": np.nan, "ci95_per_decade": np.nan,
                "first_year": None, "last_year": None, "change": np.nan, "change_ci95": np.nan}
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    se = np.sqrt((resid**2).sum() / (n - 2) / ((x - x.mean()) ** 2).sum())
    half = (1.96 + 2.4 / (n - 2)) * se
    span = x.max() - x.min()
    return {
        "n": int(n),
        "slope_per_decade": float(slope * 10),
        "ci95_per_decade": float(half * 10),
        "first_year": int(x.min()),
        "last_year": int(x.max()),
        "change": float(slope * span),  # trend line, last year minus first year
        "change_ci95": float(half * span),
    }


def trend_with_covariate(years, values, covariate) -> dict:
    """Trend per decade of `values` with a second explanatory variable held fixed."""
    x = np.asarray(years, dtype="float64")
    y = np.asarray(values, dtype="float64")
    c = np.asarray(covariate, dtype="float64")
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    if keep.sum() < 4:
        return {"n": int(keep.sum()), "slope_per_decade": np.nan, "per_unit_covariate": np.nan}
    design = np.column_stack([x[keep] - x[keep].mean(), c[keep] - c[keep].mean(), np.ones(keep.sum())])
    coef, *_ = np.linalg.lstsq(design, y[keep], rcond=None)
    return {"n": int(keep.sum()), "slope_per_decade": float(coef[0] * 10),
            "per_unit_covariate": float(coef[1])}


def split_warming(series: pd.DataFrame, urban: list[str], rural: list[str]) -> dict:
    """Split the urban summer trend into a regional part and an urban extra.

    `series` has one column per place and one row per year. Only years in which
    every listed place has a value are used, so the three trends are over the
    same summers and add up exactly: urban = rural + (urban - rural).
    """
    common = series[urban + rural].dropna()
    u, r = common[urban].mean(axis=1), common[rural].mean(axis=1)
    return {
        "urban_places": urban,
        "rural_places": rural,
        "urban": linear_trend(common.index, u),
        "regional": linear_trend(common.index, r),
        "urban_extra": linear_trend(common.index, u - r),
        "urban_minus_rural_first_decade": float((u - r).iloc[:10].mean()),
        "urban_minus_rural_last_decade": float((u - r).iloc[-10:].mean()),
    }


def decade_change(years, values, n: int = 10) -> float:
    """Mean of the last `n` years minus the mean of the first `n`."""
    s = pd.Series(np.asarray(values, dtype="float64"), index=np.asarray(years)).dropna().sort_index()
    if len(s) < 2 * n:
        return np.nan
    return float(s.iloc[-n:].mean() - s.iloc[:n].mean())


# -------------------------------------------------------- surface, before/after

EFFECT_COLUMNS = [
    "cohort", "last_green_year", "first_built_year", "pixels", "reference", "scenes_before",
    "scenes_after", "before_median_c", "after_median_c", "effect_c", "effect_same_sensor_c",
    "same_sensor_scenes",
]


def cohort_effects(
    scenes: pd.DataFrame, cohorts: pd.DataFrame, reference: str, min_scenes: int = 5
) -> pd.DataFrame:
    """Surface warming of converted land: (cohort - reference) after minus before.

    `scenes` has one row per overpass with `year`, `sensor`, the mean surface
    temperature of the reference zone and one column per cohort. `cohorts` has
    `cohort`, `last_green_year`, `first_built_year` and `pixels`. Scenes up to
    the last green edition are "before", scenes from the first sealed edition
    on are "after"; the years between are left out. `effect_same_sensor_c` uses
    only sensors that saw the cohort both before and after, each compared with
    itself, weighted by the smaller number of scenes.
    """
    rows = []
    for c in cohorts.itertuples():
        d = scenes[["year", "sensor"]].assign(delta=scenes[c.cohort] - scenes[reference]).dropna()
        before = d[d["year"] <= c.last_green_year]
        after = d[d["year"] >= c.first_built_year]
        row = {
            "cohort": c.cohort,
            "last_green_year": c.last_green_year,
            "first_built_year": c.first_built_year,
            "pixels": c.pixels,
            "reference": reference,
            "scenes_before": len(before),
            "scenes_after": len(after),
            "before_median_c": np.nan,
            "after_median_c": np.nan,
            "effect_c": np.nan,
            "effect_same_sensor_c": np.nan,
            "same_sensor_scenes": 0,
        }
        if len(before) >= min_scenes and len(after) >= min_scenes:
            row["before_median_c"] = float(before["delta"].median())
            row["after_median_c"] = float(after["delta"].median())
            row["effect_c"] = row["after_median_c"] - row["before_median_c"]
            parts, weights = [], []
            for sensor in sorted(set(before["sensor"]) & set(after["sensor"])):
                b = before.loc[before["sensor"] == sensor, "delta"]
                a = after.loc[after["sensor"] == sensor, "delta"]
                if len(b) >= min_scenes and len(a) >= min_scenes:
                    parts.append(a.median() - b.median())
                    weights.append(min(len(a), len(b)))
            if parts:
                row["effect_same_sensor_c"] = float(np.average(parts, weights=weights))
                row["same_sensor_scenes"] = int(sum(weights))
        rows.append(row)
    return pd.DataFrame(rows, columns=EFFECT_COLUMNS)


def pooled_effect(effects: pd.DataFrame, column: str = "effect_c") -> dict:
    """Area-weighted mean and the range of the cohort effects."""
    d = effects.dropna(subset=[column])
    if d.empty:
        return {"cohorts": 0, "pixels": 0, "weighted_mean_c": np.nan, "median_c": np.nan,
                "min_c": np.nan, "max_c": np.nan}
    return {
        "cohorts": len(d),
        "pixels": int(d["pixels"].sum()),
        "weighted_mean_c": float(np.average(d[column], weights=d["pixels"])),
        "median_c": float(d[column].median()),
        "min_c": float(d[column].min()),
        "max_c": float(d[column].max()),
    }


def event_study(scenes: pd.DataFrame, cohorts: pd.DataFrame, reference: str, bin_years: int = 5) -> pd.DataFrame:
    """Cohort minus reference by time since conversion, pooled over cohorts.

    Time is counted backwards from the last green edition (bins ..., -2, -1)
    and forwards from the first sealed edition (bins 0, 1, ...). Each cohort is
    first reduced to its median per bin and shifted by its own "before" median,
    so that cohorts starting from different levels can be averaged; cohorts are
    weighted by area.
    """
    rows = []
    for c in cohorts.itertuples():
        d = scenes[["year"]].assign(delta=scenes[c.cohort] - scenes[reference]).dropna()
        before = d[d["year"] <= c.last_green_year]
        after = d[d["year"] >= c.first_built_year]
        if before.empty or after.empty:
            continue
        base = before["delta"].median()
        b = before.assign(bin=-1 - (c.last_green_year - before["year"]) // bin_years)
        a = after.assign(bin=(after["year"] - c.first_built_year) // bin_years)
        for part in (b, a):
            g = part.groupby("bin")["delta"].agg(["median", "count"])
            for bin_, r in g.iterrows():
                rows.append({"cohort": c.cohort, "bin": int(bin_), "pixels": c.pixels,
                             "shift_c": r["median"] - base, "scenes": int(r["count"])})
    per = pd.DataFrame(rows, columns=["cohort", "bin", "pixels", "shift_c", "scenes"])
    if per.empty:
        return pd.DataFrame(columns=["bin", "from_year", "to_year", "cohorts", "scenes", "shift_c"])
    out = per.groupby("bin").apply(
        lambda g: pd.Series({
            "cohorts": len(g),
            "scenes": int(g["scenes"].sum()),
            "shift_c": float(np.average(g["shift_c"], weights=g["pixels"])),
        }),
        include_groups=False,
    ).reset_index()
    out["from_year"] = out["bin"] * bin_years
    out["to_year"] = out["from_year"] + bin_years - 1
    return out[["bin", "from_year", "to_year", "cohorts", "scenes", "shift_c"]].astype(
        {"cohorts": int, "scenes": int}
    )
