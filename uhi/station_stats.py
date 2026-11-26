"""Station comparison logic. Pure numpy/pandas: no network, no geo libraries."""

from __future__ import annotations

import numpy as np
import pandas as pd

GREEN_SHARE_MIN = 0.5  # a station counts as "green" when this share of its surroundings is


def interp_at(times: pd.Series, values: pd.Series, when: pd.Timestamp, max_gap_min: float = 20.0):
    """Value at `when`, interpolated linearly between the readings around it.

    Returns NaN when a neighbouring reading is missing or the two readings are
    more than `max_gap_min` minutes apart.
    """
    data = pd.DataFrame({"t": pd.to_datetime(times, utc=True), "v": values}).dropna()
    data = data.sort_values("t")
    when = pd.Timestamp(when).tz_convert("UTC") if pd.Timestamp(when).tzinfo else pd.Timestamp(when, tz="UTC")
    exact = data[data["t"] == when]
    if len(exact):
        return float(exact["v"].iloc[0])
    before, after = data[data["t"] < when], data[data["t"] > when]
    if before.empty or after.empty:
        return np.nan
    t0, v0 = before.iloc[-1]
    t1, v1 = after.iloc[0]
    if (t1 - t0).total_seconds() > max_gap_min * 60:
        return np.nan
    weight = (when - t0).total_seconds() / (t1 - t0).total_seconds()
    return float(v0 + weight * (v1 - v0))


def pixel_index(transform, shape: tuple[int, int], x: float, y: float) -> np.ndarray:
    """Flat index of the pixel that contains (x, y); empty when outside the grid."""
    col = int(np.floor((x - transform.c) / transform.a))
    row = int(np.floor((y - transform.f) / transform.e))
    if not (0 <= row < shape[0] and 0 <= col < shape[1]):
        return np.array([], dtype=int)
    return np.array([row * shape[1] + col])


def disc_indices(transform, shape: tuple[int, int], x: float, y: float, radius: float) -> np.ndarray:
    """Flat indices of the pixels whose centre lies within `radius` of (x, y)."""
    rows, cols = np.indices(shape)
    cx = transform.c + (cols + 0.5) * transform.a
    cy = transform.f + (rows + 0.5) * transform.e
    return np.flatnonzero((cx - x) ** 2 + (cy - y) ** 2 <= radius**2)


def one_series_per_station(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per scene and station name.

    The data hub lists some stations under two ids (an older and a newer
    series at the same place). Counting both would weigh those stations
    double. Per station the id with the most rows is preferred; the other
    one fills scenes the preferred id lacks.
    """
    counts = df.groupby(["station", "station_id"]).size().rename("n").reset_index()
    counts = counts.sort_values(["station", "n", "station_id"], ascending=[True, False, True])
    counts["rank"] = counts.groupby("station").cumcount()
    ranked = df.merge(counts[["station", "station_id", "rank"]], on=["station", "station_id"])
    ranked = ranked.sort_values(["scene", "station", "rank"])
    return ranked.drop_duplicates(["scene", "station"]).drop(columns="rank").reset_index(drop=True)


def surface_colder_than_air(df: pd.DataFrame, lst_cols: tuple[str, ...]) -> pd.Series:
    """True for rows where a surface temperature reads below the air temperature.

    Around a late-morning summer overpass the ground is warmer than the air,
    so such a reading is most likely cloud or cloud shadow the mask missed.
    Missing surface values do not count.
    """
    colder = pd.Series(False, index=df.index)
    for col in lst_cols:
        colder |= df[col] < df["air_c"]
    return colder


def station_class(green_share: float) -> str:
    return "green" if green_share >= GREEN_SHARE_MIN else "urban"


def _q25(s: pd.Series) -> float:
    return s.quantile(0.25)


def _q75(s: pd.Series) -> float:
    return s.quantile(0.75)


def station_summary(df: pd.DataFrame, lst_col: str) -> pd.DataFrame:
    """Per station: how surface temperature (`lst_col`) relates to air temperature.

    `gap` is surface minus air. `r_over_scenes` says whether a hotter day at the
    surface is a hotter day in the air at that station.
    """
    d = df.dropna(subset=[lst_col, "air_c"]).assign(gap=lambda t: t[lst_col] - t["air_c"])
    out = d.groupby("station").agg(
        scenes=("gap", "count"),
        air_median_c=("air_c", "median"),
        lst_median_c=(lst_col, "median"),
        gap_median_c=("gap", "median"),
        gap_q25_c=("gap", _q25),
        gap_q75_c=("gap", _q75),
    )
    out["r_over_scenes"] = d.groupby("station").apply(
        lambda t: t[lst_col].corr(t["air_c"]) if len(t) > 2 else np.nan, include_groups=False
    )
    return out.reset_index()


def spatial_anomalies(df: pd.DataFrame, lst_col: str, min_stations: int = 4) -> pd.DataFrame:
    """Per scene, each station's deviation from the mean of all stations in that scene.

    This removes the weather of the day, so what is left is the difference
    between places: the thing the park comparison is about.
    """
    d = df.dropna(subset=[lst_col, "air_c"]).copy()
    n = d.groupby("scene")["station"].transform("count")
    d = d[n >= min_stations]
    d["lst_anom_c"] = d[lst_col] - d.groupby("scene")[lst_col].transform("mean")
    d["air_anom_c"] = d["air_c"] - d.groupby("scene")["air_c"].transform("mean")
    return d[["scene", "station", "lst_anom_c", "air_anom_c"]].reset_index(drop=True)


def anomaly_relation(anom: pd.DataFrame) -> dict:
    """How much of a surface temperature difference between places shows up in the air.

    `air_per_lst` is the slope of air anomaly on surface anomaly: 0.2 means a
    place that is 5 C hotter at the surface is 1 C warmer in the air.
    """
    if len(anom) < 3 or anom["lst_anom_c"].var() == 0:
        return {"n": len(anom), "air_per_lst": np.nan, "r": np.nan}
    slope = np.polyfit(anom["lst_anom_c"], anom["air_anom_c"], 1)[0]
    return {
        "n": len(anom),
        "scenes": int(anom["scene"].nunique()),
        "stations": int(anom["station"].nunique()),
        "air_per_lst": round(float(slope), 3),
        "r": round(float(anom["lst_anom_c"].corr(anom["air_anom_c"])), 3),
        "lst_anom_sd_c": round(float(anom["lst_anom_c"].std()), 2),
        "air_anom_sd_c": round(float(anom["air_anom_c"].std()), 2),
    }


def class_gap(df: pd.DataFrame, lst_col: str) -> pd.DataFrame:
    """Per scene, the mean of the urban stations minus the mean of the green ones,
    for the surface and for the air; then the median and quartiles over scenes."""
    d = df.dropna(subset=[lst_col, "air_c"])
    per = d.groupby(["scene", "station_class"])[[lst_col, "air_c"]].mean().unstack()
    per = per.dropna()
    if per.empty or "urban" not in per[lst_col] or "green" not in per[lst_col]:
        return pd.DataFrame(columns=["measure", "scenes", "median_c", "q25_c", "q75_c"])
    diffs = {
        "surface, urban minus green": per[lst_col]["urban"] - per[lst_col]["green"],
        "air, urban minus green": per["air_c"]["urban"] - per["air_c"]["green"],
    }
    return pd.DataFrame(
        [
            {"measure": k, "scenes": len(v), "median_c": v.median(), "q25_c": v.quantile(0.25),
             "q75_c": v.quantile(0.75)}
            for k, v in diffs.items()
        ]
    )
