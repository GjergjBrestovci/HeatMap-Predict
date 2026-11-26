"""Pure logic: unit conversion, masking and statistics. No network, no geo libraries."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Landsat Collection 2 Level-2 scale factors (USGS product guide).
ST_SCALE, ST_OFFSET_K = 0.00341802, 149.0
SR_SCALE, SR_OFFSET = 2.75e-5, -0.2
# QA_PIXEL bits 0-4: fill, dilated cloud, cirrus, cloud, cloud shadow.
QA_BAD_BITS = 0b11111

# Sentinel-2 L2A: reflectance = (DN + offset) / 10000. The offset of -1000 applies
# from processing baseline 04.00 (January 2022) on.
S2_QUANT = 10000.0
S2_OFFSET_FROM_BASELINE = 4.0
# Scene classification classes that are not usable ground: no data, saturated,
# cloud shadow, cloud (medium, high), thin cirrus, snow.
S2_BAD_SCL = (0, 1, 3, 8, 9, 10, 11)

# Park size classes by area in hectares: (label, lower bound inclusive).
SIZE_CLASSES = (("large", 30.0), ("medium", 3.0), ("small", 0.0))


def landsat_st_celsius(dn: np.ndarray) -> np.ndarray:
    """Surface temperature band (ST_B10) to degrees Celsius. DN 0 is no data."""
    dn = np.asarray(dn)
    return np.where(dn == 0, np.nan, dn * ST_SCALE + ST_OFFSET_K - 273.15)


def landsat_reflectance(dn: np.ndarray) -> np.ndarray:
    """Surface reflectance band to reflectance (0-1). DN 0 is no data."""
    dn = np.asarray(dn)
    return np.where(dn == 0, np.nan, dn * SR_SCALE + SR_OFFSET)


def landsat_clear(qa: np.ndarray) -> np.ndarray:
    """True where QA_PIXEL reports no fill, cloud, cirrus or cloud shadow.

    A value of 0 never occurs in real data and marks pixels outside the scene
    after reprojection, so it counts as not clear.
    """
    qa = np.asarray(qa)
    return ((qa & QA_BAD_BITS) == 0) & (qa != 0)


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalised difference vegetation index; NaN where undefined."""
    nir = np.asarray(nir, dtype="float64")
    red = np.asarray(red, dtype="float64")
    total = nir + red
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, (nir - red) / total, np.nan)


def s2_reflectance(dn: np.ndarray, processing_baseline: str | float) -> np.ndarray:
    """Sentinel-2 L2A digital numbers to reflectance. DN 0 is no data."""
    dn = np.asarray(dn)
    offset = -1000.0 if float(processing_baseline) >= S2_OFFSET_FROM_BASELINE else 0.0
    return np.where(dn == 0, np.nan, (dn + offset) / S2_QUANT)


def s2_clear(scl: np.ndarray) -> np.ndarray:
    """True where the Sentinel-2 scene classification is usable ground."""
    return ~np.isin(np.asarray(scl), S2_BAD_SCL)


def size_class(area_ha: float) -> str:
    """Park size class for an area in hectares."""
    for label, lower in SIZE_CLASSES:
        if area_ha >= lower:
            return label
    raise ValueError(f"negative area: {area_ha}")


def site_stats(
    lst: np.ndarray,
    ndvi_arr: np.ndarray,
    clear: np.ndarray,
    site_pixels: dict[str, np.ndarray],
    min_valid_frac: float,
) -> pd.DataFrame:
    """Mean LST and NDVI per site for one scene.

    `site_pixels` maps a site name to the flat indices of its pixels. A pixel is
    valid when it is clear and has a temperature. Sites with less than
    `min_valid_frac` valid pixels are left out.
    """
    lst_flat, ndvi_flat = lst.ravel(), ndvi_arr.ravel()
    valid_flat = clear.ravel() & np.isfinite(lst_flat)
    rows = []
    for site, idx in site_pixels.items():
        if len(idx) == 0:
            continue
        valid = idx[valid_flat[idx]]
        frac = len(valid) / len(idx)
        if frac < min_valid_frac or len(valid) == 0:
            continue
        site_ndvi = ndvi_flat[valid]
        rows.append(
            {
                "site": site,
                "pixels": len(valid),
                "valid_frac": round(frac, 3),
                "lst_mean_c": float(lst_flat[valid].mean()),
                "ndvi_landsat": float(np.nanmean(site_ndvi))
                if np.isfinite(site_ndvi).any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=["site", "pixels", "valid_frac", "lst_mean_c", "ndvi_landsat"])


def add_sealed_delta(df: pd.DataFrame, sealed_sites: list[str]) -> pd.DataFrame:
    """Add `sealed_mean_c` and `delta_c` (site minus sealed mean, same scene).

    Scenes where any sealed site is missing are dropped, so the reference is
    always the mean of the same set of sites.
    """
    sealed = df[df["site"].isin(sealed_sites)]
    ref = sealed.groupby("scene")["lst_mean_c"].agg(["mean", "count"])
    ref = ref.loc[ref["count"] == len(sealed_sites), "mean"].rename("sealed_mean_c")
    out = df.merge(ref, left_on="scene", right_index=True, how="inner")
    out["delta_c"] = out["lst_mean_c"] - out["sealed_mean_c"]
    return out.reset_index(drop=True)


def _q25(s: pd.Series) -> float:
    return s.quantile(0.25)


def _q75(s: pd.Series) -> float:
    return s.quantile(0.75)


def summarize_sites(df: pd.DataFrame) -> pd.DataFrame:
    """Per site: number of scenes, median and quartiles of `delta_c`."""
    return (
        df.groupby("site")["delta_c"]
        .agg(scenes="count", delta_median_c="median", delta_q25_c=_q25, delta_q75_c=_q75)
        .reset_index()
    )


def summarize_groups(site_summary: pd.DataFrame, by: str) -> pd.DataFrame:
    """Per group of sites: count, median and quartiles of the site medians."""
    return (
        site_summary.groupby(by)["delta_median_c"]
        .agg(sites="count", delta_median_c="median", delta_q25_c=_q25, delta_q75_c=_q75)
        .reset_index()
    )
