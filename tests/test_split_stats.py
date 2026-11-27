"""Pure logic of the warming split on small arrays."""

import numpy as np
import pandas as pd
import pytest

from uhi import split_stats as st

B, F, W, G, O = st.BUILT, st.FARMLAND, st.FOREST, st.GREEN_OTHER, st.OTHER


def test_named_group():
    assert st.named_group("Grünlandnutzung", "Gewässer", "Gewässer inkl. Bachbett") == st.WATER
    assert st.named_group("Grünlandnutzung", "Landwirtschaft", "Acker") == F
    assert st.named_group("Grünlandnutzung", "Naturraum", "Wald") == W
    assert st.named_group("Grünlandnutzung", "Naturraum", "Wiese") == G
    assert st.named_group("Verkehr", "Straßenraum", "Straßenraum unbegrünt") == B
    assert st.named_group("Baulandnutzung", "x", "dichtes Wohn(misch)gebiet") == B
    assert st.named_group("Baulandnutzung", "x", "Transformationsfläche, Baustelle") == O
    assert st.named_group(None, None, None) == O


def test_code_crosswalk_takes_the_majority_and_flags_unclear_codes():
    codes = np.array([1] * 10 + [2] * 10 + [3] * 10 + [0] * 5)
    groups = np.array([F] * 9 + [B] + [B] * 10 + [F] * 5 + [B] * 5 + [B] * 5)
    out = st.code_crosswalk(codes, groups, 4, min_share=0.7).set_index("code_index")
    assert out.loc[1, "group"] == "farmland" and out.loc[1, "majority_share"] == 0.9
    assert out.loc[2, "group"] == "built"
    assert out.loc[3, "group"] == "other" and out.loc[3, "majority_share"] == 0.5
    assert out.loc[4, "pixels"] == 0 and out.loc[4, "group"] == "other"


def test_trajectories():
    #            stable  farm->built  via site  flicker  too late  built->green  forest
    columns = [
        [F, F, F, F, F, F],
        [F, F, F, B, B, B],
        [G, G, O, O, B, B],
        [F, B, F, B, B, B],
        [F, F, F, F, F, B],
        [B, B, B, F, F, F],
        [W, W, W, W, W, W],
        [B, B, B, B, B, B],
    ]
    stack = np.array(columns).T.reshape(6, 1, 8)
    t = st.trajectories(stack)
    assert t["stable_green"][0].tolist() == [True, False, False, False, False, False, True, False]
    assert t["stable_farmland"][0].tolist() == [True] + [False] * 7
    assert t["stable_forest"][0, 6] and t["stable_built"][0, 7]
    assert t["last_green"][0].tolist() == [-1, 2, 1, -1, -1, -1, -1, -1]
    assert t["first_built"][0].tolist() == [-1, 3, 4, -1, -1, -1, -1, -1]


def test_erode():
    mask = np.zeros((7, 7), dtype=bool)
    mask[1:6, 1:6] = True
    assert st.erode(mask, 1).sum() == 9
    assert st.erode(mask, 2).sum() == 1
    assert st.erode(np.ones((3, 3), dtype=bool), 1).sum() == 1  # the array edge counts as outside


def test_summer_means_needs_all_three_months():
    time = pd.to_datetime(["2000-06-01", "2000-07-01", "2000-08-01", "2000-01-01",
                           "2001-06-01", "2001-07-01"], utc=True)
    df = pd.DataFrame({"time": time, "station": "a", "t": [18.0, 20.0, 22.0, -5.0, 19.0, 21.0]})
    out = st.summer_means(df, "t", "station")
    assert out["year"].tolist() == [2000] and out["t"].tolist() == [20.0]


def test_linear_trend_recovers_slope_and_interval():
    years = np.arange(1981, 2027)
    rng = np.random.default_rng(1)
    t = st.linear_trend(years, 18 + 0.05 * (years - 1981) + rng.normal(0, 0.5, len(years)))
    assert t["slope_per_decade"] == pytest.approx(0.5, abs=0.15)
    assert 0.05 < t["ci95_per_decade"] < 0.25
    assert t["change"] == pytest.approx(t["slope_per_decade"] * 4.5)
    assert (t["first_year"], t["last_year"], t["n"]) == (1981, 2026, 46)
    assert np.isnan(st.linear_trend([2000, 2001], [1, 2])["slope_per_decade"])


def test_trend_with_covariate_separates_the_two():
    years = np.arange(1984, 2024)
    sun = 45 + 10 * np.sin(years)
    values = 0.02 * (years - 1984) + 0.1 * sun
    t = st.trend_with_covariate(years, values, sun)
    assert t["slope_per_decade"] == pytest.approx(0.2, abs=1e-6)
    assert t["per_unit_covariate"] == pytest.approx(0.1, abs=1e-6)


def test_split_warming_adds_up_over_common_years():
    years = np.arange(1981, 2027)
    regional = 18 + 0.06 * (years - 1981)
    wide = pd.DataFrame(
        {"City": regional + 1.0 + 0.01 * (years - 1981), "Late": regional + 1.5,
         "Field": regional, "Farm": regional - 0.2},
        index=years,
    )
    wide.loc[:1984, "Late"] = np.nan
    s = st.split_warming(wide, ["City", "Late"], ["Field", "Farm"])
    assert s["urban"]["first_year"] == 1985
    assert s["regional"]["slope_per_decade"] == pytest.approx(0.6)
    assert s["urban_extra"]["slope_per_decade"] == pytest.approx(0.05)
    assert s["urban"]["change"] == pytest.approx(s["regional"]["change"] + s["urban_extra"]["change"])
    assert st.decade_change(years, regional) == pytest.approx(0.06 * 36)


def _scenes():
    rows = []
    for year in range(1990, 2020):
        sensor = "TM" if year < 2012 else "OLI/TIRS"
        offset = 0.0 if sensor == "TM" else 0.5  # the newer sensor sees a larger contrast
        rows.append({"year": year, "sensor": sensor, "ref": 30.0,
                     "conv": 30.0 + offset + (3.0 if year >= 2001 else 0.0),
                     "never": 31.0})
    return pd.DataFrame(rows)


def test_cohort_effects_before_after_and_same_sensor():
    cohorts = pd.DataFrame({"cohort": ["conv"], "last_green_year": [1997],
                            "first_built_year": [2001], "pixels": [400]})
    e = st.cohort_effects(_scenes(), cohorts, "ref").iloc[0]
    assert (e["scenes_before"], e["scenes_after"]) == (8, 19)  # 1998-2000 left out
    assert e["before_median_c"] == pytest.approx(0.0)
    assert e["effect_c"] == pytest.approx(3.0)  # the median of "after" lies in the TM years
    assert e["effect_same_sensor_c"] == pytest.approx(3.0)  # TM before against TM after only
    late = pd.DataFrame({"cohort": ["conv"], "last_green_year": [1991],
                         "first_built_year": [2001], "pixels": [400]})
    assert np.isnan(st.cohort_effects(_scenes(), late, "ref").iloc[0]["effect_c"])  # 2 scenes before
    assert list(st.cohort_effects(_scenes(), cohorts.iloc[:0], "ref").columns) == st.EFFECT_COLUMNS


def test_pooled_effect_weights_by_area():
    effects = pd.DataFrame({"effect_c": [2.0, 4.0, np.nan], "pixels": [300, 100, 999]})
    p = st.pooled_effect(effects)
    assert p["weighted_mean_c"] == pytest.approx(2.5) and p["cohorts"] == 2 and p["pixels"] == 400
    assert st.pooled_effect(effects.iloc[2:])["cohorts"] == 0


def test_event_study_bins_relative_to_conversion():
    cohorts = pd.DataFrame({"cohort": ["conv"], "last_green_year": [1997],
                            "first_built_year": [2001], "pixels": [400]})
    out = st.event_study(_scenes(), cohorts, "ref", bin_years=5).set_index("bin")
    assert out.loc[-1, "shift_c"] == pytest.approx(0.0) and out.loc[-1, "scenes"] == 5
    assert out.loc[0, "shift_c"] == pytest.approx(3.0)
    assert out.loc[3, "shift_c"] == pytest.approx(3.5)  # 2016-2019, the newer sensor
    assert (out.loc[-1, "from_year"], out.loc[-1, "to_year"]) == (-5, -1)
