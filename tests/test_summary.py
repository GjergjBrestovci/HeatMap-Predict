import numpy as np
import pandas as pd
import pytest

from uhi import summary


def _raw():
    """Two scenes. West sealed sites are 40 C, the citywide extra site is 36 C."""
    rows = []
    for scene, offset in (("2020-07-01_x", 0.0), ("2021-07-01_x", 2.0)):
        rows += [
            (scene, "rail west", 41.0 + offset),
            (scene, "road west", 39.0 + offset),
            (scene, "housing south", 36.0 + offset),
            (scene, "dry small", 37.0 + offset),
            (scene, "dry medium", 35.0 + offset),
            (scene, "wet small", 30.0 + offset),
        ]
    # A third scene where the southern site is masked: usable for "west" only.
    rows += [
        ("2022-07-01_x", "rail west", 41.0),
        ("2022-07-01_x", "road west", 39.0),
        ("2022-07-01_x", "dry small", 30.0),
    ]
    return pd.DataFrame(rows, columns=["scene", "site", "lst_mean_c"])


BASELINES = {
    "citywide": ["rail west", "road west", "housing south"],
    "west": ["rail west", "road west"],
}


def _parks():
    return pd.DataFrame(
        {
            "site": ["dry small", "dry medium", "wet small"],
            "size_class": ["small", "medium", "small"],
            "near_water": [False, False, True],
        }
    )


def test_with_baselines_main_baseline_decides_scenes():
    out = summary.with_baselines(_raw(), BASELINES)
    assert sorted(out["scene"].unique()) == ["2020-07-01_x", "2021-07-01_x"]
    row = out[(out["site"] == "dry small") & (out["scene"] == "2020-07-01_x")].iloc[0]
    # citywide mean = (41 + 39 + 36) / 3 = 38.667; west mean = 40
    assert row["sealed_mean_c"] == pytest.approx(116 / 3)
    assert row["delta_c"] == pytest.approx(37 - 116 / 3)
    assert row["delta_west_c"] == pytest.approx(-3.0)


def test_with_baselines_secondary_can_be_missing():
    raw = _raw()
    raw = raw[~((raw["scene"] == "2021-07-01_x") & (raw["site"] == "road west"))]
    out = summary.with_baselines(raw, {"citywide": ["rail west", "housing south"], "west": BASELINES["west"]})
    second = out[out["scene"] == "2021-07-01_x"]
    assert len(second) > 0
    assert second["delta_west_c"].isna().all()
    assert second["delta_c"].notna().all()


def test_park_subsets_split_on_water():
    subsets = summary.park_subsets(_parks())
    assert list(subsets) == summary.SUBSETS
    assert subsets["away from water"]["site"].tolist() == ["dry small", "dry medium"]
    assert subsets["near water"]["site"].tolist() == ["wet small"]
    assert len(subsets["all parks"]) == 3


def test_class_summary_headline_is_away_from_water():
    table = summary.with_baselines(_raw(), BASELINES)
    out = summary.class_summary(table, _parks())
    assert out.iloc[0]["subset"] == "away from water"
    dry_small = out[(out["subset"] == "away from water") & (out["size_class"] == "small")].iloc[0]
    assert dry_small["sites"] == 1
    assert dry_small["scenes"] == 2
    assert dry_small["delta_median_c"] == pytest.approx(37 - 116 / 3)
    wet_small = out[(out["subset"] == "near water") & (out["size_class"] == "small")].iloc[0]
    assert wet_small["delta_median_c"] == pytest.approx(30 - 116 / 3)
    all_small = out[(out["subset"] == "all parks") & (out["size_class"] == "small")].iloc[0]
    assert all_small["sites"] == 2
    assert all_small["scenes_cooler_share"] == 1.0


def test_class_summary_for_another_baseline_uses_its_own_scenes():
    raw = _raw()
    west_only = summary.with_baselines(raw, {"west": BASELINES["west"]})
    out = summary.class_summary(west_only, _parks())
    dry_small = out[(out["subset"] == "away from water") & (out["size_class"] == "small")].iloc[0]
    assert dry_small["scenes"] == 3
    # differences: -3, -3, -10 -> median -3
    assert dry_small["delta_median_c"] == pytest.approx(-3.0)


def test_baseline_comparison_uses_common_scenes_and_dry_parks():
    table = summary.with_baselines(_raw(), BASELINES)
    out = summary.baseline_comparison(table, _parks(), "west")
    assert out["size_class"].tolist() == ["small", "medium"]
    small = out.iloc[0]
    assert small["sites"] == 1  # the waterside park is not in the headline
    assert small["scenes"] == 2
    assert small["delta_west_median_c"] == pytest.approx(-3.0)
    assert small["delta_median_c"] == pytest.approx(37 - 116 / 3)
    # The citywide reference is 1.33 C cooler, so parks look 1.33 C less cool.
    assert small["shift_c"] == pytest.approx(4 / 3)


def test_year_summary():
    table = summary.with_baselines(_raw(), BASELINES)
    out = summary.year_summary(table, _parks())
    assert sorted(out["year"].unique()) == [2020, 2021]
    row = out[(out["year"] == 2020) & (out["size_class"] == "small")].iloc[0]
    assert row["sites"] == 2
    assert row["scenes"] == 1
    assert row["delta_median_c"] == pytest.approx(np.median([37 - 116 / 3, 30 - 116 / 3]))


def test_relation_recovers_a_known_slope():
    rng = np.random.default_rng(1)
    ndvi = rng.uniform(0.3, 0.8, 40)
    area = 10 ** rng.uniform(0, 2, 40)
    parks = pd.DataFrame(
        {"ndvi_s2": ndvi, "area_ha": area, "delta_median_c": -10 * ndvi - 1.5 * np.log10(area)}
    )
    out = summary.relation(parks)
    assert out["parks"] == 40
    assert out["joint_fit_c_per_0.1_ndvi"] == pytest.approx(-1.0)
    assert out["joint_fit_c_per_10x_area"] == pytest.approx(-1.5)
    assert out["joint_fit_r2"] == pytest.approx(1.0)
