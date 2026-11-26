import numpy as np
import pandas as pd
import pytest

from uhi import core


def test_landsat_st_scale_and_nodata():
    # 44000 * 0.00341802 + 149.0 = 299.39288 K = 26.24288 C
    out = core.landsat_st_celsius(np.array([0, 44000], dtype="uint16"))
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(26.24288, abs=1e-4)


def test_landsat_reflectance_scale_and_nodata():
    # 10000 * 2.75e-5 - 0.2 = 0.075
    out = core.landsat_reflectance(np.array([0, 10000], dtype="uint16"))
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(0.075)


def test_uint16_input_does_not_overflow():
    out = core.landsat_st_celsius(np.array([65535], dtype="uint16"))
    assert out[0] == pytest.approx(65535 * 0.00341802 + 149.0 - 273.15)


@pytest.mark.parametrize(
    "qa, expected",
    [
        (21824, True),  # clear land, Landsat 8/9
        (21952, True),  # clear water
        (1, False),  # fill
        (0, False),  # outside the scene after reprojection
        (21824 | 0b00010, False),  # dilated cloud
        (21824 | 0b00100, False),  # cirrus
        (21824 | 0b01000, False),  # cloud
        (21824 | 0b10000, False),  # cloud shadow
    ],
)
def test_landsat_clear(qa, expected):
    assert bool(core.landsat_clear(np.array([qa], dtype="uint16"))[0]) is expected


def test_ndvi_values_and_undefined():
    out = core.ndvi(np.array([0.5, 0.1, 0.0, np.nan]), np.array([0.1, 0.1, 0.0, 0.2]))
    assert out[0] == pytest.approx(0.4 / 0.6)
    assert out[1] == pytest.approx(0.0)
    assert np.isnan(out[2])
    assert np.isnan(out[3])


def test_s2_reflectance_offset_depends_on_baseline():
    dn = np.array([0, 3000], dtype="uint16")
    new = core.s2_reflectance(dn, "05.11")
    old = core.s2_reflectance(dn, "03.01")
    assert np.isnan(new[0]) and np.isnan(old[0])
    assert new[1] == pytest.approx(0.2)
    assert old[1] == pytest.approx(0.3)
    assert core.s2_reflectance(np.array([500], dtype="uint16"), "04.00")[0] == pytest.approx(-0.05)


def test_s2_clear():
    scl = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    expected = [False, False, True, False, True, True, True, True, False, False, False, False]
    assert core.s2_clear(scl).tolist() == expected


@pytest.mark.parametrize(
    "area, label",
    [(0.5, "small"), (2.99, "small"), (3.0, "medium"), (29.9, "medium"), (30.0, "large"), (400, "large")],
)
def test_size_class(area, label):
    assert core.size_class(area) == label


def _scene():
    """4x4 scene: left half is a 20 C park, right half a 40 C sealed area."""
    lst = np.array([[20.0, 20.0, 40.0, 40.0]] * 4)
    ndvi = np.array([[0.8, 0.8, 0.1, 0.1]] * 4)
    clear = np.ones((4, 4), dtype=bool)
    idx = np.arange(16).reshape(4, 4)
    pixels = {"park": idx[:, :2].ravel(), "sealed": idx[:, 2:].ravel()}
    return lst, ndvi, clear, pixels


def test_site_stats_means():
    lst, ndvi, clear, pixels = _scene()
    out = core.site_stats(lst, ndvi, clear, pixels, min_valid_frac=0.8).set_index("site")
    assert out.loc["park", "lst_mean_c"] == pytest.approx(20.0)
    assert out.loc["sealed", "lst_mean_c"] == pytest.approx(40.0)
    assert out.loc["park", "ndvi_landsat"] == pytest.approx(0.8)
    assert out.loc["park", "pixels"] == 8
    assert out.loc["park", "valid_frac"] == 1.0


def test_site_stats_ignores_masked_pixels():
    lst, ndvi, clear, pixels = _scene()
    lst[0, 0] = 99.0  # a cloud edge that must not count
    clear[0, 0] = False
    out = core.site_stats(lst, ndvi, clear, pixels, min_valid_frac=0.8).set_index("site")
    assert out.loc["park", "lst_mean_c"] == pytest.approx(20.0)
    assert out.loc["park", "pixels"] == 7
    assert out.loc["park", "valid_frac"] == pytest.approx(0.875)


def test_site_stats_skips_site_with_too_much_masked():
    lst, ndvi, clear, pixels = _scene()
    clear[:2, :2] = False  # half the park is cloud
    lst[3, 3] = np.nan  # one sealed pixel has no temperature
    out = core.site_stats(lst, ndvi, clear, pixels, min_valid_frac=0.8)
    assert out["site"].tolist() == ["sealed"]
    assert out.loc[0, "pixels"] == 7


def test_site_stats_empty_result_keeps_columns():
    lst, ndvi, clear, pixels = _scene()
    out = core.site_stats(lst, ndvi, np.zeros_like(clear), pixels, min_valid_frac=0.8)
    assert out.empty
    assert "lst_mean_c" in out.columns


def _rows():
    return pd.DataFrame(
        [
            ("s1", "park", 30.0),
            ("s1", "rail", 41.0),
            ("s1", "road", 39.0),
            ("s2", "park", 25.0),
            ("s2", "rail", 30.0),  # road missing in s2: scene must be dropped
            ("s3", "park", 33.0),
            ("s3", "rail", 40.0),
            ("s3", "road", 40.0),
        ],
        columns=["scene", "site", "lst_mean_c"],
    )


def test_add_sealed_delta_uses_same_scene_reference():
    out = core.add_sealed_delta(_rows(), ["rail", "road"])
    assert sorted(out["scene"].unique()) == ["s1", "s3"]
    park = out[out["site"] == "park"].set_index("scene")
    assert park.loc["s1", "sealed_mean_c"] == pytest.approx(40.0)
    assert park.loc["s1", "delta_c"] == pytest.approx(-10.0)
    assert park.loc["s3", "delta_c"] == pytest.approx(-7.0)
    rail = out[out["site"] == "rail"].set_index("scene")
    assert rail.loc["s1", "delta_c"] == pytest.approx(1.0)


def test_summaries():
    df = pd.DataFrame(
        {
            "site": ["a"] * 5 + ["b"] * 3,
            "delta_c": [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -8.0, -10.0],
        }
    )
    sites = core.summarize_sites(df).set_index("site")
    assert sites.loc["a", "scenes"] == 5
    assert sites.loc["a", "delta_median_c"] == pytest.approx(-3.0)
    assert sites.loc["a", "delta_q25_c"] == pytest.approx(-4.0)
    assert sites.loc["a", "delta_q75_c"] == pytest.approx(-2.0)
    assert sites.loc["b", "delta_median_c"] == pytest.approx(-8.0)

    sites = sites.reset_index().assign(size_class=["small", "small"])
    groups = core.summarize_groups(sites, "size_class")
    assert groups.loc[0, "sites"] == 2
    assert groups.loc[0, "delta_median_c"] == pytest.approx(-5.5)
