import numpy as np
import pandas as pd
import pytest

from uhi import station_stats as ss


class T:  # minimal stand-in for an affine transform: 30 m pixels, origin (1000, 2000)
    a, e, c, f = 30.0, -30.0, 1000.0, 2000.0


def _series():
    times = pd.to_datetime(["2020-07-01T09:30Z", "2020-07-01T09:40Z", "2020-07-01T09:50Z"])
    return pd.Series(times), pd.Series([20.0, 22.0, 23.0])


def test_interp_between_readings():
    t, v = _series()
    assert ss.interp_at(t, v, pd.Timestamp("2020-07-01T09:44Z")) == pytest.approx(22.4)


def test_interp_exact_reading_and_naive_time_is_utc():
    t, v = _series()
    assert ss.interp_at(t, v, pd.Timestamp("2020-07-01T09:40Z")) == 22.0
    assert ss.interp_at(t, v, pd.Timestamp("2020-07-01T09:35")) == pytest.approx(21.0)


def test_interp_outside_or_across_a_hole_is_nan():
    t, v = _series()
    assert np.isnan(ss.interp_at(t, v, pd.Timestamp("2020-07-01T09:55Z")))
    assert np.isnan(ss.interp_at(t, v, pd.Timestamp("2020-07-01T09:20Z")))
    v2 = pd.Series([20.0, np.nan, np.nan])
    t2 = pd.Series(pd.to_datetime(["2020-07-01T09:00Z", "2020-07-01T09:10Z", "2020-07-01T09:20Z"]))
    assert np.isnan(ss.interp_at(t2, v2, pd.Timestamp("2020-07-01T09:05Z")))
    # readings 40 minutes apart are too far to interpolate
    t3 = pd.Series(pd.to_datetime(["2020-07-01T09:00Z", "2020-07-01T09:40Z"]))
    assert np.isnan(ss.interp_at(t3, pd.Series([20.0, 24.0]), pd.Timestamp("2020-07-01T09:20Z")))


def test_pixel_index():
    shape = (4, 5)
    assert ss.pixel_index(T, shape, 1001, 1999).tolist() == [0]
    assert ss.pixel_index(T, shape, 1000 + 30 * 4 + 1, 2000 - 30 * 3 - 1).tolist() == [3 * 5 + 4]
    assert ss.pixel_index(T, shape, 999, 1999).size == 0
    assert ss.pixel_index(T, shape, 1001, 2000 - 30 * 4 - 1).size == 0


def test_disc_indices():
    shape = (5, 5)
    centre = (1000 + 75, 2000 - 75)  # centre of pixel (2, 2)
    assert ss.disc_indices(T, shape, *centre, 1).tolist() == [12]
    assert sorted(ss.disc_indices(T, shape, *centre, 31).tolist()) == [7, 11, 12, 13, 17]
    assert len(ss.disc_indices(T, shape, *centre, 1000)) == 25


def test_station_class():
    assert ss.station_class(0.5) == "green"
    assert ss.station_class(0.49) == "urban"


def _frame():
    rows = []
    # Air follows the day; the urban surface is 12 C above air, the green one 4 C.
    for i, day in enumerate([20.0, 24.0, 28.0, 30.0]):
        scene = f"2020-07-0{i + 1}_x"
        rows += [
            (scene, "city", "urban", day + 1.0, day + 13.0),
            (scene, "yard", "urban", day + 0.5, day + 12.5),
            (scene, "park", "green", day - 0.5, day + 3.5),
            (scene, "wood", "green", day - 1.0, day + 3.0),
        ]
    return pd.DataFrame(rows, columns=["scene", "station", "station_class", "air_c", "lst_c"])


def test_station_summary():
    out = ss.station_summary(_frame(), "lst_c").set_index("station")
    assert out.loc["city", "scenes"] == 4
    assert out.loc["city", "gap_median_c"] == pytest.approx(12.0)
    assert out.loc["park", "gap_median_c"] == pytest.approx(4.0)
    assert out.loc["city", "r_over_scenes"] == pytest.approx(1.0)


def test_station_summary_ignores_missing_values():
    df = _frame()
    df.loc[0, "lst_c"] = np.nan
    out = ss.station_summary(df, "lst_c").set_index("station")
    assert out.loc["city", "scenes"] == 3


def test_spatial_anomalies_remove_the_day():
    anom = ss.spatial_anomalies(_frame(), "lst_c")
    city = anom[anom["station"] == "city"]
    # scene mean air = day, scene mean surface = day + 8
    np.testing.assert_allclose(city["air_anom_c"], 1.0)
    np.testing.assert_allclose(city["lst_anom_c"], 5.0)
    assert anom.groupby("scene")["air_anom_c"].sum().abs().max() < 1e-9


def test_spatial_anomalies_need_enough_stations():
    df = _frame()
    df = df[~((df["scene"] == "2020-07-01_x") & (df["station"] == "city"))]
    anom = ss.spatial_anomalies(df, "lst_c", min_stations=4)
    assert "2020-07-01_x" not in set(anom["scene"])


def test_anomaly_relation_slope():
    rel = ss.anomaly_relation(ss.spatial_anomalies(_frame(), "lst_c"))
    # anomalies: surface +5, +4.5, -4.5, -5; air +1, +0.5, -0.5, -1
    assert rel["n"] == 16
    assert rel["scenes"] == 4 and rel["stations"] == 4
    assert rel["air_per_lst"] == pytest.approx(0.16, abs=0.01)
    assert rel["r"] > 0.95


def test_anomaly_relation_without_data():
    rel = ss.anomaly_relation(pd.DataFrame(columns=["scene", "station", "lst_anom_c", "air_anom_c"]))
    assert rel["n"] == 0 and np.isnan(rel["air_per_lst"])


def test_class_gap():
    out = ss.class_gap(_frame(), "lst_c").set_index("measure")
    assert out.loc["surface, urban minus green", "median_c"] == pytest.approx(9.5)
    assert out.loc["air, urban minus green", "median_c"] == pytest.approx(1.5)
    assert out.loc["air, urban minus green", "scenes"] == 4


def test_class_gap_needs_both_classes():
    df = _frame()
    assert ss.class_gap(df[df["station_class"] == "urban"], "lst_c").empty


def test_one_series_per_station_prefers_the_fuller_id_and_fills_holes():
    df = pd.DataFrame(
        [
            ("s1", "Hohe Warte", "105", 20.0),
            ("s1", "Hohe Warte", "5904", 20.1),
            ("s2", "Hohe Warte", "5904", 21.0),
            ("s3", "Hohe Warte", "5904", 22.0),
            ("s4", "Hohe Warte", "105", 23.0),  # only the older series has this scene
            ("s1", "Innere Stadt", "5925", 24.0),
        ],
        columns=["scene", "station", "station_id", "air_c"],
    )
    out = ss.one_series_per_station(df)
    assert len(out) == 5
    hw = out[out["station"] == "Hohe Warte"].set_index("scene")
    assert hw.loc["s1", "station_id"] == "5904"
    assert hw.loc["s4", "station_id"] == "105"
    assert not out.duplicated(["scene", "station"]).any()
