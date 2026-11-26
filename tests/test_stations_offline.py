"""The station comparison on synthetic data. No network, no real data."""

import json
import sys
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

gpd = pytest.importorskip("geopandas")
from shapely.geometry import box

from uhi import data, stations

CLEAR = 21824
# name: (id, lon, lat, altitude, surroundings green?, surface minus air, air offset)
FAKE = {
    "City": ("1", 16.37, 48.21, 170, False, 14.0, 1.0),
    "Yard": ("2", 16.34, 48.20, 180, False, 12.0, 0.5),
    "Park": ("3", 16.40, 48.19, 160, True, 5.0, -0.5),
    "Wood": ("4", 16.30, 48.24, 200, True, 3.0, -1.0),
}


def _dn(celsius):
    return np.round((celsius + 273.15 - data.core.ST_OFFSET_K) / data.core.ST_SCALE).astype("uint16")


def _disc(lon, lat, radius):
    return data._metric(data.Point(lon, lat)).buffer(radius)


def test_parse_air_csv():
    text = "time,station,tl,substation\n2020-07-01T09:40+00:00,5904,21.5,5904\n2020-07-01T09:50+00:00,5904,,5904\n"
    out = stations.parse_air_csv(text)
    assert out["station"].tolist() == ["5904", "5904"]
    assert out["air_c"].iloc[0] == 21.5 and np.isnan(out["air_c"].iloc[1])
    assert str(out["time"].dt.tz) == "UTC"


def test_parse_air_csv_reports_missing_columns():
    with pytest.raises(SystemExit, match="lacks"):
        stations.parse_air_csv("time,station,temperature\n")


def test_station_pipeline_end_to_end(tmp_path, monkeypatch):
    grid = data.landsat_grid()
    days = [22.0 + i for i in range(12)]
    passes = {
        f"{2013 + i}-07-01_landsat-8_p190": [
            SimpleNamespace(datetime=datetime(2013 + i, 7, 1, 9, 44, tzinfo=UTC))
        ]
        for i in range(12)
    }

    from rasterio.features import geometry_mask

    masks = {
        name: geometry_mask([_disc(lon, lat, 600)], grid.shape, grid.transform, invert=True)
        for name, (_, lon, lat, *_rest) in FAKE.items()
    }

    def load(items, grid_, cache):
        day = days[items[0].datetime.year - 2013]
        surface = np.full(grid.shape, day + 9.0)
        for name, (_, _, _, _, _, gap, offset) in FAKE.items():
            surface[masks[name]] = day + offset + gap
        return {
            "lwir11": _dn(surface),
            "red": np.full(grid.shape, 10000, dtype="uint16"),
            "nir08": np.full(grid.shape, 20000, dtype="uint16"),
            "qa_pixel": np.full(grid.shape, CLEAR, dtype="uint16"),
        }

    def fetch_air(ids, when, cache):
        day = days[when.year - 2013]
        times = pd.to_datetime([when.floor("10min"), when.floor("10min") + pd.Timedelta(minutes=10)])
        frames = [
            pd.DataFrame({"time": times, "station": sid, "air_c": day + offset})
            for sid, _, _, _, _, _, offset in FAKE.values()
        ]
        frames.append(pd.DataFrame({"time": times, "station": "99", "air_c": day + 1.0}))
        return pd.concat(frames, ignore_index=True)

    green = [_disc(lon, lat, 800) for _, lon, lat, _, is_green, *_ in FAKE.values() if is_green]
    landuse = gpd.GeoDataFrame(
        {"NUTZUNG_LEVEL1": "Grünlandnutzung", "NUTZUNG_LEVEL2": "Naturraum", "geometry": green},
        crs=data.CRS,
    )
    layers = {
        "REALNUT2022OGD": landuse,
        "LANDESGRENZEOGD": gpd.GeoDataFrame({"geometry": [box(*grid.bounds)]}, crs=data.CRS),
    }
    meta = pd.DataFrame(
        [
            {"id": sid, "name": name, "lon": lon, "lat": lat, "altitude": alt,
             "valid_from": "2000", "valid_to": "2100"}
            for name, (sid, lon, lat, alt, *_rest) in FAKE.items()
        ]
    )
    monkeypatch.setattr(data, "wfs", lambda layer, cache: layers[layer])
    monkeypatch.setattr(data, "landsat_items", lambda *a, **k: [])
    monkeypatch.setattr(data, "fetch_landsat", lambda *a, **k: None)
    monkeypatch.setattr(data, "landsat_passes", lambda items: passes)
    monkeypatch.setattr(data, "load_landsat_pass", load)
    # The data hub lists some stations twice; the copy must not be counted.
    twin = meta[meta["name"] == "City"].assign(id="99")
    meta = pd.concat([meta, twin], ignore_index=True)
    monkeypatch.setattr(stations, "station_metadata", lambda cache: meta.copy())
    monkeypatch.setattr(stations, "fetch_air", fetch_air)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["stations", "--cache", str(tmp_path / "c"), "--out", str(out)])

    stations.main()

    summary = pd.read_csv(out / "station_summary.csv").set_index("station")
    assert summary["station_class"].to_dict() == {
        "City": "urban", "Yard": "urban", "Park": "green", "Wood": "green"
    }
    assert summary.loc["City", "scenes"] == 12
    for name, (_, _, _, _, _, gap, _) in FAKE.items():
        assert summary.loc[name, "gap_median_c"] == pytest.approx(gap, abs=0.05)
        assert summary.loc[name, "gap_pixel_median_c"] == pytest.approx(gap, abs=0.05)

    classes = pd.read_csv(out / "urban_vs_green.csv").set_index("measure")
    # surface: (15 + 12.5) / 2 - (4.5 + 2) / 2 = 10.5; air: 0.75 - (-0.75) = 1.5
    assert classes.loc["surface, urban minus green", "median_c"] == pytest.approx(10.5, abs=0.05)
    assert classes.loc["air, urban minus green", "median_c"] == pytest.approx(1.5, abs=0.05)

    stats = json.loads((out / "stats.json").read_text())
    assert stats["overpasses_used"] == 12
    assert stats["station_scene_pairs"] == 48
    assert stats["between_places_300m"]["air_per_lst"] == pytest.approx(0.14, abs=0.02)
    table = pd.read_csv(out / "station_scene.csv")
    assert table["overpass_utc"].str.endswith("T09:44").all()
    for name in ("surface_vs_air.png", "gap_per_station.png", "between_places.png"):
        assert (out / name).stat().st_size > 5000
