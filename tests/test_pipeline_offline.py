"""The whole multi-scene pipeline on a synthetic city. No network, no real data.

The data layer is replaced by small fakes, so this checks that the steps fit
together and that known temperatures come out as known differences.
"""

import json
import sys

import numpy as np
import pandas as pd
import pytest

gpd = pytest.importorskip("geopandas")
from rasterio.features import geometry_mask
from shapely.geometry import box

from uhi import data, multi_scene

SEALED_C, PARK_C, WATER_PARK_C, COOL_SEALED_C = 40.0, 34.0, 30.0, 38.0
CLEAR = 21824


def _dn(celsius: float) -> int:
    return round((celsius + 273.15 - data.core.ST_OFFSET_K) / data.core.ST_SCALE)


def _square(lon: float, lat: float, side_m: float):
    p = data._metric(data.Point(lon, lat))
    return box(p.x - side_m / 2, p.y - side_m / 2, p.x + side_m / 2, p.y + side_m / 2)


@pytest.fixture
def fake_city(monkeypatch):
    grid = data.landsat_grid()
    city = box(*grid.bounds)
    water = _square(16.40, 48.23, 400)
    parks = {  # name: (geometry, on water)
        "Small Park": (_square(16.36, 48.20, 120), False),  # 1.4 ha
        "Medium Park": (_square(16.38, 48.21, 300), False),  # 9 ha
        "Large Park": (_square(16.39, 48.19, 700), False),  # 49 ha
        "Water Park": (_square(16.40, 48.23, 400).buffer(150), True),  # ring around the water
    }
    green = gpd.GeoDataFrame(
        {"T_LANG": list(parks), "P_PK": 1, "geometry": [g for g, _ in parks.values()]},
        crs=data.CRS,
    )
    landuse_rows = [
        ("Verkehr", "weitere", "Bahnhöfe und Bahnanlagen", "15", data._metric(data.WESTBAHN_AOI)),
        ("Verkehr", "Straßenraum", "Straßenraum unbegrünt", "07",
         data._metric(data.GUERTEL_LINE).buffer(200)),
        ("Bauland", "Wohnen", "dichtes Wohn(misch)gebiet", "07", _square(16.348, 48.202, 600)),
        ("Grünland", "Gewässer", "Gewässer inkl. Bachbett", "02", water),
        # Rule-based candidates in Favoriten: the largest housing block sits in a
        # temperature gap, so the rule must take the second one.
        ("Bauland", "Wohnen", "dichtes Wohn(misch)gebiet", "10", _square(16.37, 48.15, 900)),
        ("Bauland", "Wohnen", "dichtes Wohn(misch)gebiet", "10", _square(16.37, 48.17, 500)),
        ("Verkehr", "weitere", "Bahnhöfe und Bahnanlagen", "10", _square(16.40, 48.16, 500)),
    ]
    landuse = gpd.GeoDataFrame(
        landuse_rows,
        columns=["NUTZUNG_LEVEL1", "NUTZUNG_LEVEL2", "NUTZUNG_LEVEL3", "BEZ", "geometry"],
        crs=data.CRS,
    )
    layers = {
        "OEFFGRUENFLOGD": green,
        "REALNUT2022OGD": landuse,
        "LANDESGRENZEOGD": gpd.GeoDataFrame({"geometry": [city]}, crs=data.CRS),
    }

    def mask(geom):
        return geometry_mask([geom], grid.shape, grid.transform, invert=True)

    lwir = np.full(grid.shape, _dn(SEALED_C), dtype="uint16")
    lwir[mask(_square(16.37, 48.17, 500))] = _dn(COOL_SEALED_C)
    lwir[mask(_square(16.40, 48.16, 500))] = _dn(COOL_SEALED_C)
    for geom, on_water in parks.values():
        lwir[mask(geom)] = _dn(WATER_PARK_C if on_water else PARK_C)
    lwir[mask(_square(16.37, 48.15, 900))] = 0  # permanent gap
    raw = {
        "lwir11": lwir,
        "red": np.full(grid.shape, 10000, dtype="uint16"),
        "nir08": np.full(grid.shape, 20000, dtype="uint16"),
        "qa_pixel": np.full(grid.shape, CLEAR, dtype="uint16"),
    }
    passes = {f"{2013 + i}-07-0{1 + i % 3}_landsat-8_p190": [] for i in range(12)}

    monkeypatch.setattr(data, "wfs", lambda layer, cache: layers[layer])
    monkeypatch.setattr(data, "landsat_items", lambda *a, **k: [])
    monkeypatch.setattr(data, "fetch_landsat", lambda *a, **k: None)
    monkeypatch.setattr(data, "landsat_passes", lambda items: passes)
    monkeypatch.setattr(data, "load_landsat_pass", lambda items, grid, cache: raw)
    monkeypatch.setattr(
        data, "s2_summer_ndvi",
        lambda grid, *a, **k: (np.full(grid.shape, 0.5, dtype="float32"), ["2024-07-01"]),
    )


def test_pipeline_end_to_end(fake_city, tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv", ["multi_scene", "--cache", str(tmp_path / "cache"), "--out", str(out)]
    )
    multi_scene.main()

    stats = json.loads((out / "stats.json").read_text())
    assert stats["overpasses_used"] == 12
    west = stats["sealed_sites"]["west"]
    citywide = stats["sealed_sites"]["citywide"]
    assert len(west) == 3
    # Two rule-based sites found; the other rules have no candidates here.
    assert set(citywide) - set(west) == {"Favoriten dense housing", "Favoriten rail yard"}

    sites = pd.read_csv(out / "site_summary.csv").set_index("site")
    # The housing rule skipped the 81 ha block in the gap and took the 25 ha one.
    assert sites.loc["Favoriten dense housing", "area_ha"] == pytest.approx(25.0, abs=0.5)
    assert bool(sites.loc["Water Park", "near_water"]) is True
    assert bool(sites.loc["Medium Park", "near_water"]) is False

    # West baseline: 40 C. Citywide: (3 * 40 + 2 * 38) / 5 = 39.2 C.
    assert sites.loc["Medium Park", "delta_west_median_c"] == pytest.approx(PARK_C - 40.0, abs=0.05)
    assert sites.loc["Medium Park", "delta_median_c"] == pytest.approx(PARK_C - 39.2, abs=0.05)
    assert sites.loc["Water Park", "delta_median_c"] == pytest.approx(WATER_PARK_C - 39.2, abs=0.05)

    classes = pd.read_csv(out / "class_summary.csv")
    first = classes.iloc[0]
    assert (first["baseline"], first["subset"]) == ("citywide", "away from water")
    assert set(classes["subset"]) == {"away from water", "near water", "all parks"}

    comparison = pd.read_csv(out / "baseline_comparison.csv")
    assert comparison["size_class"].tolist() == ["small", "medium", "large"]
    np.testing.assert_allclose(comparison["shift_c"], 0.8, atol=0.05)

    table = pd.read_csv(out / "scene_site.csv")
    assert {"delta_c", "delta_west_c", "lst_mean_c"} <= set(table.columns)
    for name in ("cooling_by_size.png", "cooling_vs_ndvi.png", "cooling_per_year.png"):
        assert (out / name).stat().st_size > 5000
