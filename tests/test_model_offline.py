"""The model pipeline and the scenario tool on a synthetic city. No network, no real data."""

import json
import sys
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
gpd = pytest.importorskip("geopandas")
from shapely.geometry import box

from uhi import data, history, layout, model, scenario
from uhi import features as ft

CLEAR = 21824
PATCH = 30  # 10 m pixels: patches of 300 m
# value of each cover for the synthetic surface temperature (°C against the mean)
HEAT = {ft.BUILDING: 4.0, ft.ROAD: 5.0, ft.GRASS: -2.0, ft.FOREST: -6.0, ft.FIELD: 1.0, ft.WATER: -8.0}
CITY_KM = (9, 6, 21, 18)  # the city: a 12 km square inside the grid
CONVERTED_KM = (12, 9, 13.2, 10.2)  # a field in the early scenes, buildings now
CAR_PARK_KM = (15, 12, 15.6, 12.6)
FIELD_KM = (17.1, 15, 18.3, 16.2)


def _dn(celsius):
    return np.round((celsius + 273.15 - data.core.ST_OFFSET_K) / data.core.ST_SCALE).astype("uint16")


def _km_box(grid, x0, y0, x1, y1):
    minx, _, _, maxy = grid.bounds
    return box(minx + x0 * 1e3, maxy - y1 * 1e3, minx + x1 * 1e3, maxy - y0 * 1e3)


def _synthetic_layers(grid30):
    grid10 = layout.fine_grid(grid30)
    rng = np.random.default_rng(3)
    covers = np.array(list(HEAT))
    ph, pw = grid10.height // PATCH + 1, grid10.width // PATCH + 1
    patches = rng.choice(covers, size=(ph, pw), p=[0.3, 0.15, 0.2, 0.15, 0.15, 0.05])
    cover = np.kron(patches, np.ones((PATCH, PATCH), dtype="uint8"))[: grid10.height, : grid10.width]
    cover = cover.astype("uint8")
    for km, code in ((CONVERTED_KM, ft.BUILDING), (CAR_PARK_KM, ft.ROAD), (FIELD_KM, ft.FIELD)):
        cover[layout.polygon_mask(_km_box(grid30, *km), grid10)] = code
    tall = np.kron(rng.uniform(6, 30, size=(ph, pw)), np.ones((PATCH, PATCH)))[: grid10.height, : grid10.width]
    leafy = np.kron(rng.random((ph, pw)) < 0.5, np.ones((PATCH, PATCH)))[: grid10.height, : grid10.width] > 0
    canopy = np.where((cover == ft.GRASS) & leafy, 0.6, 0.0).astype("float32")
    ndvi = np.select([cover == ft.GRASS, cover == ft.FOREST, cover == ft.FIELD], [0.6, 0.8, 0.4], 0.1)
    ndvi = (ndvi + 0.15 * canopy).astype("float32")
    rows = np.linspace(160, 260, grid30.height, dtype="float32")[:, None]
    return {
        "cover": cover,
        "height": np.where(cover == ft.BUILDING, tall, 0).astype("float32"),
        "canopy": canopy,
        "ndvi": ndvi,
        "elevation": np.broadcast_to(rows, grid30.shape).copy(),
    }


def _truth(layers):
    """Surface anomaly at 30 m: each cover's value, cooler under trees, blurred like the sensor."""
    value = np.zeros(layers["cover"].shape, dtype="float32")
    for code, heat in HEAT.items():
        value[layers["cover"] == code] = heat
    value -= 4.0 * layers["canopy"]
    return ft.disc_mean(ft.block_mean(value), 100 / 30)


def _landuse(grid30, converted_is_built):
    green = ("Grünlandnutzung", "Landwirtschaft", "Acker")
    built = ("Baulandnutzung", "Wohn- u. Mischnutzung", "dichtes Wohn(misch)gebiet")
    parking = ("Verkehr", "weitere verkehrliche Nutzungen", "Parkplätze u. Parkhäuser")
    rows = [
        (built, _km_box(grid30, 0, 0, 31, 26)),
        (green, _km_box(grid30, *FIELD_KM)),
        (parking, _km_box(grid30, *CAR_PARK_KM)),
        (built if converted_is_built else green, _km_box(grid30, *CONVERTED_KM)),
    ]
    return gpd.GeoDataFrame(
        [dict(zip(history.LEVEL_COLUMNS[0], names)) | {"geometry": g} for names, g in rows], crs=data.CRS
    )


def test_fmzk_tiles_are_burnt_and_cached(tmp_path, monkeypatch):
    grid10 = data.make_grid((600000, 5340000, 608000, 5344000), 10.0, 0.0)
    assert len(layout.tiles(grid10)) == 2
    calls = []

    def fetch(layer, bounds):
        calls.append(layer)
        x, y = bounds[0], bounds[1]
        if layer == layout.BUILDING_LAYER:
            return gpd.GeoDataFrame(
                {"F_KLASSE": [11, 11, 86], "O_KOTE": [30.0, 50.0, 9.0], "HOEHE_DGM": [20.0, 20.0, 5.0],
                 "geometry": [box(x, y, x + 1000, y + 1000), box(x + 500, y, x + 1000, y + 500),
                              box(x + 3000, y + 3000, x + 3100, y + 3100)]}, crs=data.CRS)
        if layer == "FMZKGRUEN2OGD":
            return gpd.GeoDataFrame({"F_KLASSE": [53], "geometry": [box(x, y, x + 4000, y + 4000)]}, crs=data.CRS)
        if layer == "FMZKVERKEHR1OGD":
            return gpd.GeoDataFrame({"F_KLASSE": [21, 31], "geometry": [
                box(x, y + 2000, x + 4000, y + 2020), box(x + 2000, y + 3000, x + 2100, y + 3100)]}, crs=data.CRS)
        return gpd.GeoDataFrame({"F_KLASSE": []}, geometry=[], crs=data.CRS)

    monkeypatch.setattr(layout, "fetch_tile", fetch)
    cover, height = layout.fmzk_layers(tmp_path, grid10, workers=1)
    assert cover.shape == grid10.shape
    assert (cover == ft.BUILDING).mean() == pytest.approx(1 / 16)  # buildings lie on top of the meadow
    assert (cover == ft.ROAD).mean() == pytest.approx(20 / 4000)
    assert (cover == ft.GRASS).mean() == pytest.approx(1 - 1 / 16 - 20 / 4000)  # street furniture is ignored
    assert height.max() == 30.0 and height[cover == ft.BUILDING].min() == 10.0  # the taller part wins
    assert (height[cover != ft.BUILDING] == 0).all()
    n = len(calls)
    again, _ = layout.fmzk_layers(tmp_path, grid10, workers=1)
    assert len(calls) == n and (again == cover).all()


def test_tree_crowns():
    assert layout.crown_diameter_m("19-21 m") == 20.0
    assert layout.crown_diameter_m("nicht definiert") == 0.0 and layout.crown_diameter_m(None) == 0.0
    grid = data.make_grid((0, 0, 500, 500), 10.0, 0.0)
    x, y = np.array([255.0, 255.0, 9000.0]), np.array([255.0, 255.0, 255.0])
    canopy = layout.canopy_fraction(x, y, np.array([20.0, 0.0, 20.0]), grid)
    assert canopy.sum() * 100 == pytest.approx(np.pi * 10**2, rel=0.01)  # one crown of 20 m, in m²
    assert canopy.max() <= 1.0 and canopy[0, 0] == 0


def test_model_and_scenarios_end_to_end(tmp_path, monkeypatch):
    grid30 = data.landsat_grid()
    layers = _synthetic_layers(grid30)
    truth_now = _truth(layers)
    before = {k: v.copy() for k, v in layers.items()}
    was_field = layout.polygon_mask(_km_box(grid30, *CONVERTED_KM), layout.fine_grid(grid30))
    before["cover"][was_field] = ft.FIELD
    before["height"][was_field] = 0
    before["ndvi"][was_field] = 0.4
    truth_early = _truth(before)
    ndvi30 = {True: ft.block_mean(before["ndvi"]), False: ft.block_mean(layers["ndvi"])}
    rng = np.random.default_rng(5)

    passes = {}
    for year in range(2013, 2027):
        for day in (5, 21):
            passes[f"{year}-07-{day:02d}_landsat-8_p190"] = [
                SimpleNamespace(datetime=datetime(year, 7, day, 9, 44, tzinfo=UTC), properties={})
            ]

    def load(items, grid_, cache):
        when = items[0].datetime
        early = when.year < 2017
        surface = 30.0 + when.day / 10 + (truth_early if early else truth_now)
        surface = surface + rng.normal(0, 0.3, grid30.shape)
        nd = ndvi30[early]
        red = np.full(grid30.shape, 0.1)
        nir = red * (1 + nd) / (1 - nd)
        to_dn = lambda r: np.round((r - data.core.SR_OFFSET) / data.core.SR_SCALE).astype("uint16")
        qa = np.full(grid30.shape, CLEAR, dtype="uint16")
        if when.year == 2020 and when.day == 5:
            qa[:] = 22280  # a cloudy scene
        return {"lwir11": _dn(surface), "red": to_dn(red), "nir08": to_dn(nir), "qa_pixel": qa}

    city = _km_box(grid30, *CITY_KM)
    wfs_layers = {
        "LANDESGRENZEOGD": gpd.GeoDataFrame({"geometry": [city]}, crs=data.CRS),
        "REALNUT2012OGD": _landuse(grid30, False), "REALNUT2014OGD": _landuse(grid30, False),
        "REALNUT2022OGD": _landuse(grid30, True), "REALNUT2024OGD": _landuse(grid30, True),
    }
    grid10 = layout.fine_grid(grid30)
    t = grid10.transform
    park_rows = []
    labels = (layers["cover"] == ft.FOREST) | ((layers["cover"] == ft.GRASS) & (layers["canopy"] > 0))
    for r in range(700, 1700, PATCH):
        for c in range(1000, 2000, PATCH):
            if labels[r + 1, c + 1] and len(park_rows) < 12:
                geom = box(t.c + c * 10, t.f - (r + PATCH) * 10, t.c + (c + PATCH) * 10, t.f - r * 10)
                park_rows.append({"site": f"Park {len(park_rows)}", "size_class": "medium", "geometry": geom})
    parks = gpd.GeoDataFrame(park_rows, crs=data.CRS)

    monkeypatch.setattr(data, "wfs", lambda layer, cache: wfs_layers[layer])
    monkeypatch.setattr(data, "landsat_items", lambda *a, **k: [])
    monkeypatch.setattr(data, "fetch_landsat", lambda *a, **k: None)
    monkeypatch.setattr(data, "landsat_passes", lambda items: passes)
    monkeypatch.setattr(data, "load_landsat_pass", load)
    monkeypatch.setattr(data, "landsat_gap_mask", lambda *a, **k: np.zeros(grid30.shape, dtype=bool))
    monkeypatch.setattr(data, "park_sites", lambda cache: parks)
    monkeypatch.setattr(layout, "build_layers", lambda *a, **k: layers)
    monkeypatch.setattr(history, "ghsl_built_fraction",
                        lambda cache, epoch, grid: ft.block_mean(
                            ((before if epoch == 2015 else layers)["cover"] == ft.BUILDING).astype("float32")))
    hot = {f"{y}-07-21" for y in range(2019, 2027)}
    monkeypatch.setattr(model, "hot_dates", lambda cache, first, last: hot)
    monkeypatch.setattr(model, "MIN_HOT_SCENES", 8)
    cache, out = tmp_path / "c", tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["model", "--cache", str(cache), "--out", str(out), "--hot-days"])

    model.main()

    stats = json.loads((out / "stats.json").read_text())
    assert stats["target"]["scenes_recent"] == 15 and stats["target"]["scenes_early"] == 8
    assert stats["target"]["cells_with_target"] == stats["target"]["cells_in_city"] > 15000
    cv = stats["cross_validation"]
    assert cv["boosting"]["away_from_training_blocks"]["r2"] > 0.9
    assert cv["boosting"]["away_from_training_blocks"]["n"] < cv["boosting"]["all_test_cells"]["n"]
    assert cv["linear"]["all_test_cells"]["r2"] > 0.5
    assert 0.3 < stats["cells_near_another_fold_share"] < 0.9
    assert stats["check_parks"]["parks"] == 12 and stats["check_parks"]["r"] > 0.8
    assert stats["check_parks"]["observed_mean_c"] < -1

    conv = stats["check_conversions"]["with_ndvi"]
    assert conv["converted_cells"] > 100  # 144 ha
    observed = conv["converted"]["observed_mean_c"]
    assert observed == pytest.approx(HEAT[ft.BUILDING] - HEAT[ft.FIELD], abs=0.6)
    assert conv["converted"]["predicted_mean_c"] > 0.4 * observed
    assert abs(conv["unchanged"]["observed_mean_c"]) < 0.1
    assert abs(conv["unchanged"]["predicted_mean_c"]) < 0.3
    assert stats["hot_days"]["fitted"] and stats["hot_days"]["hot_day_scenes"] == 8
    assert stats["hot_days"]["r_with_main_target"] > 0.95

    demo = stats["scenarios"]
    assert set(demo) == {"car_park_to_tree_park", "car_park_to_lawn", "pocket_park_in_car_park", "field_to_housing"}
    park, lawn, pocket = (demo[k]["inside"]["mean_change_c"] for k in
                          ("car_park_to_tree_park", "car_park_to_lawn", "pocket_park_in_car_park"))
    # road -> trees is -11.4 °C in the synthetic truth, road -> lawn -7; a 1 ha park is diluted
    assert park < lawn < -3 and park < pocket < 0
    assert demo["car_park_to_tree_park"]["inside"]["mean_low_c"] < park < demo["car_park_to_tree_park"]["inside"]["mean_high_c"]
    assert demo["car_park_to_tree_park"]["around"]["cells"] > demo["car_park_to_tree_park"]["inside"]["cells"]
    # field (+1) -> 40 % building, 20 % paving, 40 % lawn (+0.8): hardly any change
    assert abs(demo["field_to_housing"]["inside"]["mean_change_c"]) < 1
    for name in ("maps.png", "importance.png", "partial_dependence.png", "checks.png",
                 "scenarios/car_park_to_tree_park.png", "scenarios/examples.geojson",
                 "cv_scores.csv", "importance.csv", "parks_check.csv", "conversion_check.csv"):
        assert (out / name).stat().st_size > 100, name
    assert len(pd.read_csv(out / "cv_scores.csv")) == 4

    # The scenario tool, from a user's file, with the saved model
    np.savez_compressed(cache / "model" / "layers.npz", **layers)
    edit = gpd.GeoDataFrame(
        {"name": ["roof"], "cover": ["building"], "height_m": [200.0],
         "geometry": [_km_box(grid30, *FIELD_KM)]}, crs=data.CRS,
    ).to_crs("EPSG:4326")
    path = tmp_path / "tower.geojson"
    edit.to_file(path, driver="GeoJSON")
    monkeypatch.setattr(sys, "argv", ["scenario", str(path), "--cache", str(cache), "--out", str(tmp_path / "s")])
    scenario.main()
    summary = json.loads((tmp_path / "s" / "tower.json").read_text())
    assert summary["inside"]["mean_change_c"] > 1  # field -> building
    assert summary["cells_outside_training_range"] > 0  # nothing in the city is 200 m tall
    cells = pd.read_csv(tmp_path / "s" / "tower_cells.csv")
    assert {"change_c", "low_c", "high_c", "features_outside_training"} <= set(cells.columns)
    assert (tmp_path / "s" / "tower.png").stat().st_size > 5000
    with pytest.raises(SystemExit, match="cover"):
        edit.drop(columns="cover").to_file(tmp_path / "bad.geojson", driver="GeoJSON")
        scenario.read_edits(tmp_path / "bad.geojson")
