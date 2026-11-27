"""The warming split on a synthetic city. No network, no real data."""

import json
import sys
import zipfile
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

gpd = pytest.importorskip("geopandas")
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from shapely.geometry import box

from uhi import climate, data, history, warming_split
from uhi import split_stats as st

CLEAR = 21824
EDITIONS = ((1991, "A"), (1997, "B"), (2005, "C"), (2008, "D"), (2016, "E"), (2024, "F"))
REGIONAL, URBAN_EXTRA = 0.05, 0.01  # °C per year


def _dn(celsius):
    return np.round((celsius + 273.15 - data.core.ST_OFFSET_K) / data.core.ST_SCALE).astype("uint16")


def _blocks(grid):
    """Named rectangles of the synthetic city, in metres from the top-left corner."""
    minx, _, _, maxy = grid.bounds
    km = lambda x0, y0, x1, y1: box(minx + x0 * 1e3, maxy - y1 * 1e3, minx + x1 * 1e3, maxy - y0 * 1e3)
    return {
        "forest": km(1, 1, 7, 9),
        "farmland": km(8, 1, 14, 9),
        "town": km(15, 1, 21, 7),
        "early": km(8, 11, 10, 13),  # farmland until 1997, sealed from 2005
        "late": km(12, 11, 14, 13),  # farmland until 2008, sealed from 2016
        "site": km(16, 11, 18, 13),  # farmland, construction site in 2008, sealed from 2016
    }


def _layers(grid):
    b = _blocks(grid)
    sealed_from = {"early": 2005, "late": 2016, "site": 2016}
    layers = {}
    for year, name in EDITIONS:
        state = {"forest": "forest", "farmland": "farmland", "town": "built"}
        for block, first in sealed_from.items():
            state[block] = "built" if year >= first else "farmland"
        if year == 2008:
            state["site"] = "site"
        if year <= 2005:
            code = {"forest": "WL", "farmland": "AC", "built": "WO"}
            rows = [{"NUTCD": code[s], "geometry": b[k]} for k, s in state.items()]
            # Block outlines without a code lie on top of everything in the early editions.
            rows.append({"NUTCD": None, "geometry": box(*grid.bounds)})
            layers[name] = gpd.GeoDataFrame(rows, crs=data.CRS)
        else:
            named = {
                "forest": ("Grünlandnutzung", "Naturraum", "Wald"),
                "farmland": ("Grünlandnutzung", "Landwirtschaft", "Acker"),
                "built": ("Baulandnutzung", "Wohn- u. Mischnutzung", "dichtes Wohn(misch)gebiet"),
                "site": ("Baulandnutzung", "Technische Infrastruktur", "Transformationsfläche, Baustelle"),
            }
            cols = history.LEVEL_COLUMNS[1] if year == 2024 else history.LEVEL_COLUMNS[0]
            rows = [dict(zip(cols, named[s])) | {"geometry": b[k]} for k, s in state.items()]
            layers[name] = gpd.GeoDataFrame(rows, crs=data.CRS)
    layers["LANDESGRENZEOGD"] = gpd.GeoDataFrame({"geometry": [box(*grid.bounds)]}, crs=data.CRS)
    return layers


def test_landuse_stack_translates_codes_and_caches(tmp_path, monkeypatch):
    grid = data.landsat_grid()
    layers = _layers(grid)
    monkeypatch.setattr(history, "EDITIONS", EDITIONS)
    monkeypatch.setattr(data, "wfs", lambda layer, cache: layers[layer])
    stack, years, crosswalk = history.landuse_stack(tmp_path, grid)
    assert years == [y for y, _ in EDITIONS] and stack.shape == (6, *grid.shape)
    assert crosswalk.set_index("code")["group"].to_dict() == {
        "AC": "farmland", "WL": "forest", "WO": "built"
    }
    inside = np.ones(grid.shape, dtype=bool)
    area = history.area_by_edition(stack, years, inside, 0.09).set_index("year")
    assert area.loc[1991, "built_ha"] == pytest.approx(3600, rel=0.02)  # the town, 6 km x 6 km
    assert area.loc[2024, "built_ha"] - area.loc[1991, "built_ha"] == pytest.approx(1200, rel=0.03)
    assert area.loc[2008, "other_ha"] == pytest.approx(400, rel=0.03)
    assert area.loc[1991, "unmapped_ha"] > 0  # the outline without a code is not land use

    monkeypatch.setattr(data, "wfs", lambda layer, cache: pytest.fail("not cached"))
    again, years2, crosswalk2 = history.landuse_stack(tmp_path, grid)
    assert (again == stack).all() and years2 == years and len(crosswalk2) == 3


def test_ghsl_reads_the_tile_from_its_zip(tmp_path):
    grid = data.landsat_grid()
    minx, miny, maxx, maxy = rasterio.warp.transform_bounds(data.CRS, "ESRI:54009", *grid.bounds)
    tif = f"GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100_V1_0_{history.GHSL_TILE}.tif"
    values = np.full((400, 500), 2500, dtype="uint16")
    values[150:160, :] = 65535
    with rasterio.open(
        tmp_path / tif, "w", driver="GTiff", width=500, height=400, count=1, dtype="uint16",
        crs="ESRI:54009", transform=from_origin(minx - 5000, maxy + 5000, 100, 100), nodata=65535,
    ) as dst:
        dst.write(values, 1)
    (tmp_path / "ghsl").mkdir()
    with zipfile.ZipFile(tmp_path / "ghsl" / f"GHS_BUILT_S_E2020_{history.GHSL_TILE}.zip", "w") as z:
        z.write(tmp_path / tif, tif)
    out = history.ghsl_built_fraction(tmp_path, 2020, grid)
    assert out.shape == grid.shape
    assert np.median(out) == pytest.approx(0.25)
    assert out.min() == 0.0  # no-data cells count as not built


def test_parsers_and_series_choice():
    text = (
        "time,station,tl_mittel,tlmin_mittel,tlmax_mittel,substation\n"
        "1981-06-01T00:00+00:00,105,18.0,13.0,23.5,5901\n"
        "1981-07-01T00:00+00:00,105,,14.0,25.0,\n"
    )
    out = climate.parse_station_csv(text)
    assert out["station"].tolist() == ["105", "105"] and out["substation"].tolist() == ["5901", ""]
    assert out["mean_c"].iloc[0] == 18.0 and np.isnan(out["mean_c"].iloc[1])
    with pytest.raises(SystemExit, match="lacks"):
        climate.parse_station_csv("time,station,tl\n")
    grid = climate.parse_spartacus_csv(
        "time,TM [degC],TN [degC],TX [degC],lat,lon\n1981-06-01T00:00+00:00,19.1,13.9,24.4,48.2,16.9\n"
    )
    assert grid[["mean_c", "min_c", "max_c"]].iloc[0].tolist() == [19.1, 13.9, 24.4]
    assert climate.lattice((0, 0, 4, 2), 2) == [(1.0, 1.0), (3.0, 1.0)]
    hom = climate.parse_histalp_csv(
        "# unit: 0,1 °C\n# missing-value: 999999\n#-----\n"
        "year;jan;feb;mar;apr;may;jun;jul;aug;sep;oct;nov;dec;jun-aug\n"
        "2025;18;17;79;129;144;220;212;212;172;106;51;26;215\n"
        "2026;-12;40;83;119;173;999999;999999;999999;999999;999999;999999;999999;999999\n"
    )
    assert hom["year"].tolist() == [2025, 2026]
    assert hom["mean_c"].iloc[0] == pytest.approx(21.467, abs=0.001) and np.isnan(hom["mean_c"].iloc[1])
    with pytest.raises(SystemExit, match="lacks"):
        climate.parse_histalp_csv("year;jan\n2025;1\n")

    months = pd.date_range("1981-01-01", "1990-12-01", freq="MS", tz="UTC")
    def series(sid, sites):
        return pd.DataFrame({"time": months, "station": sid, "mean_c": 20.0,
                             "substation": [sites[0] if t.year < 1986 else sites[-1] for t in months]})
    monthly = pd.concat([series("31", ["5970", "5972"]), series("5972", ["5972"]).iloc[60:]])
    stations = pd.DataFrame({"id": ["31", "5972"], "name": ["Feld", "Feld"]})
    summers = st.summer_means(monthly, "mean_c", "station")
    picked = climate.pick_series(monthly, stations, summers, min_summers=5)
    assert picked["id"].tolist() == ["31"] and picked["summers"].tolist() == [10]
    assert picked["sites"].tolist() == [2] and picked["site_change_years"].tolist() == [[1986]]


def test_warming_split_end_to_end(tmp_path, monkeypatch):
    grid = data.landsat_grid()
    blocks = _blocks(grid)
    layers = _layers(grid)
    mask = {k: geometry_mask([g], grid.shape, grid.transform, invert=True) for k, g in blocks.items()}
    sealed_from = {"early": 2001, "late": 2012, "site": 2012}  # on the ground, between the maps

    def platform(year):
        return "landsat-5" if year < 2012 else "landsat-7" if year == 2012 else "landsat-8"

    passes = {
        f"{y}-07-01_{platform(y)}_p190": [
            SimpleNamespace(datetime=datetime(y, 7, 1, 9, 30, tzinfo=UTC),
                            properties={"platform": platform(y), "view:sun_elevation": 55.0})
        ]
        for y in range(1986, 2025)
    }

    def load(items, grid_, cache):
        year = items[0].datetime.year
        day = 25.0 + (year % 5)
        surface = np.full(grid.shape, day)
        surface[mask["forest"]] = day - 2
        surface[mask["farmland"]] = day + 2
        surface[mask["town"]] = day + 8
        for block, first in sealed_from.items():
            surface[mask[block]] = day + 7 if year >= first else day + 2
        qa = np.full(grid.shape, CLEAR, dtype="uint16")
        if year == 2012:
            qa[::7] = 1  # striped gaps of Landsat 7
        if year == 1990:
            qa[:] = 22280  # cloud everywhere
        return {"lwir11": _dn(surface), "red": np.full(grid.shape, 10000, dtype="uint16"),
                "nir08": np.full(grid.shape, 20000, dtype="uint16"), "qa_pixel": qa}

    def ghsl(cache, epoch, grid_):
        town = geometry_mask([blocks["town"]], grid_.shape, grid_.transform, invert=True)
        return np.where(town, 0.6, 0.0).astype("float32")

    centre = blocks["town"].centroid
    lon, lat = gpd.GeoSeries([centre], crs=data.CRS).to_crs("EPSG:4326").iloc[0].coords[0]
    meta = pd.DataFrame(
        [("1", "Wien Innere Stadt", lon, lat, 170.0), ("2", "Wien Hohe Warte", lon + 0.01, lat, 200.0),
         ("3", "Feld", 16.75, 48.20, 150.0), ("4", "Acker", 16.80, 48.30, 160.0),
         ("99", "Feld", 16.75, 48.20, 150.0), ("5", "Berg", 16.75, 48.25, 500.0),
         ("6", "Fern", 19.0, 48.2, 150.0)],
        columns=["id", "name", "lon", "lat", "altitude"],
    )

    def monthly(ids, first, last, cache):
        assert "5" not in ids and "6" not in ids  # too high, too far
        months = pd.date_range(f"{first}-01-01", f"{last}-12-01", freq="MS", tz="UTC")
        frames = []
        for sid in ids:
            t = 18.0 + REGIONAL * (months.year - 1981)
            if sid in ("1", "2"):
                t = t + 1.0 + URBAN_EXTRA * (months.year - 1981)
            f = pd.DataFrame({"time": months, "station": sid, "substation": sid,
                              "mean_c": t, "min_c": t - 5, "max_c": t + 5})
            if sid == "1":
                f = f[f["time"].dt.year >= 1985]  # Innere Stadt starts in 1985
            if sid == "99":
                f = f[f["time"].dt.year >= 2015]
            frames.append(f)
        return pd.concat(frames, ignore_index=True)

    def spartacus(points, first, last, cache, tag):
        months = pd.date_range(f"{first}-01-01", f"{last}-12-01", freq="MS", tz="UTC")
        t = 18.0 + REGIONAL * (months.year - 1981) + (0.5 if tag == "built-up" else 0.0)
        return pd.concat(
            [pd.DataFrame({"time": months, "lat": la, "lon": lo, "mean_c": t, "min_c": t - 5,
                           "max_c": t + 5}) for la, lo in points],
            ignore_index=True,
        )

    def histalp(first, last, cache):
        years = np.arange(first, last + 1)
        base = 18.0 + REGIONAL * (years - 1981)
        table = pd.DataFrame({n: base for n in climate.HISTALP_STATIONS}, index=years)
        table["Wien-Hohe Warte"] += 0.8 + 0.004 * (years - 1981)
        table.loc[last, "Retz"] = np.nan  # the current summer is not complete yet
        return table

    monkeypatch.setattr(climate, "fetch_histalp", histalp)
    monkeypatch.setattr(history, "EDITIONS", EDITIONS)
    monkeypatch.setattr(history, "ghsl_built_fraction", ghsl)
    monkeypatch.setattr(data, "wfs", lambda layer, cache: layers[layer])
    monkeypatch.setattr(data, "landsat_items", lambda *a, **k: [])
    monkeypatch.setattr(data, "fetch_landsat", lambda *a, **k: None)
    monkeypatch.setattr(data, "landsat_passes", lambda items: passes)
    monkeypatch.setattr(data, "load_landsat_pass", load)
    monkeypatch.setattr(climate, "station_metadata", lambda cache: meta.copy())
    monkeypatch.setattr(climate, "fetch_station_monthly", monthly)
    monkeypatch.setattr(climate, "fetch_spartacus", spartacus)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["warming_split", "--years", "1981-2024",
                                      "--cache", str(tmp_path / "c"), "--out", str(out)])

    warming_split.main()

    stats = json.loads((out / "stats.json").read_text())
    air = stats["air"]
    assert air["urban_stations"] == ["Wien Innere Stadt", "Wien Hohe Warte"]
    assert air["rural_stations"] == ["Feld", "Acker"]
    split = air["split"]["mean_c"]
    assert split["urban"]["first_year"] == 1985 and split["urban"]["last_year"] == 2024
    assert split["regional"]["slope_per_decade"] == pytest.approx(0.5, abs=0.002)
    assert split["urban_extra"]["slope_per_decade"] == pytest.approx(0.1, abs=0.002)
    head = stats["headline"]
    assert head["raw_stations_regional_air_warming_c"] == pytest.approx(REGIONAL * 39, abs=0.01)
    assert head["raw_stations_urban_extra_air_warming_c"] == pytest.approx(URBAN_EXTRA * 39, abs=0.01)
    # The homogenised series lead the headline: 1981-2023, the last summer has a gap.
    assert head["summers"] == [1981, 2023]
    assert head["regional_air_warming_c"] == pytest.approx(REGIONAL * 42, abs=0.01)
    assert head["urban_extra_air_warming_c"] == pytest.approx(0.004 * 42, abs=0.01)
    assert head["urban_air_warming_c"] == pytest.approx((REGIONAL + 0.004) * 42, abs=0.01)
    assert set(air["split_homogenised"]["urban_extra_per_decade_against_each_rural_station"]) == {
        "Eisenstadt", "Retz", "St. Pölten"
    }
    assert len(pd.read_csv(out / "air_homogenised.csv")) == 44
    assert air["grid_city_against_grid_ring"]["mean_c"]["urban_extra"]["slope_per_decade"] == pytest.approx(0, abs=0.002)
    assert air["grid_points"]["rural_ring"] > 50 and air["grid_points"]["built_up_city"] >= 4

    surface = stats["surface"]
    assert surface["overpasses_used"] == 38
    assert surface["overpasses_skipped"] == {"city too cloudy": 1}
    assert surface["overpasses_used_by_sensor"] == {"TM": 25, "OLI/TIRS": 12, "ETM+": 1}
    gap = surface["sealed_minus_green"]["built_minus_farmland_c"]
    assert gap["median_by_sensor"]["TM"] == pytest.approx(6.0, abs=0.05)
    assert gap["trend_of_yearly_medians"]["slope_per_decade"] == pytest.approx(0, abs=0.01)
    assert surface["ghsl_built_share_in_zones"]["stable_built"]["built_share_2020"] == pytest.approx(0.6)
    assert surface["ghsl_built_share_in_zones"]["converted"]["built_share_2020"] == pytest.approx(0.0)
    assert surface["converted_ha"] == pytest.approx(1200, rel=0.03)
    for ref in ("stable_green", "stable_farmland"):
        pooled = surface["converted_land"][ref]
        assert pooled["effect"]["cohorts"] == 3
        assert pooled["effect"]["weighted_mean_c"] == pytest.approx(5.0, abs=0.05)
    assert head["converted_land_surface_warming_c"] == pytest.approx(5.0, abs=0.05)

    cohorts = pd.read_csv(out / "cohorts.csv")
    farm = cohorts[cohorts["reference"] == "stable_farmland"].set_index("cohort")
    assert sorted(farm.index) == ["conv_1997_2005", "conv_2005_2016", "conv_2008_2016"]
    assert farm.loc["conv_1997_2005", "before_median_c"] == pytest.approx(0.0, abs=0.05)
    assert farm.loc["conv_1997_2005", "effect_same_sensor_c"] == pytest.approx(5.0, abs=0.05)
    assert not farm["scheme_change"].any()  # 2005 -> 2016 passes through a construction site
    assert (farm["main_origin"] == "farmland").all()

    event = pd.read_csv(out / "event_study.csv").set_index("bin")
    assert event.loc[-1, "shift_c"] == pytest.approx(0.0, abs=0.05)
    assert event.loc[0, "shift_c"] == pytest.approx(5.0, abs=0.05)
    landuse = pd.read_csv(out / "landuse_by_edition.csv").set_index("year")
    assert landuse.loc[2020, "ghsl_built_ha"] == pytest.approx(0.6 * 3600, rel=0.02)
    assert landuse.loc[2024, "built_ha"] > landuse.loc[1991, "built_ha"]
    trends = pd.read_csv(out / "air_trends.csv")
    assert set(trends["measure"]) == {"mean_c", "min_c", "max_c"}
    assert len(pd.read_csv(out / "scenes.csv")) == 39
    for name in ("air_trends.png", "land_use.png", "surface_gap.png", "converted_land.png"):
        assert (out / name).stat().st_size > 5000
