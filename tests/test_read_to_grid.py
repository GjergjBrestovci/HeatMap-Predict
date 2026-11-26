"""`read_to_grid` against a small local GeoTIFF. No network."""

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from affine import Affine

from uhi import data


def _write(path, array, nodata, x0=600000.0, y1=5340000.0):
    with rasterio.open(
        path, "w", driver="GTiff", width=array.shape[1], height=array.shape[0], count=1,
        dtype="uint16", crs=data.CRS, transform=Affine(30, 0, x0, 0, -30, y1), nodata=nodata,
    ) as dst:
        dst.write(array, 1)


def test_values_and_nodata_survive_on_the_same_grid(tmp_path):
    array = np.array([[0, 44000, 45000], [46000, 0, 47000]], dtype="uint16")
    _write(tmp_path / "st.tif", array, nodata=0)
    grid = data.Grid(Affine(30, 0, 600000.0, 0, -30, 5340000.0), 3, 2)
    out = data.read_to_grid(tmp_path / "st.tif", grid)
    # No-data zeros must stay 0, not become 1.
    np.testing.assert_array_equal(out, array)


def test_cells_outside_the_raster_are_zero(tmp_path):
    array = np.full((2, 2), 44000, dtype="uint16")
    _write(tmp_path / "st.tif", array, nodata=0)
    grid = data.Grid(Affine(30, 0, 600000.0 - 30, 0, -30, 5340000.0), 4, 2)  # one column wider each side
    out = data.read_to_grid(tmp_path / "st.tif", grid)
    np.testing.assert_array_equal(out, [[0, 44000, 44000, 0], [0, 44000, 44000, 0]])


def test_qa_fill_value_becomes_zero(tmp_path):
    # QA_PIXEL uses 1 for fill; 21824 is clear land.
    array = np.array([[1, 21824], [21824, 1]], dtype="uint16")
    _write(tmp_path / "qa.tif", array, nodata=1)
    grid = data.Grid(Affine(30, 0, 600000.0, 0, -30, 5340000.0), 2, 2)
    out = data.read_to_grid(tmp_path / "qa.tif", grid)
    np.testing.assert_array_equal(out, [[0, 21824], [21824, 0]])


def test_make_grid_aligns_to_landsat_pixel_edges():
    grid = data.make_grid((587450.0, 5330000.0, 587500.0, 5330100.0), 30.0, 15.0)
    assert (grid.transform.c - 15) % 30 == 0
    assert (grid.transform.f - 15) % 30 == 0
    minx, miny, maxx, maxy = grid.bounds
    assert minx <= 587450.0 and maxx >= 587500.0 and miny <= 5330000.0 and maxy >= 5330100.0
