"""Layout features, scenario edits and model checks on small arrays."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
from uhi import features as ft
from uhi import model_stats as ms


def test_block_mean_ignores_nan_and_needs_enough_values():
    a = np.arange(36, dtype="float32").reshape(6, 6)
    a[0, 0] = np.nan
    out = ft.block_mean(a, 3)
    assert out.shape == (2, 2)
    assert out[0, 0] == pytest.approx(np.nanmean(a[:3, :3]))
    assert np.isnan(ft.block_mean(a, 3, min_valid=9)[0, 0])
    assert ft.block_mean(np.ones((7, 8)), 3).shape == (2, 2)  # incomplete blocks are cut off


def test_disc_mean_matches_a_direct_count():
    a = np.zeros((21, 21), dtype="float32")
    a[10, 10] = 1.0
    out = ft.disc_mean(a, 2.0)
    assert out[10, 10] == pytest.approx(1 / 13)  # 13 pixels lie within 2 pixels
    assert out[10, 13] == pytest.approx(0, abs=1e-6)
    assert ft.disc_mean(np.ones((9, 9), dtype="float32"), 3.0) == pytest.approx(1.0)  # edges too
    b = np.ones((9, 9), dtype="float32")
    b[4, 4] = np.nan
    assert ft.disc_mean(b, 1.0)[4, 4] == pytest.approx(1.0)  # missing values do not count


def _layers(shape30=(18, 18)):
    h, w = shape30[0] * 3, shape30[1] * 3
    cover = np.full((h, w), ft.GRASS, dtype="uint8")
    cover[:, : w // 2] = ft.BUILDING
    cover[:9, -9:] = ft.WATER
    return {
        "cover": cover,
        "height": np.where(cover == ft.BUILDING, 20.0, 0.0).astype("float32"),
        "canopy": np.where(cover == ft.GRASS, 0.5, 0.0).astype("float32"),
        "ndvi": np.where(cover == ft.GRASS, 0.7, 0.1).astype("float32"),
        "elevation": np.full(shape30, 170.0, dtype="float32"),
    }


def test_build_features_shapes_scales_and_no_coordinates():
    f = ft.build_features(_layers())
    assert all(v.shape == (6, 6) for v in f.values())
    assert f["building"][2, 0] == 1.0 and f["building"][2, 5] == 0.0
    assert f["height"][2, 0] == pytest.approx(20.0)
    assert 0 < f["building_500m"][2, 5] < f["building_500m"][2, 0] <= 1
    assert f["water"][0, 5] == 1.0 and f["water_dist_m"][0, 5] == 0.0
    assert f["water_dist_m"][5, 0] > f["water_dist_m"][1, 4] > 0
    assert "canopy_150m" in f and "canopy_500m" not in f
    assert not any(n in f for n in ("x", "y", "row", "col", "district"))
    assert ft.matrix(f, sorted(f)).shape == (36, len(f))


def test_scene_anomaly_and_composite():
    lst = np.array([[30.0, 34.0], [np.nan, 40.0]])
    clear = np.array([[True, True], [True, False]])
    city = np.array([[True, True], [True, True]])
    a = ft.scene_anomaly(lst, clear, city)
    assert a[0, 0] == pytest.approx(-2.0) and a[0, 1] == pytest.approx(2.0)
    assert np.isnan(a[1, 0]) and np.isnan(a[1, 1])
    comp = ft.composite([a, a + 1, np.full((2, 2), np.nan, dtype="float32")], min_scenes=2)
    assert comp[0, 0] == pytest.approx(-1.5) and np.isnan(comp[1, 1])
    cells = ft.to_cells(np.where(np.arange(81).reshape(9, 9) % 9 < 3, np.nan, 1.0))
    assert cells.shape == (3, 3) and np.isnan(cells[0, 0]) and cells[0, 1] == 1.0


def test_apply_edit_and_typical_ndvi():
    layers = _layers()
    ndvi_of = ft.typical_ndvi(layers)
    assert ndvi_of["building"] == pytest.approx(0.1) and ndvi_of["trees"] == pytest.approx(0.7)
    assert np.isnan(ndvi_of["lawn"])  # no grass without trees in this city
    ndvi_of["lawn"] = 0.5
    mask = np.zeros(layers["cover"].shape, dtype=bool)
    mask[20:30, 0:10] = True
    park = ft.apply_edit(layers, mask, "trees", ndvi_of)
    assert (park["cover"][mask] == ft.GRASS).all() and (park["height"][mask] == 0).all()
    assert (park["canopy"][mask] == pytest.approx(0.7)) and (layers["cover"][mask] == ft.BUILDING).all()
    block = ft.apply_edit(park, mask, "building", ndvi_of, height_m=30)
    assert (block["height"][mask] == 30).all() and (block["canopy"][mask] == 0).all()
    housing = ft.apply_edit(layers, mask, "housing", ndvi_of, height_m=15)
    share = (housing["cover"][mask] == ft.BUILDING).mean()
    assert share == pytest.approx(0.4) and (housing["cover"][mask] == ft.PAVED).mean() == pytest.approx(0.2)
    assert housing["height"][mask].max() == 15
    with pytest.raises(SystemExit, match="Unknown cover"):
        ft.apply_edit(layers, mask, "volcano", ndvi_of)


def test_block_folds_keep_blocks_together_and_mark_edges():
    folds = ms.block_folds((60, 80), block=20, n_folds=3)
    assert set(np.unique(folds)) == {0, 1, 2}
    for r in range(0, 60, 20):
        for c in range(0, 80, 20):
            assert len(np.unique(folds[r : r + 20, c : c + 20])) == 1
    edge = ms.near_other_fold(folds, radius_cells=5)
    assert not edge[10, 10] or folds[10, 10] != folds[10, 30]  # the middle of a block is safe ...
    assert not edge[9:11, 9:11].any()
    diff = folds[:, 19] != folds[:, 20]
    assert edge[diff, 19].all() and edge[diff, 20].all()  # ... its rim next to another fold is not
    assert 0.2 < edge.mean() < 0.9


def test_cross_validation_never_tests_on_training_rows():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 3)).astype("float32")
    y = 2 * X[:, 0] - X[:, 1] + rng.normal(0, 0.1, 600)
    fold = np.repeat(np.arange(5), 120)
    pred, models = ms.cross_validate(X, y, fold, "linear")
    assert len(models) == 5 and ms.scores(y, pred)["r2"] > 0.98
    shuffled = y.copy()
    shuffled[fold == 0] = 100.0  # the model for fold 0 must not learn this
    pred2, _ = ms.cross_validate(X, shuffled, fold, "linear")
    assert np.abs(pred2[fold == 0]).max() < 20
    report = ms.cv_report(y, pred, fold == 1)
    assert report["all_test_cells"]["n"] == 600 and report["away_from_training_blocks"]["n"] == 480
    boosted, models_b = ms.cross_validate(X, y, fold, "boosting")
    assert ms.scores(y, boosted)["r2"] > 0.9
    imp = ms.held_out_importance(models_b, X, y, fold, ["a_150m", "a", "noise"], repeats=1)
    assert imp["feature"].iloc[-1] == "noise"
    grouped = ms.group_importance(imp)
    assert grouped["quantity"].tolist()[0] == "a" and len(grouped) == 2
    curve = ms.partial_dependence(models_b, X, 0, np.array([-1.0, 0.0, 1.0]))
    assert curve[0] < curve[1] < curve[2]


def test_change_band_and_training_range():
    before = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    after = np.array([[0.0, -1.0], [0.0, -2.0], [0.0, -3.0]])
    mean, low, high = ms.change_band(before, after, held_out_mae=0.5)
    assert mean.tolist() == [-1.0, -3.0]
    assert (high[0] - mean[0]) == pytest.approx(0.5)  # models agree: only the held-out error
    assert (high[1] - mean[1]) > 1.5 and low[1] < mean[1]
    X = np.column_stack([np.linspace(0, 1, 101), np.linspace(10, 20, 101)])
    limits = ms.training_range(X)
    assert ms.outside_range(np.array([[0.5, 15.0], [1.5, 15.0], [-1.0, 30.0]]), limits).tolist() == [0, 1, 2]
