"""Model fitting and checking: spatial folds, cross-validation, uncertainty band.

Works on plain arrays; the maps are flattened by the caller.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import disc

N_FOLDS = 5
BLOCK_CELLS = 22  # 22 cells of 90 m: blocks of about 2 km
BUFFER_CELLS = 500 / 90  # the widest feature reaches 500 m
SEED = 0


def block_folds(shape: tuple[int, int], block: int = BLOCK_CELLS, n_folds: int = N_FOLDS,
                seed: int = SEED) -> np.ndarray:
    """Fold number per cell: square blocks, each block as a whole in one fold."""
    rows = np.arange(shape[0]) // block
    cols = np.arange(shape[1]) // block
    n_cols = cols.max() + 1
    block_id = rows[:, None] * n_cols + cols[None, :]
    n_blocks = int(block_id.max()) + 1
    assignment = np.random.default_rng(seed).permutation(n_blocks) % n_folds
    return assignment[block_id].astype("int16")


def near_other_fold(folds: np.ndarray, radius_cells: float = BUFFER_CELLS) -> np.ndarray:
    """True for cells that have a cell of another fold within `radius_cells`.

    Their neighbourhood features overlap with training data, so they are an
    easier test than a place the model has never seen.
    """
    footprint = disc(radius_cells) > 0
    highest = ndimage.maximum_filter(folds, footprint=footprint, mode="nearest")
    lowest = ndimage.minimum_filter(folds, footprint=footprint, mode="nearest")
    return highest != lowest


def make_model(kind: str):
    if kind == "linear":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    if kind == "boosting":
        return HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=1.0, random_state=SEED,
        )
    raise ValueError(kind)


def scores(y: np.ndarray, pred: np.ndarray) -> dict:
    keep = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[keep], pred[keep]
    if len(y) < 2:
        return {"n": int(len(y)), "r2": np.nan, "mae": np.nan, "rmse": np.nan}
    resid = y - pred
    return {
        "n": int(len(y)),
        "r2": float(1 - (resid**2).sum() / ((y - y.mean()) ** 2).sum()),
        "mae": float(np.abs(resid).mean()),
        "rmse": float(np.sqrt((resid**2).mean())),
    }


def cross_validate(X: np.ndarray, y: np.ndarray, fold: np.ndarray, kind: str):
    """Fit one model per fold on the other folds.

    Returns the out-of-fold prediction per row and the fitted models, in fold
    order: model k has never seen fold k.
    """
    pred = np.full(len(y), np.nan)
    models = []
    for k in sorted(set(fold.tolist())):
        test = fold == k
        model = make_model(kind).fit(X[~test], y[~test])
        pred[test] = model.predict(X[test])
        models.append(model)
    return pred, models


def cv_report(y: np.ndarray, pred: np.ndarray, edge: np.ndarray) -> dict:
    """Scores on all held-out cells and on those at least 500 m from any training block."""
    return {"all_test_cells": scores(y, pred), "away_from_training_blocks": scores(y[~edge], pred[~edge])}


def held_out_importance(models, X, y, fold, names, repeats: int = 3) -> pd.DataFrame:
    """Permutation importance on each model's own held-out fold, averaged over folds."""
    total = np.zeros(len(names))
    for k, model in zip(sorted(set(fold.tolist())), models):
        test = fold == k
        result = permutation_importance(
            model, X[test], y[test], n_repeats=repeats, random_state=SEED,
            scoring="neg_mean_absolute_error",
        )
        total += result.importances_mean
    out = pd.DataFrame({"feature": names, "mae_increase_c": total / len(models)})
    return out.sort_values("mae_increase_c", ascending=False).reset_index(drop=True)


def group_importance(importance: pd.DataFrame) -> pd.DataFrame:
    """Importance summed over a quantity's three scales (cell, 150 m, 500 m)."""
    base = importance["feature"].str.replace(r"_(150|500)m$", "", regex=True)
    out = importance.groupby(base)["mae_increase_c"].sum().sort_values(ascending=False)
    return out.rename_axis("quantity").reset_index()


def partial_dependence(models, X: np.ndarray, column: int, grid: np.ndarray, sample: int = 3000) -> np.ndarray:
    """Mean prediction when one feature is set to each grid value, averaged over the fold models."""
    rng = np.random.default_rng(SEED)
    rows = X[rng.choice(len(X), size=min(sample, len(X)), replace=False)].copy()
    out = []
    for value in grid:
        rows[:, column] = value
        out.append(np.mean([m.predict(rows).mean() for m in models]))
    return np.array(out)


def ensemble_predict(models, X: np.ndarray) -> np.ndarray:
    """Predictions of every fold model: shape (models, rows)."""
    return np.stack([m.predict(X) for m in models])


def change_band(before: np.ndarray, after: np.ndarray, held_out_mae: float) -> tuple:
    """Predicted change with a band, from predictions of shape (models, rows).

    The band combines how much the fold models disagree about the change (1.96
    standard deviations) with the model's typical held-out error. The second
    part is cautious: errors of "before" and "after" at the same place partly
    cancel, and that is not credited here.
    """
    delta = after - before
    mean = delta.mean(axis=0)
    half = np.sqrt((1.96 * delta.std(axis=0)) ** 2 + held_out_mae**2)
    return mean, mean - half, mean + half


def training_range(X: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Per feature, the 1st and 99th percentile of the training data: shape (2, features)."""
    return np.nanpercentile(X, [low, high], axis=0)


def outside_range(X: np.ndarray, limits: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    """Number of features per row that lie outside what the model was trained on."""
    with np.errstate(invalid="ignore"):
        return ((X < limits[0] - tolerance) | (X > limits[1] + tolerance)).sum(axis=1)
