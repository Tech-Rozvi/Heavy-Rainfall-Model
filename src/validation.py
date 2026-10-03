"""
validation.py
Validate model outputs against OBSERVED rainfall — station observations are
tagged and reported separately from satellite/reanalysis products, per the
requirement that they never be treated as equivalent (project section 8).

Two separate metric families are implemented, matched to what each model
actually produces (section 8/25):
  - Continuous metrics for the susceptibility score / rainfall-amount
    prediction: Pearson r, Spearman rho, RMSE, MAE, R^2, bias.
  - Classification metrics for a binary heavy-rain-event prediction:
    ROC-AUC, precision, recall, F1.
It is a methodological error to compute ROC-AUC on a continuous
susceptibility score without first thresholding it into an event
prediction — this module keeps the two paths explicit and separate.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .utils import get_logger

logger = get_logger(__name__)


def extract_at_points(raster: np.ndarray, transform, points_xy: list[tuple[float, float]]
                       ) -> np.ndarray:
    """Sample a raster's values at a list of (x, y) point coordinates in the
    raster's own CRS, using the affine transform to convert to row/col."""
    from rasterio.transform import rowcol
    values = []
    for x, y in points_xy:
        row, col = rowcol(transform, x, y)
        if 0 <= row < raster.shape[0] and 0 <= col < raster.shape[1]:
            values.append(raster[row, col])
        else:
            values.append(np.nan)
    return np.array(values)


def continuous_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    """Pearson, Spearman, RMSE, MAE, R^2, bias. NaNs are pairwise-dropped."""
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    mask = np.isfinite(observed) & np.isfinite(predicted)
    obs, pred = observed[mask], predicted[mask]
    if len(obs) < 3:
        raise ValueError(f"Only {len(obs)} valid paired observations — need at "
                          f"least 3 for meaningful validation statistics.")

    pearson_r, pearson_p = pearsonr(obs, pred)
    spearman_r, spearman_p = spearmanr(obs, pred)
    rmse = float(np.sqrt(mean_squared_error(obs, pred)))

    return {
        "n": int(len(obs)),
        "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
        "rmse": rmse,
        "mae": float(mean_absolute_error(obs, pred)),
        "r2": float(r2_score(obs, pred)),
        "bias": float(np.mean(pred - obs)),
    }


def classification_metrics(observed_event: np.ndarray, predicted_score: np.ndarray,
                            decision_threshold: float = 0.5) -> dict:
    """ROC-AUC (threshold-independent) plus precision/recall/F1 at a chosen
    decision threshold applied to predicted_score (which may be a
    susceptibility score or a predicted event probability)."""
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

    mask = np.isfinite(observed_event) & np.isfinite(predicted_score)
    obs, pred = observed_event[mask], predicted_score[mask]
    if len(np.unique(obs)) < 2:
        raise ValueError("Observed event array must contain both classes (0 and 1) "
                          "to compute classification metrics.")

    pred_binary = (pred >= decision_threshold).astype(int)
    return {
        "n": int(len(obs)),
        "roc_auc": float(roc_auc_score(obs, pred)),
        "precision": float(precision_score(obs, pred_binary, zero_division=0)),
        "recall": float(recall_score(obs, pred_binary, zero_division=0)),
        "f1": float(f1_score(obs, pred_binary, zero_division=0)),
        "decision_threshold": decision_threshold,
    }


def validate_against_stations(model_raster: np.ndarray, transform,
                               stations_df: pd.DataFrame,
                               heavy_rain_definition: dict) -> dict:
    """Full validation workflow against a station observation table with
    columns: station_id, x, y, source ('station'|'CHIRPS'|'ERA5-Land'|...),
    observed_value, observed_event (0/1, optional).

    Reports metrics PER SOURCE so ground station accuracy is never blended
    with, or mistaken for, satellite/reanalysis product accuracy.
    """
    required = {"station_id", "x", "y", "source", "observed_value"}
    missing = required - set(stations_df.columns)
    if missing:
        raise ValueError(f"stations_df missing required columns: {missing}")

    points = list(zip(stations_df["x"], stations_df["y"]))
    predicted = extract_at_points(model_raster, transform, points)
    stations_df = stations_df.copy()
    stations_df["predicted_value"] = predicted

    report = {}
    for source, group in stations_df.groupby("source"):
        try:
            cont = continuous_metrics(group["observed_value"].values,
                                       group["predicted_value"].values)
        except ValueError as exc:
            logger.warning("Skipping continuous metrics for source=%s: %s", source, exc)
            cont = None

        cls = None
        if "observed_event" in group.columns and group["observed_event"].notna().any():
            try:
                cls = classification_metrics(group["observed_event"].values,
                                              group["predicted_value"].values)
            except ValueError as exc:
                logger.warning("Skipping classification metrics for source=%s: %s", source, exc)

        report[source] = {"continuous": cont, "classification": cls, "n_stations": len(group)}
        logger.info("Validation [%s]: n=%d, continuous=%s", source, len(group),
                    "OK" if cont else "insufficient data")

    return report
