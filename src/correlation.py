"""
correlation.py
Detect redundant/collinear explanatory variables before they are combined
into the weighted overlay, so the model does not silently double-count the
same physical signal (e.g. elevation and temperature, humidity and
precipitation).

Implements Pearson, Spearman, and Variance Inflation Factor (VIF). The
pipeline treats a flagged pair as something requiring a decision (drop one,
or combine via PCA) -- it does not auto-drop variables silently, since which
variable to keep is a scientific judgment (see README 'Handling collinearity').
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .utils import get_logger

logger = get_logger(__name__)


def factors_to_dataframe(factor_arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    """Flatten a dict of equal-shape 2D arrays into a tidy DataFrame of valid
    (non-NaN in ALL layers) pixels, ready for correlation/VIF/ML."""
    shapes = {k: v.shape for k, v in factor_arrays.items()}
    if len(set(shapes.values())) > 1:
        raise ValueError(f"Cannot build correlation dataframe: unaligned shapes {shapes}")
    stacked = {k: v.ravel() for k, v in factor_arrays.items()}
    df = pd.DataFrame(stacked)
    df = df.dropna(how="any")
    if df.empty:
        raise ValueError("No pixels with valid data across all factors after dropna.")
    return df


def correlation_matrices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (Pearson, Spearman) correlation matrices."""
    pearson = df.corr(method="pearson")
    spearman = df.corr(method="spearman")
    return pearson, spearman


def variance_inflation_factors(df: pd.DataFrame) -> pd.Series:
    """Compute VIF per variable. VIF > threshold (typically 5) indicates a
    variable is highly explainable from the others -- i.e. redundant."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = df.values.astype(float)
    vifs = {}
    for i, col in enumerate(df.columns):
        try:
            vifs[col] = variance_inflation_factor(X, i)
        except Exception as exc:  # singular matrix etc.
            logger.warning("VIF failed for %s: %s", col, exc)
            vifs[col] = np.nan
    return pd.Series(vifs, name="VIF")


def flag_redundant_pairs(pearson: pd.DataFrame, spearman: pd.DataFrame,
                          pearson_thresh: float = 0.70,
                          spearman_thresh: float = 0.70) -> list[dict]:
    """List variable pairs exceeding either correlation threshold."""
    flags = []
    cols = pearson.columns
    for a, b in combinations(cols, 2):
        p = pearson.loc[a, b]
        s = spearman.loc[a, b]
        if abs(p) >= pearson_thresh or abs(s) >= spearman_thresh:
            flags.append({"factor_a": a, "factor_b": b, "pearson_r": round(p, 3),
                          "spearman_r": round(s, 3)})
    return flags


def run_full_diagnostics(factor_arrays: dict[str, np.ndarray],
                          pearson_thresh: float = 0.70,
                          spearman_thresh: float = 0.70,
                          vif_thresh: float = 5.0,
                          sample_size: int = 20000,
                          random_seed: int = 42) -> dict:
    """Full correlation/VIF diagnostic report. Subsamples pixels for
    tractability on large national-scale grids."""
    df = factors_to_dataframe(factor_arrays)
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=random_seed)

    pearson, spearman = correlation_matrices(df)
    vif = variance_inflation_factors(df)
    flags = flag_redundant_pairs(pearson, spearman, pearson_thresh, spearman_thresh)
    vif_flags = vif[vif > vif_thresh].to_dict()

    if flags:
        logger.warning("%d correlated factor pair(s) flagged (|r| >= %.2f): %s",
                        len(flags), pearson_thresh, flags)
    if vif_flags:
        logger.warning("%d factor(s) flagged for high VIF (>%.1f): %s",
                        len(vif_flags), vif_thresh, vif_flags)
    if not flags and not vif_flags:
        logger.info("No significant collinearity detected among selected factors.")

    return {
        "pearson": pearson,
        "spearman": spearman,
        "vif": vif,
        "flagged_pairs": flags,
        "flagged_vif": vif_flags,
        "n_samples": len(df),
    }
