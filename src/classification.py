"""
classification.py
Convert raw factor rasters into standardized [0,1] scores and, separately,
into discrete 1-5 risk classes for interpretability/mapping.

Each factor in config.yaml declares its own classification/standardization
method -- this module does NOT decide the method itself, it only implements
each method correctly, per the requirement that classification choice be
scientifically justified per-variable rather than applied blindly and
uniformly (e.g. equal-interval everywhere).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .utils import get_logger, qc_check_array

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Standardization (continuous -> [0,1])
# ---------------------------------------------------------------------------
def minmax_standardize(arr: np.ndarray, nodata: float = -9999.0) -> np.ndarray:
    """Min-max normalization. Appropriate for bounded continuous variables
    (e.g. elevation, slope) where the absolute range within the study area
    is the meaningful reference frame.

    All-NaN or uniform/nodata inputs return a zero field instead of raising,
    so the pipeline can continue when a remote layer has no finite pixels.
    """
    a = np.asarray(arr, dtype="float64")
    mask = np.isfinite(a) & (a != nodata)
    valid = a[mask]
    if valid.size == 0:
        logger.warning("minmax_standardize: no valid values; returning zeros.")
        return np.zeros_like(a, dtype="float64")
    lo = float(np.nanmin(valid))
    hi = float(np.nanmax(valid))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        out = np.zeros_like(a, dtype="float64")
        out[mask] = 0.0
        return out
    out = np.full_like(a, np.nan, dtype="float64")
    out[mask] = (a[mask] - lo) / (hi - lo)
    out = np.where(mask, np.clip(out, 0.0, 1.0), 0.0)
    return out


def zscore_standardize(arr: np.ndarray, nodata: float = -9999.0,
                        clip_sigma: float = 3.0) -> np.ndarray:
    """Z-score standardization, then rescaled to [0,1] via clipping at
    +/- clip_sigma. Appropriate for approximately-Gaussian climatic fields
    (humidity, temperature, pressure) where we care about departure from the
    local mean, not an arbitrary min/max in the current sample.

    All-NaN inputs return a constant 0.5 field instead of raising.
    """
    a = np.asarray(arr, dtype="float64")
    mask = np.isfinite(a) & (a != nodata)
    valid = a[mask]
    if valid.size == 0:
        logger.warning("zscore_standardize: no valid values; returning 0.5 field.")
        return np.full_like(a, 0.5, dtype="float64")
    mu = float(np.nanmean(valid))
    sigma = float(np.nanstd(valid))
    if not np.isfinite(sigma) or sigma == 0:
        out = np.full_like(a, 0.5, dtype="float64")
        return out
    z = np.full_like(a, np.nan, dtype="float64")
    z[mask] = (a[mask] - mu) / sigma
    z = np.clip(z, -clip_sigma, clip_sigma)
    out = (z + clip_sigma) / (2 * clip_sigma)
    return np.where(mask, out, 0.5)


def percentile_rank_standardize(arr: np.ndarray, nodata: float = -9999.0) -> np.ndarray:
    """Rank each pixel by its percentile position within the study-area
    distribution. Preferred for precipitation/extreme-index variables since
    it is robust to skewed, heavy-tailed rainfall distributions where
    min-max would be dominated by a handful of extreme outlier pixels."""
    valid_mask = np.isfinite(arr) & (arr != nodata)
    out = np.full(arr.shape, np.nan)
    if not valid_mask.any():
        logger.warning("standardize: no valid values; returning zeros.")
        return np.zeros_like(arr, dtype="float64")
    from scipy.stats import rankdata
    ranks = rankdata(arr[valid_mask]) / valid_mask.sum()
    out[valid_mask] = ranks
    return np.where(valid_mask, out, 0.0)


def fuzzy_decay_standardize(distance_arr: np.ndarray, decay_km: float,
                             nodata: float = -9999.0) -> np.ndarray:
    """Fuzzy (exponential decay) membership for distance-based factors:
    score = exp(-distance_km / decay_km), i.e. influence of a moisture
    source decays continuously rather than via an arbitrary hard cutoff.
    Score of 1.0 at distance=0, ~0.37 at distance=decay_km.

    Accepts distances already in km or metres (values > 500 treated as m).
    All-NaN / nodata inputs return zeros.
    """
    a = np.asarray(distance_arr, dtype="float64")
    mask = np.isfinite(a) & (a != nodata)
    if not mask.any():
        logger.warning("fuzzy_decay_standardize: no valid values; returning zeros.")
        return np.zeros_like(a, dtype="float64")
    # Heuristic: if median finite value > 500, treat as metres
    med = float(np.nanmedian(a[mask]))
    dist_km = a / 1000.0 if med > 500 else a
    score = np.exp(-np.where(mask, dist_km, 0.0) / max(float(decay_km), 1e-6))
    return np.where(mask, np.clip(score, 0.0, 1.0), 0.0)


STANDARDIZERS = {
    "minmax": minmax_standardize,
    "zscore": zscore_standardize,
    "percentile": percentile_rank_standardize,
    # fuzzy_decay requires the extra decay_km arg and is called directly,
    # not dispatched through this generic table.
}


def standardize(arr: np.ndarray, method: str, nodata: float = -9999.0,
                 **kwargs) -> np.ndarray:
    if method == "fuzzy_decay":
        return fuzzy_decay_standardize(arr, kwargs.get("decay_km", 100.0), nodata)
    if method not in STANDARDIZERS:
        raise ValueError(f"Unknown standardization method '{method}'. "
                          f"Choose from {list(STANDARDIZERS) + ['fuzzy_decay']}.")
    return STANDARDIZERS[method](arr, nodata)


# ---------------------------------------------------------------------------
# Discrete classification (for legend/mapping, 1..n_classes)
# ---------------------------------------------------------------------------
def classify_quantile(standardized: np.ndarray, n_classes: int = 5) -> np.ndarray:
    """Quantile classification: each class holds an equal COUNT of pixels.
    Preferred for terrain variables (elevation, slope) since orographic
    effects are relative to the local terrain distribution, not a universal
    absolute threshold."""
    valid = standardized[np.isfinite(standardized)]
    edges = np.nanpercentile(valid, np.linspace(0, 100, n_classes + 1))
    edges[-1] += 1e-9  # ensure max value falls in last bin
    classes = np.digitize(standardized, edges[1:-1]) + 1
    return np.where(np.isfinite(standardized), classes, np.nan)


def classify_percentile(standardized: np.ndarray, n_classes: int = 5) -> np.ndarray:
    """Same mechanics as quantile but framed around climatological
    percentile breakpoints (e.g. <50th, 50-75th, 75-90th, 90-95th, >95th) --
    the convention used for extreme-precipitation indices in the
    climate-science literature, emphasizing separation of the extreme tail
    rather than equal-sized bins throughout."""
    valid = standardized[np.isfinite(standardized)]
    breakpoints = [0, 50, 75, 90, 95, 100][: n_classes + 1] if n_classes == 5 else \
        list(np.linspace(0, 100, n_classes + 1))
    edges = np.nanpercentile(valid, breakpoints)
    edges[-1] += 1e-9
    classes = np.digitize(standardized, edges[1:-1]) + 1
    return np.where(np.isfinite(standardized), classes, np.nan)


def classify_zscore(standardized_01: np.ndarray, n_classes: int = 5) -> np.ndarray:
    """Classify an already [0,1]-rescaled z-score using fixed sigma-based
    breakpoints (equivalent to roughly -1.5, -0.5, 0.5, 1.5 sigma), which is
    appropriate because z-score standardization already encodes a physically
    meaningful scale (standard deviations from the mean)."""
    edges = [0, 0.25, 0.4166, 0.5833, 0.75, 1.0]  # maps back to ~-3,-1.5,-0.5,0.5,1.5,3 sigma
    classes = np.digitize(standardized_01, edges[1:-1]) + 1
    return np.where(np.isfinite(standardized_01), classes, np.nan)


def classify_equal_width_fallback(standardized: np.ndarray, n_classes: int = 5) -> np.ndarray:
    """Equal-interval classification. Included ONLY as an explicit fallback
    (e.g. categorical aspect-exposure scores where quantiles would be
    meaningless) -- never used as the default per the project requirements.
    """
    edges = np.linspace(0, 1, n_classes + 1)
    classes = np.digitize(standardized, edges[1:-1]) + 1
    return np.where(np.isfinite(standardized), classes, np.nan)


CLASSIFIERS = {
    "quantile": classify_quantile,
    "percentile": classify_percentile,
    "zscore": classify_zscore,
    "fuzzy": classify_equal_width_fallback,       # fuzzy scores are already continuous;
    "categorical": classify_equal_width_fallback,  # binned only for display purposes
}


def classify(standardized: np.ndarray, method: str, n_classes: int = 5) -> np.ndarray:
    if method not in CLASSIFIERS:
        raise ValueError(f"Unknown classification method '{method}'. "
                          f"Choose from {list(CLASSIFIERS)}.")
    return CLASSIFIERS[method](standardized, n_classes)
