"""
rainfall_indices.py
Compute the historical rainfall baseline and extreme-precipitation indices
from a daily precipitation stack (time, y, x).

Indices implemented follow the ETCCDI (Expert Team on Climate Change
Detection and Indices) definitions, which are the standard used in
peer-reviewed extreme-precipitation literature:

  Rx1day  - max 1-day precipitation in the period
  Rx5day  - max consecutive 5-day precipitation
  R95p    - total precip on days exceeding the local 95th percentile
            (wet-day climatological reference)
  R99p    - as above, 99th percentile
  R10mm   - count of days with precip >= 10 mm
  R20mm   - count of days with precip >= 20 mm
  CDD     - max length of consecutive dry days (<1 mm)
  CWD     - max length of consecutive wet days (>=1 mm)

Rationale for inclusion (see methodology write-up): totals alone (mean
annual precipitation) do not describe heavy-rainfall *character* -- two
locations can share the same annual total while one delivers it in
extreme bursts (high Rx1day/R95p) and the other in steady moderate rain
(low CDD/CWD variability). The extreme indices are what actually
differentiate heavy-rainfall susceptibility from generic wetness.
"""
from __future__ import annotations

import numpy as np

from .utils import get_logger, qc_check_array

logger = get_logger(__name__)

WET_DAY_MM = 1.0  # ETCCDI standard wet-day threshold


def _safe_nanmax(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Max along axis; all-NaN slices become 0.0 (no All-NaN RuntimeWarning)."""
    a = np.asarray(arr, dtype="float64")
    filled = np.where(np.isfinite(a), a, -np.inf)
    out = np.max(filled, axis=axis)
    out = np.where(np.isfinite(out) & (out > -np.inf / 2), out, 0.0)
    return out


def _safe_nanpercentile(arr: np.ndarray, percentile: float, axis: int = 0) -> np.ndarray:
    """Percentile along axis; all-NaN slices become 0.0 without warnings."""
    a = np.asarray(arr, dtype="float64")
    # Replace NaN with a sentinel that rank-based percentile ignores via mask count
    finite = np.isfinite(a)
    if not finite.any():
        spatial = tuple(s for i, s in enumerate(a.shape) if i != axis % a.ndim)
        return np.zeros(spatial if spatial else (), dtype="float64")
    # Use nanpercentile under warnings suppressed + post-fill
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            out = np.nanpercentile(a, percentile, axis=axis)
    return np.where(np.isfinite(out), out, 0.0)

def _rolling_sum(arr: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """Rolling sum along the time axis using cumulative-sum trick (fast,
    memory-light, avoids an explicit Python loop over potentially thousands
    of days for large national-scale stacks)."""
    csum = np.cumsum(arr, axis=axis)
    csum = np.insert(csum, 0, 0, axis=axis)
    slicer_hi = [slice(None)] * arr.ndim
    slicer_lo = [slice(None)] * arr.ndim
    slicer_hi[axis] = slice(window, None)
    slicer_lo[axis] = slice(None, -window)
    return csum[tuple(slicer_hi)] - csum[tuple(slicer_lo)]


def mean_annual_precip(daily_precip: np.ndarray, n_years: float) -> np.ndarray:
    # Soft QC: allow NaNs from remote streams; do not require every pixel finite
    with np.errstate(all="ignore"):
        total = np.nansum(daily_precip, axis=0)
    out = total / max(float(n_years), 1e-6)
    return np.where(np.isfinite(out), out, 0.0)


def rx1day(daily_precip: np.ndarray) -> np.ndarray:
    return _safe_nanmax(daily_precip, axis=0)


def rx5day(daily_precip: np.ndarray) -> np.ndarray:
    if daily_precip.shape[0] < 5:
        raise ValueError("Rx5day requires at least 5 daily time steps.")
    # Treat NaN days as 0 for rolling wet sums so arid gaps do not poison the window
    filled = np.where(np.isfinite(daily_precip), daily_precip, 0.0)
    roll5 = _rolling_sum(filled, window=5, axis=0)
    return _safe_nanmax(roll5, axis=0)


def percentile_threshold_index(daily_precip: np.ndarray, percentile: float,
                                wet_days_only: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return (per-pixel percentile threshold, total precip on days exceeding
    it) — i.e. R95p/R99p style indices. The percentile is computed on wet
    days only (>= WET_DAY_MM), per ETCCDI convention, to avoid the threshold
    collapsing toward zero in arid pixels with many dry days.
    """
    wet_mask = daily_precip >= WET_DAY_MM if wet_days_only else np.isfinite(daily_precip)
    masked = np.where(wet_mask, daily_precip, np.nan)
    threshold = _safe_nanpercentile(masked, percentile, axis=0)
    exceed = np.where(daily_precip >= threshold[np.newaxis, ...], daily_precip, 0.0)
    total_exceed = np.nansum(exceed, axis=0)
    return threshold, total_exceed


def r10mm(daily_precip: np.ndarray) -> np.ndarray:
    return np.nansum(np.where(np.isfinite(daily_precip), daily_precip >= 10.0, 0).astype("float64"), axis=0)


def r20mm(daily_precip: np.ndarray) -> np.ndarray:
    return np.nansum(np.where(np.isfinite(daily_precip), daily_precip >= 20.0, 0).astype("float64"), axis=0)


def _max_consecutive_run(mask: np.ndarray, axis: int = 0) -> np.ndarray:
    """Max length of consecutive True values along `axis`, vectorised over
    the spatial dimensions (avoids per-pixel Python loops for large grids).
    """
    mask = mask.astype(int)
    # reset counter to 0 wherever mask is False, else increment
    out_shape = mask.shape
    run = np.zeros(out_shape[1:], dtype=int)
    best = np.zeros(out_shape[1:], dtype=int)
    for t in range(out_shape[0]):
        run = np.where(mask[t] == 1, run + 1, 0)
        best = np.maximum(best, run)
    return best


def cdd(daily_precip: np.ndarray) -> np.ndarray:
    """Consecutive Dry Days (max run of days with precip < 1 mm)."""
    return _max_consecutive_run(daily_precip < WET_DAY_MM)


def cwd(daily_precip: np.ndarray) -> np.ndarray:
    """Consecutive Wet Days (max run of days with precip >= 1 mm)."""
    return _max_consecutive_run(daily_precip >= WET_DAY_MM)


def compute_all_indices(daily_precip: np.ndarray, n_years: float,
                         extreme_percentile: float = 95.0) -> dict[str, np.ndarray]:
    """Compute the full recommended index set from a (time, y, x) daily
    precipitation stack. Returns a dict of 2D arrays keyed by index name.
    """
    logger.info("Computing rainfall indices over %d time steps (%.1f years).",
                daily_precip.shape[0], n_years)
    p95_thresh, r95p = percentile_threshold_index(daily_precip, 95.0)
    p99_thresh, r99p = percentile_threshold_index(daily_precip, 99.0)

    return {
        "mean_annual_precip": mean_annual_precip(daily_precip, n_years),
        "rx1day": rx1day(daily_precip),
        "rx5day": rx5day(daily_precip),
        "r95p": r95p,
        "r99p": r99p,
        "r10mm": r10mm(daily_precip).astype("float64"),
        "r20mm": r20mm(daily_precip).astype("float64"),
        "cdd": cdd(daily_precip).astype("float64"),
        "cwd": cwd(daily_precip).astype("float64"),
    }
