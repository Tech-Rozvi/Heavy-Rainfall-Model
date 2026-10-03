"""
weighting.py
Two independent factor-weighting methods, per project requirement 6:

  Method A - AHP (Analytic Hierarchy Process): expert pairwise comparison
             matrix -> principal eigenvector -> normalized weights ->
             Consistency Ratio (CR) validation.

  Method B - Data-driven: Random Forest trained against an observed
             heavy-rainfall target, using permutation importance (robust to
             the bias built into RF's default impurity importance) as the
             weight proxy.

The two are compared, not merged silently -- see `compare_weights`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .utils import get_logger

logger = get_logger(__name__)

# Saaty (1980) Random Index table for consistency-ratio calculation
_SAATY_RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32,
             8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51}


# ---------------------------------------------------------------------------
# Method A: AHP
# ---------------------------------------------------------------------------
def ahp_weights(pairwise_matrix: np.ndarray, factor_names: list[str],
                 max_acceptable_cr: float = 0.10) -> dict:
    """Compute AHP weights from a Saaty pairwise comparison matrix.

    Returns a dict with weights, consistency index (CI), consistency ratio
    (CR), and an `is_consistent` flag. Per project requirement, a CR above
    max_acceptable_cr (Saaty's own convention: 0.10) means the expert
    judgments are too inconsistent to trust and MUST be revised -- the
    function still returns the numbers but flags them, it does not silently
    approve an inconsistent matrix.
    """
    n = pairwise_matrix.shape[0]
    if pairwise_matrix.shape != (n, n):
        raise ValueError("Pairwise comparison matrix must be square.")
    if len(factor_names) != n:
        raise ValueError("factor_names length must match matrix dimension.")
    if not np.allclose(pairwise_matrix, pairwise_matrix.T ** -1, rtol=0.05):
        logger.warning("Pairwise matrix does not appear reciprocal "
                        "(A[j,i] != 1/A[i,j]); verify the input matrix.")

    # Principal eigenvector via the standard normalized-column-average
    # approximation (fast, numerically stable, and the conventional AHP
    # method for matrices up to ~15x15).
    col_sums = pairwise_matrix.sum(axis=0)
    normalized = pairwise_matrix / col_sums
    weights = normalized.mean(axis=1)
    weights = weights / weights.sum()

    # Consistency check
    weighted_sum_vector = pairwise_matrix @ weights
    lambda_max = np.mean(weighted_sum_vector / weights)
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = _SAATY_RI.get(n, 1.51)
    cr = ci / ri if ri > 0 else 0.0

    result = {
        "weights": dict(zip(factor_names, weights)),
        "lambda_max": float(lambda_max),
        "consistency_index": float(ci),
        "consistency_ratio": float(cr),
        "is_consistent": bool(cr <= max_acceptable_cr),
    }
    if not result["is_consistent"]:
        logger.warning("AHP Consistency Ratio = %.3f exceeds acceptable threshold "
                        "%.2f. Revise the pairwise comparison matrix before using "
                        "these weights operationally.", cr, max_acceptable_cr)
    else:
        logger.info("AHP Consistency Ratio = %.3f (acceptable, <= %.2f).",
                    cr, max_acceptable_cr)
    return result


def build_default_pairwise_matrix(factor_order: list[str],
                                   importance_rank: Optional[dict[str, float]] = None
                                   ) -> np.ndarray:
    """Construct a documented, literature-consistent default pairwise matrix
    when the user has not supplied expert judgments of their own. Ordering
    reflects the standard heavy-rainfall literature consensus: direct
    historical/extreme precipitation signals > thermodynamic atmospheric
    fields > orographic modifiers. This default is a STARTING POINT, meant
    to be edited by a domain expert for the specific study area, not treated
    as ground truth.
    """
    n = len(factor_order)
    if importance_rank is None:
        # A simple, monotonic default: earlier factors in factor_order are
        # assumed moderately more important than later ones (Saaty scale
        # capped at 5 = "strong importance" to avoid overconfident extremes).
        importance_rank = {f: n - i for i, f in enumerate(factor_order)}

    matrix = np.ones((n, n))
    for i, fi in enumerate(factor_order):
        for j, fj in enumerate(factor_order):
            if i == j:
                continue
            rank_diff = importance_rank[fi] - importance_rank[fj]
            # Map rank difference to Saaty 1-5 scale (avoids 9, which is an
            # extreme judgment inappropriate for a generic default)
            saaty_val = 1 + min(4, abs(rank_diff))
            matrix[i, j] = saaty_val if rank_diff > 0 else 1.0 / saaty_val
    return matrix


# ---------------------------------------------------------------------------
# Method B: Data-driven (Random Forest)
# ---------------------------------------------------------------------------
def random_forest_weights(X: pd.DataFrame, y: np.ndarray, random_seed: int = 42,
                           n_estimators: int = 300, task: str = "regression"
                           ) -> dict:
    """Fit a Random Forest against an observed heavy-rainfall target and
    derive weights from permutation importance (more reliable than the
    default impurity-based importance, which is biased toward
    high-cardinality/continuous features).

    task: "regression" (continuous rainfall amount/index) or
          "classification" (binary heavy-rain event).
    """
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    if task not in ("regression", "classification"):
        raise ValueError("task must be 'regression' or 'classification'.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_seed)

    if task == "regression":
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_seed,
                                       n_jobs=-1)
    else:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_seed,
                                        n_jobs=-1, class_weight="balanced")
    model.fit(X_train, y_train)

    perm = permutation_importance(model, X_test, y_test, n_repeats=15,
                                   random_state=random_seed, n_jobs=-1)
    importances = np.clip(perm.importances_mean, 0, None)  # negative importance -> 0
    if importances.sum() == 0:
        logger.warning("All permutation importances are zero or negative; "
                        "data-driven weights are unreliable with this sample.")
        importances = np.ones_like(importances)
    weights = importances / importances.sum()

    return {
        "weights": dict(zip(X.columns, weights)),
        "model": model,
        "train_score": model.score(X_train, y_train),
        "test_score": model.score(X_test, y_test),
        "raw_importances": dict(zip(X.columns, perm.importances_mean)),
    }


def shap_importance(model, X: pd.DataFrame, sample_size: int = 500,
                     random_seed: int = 42) -> Optional[pd.Series]:
    """Optional SHAP-based feature importance (mean |SHAP value|). Returns
    None gracefully if the shap package is not installed, rather than
    crashing the whole pipeline over an optional diagnostic.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed; skipping SHAP importance (optional step).")
        return None
    sample = X.sample(min(sample_size, len(X)), random_state=random_seed)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):  # classifier multi-output
        shap_values = shap_values[-1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.Series(mean_abs, index=X.columns, name="mean_abs_shap").sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def compare_weights(ahp_result: dict, rf_result: Optional[dict] = None) -> pd.DataFrame:
    """Side-by-side comparison of AHP vs. RF-derived weights. Large
    divergence for a given factor is a diagnostic worth investigating (e.g.
    the expert view may reflect a physical mechanism not well captured by
    the observed sample, or the RF may be picking up local sampling
    artefacts) -- it is reported, not resolved automatically.
    """
    ahp_w = pd.Series(ahp_result["weights"], name="AHP_weight")
    if rf_result is None:
        df = ahp_w.to_frame()
        df["RF_weight"] = np.nan
    else:
        rf_w = pd.Series(rf_result["weights"], name="RF_weight")
        df = pd.concat([ahp_w, rf_w], axis=1)
    df["abs_difference"] = (df["AHP_weight"] - df["RF_weight"]).abs()
    return df.sort_values("AHP_weight", ascending=False)


def resolve_final_weights(ahp_result: dict, rf_result: Optional[dict],
                           method: str = "ahp") -> dict[str, float]:
    """Select the final weight set to use in the weighted-overlay model,
    per config.yaml `weighting.method`."""
    if method == "ahp":
        return ahp_result["weights"]
    if method == "random_forest":
        if rf_result is None:
            raise ValueError("method='random_forest' requires rf_result (observed "
                              "target data).")
        return rf_result["weights"]
    if method == "average_of_both":
        if rf_result is None:
            raise ValueError("method='average_of_both' requires rf_result.")
        keys = ahp_result["weights"].keys()
        combined = {k: (ahp_result["weights"][k] + rf_result["weights"].get(k, 0)) / 2
                    for k in keys}
        total = sum(combined.values())
        return {k: v / total for k, v in combined.items()}
    raise ValueError(f"Unknown weighting method '{method}'.")
