"""
model.py
Two distinct, clearly labelled model outputs (see methodology section 25):

  1. HeavyRainSusceptibilityModel — the transparent weighted-overlay score
     Susceptibility(p) = sum(Wi * Xi(p)), sum(Wi) = 1. This is a RELATIVE
     susceptibility ranking, not a quantitative rainfall forecast.

  2. HeavyRainPredictionModel — an optional supervised ML model (RF /
     Gradient Boosting / regularized regression / logistic regression) that
     genuinely predicts a rainfall quantity or event probability from
     observed data, WHEN sufficient observed target data exists. The two are
     never conflated in outputs, filenames, or reported statistics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .utils import RasterLayer, get_logger, qc_check_array

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. Weighted-overlay susceptibility model
# ---------------------------------------------------------------------------
@dataclass
class HeavyRainSusceptibilityModel:
    """Transparent weighted linear combination of standardized factor
    layers. This is THE deliverable described in methodology section 4-5:
    Rainfall Susceptibility Score = sum(Wi * Xi).
    """
    weights: dict[str, float]
    reference_layer: Optional[RasterLayer] = None  # for transform/crs metadata

    def __post_init__(self):
        total = sum(self.weights.values())
        if not np.isclose(total, 1.0, atol=1e-3):
            raise ValueError(f"Weights must sum to 1.0 (got {total:.4f}). "
                              f"Check weighting.resolve_final_weights output.")

    def compute(self, standardized_factors: dict[str, np.ndarray]) -> np.ndarray:
        """Combine standardized [0,1] factor arrays into the final score."""
        missing = set(self.weights) - set(standardized_factors)
        if missing:
            raise ValueError(f"Missing standardized layers for weighted factors: {missing}")

        shape = next(iter(standardized_factors.values())).shape
        score = np.zeros(shape, dtype="float64")
        valid_mask = np.ones(shape, dtype=bool)

        for name, w in self.weights.items():
            arr = standardized_factors[name]
            if arr.shape != shape:
                raise ValueError(f"Shape mismatch for factor '{name}': {arr.shape} vs {shape}")
            score += w * np.nan_to_num(arr, nan=0.0)
            valid_mask &= np.isfinite(arr)

        score = np.where(valid_mask, score, np.nan)
        qc_check_array(score[valid_mask], "susceptibility_score", valid_range=(0, 1.01))
        return score

    def classify_result(self, score: np.ndarray, n_classes: int = 5,
                         labels: Optional[list[str]] = None) -> tuple[np.ndarray, dict]:
        """Classify the continuous score into risk categories using quantile
        breaks of the score itself (the final combined score's own
        distribution, not any single input factor's)."""
        from .classification import classify_quantile
        classes = classify_quantile(score, n_classes)
        if labels:
            legend = {i + 1: labels[i] for i in range(n_classes)}
        else:
            legend = {i + 1: f"Class {i+1}" for i in range(n_classes)}
        return classes, legend


# ---------------------------------------------------------------------------
# 2. Data-driven prediction model (optional, requires observed target)
# ---------------------------------------------------------------------------
@dataclass
class HeavyRainPredictionModel:
    """Supervised model that predicts a rainfall QUANTITY (regression) or a
    heavy-rain EVENT (classification) from environmental predictors. Unlike
    the susceptibility model, this makes a genuine quantitative claim and is
    therefore held to a stricter validation standard (see validation.py) and
    MUST use spatial cross-validation (see spatial_cv_folds) to avoid
    leakage from spatially autocorrelated neighbouring pixels.
    """
    task: str  # "regression" | "classification"
    model_type: str = "random_forest"  # "random_forest" | "gradient_boosting" | "linear"
    random_seed: int = 42
    model: object = field(default=None, repr=False)

    def _build_estimator(self):
        if self.task == "regression":
            if self.model_type == "random_forest":
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(n_estimators=300, random_state=self.random_seed,
                                              n_jobs=-1)
            if self.model_type == "gradient_boosting":
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor(random_state=self.random_seed)
            if self.model_type == "linear":
                from sklearn.linear_model import ElasticNetCV
                return ElasticNetCV(cv=5, random_state=self.random_seed)
        elif self.task == "classification":
            if self.model_type == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_estimators=300, random_state=self.random_seed,
                                               n_jobs=-1, class_weight="balanced")
            if self.model_type == "gradient_boosting":
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(random_state=self.random_seed)
            if self.model_type == "linear":
                from sklearn.linear_model import LogisticRegressionCV
                return LogisticRegressionCV(cv=5, random_state=self.random_seed,
                                             class_weight="balanced", max_iter=2000)
        raise ValueError(f"Unsupported combination task={self.task}, model_type={self.model_type}")

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.model = self._build_estimator()
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fit yet.")
        return self.model.predict(X)


def spatial_cv_folds(coords: np.ndarray, n_folds: int = 5, random_seed: int = 42) -> np.ndarray:
    """Assign spatial cross-validation fold IDs using KMeans clustering on
    pixel coordinates, so that spatially clustered/autocorrelated
    observations (e.g. all pixels from the same storm footprint) stay
    together in a single fold rather than leaking between train and test --
    see methodology section 18.
    """
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_folds, random_state=random_seed, n_init=10)
    fold_ids = km.fit_predict(coords)
    return fold_ids


def build_feature_matrix(factor_arrays: dict[str, np.ndarray]) -> tuple[pd.DataFrame, np.ndarray]:
    """Flatten aligned 2D factor arrays into a feature matrix plus the
    (row, col) pixel coordinates of each retained (fully valid) sample --
    coordinates are needed for spatial_cv_folds.
    """
    shape = next(iter(factor_arrays.values())).shape
    rows, cols = np.indices(shape)
    flat = {k: v.ravel() for k, v in factor_arrays.items()}
    flat["_row"] = rows.ravel()
    flat["_col"] = cols.ravel()
    df = pd.DataFrame(flat).dropna(how="any")
    coords = df[["_row", "_col"]].values
    X = df.drop(columns=["_row", "_col"])
    return X, coords
