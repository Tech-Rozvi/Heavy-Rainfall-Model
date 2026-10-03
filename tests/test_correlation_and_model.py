import numpy as np
import pytest

from src import correlation as corr
from src.model import HeavyRainSusceptibilityModel


def test_flag_redundant_pairs_detects_collinearity():
    rng = np.random.default_rng(0)
    n = 500
    a = rng.normal(size=n)
    b = a + rng.normal(scale=0.01, size=n)  # near-duplicate of a
    c = rng.normal(size=n)                  # independent
    import pandas as pd
    df = pd.DataFrame({"a": a, "b": b, "c": c})
    pearson, spearman = corr.correlation_matrices(df)
    flags = corr.flag_redundant_pairs(pearson, spearman, 0.7, 0.7)
    flagged_names = {(f["factor_a"], f["factor_b"]) for f in flags}
    assert ("a", "b") in flagged_names
    assert ("a", "c") not in flagged_names


def test_factors_to_dataframe_shape_mismatch_raises():
    arrays = {"a": np.zeros((3, 3)), "b": np.zeros((4, 4))}
    with pytest.raises(ValueError):
        corr.factors_to_dataframe(arrays)


def test_susceptibility_model_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        HeavyRainSusceptibilityModel(weights={"a": 0.5, "b": 0.6})


def test_susceptibility_model_compute_bounds():
    weights = {"a": 0.6, "b": 0.4}
    model = HeavyRainSusceptibilityModel(weights=weights)
    factors = {
        "a": np.array([[0.0, 1.0], [0.5, 1.0]]),
        "b": np.array([[1.0, 0.0], [0.5, 1.0]]),
    }
    score = model.compute(factors)
    assert score.shape == (2, 2)
    assert np.nanmax(score) <= 1.0 + 1e-6
    assert np.nanmin(score) >= 0.0 - 1e-6
    # pixel (1,1): both factors = 1.0 -> score should be 1.0
    assert score[1, 1] == pytest.approx(1.0)


def test_susceptibility_model_missing_factor_raises():
    model = HeavyRainSusceptibilityModel(weights={"a": 0.5, "b": 0.5})
    with pytest.raises(ValueError):
        model.compute({"a": np.zeros((2, 2))})
