import numpy as np
import pytest

from src import classification as clf


def test_minmax_standardize_range():
    arr = np.array([[1.0, 5.0], [10.0, 20.0]])
    out = clf.minmax_standardize(arr)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_minmax_standardize_constant_array():
    arr = np.full((3, 3), 5.0)
    out = clf.minmax_standardize(arr)
    assert np.all(out == 0.0)


def test_zscore_standardize_bounds():
    rng = np.random.default_rng(0)
    arr = rng.normal(10, 2, size=(50, 50))
    out = clf.zscore_standardize(arr, clip_sigma=3.0)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_percentile_rank_standardize_monotonic():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = clf.percentile_rank_standardize(arr)
    # Highest raw value should get the highest rank
    assert out[1, 1] > out[0, 0]


def test_fuzzy_decay_standardize_decreasing():
    dist = np.array([0.0, 50_000.0, 200_000.0])
    out = clf.fuzzy_decay_standardize(dist, decay_km=100)
    assert out[0] > out[1] > out[2]
    assert out[0] == pytest.approx(1.0)


def test_classify_quantile_produces_n_classes():
    rng = np.random.default_rng(1)
    arr = rng.uniform(0, 1, size=(100, 100))
    classes = clf.classify_quantile(arr, n_classes=5)
    valid = classes[np.isfinite(classes)]
    assert set(np.unique(valid)).issubset({1, 2, 3, 4, 5})
    assert len(np.unique(valid)) == 5


def test_standardize_unknown_method_raises():
    arr = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        clf.standardize(arr, method="not_a_real_method")


def test_classify_unknown_method_raises():
    arr = np.array([[0.1, 0.9]])
    with pytest.raises(ValueError):
        clf.classify(arr, method="not_a_real_method")
