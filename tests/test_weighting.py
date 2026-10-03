import numpy as np
import pytest

from src import weighting as wt


def test_ahp_weights_sum_to_one():
    names = ["a", "b", "c"]
    matrix = np.array([
        [1, 3, 5],
        [1/3, 1, 3],
        [1/5, 1/3, 1],
    ])
    result = wt.ahp_weights(matrix, names)
    total = sum(result["weights"].values())
    assert total == pytest.approx(1.0, abs=1e-6)


def test_ahp_weights_consistent_matrix_flagged_consistent():
    # A perfectly consistent 3x3 matrix (derived from true ratios 4:2:1)
    names = ["a", "b", "c"]
    true_w = np.array([4.0, 2.0, 1.0])
    matrix = np.outer(true_w, 1 / true_w)
    result = wt.ahp_weights(matrix, names)
    assert result["consistency_ratio"] < 1e-6
    assert result["is_consistent"] is True


def test_ahp_weights_ordering_matches_dominance():
    # 'a' is judged much more important than 'b' and 'c'
    names = ["a", "b", "c"]
    matrix = np.array([
        [1, 5, 7],
        [1/5, 1, 3],
        [1/7, 1/3, 1],
    ])
    result = wt.ahp_weights(matrix, names)
    w = result["weights"]
    assert w["a"] > w["b"] > w["c"]


def test_ahp_rejects_non_square_matrix():
    matrix = np.ones((2, 3))
    with pytest.raises(ValueError):
        wt.ahp_weights(matrix, ["a", "b"])


def test_build_default_pairwise_matrix_shape():
    order = ["mean_annual_precip", "rx1day", "elevation", "slope"]
    matrix = wt.build_default_pairwise_matrix(order)
    assert matrix.shape == (4, 4)
    assert np.allclose(np.diag(matrix), 1.0)


def test_compare_weights_handles_missing_rf():
    ahp_result = {"weights": {"a": 0.6, "b": 0.4}}
    df = wt.compare_weights(ahp_result, rf_result=None)
    assert "AHP_weight" in df.columns
    assert df["RF_weight"].isna().all()


def test_resolve_final_weights_ahp_method():
    ahp_result = {"weights": {"a": 0.6, "b": 0.4}}
    final = wt.resolve_final_weights(ahp_result, None, method="ahp")
    assert final == {"a": 0.6, "b": 0.4}


def test_resolve_final_weights_invalid_method_raises():
    ahp_result = {"weights": {"a": 1.0}}
    with pytest.raises(ValueError):
        wt.resolve_final_weights(ahp_result, None, method="not_a_method")
