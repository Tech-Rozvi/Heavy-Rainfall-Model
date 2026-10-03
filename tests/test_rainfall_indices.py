import numpy as np
import pytest

from src import rainfall_indices as ri


@pytest.fixture
def synthetic_stack():
    rng = np.random.default_rng(0)
    # 100 days, 4x4 grid
    stack = rng.gamma(0.6, 5.0, size=(100, 4, 4)).astype("float32")
    stack[rng.random(stack.shape) < 0.4] = 0.0  # dry days
    return stack


def test_rx1day_equals_max(synthetic_stack):
    out = ri.rx1day(synthetic_stack)
    assert out.shape == (4, 4)
    assert np.allclose(out, synthetic_stack.max(axis=0))


def test_rx5day_at_least_rx1day(synthetic_stack):
    rx1 = ri.rx1day(synthetic_stack)
    rx5 = ri.rx5day(synthetic_stack)
    # 5-day total must be >= the single wettest day (precip is non-negative)
    assert np.all(rx5 >= rx1 - 1e-6)


def test_rx5day_requires_min_5_days():
    short_stack = np.zeros((3, 2, 2))
    with pytest.raises(ValueError):
        ri.rx5day(short_stack)


def test_cdd_cwd_sum_bounds(synthetic_stack):
    dry_run = ri.cdd(synthetic_stack)
    wet_run = ri.cwd(synthetic_stack)
    n_days = synthetic_stack.shape[0]
    assert np.all(dry_run <= n_days)
    assert np.all(wet_run <= n_days)
    assert np.all(dry_run >= 0)
    assert np.all(wet_run >= 0)


def test_r10mm_r20mm_ordering(synthetic_stack):
    r10 = ri.r10mm(synthetic_stack)
    r20 = ri.r20mm(synthetic_stack)
    # Fewer (or equal) days should exceed the higher 20mm threshold
    assert np.all(r20 <= r10)


def test_percentile_threshold_index_95_le_99(synthetic_stack):
    thresh95, r95p = ri.percentile_threshold_index(synthetic_stack, 95.0)
    thresh99, r99p = ri.percentile_threshold_index(synthetic_stack, 99.0)
    assert np.all(thresh99 >= thresh95 - 1e-6)


def test_compute_all_indices_keys_present(synthetic_stack):
    out = ri.compute_all_indices(synthetic_stack, n_years=1.0)
    expected_keys = {"mean_annual_precip", "rx1day", "rx5day", "r95p", "r99p",
                      "r10mm", "r20mm", "cdd", "cwd"}
    assert expected_keys.issubset(out.keys())
    for k, v in out.items():
        assert v.shape == (4, 4)


def test_mean_annual_precip_scales_with_years(synthetic_stack):
    out_1yr = ri.mean_annual_precip(synthetic_stack, n_years=1.0)
    out_2yr = ri.mean_annual_precip(synthetic_stack, n_years=2.0)
    assert np.allclose(out_1yr, out_2yr * 2, rtol=1e-6)
