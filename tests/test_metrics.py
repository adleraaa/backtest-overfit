import numpy as np
import pandas as pd
import pytest
from scipy import stats

from btoverfit.metrics import (
    bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    effective_number_of_trials,
    expected_max_sharpe,
    max_drawdown,
    probabilistic_sharpe_ratio,
    psr_from_moments,
    sharpe_ratio,
    sign_aligned,
)


def test_sharpe_ratio_known_value():
    # mean 0.02, sample std 0.02 * sqrt(2) / 2 ... computed by hand below.
    r = [0.01, 0.03]
    expected = 0.02 / np.std(r, ddof=1) * np.sqrt(12)
    assert sharpe_ratio(r) == pytest.approx(expected)
    assert sharpe_ratio([0.01, np.nan, 0.03]) == pytest.approx(expected)  # NaNs ignored


def test_sharpe_ratio_scale_invariant_sign_flip():
    rng = np.random.default_rng(0)
    r = rng.normal(0.01, 0.05, 100)
    assert sharpe_ratio(3 * r) == pytest.approx(sharpe_ratio(r))
    assert sharpe_ratio(-r) == pytest.approx(-sharpe_ratio(r))


def test_max_drawdown_known_path():
    # Wealth 1.1 -> 0.55 -> 0.66: worst drop is 0.55 / 1.1 - 1 = -50%.
    assert max_drawdown([0.1, -0.5, 0.2]) == pytest.approx(-0.5)
    assert max_drawdown([0.01, 0.02]) == 0.0
    # A loss in the very first month counts from the starting wealth of 1.
    assert max_drawdown([-0.2, 0.1]) == pytest.approx(-0.2)


def test_psr_equals_half_at_own_sharpe():
    rng = np.random.default_rng(1)
    r = rng.normal(0.01, 0.04, 240)
    own = r.mean() / r.std(ddof=1)
    assert probabilistic_sharpe_ratio(r, benchmark_sr=own) == pytest.approx(0.5)


def test_psr_matches_closed_form_for_two_point_sample():
    # An alternating two-point series has skew 0 and raw kurtosis 1 (not 3), so only the
    # kurtosis term of the standard error matters; compare with the formula directly.
    r = np.tile([0.03, -0.01], 60)
    sr = r.mean() / r.std(ddof=1)
    kurt = stats.kurtosis(r, fisher=False)
    expected = stats.norm.cdf(sr * np.sqrt(len(r) - 1) / np.sqrt(1 + (kurt - 1) / 4 * sr**2))
    assert probabilistic_sharpe_ratio(r) == pytest.approx(expected)


@pytest.mark.parametrize("n,exact", [(10, 1.538753), (100, 2.507594), (1000, 3.241436)])
def test_expected_max_sharpe_close_to_exact_normal_order_statistic(n, exact):
    # Exact E[max of n standard normals] from order-statistic tables; the extreme-value
    # approximation in the DSR paper should be within a few percent.
    assert expected_max_sharpe(n, 1.0) == pytest.approx(exact, rel=0.03)


def test_expected_max_sharpe_scales_with_std_and_shifts_with_mean():
    base = expected_max_sharpe(50, 1.0)
    assert expected_max_sharpe(50, 0.25) == pytest.approx(0.5 * base)
    assert expected_max_sharpe(50, 0.25, sr_mean=0.1) == pytest.approx(0.1 + 0.5 * base)
    assert expected_max_sharpe(1, 0.25) == 0.0


def test_deflated_sharpe_single_trial_equals_psr_and_falls_with_trials():
    rng = np.random.default_rng(2)
    r = rng.normal(0.01, 0.04, 300)
    assert deflated_sharpe_ratio(r, 1, 0.01) == pytest.approx(probabilistic_sharpe_ratio(r))
    dsrs = [deflated_sharpe_ratio(r, n, 0.01) for n in (1, 10, 100, 1000)]
    assert all(a > b for a, b in zip(dsrs, dsrs[1:], strict=False))


def test_deflated_sharpe_rejects_best_of_many_noise_strategies():
    """Selecting the best of 200 pure-noise strategies should not look significant."""
    rng = np.random.default_rng(3)
    trials = rng.normal(0.0, 0.04, (240, 200))
    srs = trials.mean(0) / trials.std(0, ddof=1)
    best = trials[:, srs.argmax()]
    assert probabilistic_sharpe_ratio(best) > 0.95  # naive test is fooled
    assert deflated_sharpe_ratio(best, 200, srs.var(ddof=1)) < 0.95  # DSR is not


def test_deflated_sharpe_reproduces_paper_worked_example():
    """Bailey & Lopez de Prado (2014), numerical example: N = 100 trials whose annualized
    Sharpe ratios have variance 1/2, a selected strategy with annualized SR 2.5 over
    T = 1250 daily observations, skew -3 and raw kurtosis 10. The paper reports a
    per-period threshold SR_0 = 0.1132 and DSR = 0.9004 (250 trading days a year)."""
    days = 250
    sr0 = expected_max_sharpe(100, 0.5) / np.sqrt(days)
    assert sr0 == pytest.approx(0.1132, abs=5e-4)
    dsr = psr_from_moments(2.5 / np.sqrt(days), 1250, skew=-3.0, kurtosis=10.0, benchmark_sr=sr0)
    assert dsr == pytest.approx(0.9004, abs=5e-4)


def test_sharpe_is_nan_not_inf_when_undefined():
    assert np.isnan(sharpe_ratio([0.01]))
    assert np.isnan(sharpe_ratio([0.01, 0.01, 0.01]))  # zero variance
    assert np.isnan(probabilistic_sharpe_ratio([0.0, 0.0, 0.0]))


def test_effective_number_of_trials_counts_copies_and_mirrors_once():
    rng = np.random.default_rng(5)
    independent = pd.DataFrame(rng.normal(0, 1, (5000, 4)))
    assert effective_number_of_trials(independent) == pytest.approx(4, rel=0.02)
    base = independent[0]
    copies = pd.DataFrame({"a": base, "b": 2 * base, "c": -base})  # scaled copy + mirror
    assert effective_number_of_trials(copies) == pytest.approx(1.0)
    # Two independent ideas, each tried three times (once as its mirror image).
    pairs = pd.DataFrame({"a": base, "b": -base, "c": independent[1], "d": -independent[1]})
    assert effective_number_of_trials(pairs) == pytest.approx(2, rel=0.02)


def test_sign_aligned_flips_only_losing_families():
    sharpes = pd.Series([-0.5, -0.3, 0.1, 0.2, -0.1], index=list("abcde"))
    families = pd.Series(["rev", "rev", "mom", "mom", "mom"], index=list("abcde"))
    aligned = sign_aligned(sharpes, families)
    # "rev" has a negative median and is flipped as a whole; "mom" keeps its losing trial.
    assert aligned.tolist() == pytest.approx([0.5, 0.3, 0.1, 0.2, -0.1])


def test_bootstrap_ci_brackets_estimate_and_is_reproducible():
    rng = np.random.default_rng(4)
    r = rng.normal(0.01, 0.04, 360)
    lo, hi = bootstrap_sharpe_ci(r, n_boot=2000, seed=11)
    assert lo < sharpe_ratio(r) < hi
    assert (lo, hi) == bootstrap_sharpe_ci(r, n_boot=2000, seed=11)
    # Width should be near the analytic iid value 2 * 1.96 * sqrt((1 + SR^2/2) / T) * sqrt(12).
    sr_m = r.mean() / r.std(ddof=1)
    analytic = 2 * 1.96 * np.sqrt((1 + sr_m**2 / 2) / len(r)) * np.sqrt(12)
    assert (hi - lo) == pytest.approx(analytic, rel=0.25)
