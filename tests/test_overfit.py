import itertools

import numpy as np
import pandas as pd
import pytest

from btoverfit.overfit import cscv_pbo, walk_forward


def reference_logits(x, n_blocks):
    """Slow, obviously-correct CSCV used to check the vectorized version."""
    x = x[len(x) % n_blocks :]
    blocks = np.array_split(np.arange(len(x)), n_blocks)
    out = []
    for combo in itertools.combinations(range(n_blocks), n_blocks // 2):
        is_rows = np.concatenate([blocks[i] for i in combo])
        oos_rows = np.setdiff1d(np.arange(len(x)), is_rows)
        is_sr = x[is_rows].mean(0) / x[is_rows].std(0, ddof=1)
        oos_sr = x[oos_rows].mean(0) / x[oos_rows].std(0, ddof=1)
        best = is_sr.argmax()
        rank = (oos_sr < oos_sr[best]).sum() + 1
        omega = rank / (x.shape[1] + 1)
        out.append(np.log(omega / (1 - omega)))
    return np.array(out)


def test_cscv_matches_reference_implementation():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.02, (123, 7))  # 123 rows: 3 get dropped for 6 blocks
    result = cscv_pbo(x, n_blocks=6)
    np.testing.assert_allclose(result.logits, reference_logits(x, 6))
    assert result.n_splits == 20 and result.rows_used == 120


def test_pbo_is_one_when_in_sample_winner_is_always_out_of_sample_loser():
    # Demeaned columns: whatever a trial gains in-sample it loses out-of-sample, so the
    # in-sample best by mean is always the out-of-sample worst.
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, (160, 10))
    x -= x.mean(axis=0)
    assert cscv_pbo(x, n_blocks=8, metric="mean").pbo == 1.0


def test_pbo_near_zero_with_one_genuinely_skilled_trial():
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 0.02, (480, 20))
    x[:, 5] += 0.015  # monthly Sharpe ~0.75: skill dwarfs noise
    result = cscv_pbo(x, n_blocks=16)
    assert result.pbo < 0.01
    assert result.prob_oos_loss < 0.01


def test_pbo_near_half_for_pure_noise():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 0.02, (480, 50))
    assert 0.3 < cscv_pbo(x, n_blocks=16).pbo < 0.7


def test_cscv_rejects_nan_and_odd_blocks():
    with pytest.raises(ValueError):
        cscv_pbo(np.full((32, 3), np.nan))
    with pytest.raises(ValueError):
        cscv_pbo(np.zeros((30, 3)), n_blocks=5)


def _matrix(values: dict):
    n = len(next(iter(values.values())))
    return pd.DataFrame(values, index=pd.period_range("2000-01", periods=n, freq="M"))


def test_walk_forward_follows_the_trailing_winner():
    rng = np.random.default_rng(4)
    noise = rng.normal(0, 0.01, (96, 2))
    a = noise[:, 0] + np.where(np.arange(96) < 48, 0.02, -0.02)
    b = noise[:, 1] + np.where(np.arange(96) < 48, -0.02, 0.02)
    res = walk_forward(_matrix({"a": a, "b": b}), train=12, test=12)
    assert res.selections["trial"].tolist() == ["a", "a", "a", "a", "b", "b", "b"]
    # The window starting at month 48 (when b takes over) was chosen on months 36-47, so
    # walk-forward is still holding the old winner: it pays for the regime change once.
    assert res.selections.loc[3, "test_start"] == "2004-01"
    assert res.selections.loc[3, "trial"] == "a"
    assert len(res.oos_returns) == 96 - 12


def test_walk_forward_selection_ignores_future_data():
    rng = np.random.default_rng(5)
    m = _matrix({k: rng.normal(0, 0.02, 120) for k in "abcd"})
    base = walk_forward(m, train=36, test=12).selections
    shocked = m.copy()
    shocked.iloc[60:, :] = rng.normal(0, 0.02, (60, 4))  # rewrite everything from month 60
    after = walk_forward(shocked, train=36, test=12).selections
    # Windows starting at 36 and 48 were chosen from data before month 60.
    assert base["trial"].iloc[:2].tolist() == after["trial"].iloc[:2].tolist()
    pd.testing.assert_series_equal(
        base["train_sharpe_ann"].iloc[:2], after["train_sharpe_ann"].iloc[:2]
    )


def test_walk_forward_skips_a_flat_trial_instead_of_dividing_by_zero():
    rng = np.random.default_rng(9)
    index = pd.period_range("2000-01", periods=60, freq="M")
    returns = pd.DataFrame(
        {"noise": rng.normal(0.001, 0.02, 60), "flat": np.full(60, 0.001)}, index=index
    )
    result = walk_forward(returns, train=24, test=12)
    # A constant positive return has infinite Sharpe if the zero std is not guarded.
    assert (result.selections["trial"] == "noise").all()
    assert np.isfinite(result.selections["train_sharpe_ann"]).all()
