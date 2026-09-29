"""Tests that fail if any signal or the backtester uses future data."""

import numpy as np
import pandas as pd
import pytest

from btoverfit.backtest import backtest
from btoverfit.grid import signal_grid
from btoverfit.strategies import FAMILIES, normalize_gross, trailing_return


@pytest.fixture(scope="module")
def returns():
    rng = np.random.default_rng(7)
    index = pd.period_range("1990-01", periods=90, freq="M")
    df = pd.DataFrame(rng.normal(0.005, 0.05, (90, 8)), index=index, columns=list("abcdefgh"))
    df.iloc[:20, 0] = np.nan  # an industry that starts late, like several in the real data
    return df


def assert_causal(signal_fn, returns, check_points=(30, 45, 60, 75)):
    """Weights at month t must not change when everything after t is deleted."""
    full = signal_fn(returns)
    for t in check_points:
        truncated = signal_fn(returns.iloc[: t + 1])
        pd.testing.assert_series_equal(full.iloc[t], truncated.iloc[t], check_names=False)


@pytest.mark.parametrize("family,params", signal_grid(), ids=lambda x: str(x))
def test_every_grid_signal_is_causal(family, params, returns):
    assert_causal(lambda r: FAMILIES[family](r, **params), returns)


def test_checker_catches_a_leaky_signal(returns):
    """Sanity check of the checker itself: peeking one month ahead must be detected."""

    def leaky(r):
        return normalize_gross(np.sign(trailing_return(r, 3).shift(-1)).fillna(0.0))

    with pytest.raises(AssertionError):
        assert_causal(leaky, returns)


def test_backtest_positions_are_causal(returns):
    """Positions held in month t+1 must not depend on returns after month t."""
    weights = FAMILIES["tsmom"](returns, lookback=6)
    full = backtest(weights, returns).positions
    for t in (50, 70):
        cut = returns.iloc[: t + 1]
        partial = backtest(weights.iloc[: t + 1], cut).positions
        pd.testing.assert_series_equal(full.iloc[t], partial.iloc[t], check_names=False)


def test_backtest_lags_weights_by_one_month():
    """A same-month signal on an alternating series must lose money if lagged correctly.

    weights_t = sign(r_t) and r_{t+1} = -r_t, so trading at t+1 always takes the wrong
    side. A backtester that forgot the lag would show a perfect winning record instead.
    """
    index = pd.period_range("2000-01", periods=40, freq="M")
    r = pd.DataFrame({"a": [0.01, -0.01] * 20}, index=index)
    w = np.sign(r)
    result = backtest(w, r, cost_bps=0.0, vol_target=None)
    live = result.gross_returns.dropna()
    assert (live < 0).all()
