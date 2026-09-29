"""The grid must not contain the same idea twice (a copy or a sign-flipped mirror).

Near-duplicate trials inflate the trial count and, for mirror images, the cross-trial
variance of Sharpe ratios that the Deflated Sharpe Ratio uses as its benchmark.
"""

import numpy as np
import pandas as pd
import pytest

from btoverfit.grid import build_grid, signal_grid
from btoverfit.strategies import cross_sectional_momentum, short_term_reversal
from btoverfit.study import run_trials


@pytest.fixture(scope="module")
def returns():
    # Fat-tailed, cross-sectionally correlated synthetic returns for 20 "industries".
    rng = np.random.default_rng(11)
    market = rng.standard_t(5, size=(400, 1)) * 0.03
    idio = rng.standard_t(5, size=(400, 20)) * 0.04
    index = pd.period_range("1980-01", periods=400, freq="M")
    return pd.DataFrame(market + idio, index=index, columns=[f"i{k}" for k in range(20)])


def test_no_two_signals_are_near_duplicates(returns):
    # Signals only (no vol targeting): the vol window is a deliberate second dimension on
    # top of each signal, and a copy or mirror must show up at the signal level.
    matrix, _ = run_trials(build_grid(vol_windows=(None,)), returns, "1983-01", cost_bps=0.0)
    corr = matrix.corr().to_numpy()
    off_diagonal = np.abs(corr[~np.eye(len(corr), dtype=bool)])
    assert off_diagonal.max() < 0.99


def test_reversal_with_an_xsmom_lookback_would_be_an_exact_mirror(returns):
    """Why reversal stops at lookback 2: at lookback 3 it is xsmom(3, skip=0) negated."""
    for q in (0.1, 0.2, 0.3):
        mirror = short_term_reversal(returns, lookback=3, quantile=q)
        momentum = cross_sectional_momentum(returns, lookback=3, skip=0, quantile=q)
        pd.testing.assert_frame_equal(mirror, -momentum + 0.0)
    xsmom_lookbacks = {p["lookback"] for f, p in signal_grid() if f == "xsmom"}
    reversal_lookbacks = {p["lookback"] for f, p in signal_grid() if f == "reversal"}
    assert not xsmom_lookbacks & reversal_lookbacks


def test_trial_names_are_unique():
    names = [t.name for t in build_grid()] + [t.name for t in build_grid(vol_windows=(None,))]
    assert len(names) == len(set(names))
