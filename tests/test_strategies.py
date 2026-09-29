import numpy as np
import pandas as pd
import pytest

from btoverfit.strategies import (
    long_short_quantiles,
    moving_average_crossover,
    short_term_reversal,
    time_series_momentum,
    trailing_return,
)


def _frame(values, columns=None):
    df = pd.DataFrame(values, columns=columns)
    df.index = pd.period_range("2000-01", periods=len(df), freq="M")
    return df


def test_trailing_return_compounds_and_skips():
    r = _frame({"a": [0.10, 0.10, -0.50, 0.20]})
    tr = trailing_return(r, lookback=2)
    assert np.isnan(tr["a"].iloc[0])
    assert tr["a"].iloc[1] == pytest.approx(1.1 * 1.1 - 1)
    assert tr["a"].iloc[2] == pytest.approx(1.1 * 0.5 - 1)
    skipped = trailing_return(r, lookback=2, skip=1)
    assert skipped["a"].iloc[2] == pytest.approx(tr["a"].iloc[1])


def test_long_short_quantiles_picks_extremes_with_unit_gross():
    score = _frame([[5.0, 1.0, 3.0, 2.0, 4.0]], columns=list("abcde"))
    w = long_short_quantiles(score, quantile=0.2).iloc[0]
    assert w["a"] == pytest.approx(0.5)
    assert w["b"] == pytest.approx(-0.5)
    assert w[["c", "d", "e"]].abs().sum() == 0
    assert w.sum() == pytest.approx(0.0)  # dollar neutral


def test_long_short_quantiles_ignores_missing_scores():
    score = _frame([[np.nan, 1.0, 3.0, np.nan]], columns=list("abcd"))
    w = long_short_quantiles(score, quantile=0.5).iloc[0]
    assert w["c"] == pytest.approx(0.5) and w["b"] == pytest.approx(-0.5)
    assert w["a"] == 0 and w["d"] == 0


def test_long_short_quantiles_single_asset_has_no_position():
    score = _frame([[np.nan, 1.0]], columns=list("ab"))
    assert long_short_quantiles(score, 0.5).abs().sum().sum() == 0


def test_time_series_momentum_signs():
    r = _frame({"up": [0.01, 0.02, 0.01], "down": [-0.01, -0.02, -0.03]})
    w = time_series_momentum(r, lookback=2)
    assert (w.iloc[0] == 0).all()  # not enough history yet
    assert w["up"].iloc[2] == pytest.approx(0.5)
    assert w["down"].iloc[2] == pytest.approx(-0.5)


def test_reversal_is_opposite_of_recent_winners():
    r = _frame([[0.05, -0.05, 0.0, 0.01]], columns=list("abcd"))
    w = short_term_reversal(r, lookback=1, quantile=0.25).iloc[0]
    assert w["b"] > 0 and w["a"] < 0


def test_moving_average_crossover_trend_and_validation():
    rising = [0.02] * 8
    falling = [-0.02] * 8
    w = moving_average_crossover(_frame({"up": rising, "dn": falling}), fast=1, slow=3)
    assert (w.iloc[:2] == 0).all().all()  # needs `slow` months of history
    assert w["up"].iloc[-1] == pytest.approx(0.5)
    assert w["dn"].iloc[-1] == pytest.approx(-0.5)
    with pytest.raises(ValueError):
        moving_average_crossover(_frame({"a": rising}), fast=6, slow=6)
