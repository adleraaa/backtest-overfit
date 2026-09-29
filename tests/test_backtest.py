import numpy as np
import pandas as pd
import pytest

from btoverfit.backtest import backtest


def _single_asset(values):
    index = pd.period_range("2000-01", periods=len(values), freq="M")
    return pd.DataFrame({"a": values}, index=index)


def test_constant_weight_earns_next_month_return_and_pays_entry_cost_once():
    r = _single_asset([0.01, 0.02, -0.03, 0.04])
    w = pd.DataFrame({"a": [1.0] * 4}, index=r.index)
    res = backtest(w, r, cost_bps=50.0, vol_target=None)
    assert np.isnan(res.returns.iloc[0])  # no position during the first month
    assert res.gross_returns.iloc[1:].tolist() == pytest.approx([0.02, -0.03, 0.04])
    # 50 bps charged only when the position is opened.
    assert res.returns.iloc[1] == pytest.approx(0.02 - 0.005)
    assert res.returns.iloc[2:].tolist() == pytest.approx([-0.03, 0.04])
    assert res.turnover.iloc[1:].tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_flipping_position_costs_twice_the_gross():
    r = _single_asset([0.0] * 4)
    w = pd.DataFrame({"a": [1.0, -1.0, 1.0, -1.0]}, index=r.index)
    res = backtest(w, r, cost_bps=10.0, vol_target=None)
    assert res.turnover.iloc[2:].tolist() == pytest.approx([2.0, 2.0])
    assert res.returns.iloc[2] == pytest.approx(-2.0 * 10 / 1e4)


def test_vol_targeting_hits_target_on_stationary_series():
    rng = np.random.default_rng(0)
    r = _single_asset(rng.normal(0.0, 0.02, 2000))  # ~6.9% annual vol unscaled
    w = pd.DataFrame({"a": 1.0}, index=r.index)
    res = backtest(w, r, cost_bps=0.0, vol_target=0.10, vol_window=36)
    realized = res.returns.dropna().std() * np.sqrt(12)
    assert realized == pytest.approx(0.10, rel=0.05)


def test_vol_targeting_respects_leverage_cap():
    rng = np.random.default_rng(1)
    r = _single_asset(rng.normal(0.0, 0.001, 200))  # tiny vol would imply huge leverage
    w = pd.DataFrame({"a": 1.0}, index=r.index)
    res = backtest(w, r, cost_bps=0.0, vol_target=0.10, vol_window=12, max_leverage=3.0)
    assert res.positions.abs().sum(axis=1).max() == pytest.approx(3.0)


def test_one_flat_month_does_not_switch_the_strategy_off():
    rng = np.random.default_rng(3)
    r = _single_asset(rng.normal(0.0, 0.02, 80))
    w = pd.DataFrame({"a": 1.0}, index=r.index)
    w.iloc[40] = 0.0  # one month with no signal, well after the vol window has filled
    res = backtest(w, r, cost_bps=0.0, vol_window=12)
    held = res.positions["a"]
    assert held.iloc[41] == 0.0  # the flat decision is held in month 41
    # From month 42 on the strategy trades again; a NaN-masked flat month would have made
    # the rolling volatility NaN (and the position 0) for the next 12 months.
    assert (held.iloc[42:] > 0).all()


def test_no_position_until_vol_window_filled():
    rng = np.random.default_rng(2)
    r = _single_asset(rng.normal(0.0, 0.02, 60))
    w = pd.DataFrame({"a": 1.0}, index=r.index)
    res = backtest(w, r, vol_window=24)
    # Weights start in month 0, first P&L month 1, so the estimate exists at month 24
    # and the first scaled position is held in month 25.
    assert res.returns.first_valid_index() == r.index[25]
