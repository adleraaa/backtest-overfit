"""Vectorized monthly backtester with lagged execution, costs and volatility targeting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12
DEFAULT_VOL_TARGET = 0.10  # annualized


@dataclass(frozen=True)
class BacktestResult:
    returns: pd.Series  # net monthly excess return; NaN before the first position
    gross_returns: pd.Series  # before transaction costs
    turnover: pd.Series  # sum of |change in held weight| per month
    positions: pd.DataFrame  # weights held during each month (after vol scaling)


def backtest(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float = 10.0,
    vol_target: float | None = DEFAULT_VOL_TARGET,
    vol_window: int = 36,
    max_leverage: float = 5.0,
) -> BacktestResult:
    """Simulate holding ``weights`` decided at the end of month t during month t+1.

    Parameters
    ----------
    weights:
        Target weights indexed by decision month (row t uses information up to t).
    returns:
        Asset excess returns for the same months.
    cost_bps:
        One-way cost in basis points per unit of turnover (|delta w| summed over assets).
    vol_target:
        Annualized volatility target; ``None`` disables scaling. The scale used for the
        position decided at t is based on the realized volatility of the *unscaled*
        strategy up to t, so it never peeks at month t+1.
    vol_window:
        Months of history in the volatility estimate.
    max_leverage:
        Cap on gross exposure after scaling (low-vol periods would otherwise explode it).
    """
    weights = weights.reindex(index=returns.index, columns=returns.columns).fillna(0.0)
    asset_returns = returns.fillna(0.0)

    # Unscaled P&L: weights decided at t-1 earn returns of month t. Months with no
    # position are NaN rather than 0 so that signal warm-up does not drag the volatility
    # estimate toward zero (which would max out leverage).
    lagged = weights.shift(1)
    unscaled = (lagged * asset_returns).sum(axis=1).where(lagged.abs().sum(axis=1) > 0)

    if vol_target is None:
        decided = weights
    else:
        realized_vol = unscaled.rolling(vol_window, min_periods=vol_window).std()
        realized_vol *= np.sqrt(MONTHS_PER_YEAR)
        gross_exposure = weights.abs().sum(axis=1)
        scale = (vol_target / realized_vol).clip(upper=max_leverage / gross_exposure)
        # Before the volatility window fills there is no estimate, so no position.
        decided = weights.mul(scale.fillna(0.0), axis=0)

    held = decided.shift(1).fillna(0.0)
    gross = (held * asset_returns).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1)
    turnover.iloc[0] = held.iloc[0].abs().sum()
    net = gross - turnover * cost_bps / 1e4

    active = held.abs().sum(axis=1) > 0
    if active.any():
        first_active = active.idxmax()
        live = net.index >= first_active
    else:
        live = np.zeros(len(net), dtype=bool)
    return BacktestResult(
        returns=net.where(live),
        gross_returns=gross.where(live),
        turnover=turnover.where(live),
        positions=held,
    )
