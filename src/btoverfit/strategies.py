"""Signal functions for the four strategy families.

Every function maps a (months x assets) DataFrame of excess returns to target weights of
the same shape. The weight in row t may only use returns up to and including month t;
the backtester then holds those weights during month t+1. ``tests/test_no_lookahead.py``
enforces this by recomputing each signal on truncated data.

All raw weights are scaled to gross exposure 1 (sum of |w| = 1) so that families are
comparable before volatility targeting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trailing_return(returns: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    """Compounded return over months [t-skip-lookback+1, t-skip], known at the end of t."""
    log_growth = np.log1p(returns).rolling(lookback, min_periods=lookback).sum()
    return np.expm1(log_growth).shift(skip)


def normalize_gross(weights: pd.DataFrame) -> pd.DataFrame:
    """Scale each row to sum(|w|) = 1; rows with no position stay all zero."""
    gross = weights.abs().sum(axis=1)
    return weights.div(gross.where(gross > 0), axis=0).fillna(0.0)


def long_short_quantiles(score: pd.DataFrame, quantile: float) -> pd.DataFrame:
    """Long the top ``quantile`` of assets by score, short the bottom, equal weight.

    Assets with a NaN score are skipped, so the number of names per leg adapts to how
    many industries have data that month (some start late in the sample).
    """
    available = score.notna().sum(axis=1)
    k = np.floor(available * quantile).clip(lower=1)
    # method="first" breaks ties deterministically so the legs never overlap.
    top = score.rank(axis=1, ascending=False, method="first").le(k, axis=0)
    bottom = score.rank(axis=1, ascending=True, method="first").le(k, axis=0)
    weights = top.astype(float) - bottom.astype(float)
    weights[available < 2] = 0.0  # a single asset cannot be both long and short
    return normalize_gross(weights)


def time_series_momentum(returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Long industries with a positive trailing excess return, short the rest."""
    signal = np.sign(trailing_return(returns, lookback)).fillna(0.0)
    return normalize_gross(signal)


def cross_sectional_momentum(
    returns: pd.DataFrame, lookback: int, skip: int, quantile: float
) -> pd.DataFrame:
    """Long past winners, short past losers (optionally skipping the latest month)."""
    return long_short_quantiles(trailing_return(returns, lookback, skip), quantile)


def short_term_reversal(returns: pd.DataFrame, lookback: int, quantile: float) -> pd.DataFrame:
    """Long recent losers, short recent winners."""
    return long_short_quantiles(-trailing_return(returns, lookback), quantile)


def moving_average_crossover(returns: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """Long when the fast moving average of the cumulative-return index is above the slow one.

    ``fast=1`` is the classic "price above its moving average" rule.
    """
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be shorter than slow ({slow})")
    # Missing months (an industry that does not exist yet) are treated as flat so the index
    # is defined; the has_history mask below keeps those industries out of the portfolio.
    level = np.exp(np.log1p(returns.fillna(0.0)).cumsum())
    signal = np.sign(level.rolling(fast).mean() - level.rolling(slow).mean())
    has_history = returns.rolling(slow, min_periods=1).count() == slow
    return normalize_gross(signal.where(has_history, 0.0).fillna(0.0))


FAMILIES = {
    "tsmom": time_series_momentum,
    "xsmom": cross_sectional_momentum,
    "reversal": short_term_reversal,
    "ma_cross": moving_average_crossover,
}
