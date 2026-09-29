"""Selection-bias diagnostics: CSCV / Probability of Backtest Overfitting and walk-forward.

References
----------
Bailey, Borwein, Lopez de Prado, Zhu (2015), "The Probability of Backtest Overfitting",
Journal of Computational Finance 20(4).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PBOResult:
    pbo: float  # share of splits where the in-sample winner ranks below the OOS median
    logits: np.ndarray  # lambda = log(omega / (1 - omega)) per split
    is_best_perf: np.ndarray  # in-sample performance of the chosen trial, per split
    oos_best_perf: np.ndarray  # out-of-sample performance of the same trial
    prob_oos_loss: float  # share of splits where the chosen trial loses money OOS
    n_splits: int
    n_trials: int
    rows_used: int


def _performance(total: np.ndarray, total_sq: np.ndarray, n: int, metric: str) -> np.ndarray:
    mean = total / n
    if metric == "mean":
        return mean
    var = (total_sq - n * mean**2) / (n - 1)
    return mean / np.sqrt(var)


def cscv_pbo(returns: pd.DataFrame | np.ndarray, n_blocks: int = 16, metric: str = "sharpe"):
    """Combinatorially Symmetric Cross-Validation.

    The (T x N) matrix of trial returns is cut into ``n_blocks`` contiguous blocks. For
    every way of choosing half the blocks as in-sample (IS), the trial with the best IS
    performance is "selected", and we record where it ranks among all trials on the
    remaining out-of-sample (OOS) blocks. PBO is the fraction of splits where it ranks at
    or below the OOS median.

    Performance per split is computed from per-block sums and sums of squares, so the
    C(16, 8) = 12,870 splits cost a few matrix products instead of re-slicing the data.
    """
    if metric not in {"sharpe", "mean"}:
        raise ValueError("metric must be 'sharpe' or 'mean'")
    if n_blocks % 2:
        raise ValueError("n_blocks must be even")
    x = np.asarray(returns, dtype=float)
    if np.isnan(x).any():
        raise ValueError("returns must not contain NaN; align trials to a common window")
    # Drop the earliest rows so the sample splits into equal blocks.
    x = x[len(x) % n_blocks :]
    rows, n_trials = x.shape
    per_block = rows // n_blocks

    blocks = x.reshape(n_blocks, per_block, n_trials)
    block_sum = blocks.sum(axis=1)
    block_sq = (blocks**2).sum(axis=1)

    combos = np.array(list(itertools.combinations(range(n_blocks), n_blocks // 2)))
    is_mask = np.zeros((len(combos), n_blocks))
    np.put_along_axis(is_mask, combos, 1.0, axis=1)
    n_half = per_block * n_blocks // 2

    is_sum, is_sq = is_mask @ block_sum, is_mask @ block_sq
    oos_sum, oos_sq = block_sum.sum(0) - is_sum, block_sq.sum(0) - is_sq
    is_perf = _performance(is_sum, is_sq, n_half, metric)
    oos_perf = _performance(oos_sum, oos_sq, n_half, metric)

    split = np.arange(len(combos))
    best = is_perf.argmax(axis=1)
    oos_best = oos_perf[split, best]
    # Rank 1 = worst OOS, N = best OOS; omega is the relative rank in (0, 1).
    rank = (oos_perf < oos_best[:, None]).sum(axis=1) + 1
    omega = rank / (n_trials + 1)
    logits = np.log(omega / (1 - omega))

    return PBOResult(
        pbo=float((logits <= 0).mean()),
        logits=logits,
        is_best_perf=is_perf[split, best],
        oos_best_perf=oos_best,
        prob_oos_loss=float((oos_best < 0).mean()),
        n_splits=len(combos),
        n_trials=n_trials,
        rows_used=rows,
    )


@dataclass(frozen=True)
class WalkForwardResult:
    oos_returns: pd.Series  # stitched returns of whichever trial was selected each window
    selections: pd.DataFrame  # one row per test window: dates, chosen trial, train Sharpe


def walk_forward(returns: pd.DataFrame, train: int = 120, test: int = 12) -> WalkForwardResult:
    """Rolling selection: pick the best trailing-``train``-month Sharpe, hold it ``test`` months.

    Only data strictly before each test window is used to choose, so the stitched series is
    a genuine out-of-sample record of "always trade the current backtest winner".
    """
    if returns.isna().to_numpy().any():
        raise ValueError("returns must not contain NaN; align trials to a common window")
    pieces = []
    rows = []
    for start in range(train, len(returns), test):
        window = returns.iloc[start - train : start]
        train_sharpe = window.mean() / window.std(ddof=1)
        chosen = train_sharpe.idxmax()
        test_slice = returns[chosen].iloc[start : start + test]
        pieces.append(test_slice)
        rows.append(
            {
                "test_start": str(test_slice.index[0]),
                "test_end": str(test_slice.index[-1]),
                "trial": chosen,
                "train_sharpe_ann": float(train_sharpe[chosen] * np.sqrt(12)),
            }
        )
    return WalkForwardResult(oos_returns=pd.concat(pieces), selections=pd.DataFrame(rows))
