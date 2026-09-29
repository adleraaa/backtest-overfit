"""The parameter grid: every combination here counts as one "trial" for PBO and DSR."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import pandas as pd

from .backtest import DEFAULT_VOL_TARGET, BacktestResult, backtest
from .strategies import FAMILIES


@dataclass(frozen=True)
class Trial:
    family: str
    params: dict = field(hash=False)
    vol_window: int | None = 36  # None = no volatility targeting (gross exposure 1)

    @property
    def name(self) -> str:
        args = ",".join(f"{k}={v}" for k, v in self.params.items())
        suffix = "novol" if self.vol_window is None else f"vol{self.vol_window}"
        return f"{self.family}({args})|{suffix}"

    def run(self, returns: pd.DataFrame, cost_bps: float = 10.0) -> BacktestResult:
        weights = FAMILIES[self.family](returns, **self.params)
        if self.vol_window is None:
            return backtest(weights, returns, cost_bps=cost_bps, vol_target=None)
        return backtest(
            weights,
            returns,
            cost_bps=cost_bps,
            vol_target=DEFAULT_VOL_TARGET,
            vol_window=self.vol_window,
        )


def signal_grid() -> list[tuple[str, dict]]:
    """Signal families and parameters, before the volatility-window dimension."""
    specs: list[tuple[str, dict]] = []
    for lookback in (1, 3, 6, 9, 12, 18, 24):
        specs.append(("tsmom", {"lookback": lookback}))
    for lookback, skip, q in itertools.product((3, 6, 9, 12), (0, 1), (0.1, 0.2, 0.3)):
        specs.append(("xsmom", {"lookback": lookback, "skip": skip, "quantile": q}))
    # Reversal stops at a 2-month lookback on purpose: reversal(lookback=3) would rank
    # -trailing_return(3), the exact negative of xsmom(lookback=3, skip=0), and a trial and
    # its mirror image double-count one idea and inflate the cross-trial Var[SR] in the DSR.
    # tests/test_grid.py checks that no two trials are near-duplicates.
    for lookback, q in itertools.product((1, 2), (0.1, 0.2, 0.3)):
        specs.append(("reversal", {"lookback": lookback, "quantile": q}))
    for fast, slow in itertools.product((1, 2, 3, 6), (6, 9, 12, 18, 24)):
        if fast < slow:
            specs.append(("ma_cross", {"fast": fast, "slow": slow}))
    return specs


def build_grid(vol_windows: tuple[int | None, ...] = (12, 36)) -> list[Trial]:
    return [
        Trial(family, params, vol_window)
        for family, params in signal_grid()
        for vol_window in vol_windows
    ]
