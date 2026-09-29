"""Performance metrics and Sharpe-ratio inference.

Conventions: inputs are monthly returns; NaNs are dropped. ``sharpe_ratio`` is annualized.
The probabilistic and deflated Sharpe functions work with *per-period* (monthly) Sharpe
ratios, as in the original papers, because their standard errors are derived per period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MONTHS_PER_YEAR = 12


def _clean(returns) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    return r[~np.isnan(r)]


def _per_period_sharpe(r: np.ndarray) -> float:
    """mean / sample std, or NaN when it is undefined (fewer than 2 points or zero std).

    Returning NaN explicitly (instead of letting numpy produce inf or a warning) lets the
    experiment script fail loudly when it serializes results with allow_nan=False.
    """
    if len(r) < 2:
        return float("nan")
    std = r.std(ddof=1)
    return float(r.mean() / std) if std > 0 else float("nan")


def sharpe_ratio(returns, periods_per_year: int = MONTHS_PER_YEAR) -> float:
    """Annualized Sharpe ratio of excess returns (sample std with ddof=1)."""
    return _per_period_sharpe(_clean(returns)) * float(np.sqrt(periods_per_year))


def annualized_return(returns, periods_per_year: int = MONTHS_PER_YEAR) -> float:
    """Arithmetic mean return times periods per year (consistent with the Sharpe ratio)."""
    return float(_clean(returns).mean() * periods_per_year)


def annualized_volatility(returns, periods_per_year: int = MONTHS_PER_YEAR) -> float:
    return float(_clean(returns).std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(returns) -> float:
    """Largest peak-to-trough loss of the compounded wealth curve, as a negative number."""
    wealth = np.cumprod(1.0 + _clean(returns))
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    return float((wealth / peak - 1.0).min())


def probabilistic_sharpe_ratio(returns, benchmark_sr: float = 0.0) -> float:
    """P(true per-period Sharpe > benchmark_sr) -- Bailey & Lopez de Prado (2012).

    Uses the non-normal standard error of the Sharpe ratio (Mertens 2002), so fat tails
    and negative skew make the estimate less trustworthy.
    """
    r = _clean(returns)
    sr = _per_period_sharpe(r)
    if np.isnan(sr):
        return float("nan")
    # raw kurtosis (3 for a normal), as in the paper
    return psr_from_moments(
        sr, len(r), stats.skew(r), stats.kurtosis(r, fisher=False), benchmark_sr
    )


def psr_from_moments(
    sr: float, n_obs: int, skew: float, kurtosis: float, benchmark_sr: float = 0.0
) -> float:
    """PSR from summary statistics: per-period Sharpe, sample size, skew, raw kurtosis.

    Z[(SR - SR*) * sqrt(T - 1) / sqrt(1 - skew * SR + (kurt - 1) / 4 * SR^2)]. Split out
    from ``probabilistic_sharpe_ratio`` so it can be checked against the paper's worked
    example, which is stated in moments rather than as a return series.
    """
    se_scale = np.sqrt(1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2)
    return float(stats.norm.cdf((sr - benchmark_sr) * np.sqrt(n_obs - 1) / se_scale))


def expected_max_sharpe(n_trials: float, sr_variance: float, sr_mean: float = 0.0) -> float:
    """Expected maximum of ``n_trials`` Sharpe estimates whose true value is ``sr_mean``.

    Extreme-value approximation from Bailey & Lopez de Prado (2014), eq. for SR_0:
    E[max] ~ mean + std * ((1 - g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N e))), g = Euler's
    constant. Units follow the inputs (per-period in this project). ``n_trials`` may be
    fractional, e.g. an effective number of independent trials.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return float(sr_mean)
    g = np.euler_gamma
    z = (1 - g) * stats.norm.ppf(1 - 1 / n_trials) + g * stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return float(sr_mean + np.sqrt(sr_variance) * z)


def deflated_sharpe_ratio(returns, n_trials: float, trial_sr_variance: float) -> float:
    """Deflated Sharpe Ratio -- Bailey & Lopez de Prado (2014).

    The PSR of the selected strategy against the Sharpe ratio one would expect from the
    best of ``n_trials`` skill-less strategies. ``trial_sr_variance`` is the variance of the
    per-period Sharpe ratios across all trials that were tried.
    """
    benchmark = expected_max_sharpe(n_trials, trial_sr_variance)
    return probabilistic_sharpe_ratio(returns, benchmark_sr=benchmark)


def effective_number_of_trials(returns: pd.DataFrame) -> float:
    """How many independent trials a correlated set of trials is worth.

    Uses the participation ratio of the correlation matrix's eigenvalues,
    (sum lambda)^2 / sum lambda^2: it equals N for N uncorrelated trials and 1 when every
    trial is a copy (or a sign-flipped copy) of the same return stream. The DSR paper
    asks for the number of *independent* trials; counting 118 near-duplicates as 118 would
    overstate how much searching was done.
    """
    eigenvalues = np.linalg.eigvalsh(returns.corr().to_numpy())
    return float(eigenvalues.sum() ** 2 / (eigenvalues**2).sum())


def sign_aligned(sharpes: pd.Series, families: pd.Series) -> pd.Series:
    """Flip the sign of every family whose median Sharpe ratio is negative.

    A researcher who sees a whole family lose money would trade it the other way, so the
    spread of Sharpe ratios the DSR should penalize is the spread of the sign-aligned
    family, not the gap between a family and its own mirror image.
    """
    median = sharpes.groupby(families).transform("median")
    return sharpes.where(median >= 0, -sharpes)


def bootstrap_sharpe_ci(
    returns,
    n_boot: int = 5000,
    block_size: int = 12,
    confidence: float = 0.95,
    seed: int = 0,
    periods_per_year: int = MONTHS_PER_YEAR,
) -> tuple[float, float]:
    """Percentile CI for the annualized Sharpe ratio via a circular block bootstrap.

    Blocks of ``block_size`` consecutive months keep short-range autocorrelation and
    volatility clustering that an i.i.d. bootstrap would destroy.
    """
    r = _clean(returns)
    n = len(r)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_size)) % n
    samples = r[idx.reshape(n_boot, -1)[:, :n]]
    srs = samples.mean(axis=1) / samples.std(axis=1, ddof=1) * np.sqrt(periods_per_year)
    tail = (1 - confidence) / 2 * 100
    lo, hi = np.percentile(srs, [tail, 100 - tail])
    return float(lo), float(hi)


def summarize(returns: pd.Series) -> dict[str, float]:
    """Headline statistics for one return series."""
    r = returns.dropna()
    return {
        "n_months": int(len(r)),
        "ann_return": annualized_return(r),
        "ann_vol": annualized_volatility(r),
        "sharpe": sharpe_ratio(r),
        "max_drawdown": max_drawdown(r),
    }
