"""Factor-model alpha with Newey-West (HAC) standard errors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def newey_west_lags(n_obs: int) -> int:
    """Newey & West (1994) plug-in rule: floor(4 * (T / 100) ** (2 / 9))."""
    return int(np.floor(4 * (n_obs / 100) ** (2 / 9)))


def factor_alpha(returns: pd.Series, factors: pd.DataFrame, lags: int | None = None) -> dict:
    """Regress strategy excess returns on factor returns; report alpha and HAC t-stats.

    ``factors`` should contain only factor columns (drop RF). Strategy returns are already
    excess returns (long-short or T-bill funded), so no RF adjustment is applied here.
    """
    data = pd.concat([returns.rename("strategy"), factors], axis=1, join="inner").dropna()
    y = data["strategy"]
    x = sm.add_constant(data.drop(columns="strategy"))
    lags = newey_west_lags(len(data)) if lags is None else lags
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {
        "alpha_monthly": float(fit.params["const"]),
        "alpha_annual": float(fit.params["const"] * 12),
        "alpha_t": float(fit.tvalues["const"]),
        "betas": {k: float(v) for k, v in fit.params.drop("const").items()},
        "beta_t": {k: float(v) for k, v in fit.tvalues.drop("const").items()},
        "r_squared": float(fit.rsquared),
        "n_obs": int(fit.nobs),
        "nw_lags": lags,
    }
