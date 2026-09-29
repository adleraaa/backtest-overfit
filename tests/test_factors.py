import numpy as np
import pandas as pd
import pytest

from btoverfit.factors import factor_alpha, newey_west_lags


def test_newey_west_lag_rule():
    assert newey_west_lags(100) == 4
    assert newey_west_lags(700) == 6


def test_factor_alpha_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 600
    index = pd.period_range("1970-01", periods=n, freq="M")
    factors = pd.DataFrame(rng.normal(0, 0.04, (n, 2)), index=index, columns=["Mkt-RF", "HML"])
    y = 0.004 + 0.8 * factors["Mkt-RF"] - 0.3 * factors["HML"] + rng.normal(0, 0.005, n)
    out = factor_alpha(pd.Series(y, index=index), factors)
    assert out["alpha_monthly"] == pytest.approx(0.004, abs=0.0005)
    assert out["alpha_annual"] == pytest.approx(12 * out["alpha_monthly"])
    assert out["betas"]["Mkt-RF"] == pytest.approx(0.8, abs=0.02)
    assert out["betas"]["HML"] == pytest.approx(-0.3, abs=0.02)
    assert out["alpha_t"] > 10
    assert out["n_obs"] == n


def test_factor_alpha_insignificant_for_pure_beta():
    rng = np.random.default_rng(1)
    n = 600
    index = pd.period_range("1970-01", periods=n, freq="M")
    mkt = pd.DataFrame({"Mkt-RF": rng.normal(0.005, 0.045, n)}, index=index)
    y = pd.Series(1.2 * mkt["Mkt-RF"] + rng.normal(0, 0.01, n), index=index)
    assert abs(factor_alpha(y, mkt)["alpha_t"]) < 2.5
