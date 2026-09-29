"""End-to-end smoke test of the study pipeline on synthetic data (no network)."""

import json

import numpy as np
import pandas as pd
import pytest

from btoverfit.plots import make_figures
from btoverfit.study import StudyConfig, run_study


@pytest.fixture(scope="module")
def output():
    rng = np.random.default_rng(0)
    index = pd.period_range("1990-01", periods=200, freq="M")
    excess = pd.DataFrame(rng.normal(0.004, 0.05, (200, 6)), index=index, columns=list("abcdef"))
    ff5 = pd.DataFrame(
        rng.normal(0.0, 0.03, (200, 6)),
        index=index,
        columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
    )
    ff3 = ff5[["Mkt-RF", "SMB", "HML", "RF"]]
    config = StudyConfig(
        eval_start="1996-01", n_blocks=4, wf_train=36, wf_test=12, n_permutations=3, n_boot=200
    )
    return run_study(excess, ff3, ff5, config)


def test_summary_is_complete_and_json_safe(output):
    s = output.summary
    assert s["sample"]["n_months"] == 200 - 72
    assert s["sample"]["n_trials"] == len(output.trials) == output.matrix.shape[1]
    assert set(s["families"]) == {"tsmom", "xsmom", "reversal", "ma_cross"}
    json.dumps(s, allow_nan=False)  # raises on NaN/inf


def test_selected_is_the_in_sample_best(output):
    s = output.summary
    # Compare Sharpe values, not names: with 6 assets some grid points are identical
    # (e.g. quantile 0.1 and 0.3 both pick one name per leg), so the top can be a tie.
    chosen = output.trials.set_index("trial").loc[s["selected"]["trial"], "sharpe"]
    assert chosen == pytest.approx(output.trials["sharpe"].max())
    assert s["selected"]["sharpe"] == pytest.approx(chosen)
    lo, hi = s["selected"]["sharpe_ci95"]
    assert lo < s["selected"]["sharpe"] < hi
    assert 0 <= s["selected"]["deflated_sharpe"] <= s["selected"]["psr_vs_zero"] <= 1


def test_walk_forward_and_pbo_ranges(output):
    s = output.summary
    assert 0 <= s["pbo"]["pbo"] <= 1
    assert s["pbo"]["n_splits"] == 6  # C(4, 2)
    assert s["walk_forward"]["oos"]["n_months"] == output.matrix.shape[0] - 36
    assert s["permutation_null"]["n_permutations"] == 3
    assert 0 < s["permutation_null"]["p_value"] <= 1


def test_figures_are_written(output, tmp_path):
    paths = make_figures(output, tmp_path)
    assert len(paths) == 5
    assert all(p.stat().st_size > 1000 for p in paths)


def test_rejects_too_few_permutations():
    with pytest.raises(ValueError):
        run_study(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), StudyConfig("2000-01", n_permutations=1)
        )
