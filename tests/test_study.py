"""End-to-end smoke test of the study pipeline on synthetic data (no network)."""

import json

import numpy as np
import pandas as pd
import pytest

from btoverfit.plots import make_figures
from btoverfit.report import build
from btoverfit.study import StudyConfig, run_study, shuffle_eval_months


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


def test_dsr_variants_and_vol_ablation_are_reported(output):
    s = output.summary
    sel = s["selected"]
    assert set(sel["dsr"]) == {"raw", "sign_aligned", "sign_aligned_effective_n", "null_variance"}
    assert 1 <= sel["effective_n_trials"] <= sel["n_trials"]
    assert sel["deflated_sharpe"] == sel["dsr"]["sign_aligned"]["dsr"]
    # Fewer effective trials means a lower bar, so the DSR can only go up.
    assert sel["dsr"]["sign_aligned_effective_n"]["dsr"] >= sel["dsr"]["sign_aligned"]["dsr"]
    novol = s["no_vol_target"]
    assert novol["selected_signal"].endswith("|novol")
    assert novol["permutation_null"]["n_trials"] == s["sample"]["n_trials"] // 2
    assert "is_oos_slope" in s["pbo"]


def test_shuffle_keeps_warmup_in_place_and_permutes_eval_months():
    index = pd.period_range("2000-01", periods=30, freq="M")
    excess = pd.DataFrame({"a": np.arange(30.0), "b": np.arange(30.0) + 100}, index=index)
    excess.iloc[:6, 0] = np.nan  # an industry that starts during the warm-up
    shuffled = shuffle_eval_months(excess, "2000-11", np.random.default_rng(0))
    pd.testing.assert_frame_equal(shuffled.iloc[:10], excess.iloc[:10])
    assert not shuffled.iloc[10:].isna().to_numpy().any()  # no NaN moved into the window
    assert sorted(shuffled["a"].iloc[10:]) == list(range(10, 30))
    assert not shuffled["a"].iloc[10:].equals(excess["a"].iloc[10:])  # actually shuffled
    # Whole months move together: each row keeps its cross-section.
    assert (shuffled["b"] - shuffled["a"]).iloc[10:].eq(100).all()


def test_figures_are_written(output, tmp_path):
    paths = make_figures(output, tmp_path)
    assert len(paths) == 5
    assert all(p.stat().st_size > 1000 for p in paths)


def test_rejects_too_few_permutations():
    with pytest.raises(ValueError):
        run_study(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), StudyConfig("2000-01", n_permutations=1)
        )


def test_report_builds_from_results(output, tmp_path):
    results = tmp_path / "results"
    summary = dict(output.summary)
    summary["meta"] = {"data_source": "synthetic", "data_vintage": "test"}
    results.mkdir()
    (results / "summary.json").write_text(json.dumps(summary))
    output.trials.to_csv(results / "trials.csv", index=False)
    make_figures(output, results / "figures")

    index = build(results, tmp_path / "site")
    html = index.read_text(encoding="utf-8")
    assert f"{output.summary['selected']['sharpe']:.2f}" in html
    assert output.summary["selected"]["trial"] in html
    assert (tmp_path / "site" / "figures" / "pbo_logits.png").exists()
