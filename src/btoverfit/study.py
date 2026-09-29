"""The full study: run the grid, pick the winner, and ask how much of it is luck.

Kept separate from the command-line script so a synthetic-data smoke test can exercise
the whole pipeline in CI without downloading anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import DEFAULT_VOL_TARGET
from .factors import factor_alpha
from .grid import Trial, build_grid
from .metrics import (
    bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    summarize,
)
from .overfit import PBOResult, WalkForwardResult, cscv_pbo, walk_forward

ANNUALIZE = np.sqrt(12)


@dataclass(frozen=True)
class StudyConfig:
    eval_start: str
    cost_bps: float = 10.0
    n_blocks: int = 16
    wf_train: int = 120
    wf_test: int = 12
    n_permutations: int = 100
    n_boot: int = 5000
    seed: int = 0
    cost_grid: tuple[float, ...] = (0.0, 10.0, 25.0, 50.0)


@dataclass
class StudyOutput:
    summary: dict
    trials: pd.DataFrame  # one row of statistics per trial
    matrix: pd.DataFrame  # months x trials net returns on the evaluation window
    pbo: PBOResult
    wf: WalkForwardResult
    null_max_sharpes: np.ndarray


def run_trials(
    trials: list[Trial], excess: pd.DataFrame, eval_start: str, cost_bps: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest every trial; return (net-return matrix on the eval window, turnover)."""
    nets, turnovers = {}, {}
    for trial in trials:
        result = trial.run(excess, cost_bps=cost_bps)
        nets[trial.name] = result.returns
        turnovers[trial.name] = result.turnover
    matrix = pd.DataFrame(nets).loc[eval_start:]
    if matrix.isna().to_numpy().any():
        raise ValueError("some trial is still warming up after eval_start; start later")
    return matrix, pd.DataFrame(turnovers).loc[eval_start:]


def _monthly_sharpes(matrix: pd.DataFrame) -> pd.Series:
    return matrix.mean() / matrix.std(ddof=1)


def permutation_null(
    trials: list[Trial], excess: pd.DataFrame, config: StudyConfig
) -> pd.DataFrame:
    """Monthly Sharpe ratio of every trial on month-shuffled data (permutations x trials).

    Shuffling whole months keeps each month's cross-section (industry correlations, fat
    tails) but destroys every time-series pattern the four families try to exploit, so
    whatever the grid "finds" here is pure selection luck.
    """
    rng = np.random.default_rng(config.seed)
    rows = []
    for _ in range(config.n_permutations):
        shuffled = excess.iloc[rng.permutation(len(excess))]
        shuffled.index = excess.index
        matrix, _ = run_trials(trials, shuffled, config.eval_start, config.cost_bps)
        rows.append(_monthly_sharpes(matrix))
    return pd.DataFrame(rows, columns=[t.name for t in trials])


def _stats_with_ci(returns: pd.Series, config: StudyConfig) -> dict:
    out = summarize(returns)
    lo, hi = bootstrap_sharpe_ci(returns, n_boot=config.n_boot, seed=config.seed)
    out["sharpe_ci95"] = [lo, hi]
    return out


def _alphas(returns: pd.Series, ff3: pd.DataFrame, ff5: pd.DataFrame) -> dict:
    return {
        "ff3": factor_alpha(returns, ff3.drop(columns="RF")),
        "ff5": factor_alpha(returns, ff5.drop(columns="RF")),
    }


def run_study(
    excess: pd.DataFrame, ff3: pd.DataFrame, ff5: pd.DataFrame, config: StudyConfig
) -> StudyOutput:
    if config.n_permutations < 2:
        raise ValueError("need at least 2 permutations to estimate the null variance")
    trials = build_grid()
    by_name = {t.name: t for t in trials}
    matrix, turnover = run_trials(trials, excess, config.eval_start, config.cost_bps)
    n_trials = matrix.shape[1]

    # ---- per-trial table -------------------------------------------------------------
    rows = []
    for t in trials:
        row = {"trial": t.name, "family": t.family, "vol_window": t.vol_window, **t.params}
        row.update(summarize(matrix[t.name]))
        row["ann_turnover"] = float(turnover[t.name].mean() * 12)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("sharpe", ascending=False, ignore_index=True)

    # ---- permutation null: what does the best of this grid look like with no signal? --
    null = permutation_null(trials, excess, config)
    null_max = null.max(axis=1).to_numpy() * ANNUALIZE

    # ---- the in-sample winner and how much its Sharpe should be deflated --------------
    sr_m = _monthly_sharpes(matrix)
    best_name = sr_m.idxmax()
    best = matrix[best_name]
    # Two choices for Var[SR] in the DSR benchmark: the spread of Sharpe ratios across our
    # trials (the paper's estimator; inflated here because families genuinely differ), and
    # the sampling variance of a trial's Sharpe when there is no signal (from the null).
    sr_var_trials = float(sr_m.var(ddof=1))
    sr_var_null = float(null.var(ddof=1).mean())
    dsr_by_n = {
        str(n): deflated_sharpe_ratio(best, n, sr_var_trials)
        for n in sorted({1, 4, 10, 30, n_trials})
        if n <= n_trials
    }
    selected = {
        "trial": best_name,
        **_stats_with_ci(best, config),
        "psr_vs_zero": probabilistic_sharpe_ratio(best),
        "n_trials": n_trials,
        "trial_sharpe_std_ann": float(np.sqrt(sr_var_trials) * ANNUALIZE),
        "expected_max_sharpe_ann": expected_max_sharpe(n_trials, sr_var_trials) * ANNUALIZE,
        "deflated_sharpe": dsr_by_n[str(n_trials)],
        "deflated_sharpe_by_n_trials": dsr_by_n,
        "null_sharpe_std_ann": float(np.sqrt(sr_var_null) * ANNUALIZE),
        "expected_max_sharpe_null_var_ann": expected_max_sharpe(n_trials, sr_var_null) * ANNUALIZE,
        "deflated_sharpe_null_var": deflated_sharpe_ratio(best, n_trials, sr_var_null),
        "alpha": _alphas(best, ff3, ff5),
    }
    half = len(best) // 2
    selected["sharpe_first_half"] = summarize(best.iloc[:half])["sharpe"]
    selected["sharpe_second_half"] = summarize(best.iloc[half:])["sharpe"]
    selected["halves_split_month"] = str(best.index[half])
    cost_sensitivity = {}
    for bps in config.cost_grid:
        rerun = by_name[best_name].run(excess, cost_bps=bps).returns.loc[config.eval_start :]
        cost_sensitivity[str(bps)] = summarize(rerun)["sharpe"]
    selected["sharpe_by_cost_bps"] = cost_sensitivity

    # ---- CSCV / PBO ------------------------------------------------------------------
    pbo = cscv_pbo(matrix, n_blocks=config.n_blocks)
    pbo_summary = {
        "pbo": pbo.pbo,
        "prob_oos_loss": pbo.prob_oos_loss,
        "n_splits": pbo.n_splits,
        "n_blocks": config.n_blocks,
        "rows_used": pbo.rows_used,
        "median_is_sharpe_ann": float(np.median(pbo.is_best_perf) * ANNUALIZE),
        "median_oos_sharpe_ann": float(np.median(pbo.oos_best_perf) * ANNUALIZE),
        "median_logit": float(np.median(pbo.logits)),
    }

    # ---- walk-forward ----------------------------------------------------------------
    wf = walk_forward(matrix, train=config.wf_train, test=config.wf_test)
    wf_index = wf.oos_returns.index
    chosen_family = wf.selections["trial"].map(lambda n: by_name[n].family)
    wf_summary = {
        "train_months": config.wf_train,
        "test_months": config.wf_test,
        "n_windows": len(wf.selections),
        "distinct_trials_selected": int(wf.selections["trial"].nunique()),
        "family_share": chosen_family.value_counts(normalize=True).round(4).to_dict(),
        "oos": _stats_with_ci(wf.oos_returns, config),
        "hindsight_best_same_months": summarize(best.loc[wf_index]),
        "equal_weight_all_trials_same_months": summarize(matrix.loc[wf_index].mean(axis=1)),
        "median_trial_sharpe_same_months": float(
            _monthly_sharpes(matrix.loc[wf_index]).median() * ANNUALIZE
        ),
        "alpha": _alphas(wf.oos_returns, ff3, ff5),
    }

    # ---- per family ------------------------------------------------------------------
    families = {}
    for family, group in table.groupby("family", sort=False):
        sub = matrix[group["trial"]]
        families[family] = {
            "n_trials": len(group),
            "best_trial": group["trial"].iloc[0],
            "best_sharpe": float(group["sharpe"].iloc[0]),
            "median_sharpe": float(group["sharpe"].median()),
            "pbo": cscv_pbo(sub, n_blocks=config.n_blocks).pbo,
        }

    observed = selected["sharpe"]
    null_summary = {
        "n_permutations": config.n_permutations,
        "max_sharpe_mean": float(null_max.mean()),
        "max_sharpe_p95": float(np.percentile(null_max, 95)),
        "max_sharpe_max": float(null_max.max()),
        # +1 in numerator and denominator: the observed data counts as one permutation.
        "p_value": float((1 + (null_max >= observed).sum()) / (1 + len(null_max))),
    }

    summary = {
        "sample": {
            "eval_start": str(matrix.index[0]),
            "eval_end": str(matrix.index[-1]),
            "n_months": len(matrix),
            "n_assets": excess.shape[1],
            "n_trials": n_trials,
            "cost_bps": config.cost_bps,
            "vol_target": DEFAULT_VOL_TARGET,
        },
        "selected": selected,
        "pbo": pbo_summary,
        "walk_forward": wf_summary,
        "families": families,
        "permutation_null": null_summary,
    }
    return StudyOutput(summary, table, matrix, pbo, wf, null_max)
