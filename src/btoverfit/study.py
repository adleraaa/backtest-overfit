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
    effective_number_of_trials,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sign_aligned,
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
    n_permutations: int = 200
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
    # A flat (zero-variance) trial gets NaN rather than +/-inf.
    return matrix.mean() / matrix.std(ddof=1).replace(0.0, np.nan)


def shuffle_eval_months(
    excess: pd.DataFrame, eval_start: str, rng: np.random.Generator
) -> pd.DataFrame:
    """Randomly reorder the months from ``eval_start`` on; leave the warm-up in place.

    Some industries start late (in the real data Hlth has no returns before 1969-07).
    Shuffling warm-up rows into the evaluation window would scatter those NaN months
    through it and knock the industry out of every lookback window that touches one, so
    the null would trade a different universe than the real run.
    """
    n_warmup = int((excess.index < pd.Period(eval_start, freq="M")).sum())
    n_eval = len(excess) - n_warmup
    order = np.concatenate([np.arange(n_warmup), n_warmup + rng.permutation(n_eval)])
    shuffled = excess.iloc[order]
    shuffled.index = excess.index
    return shuffled


def permutation_null(
    trials: list[Trial], excess: pd.DataFrame, config: StudyConfig
) -> pd.DataFrame:
    """Monthly Sharpe ratio of every trial on month-shuffled data (permutations x trials).

    Shuffling whole months keeps each month's cross-section (industry correlations, fat
    tails) but destroys every time-series pattern the four families try to exploit, so
    whatever the grid "finds" here is pure selection luck. The same seed gives the same
    permutations on every call, so nulls for different trial sets are paired.
    """
    rng = np.random.default_rng(config.seed)
    rows = []
    for _ in range(config.n_permutations):
        shuffled = shuffle_eval_months(excess, config.eval_start, rng)
        matrix, _ = run_trials(trials, shuffled, config.eval_start, config.cost_bps)
        rows.append(_monthly_sharpes(matrix))
    return pd.DataFrame(rows, columns=[t.name for t in trials])


def _stats_with_ci(returns: pd.Series, config: StudyConfig) -> dict:
    out = summarize(returns)
    lo, hi = bootstrap_sharpe_ci(returns, n_boot=config.n_boot, seed=config.seed)
    out["sharpe_ci95"] = [lo, hi]
    return out


def _null_summary(null: pd.DataFrame, observed_max: float) -> dict:
    """Distribution of the best-of-grid Sharpe on shuffled data and the permutation p-value."""
    null_max = null.max(axis=1).to_numpy() * ANNUALIZE
    return {
        "n_permutations": len(null_max),
        "n_trials": null.shape[1],
        "observed_max_sharpe": float(observed_max),
        "max_sharpe_mean": float(null_max.mean()),
        "max_sharpe_p95": float(np.percentile(null_max, 95)),
        "max_sharpe_max": float(null_max.max()),
        # +1 in numerator and denominator: the observed data counts as one permutation.
        "p_value": float((1 + (null_max >= observed_max).sum()) / (1 + len(null_max))),
    }


def _dsr_entry(best: pd.Series, n_trials: float, sr_var: float) -> dict:
    return {
        "n_trials": float(n_trials),
        "sharpe_std_ann": float(np.sqrt(sr_var) * ANNUALIZE),
        "expected_max_sharpe_ann": expected_max_sharpe(n_trials, sr_var) * ANNUALIZE,
        "dsr": deflated_sharpe_ratio(best, n_trials, sr_var),
    }


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

    # ---- the in-sample winner and how much its Sharpe should be deflated --------------
    sr_m = _monthly_sharpes(matrix)
    best_name = sr_m.idxmax()
    best = matrix[best_name]
    # The DSR benchmark E[max SR] needs Var[SR] across trials and the number of trials N.
    # Reported variants:
    #   raw:           Var of the raw trial Sharpes, N = number of trials (paper recipe).
    #   sign_aligned:  families with a negative median Sharpe are flipped first, so a
    #                  losing family is not counted as "spread" against its own mirror.
    #                  Headline: N = all trials treats correlated trials as independent,
    #                  which overstates N, so this DSR errs on the low side.
    #   sign_aligned_effective_n: as above, with N = effective number of independent
    #                  trials from the eigenvalues of the trial correlation matrix (errs
    #                  on the high side: one dominant eigenvalue makes N_eff small).
    #   null_variance: Var[SR] of each trial across shuffles, i.e. pure sampling noise.
    family_of = pd.Series({t.name: t.family for t in trials})
    sr_aligned = sign_aligned(sr_m, family_of[sr_m.index])
    n_eff = effective_number_of_trials(matrix)
    corr = matrix.corr().to_numpy()
    max_abs_corr = float(np.abs(corr[~np.eye(len(corr), dtype=bool)]).max())
    sr_var_aligned = float(sr_aligned.var(ddof=1))
    dsr = {
        "raw": _dsr_entry(best, n_trials, float(sr_m.var(ddof=1))),
        "sign_aligned": _dsr_entry(best, n_trials, sr_var_aligned),
        "sign_aligned_effective_n": _dsr_entry(best, n_eff, sr_var_aligned),
        "null_variance": _dsr_entry(best, n_trials, float(null.var(ddof=1).mean())),
    }
    dsr_by_n = {
        str(n): deflated_sharpe_ratio(best, n, sr_var_aligned)
        for n in sorted({1, 4, 10, 30, n_trials})
        if n <= n_trials
    }
    selected = {
        "trial": best_name,
        **_stats_with_ci(best, config),
        "psr_vs_zero": probabilistic_sharpe_ratio(best),
        "n_trials": n_trials,
        "effective_n_trials": n_eff,
        "max_abs_corr_between_trials": max_abs_corr,
        "flipped_families": sorted(set(family_of[sr_aligned.index[sr_aligned != sr_m]])),
        "deflated_sharpe": dsr["sign_aligned"]["dsr"],
        "dsr": dsr,
        "deflated_sharpe_by_n_trials_aligned_var": dsr_by_n,
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
        # OLS slope of OOS on IS Sharpe of each split's IS winner (the is_vs_oos figure).
        "is_oos_slope": float(
            np.polyfit(pbo.is_best_perf * ANNUALIZE, pbo.oos_best_perf * ANNUALIZE, 1)[0]
        ),
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

    null_summary = _null_summary(null, selected["sharpe"])

    # ---- ablation: the same signals without volatility targeting ----------------------
    # Month shuffling also destroys volatility clustering, which the trailing-vol scaling
    # exploits on real data, so the null above rejects "no predictability and no
    # vol-timing benefit" jointly. Re-running signals and null at constant gross exposure
    # isolates the signal. Without scaling the vol window is irrelevant: each signal runs once.
    signals = build_grid(vol_windows=(None,))
    novol, _ = run_trials(signals, excess, config.eval_start, config.cost_bps)
    novol_sr = _monthly_sharpes(novol) * ANNUALIZE
    chosen = by_name[best_name]
    chosen_novol = Trial(chosen.family, chosen.params, None).name
    vol_ablation = {
        "selected_signal": chosen_novol,
        "selected_signal_sharpe": float(novol_sr[chosen_novol]),
        "best_trial": novol_sr.idxmax(),
        "permutation_null": _null_summary(
            permutation_null(signals, excess, config), float(novol_sr.max())
        ),
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
        "no_vol_target": vol_ablation,
    }
    null_max = null.max(axis=1).to_numpy() * ANNUALIZE
    return StudyOutput(summary, table, matrix, pbo, wf, null_max)
