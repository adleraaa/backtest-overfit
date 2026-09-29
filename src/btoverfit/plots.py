"""Static matplotlib figures for the README and the Pages report (Agg backend, no GUI)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .study import ANNUALIZE, StudyOutput  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, INK_2 = "#9a9993", "#0b0b0b", "#52514e"

plt.rcParams.update(
    {
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#d4d3cd",
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#ebeae6",
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 9,
        "legend.frameon": False,
        "savefig.dpi": 150,
    }
)


def _vline(ax, x, color, label, y=0.95):
    ax.axvline(x, color=color, linewidth=2)
    ax.annotate(
        label,
        (x, y),
        xycoords=("data", "axes fraction"),
        xytext=(4, 0),
        textcoords="offset points",
        color=INK_2,
        va="top",
        fontsize=8,
    )


def sharpe_distribution(out: StudyOutput, path: Path) -> None:
    s = out.summary
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(out.trials["sharpe"], bins=30, color=BLUE, edgecolor="#fcfcfb", linewidth=1)
    null_p95 = s["permutation_null"]["max_sharpe_p95"]
    _vline(ax, null_p95, GRAY, "95th pct of best-of-grid\non shuffled data", 0.95)
    _vline(ax, s["selected"]["sharpe"], ORANGE, "selected\n(in-sample best)", 0.72)
    ax.set_title(f"Net Sharpe ratios of all {s['sample']['n_trials']} trials")
    ax.set_xlabel("annualized Sharpe ratio, net of costs")
    ax.set_ylabel("number of trials")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def pbo_logits(out: StudyOutput, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.4))
    logits = out.pbo.logits
    finite = logits[np.isfinite(logits)]
    ax.hist(finite, bins=40, color=BLUE, edgecolor="#fcfcfb", linewidth=1)
    ax.axvline(0, color=INK_2, linewidth=1.2, linestyle="--")
    ax.set_title(
        f"CSCV: OOS relative rank of the IS winner  (PBO = {out.pbo.pbo:.2f}, "
        f"{out.pbo.n_splits:,} splits)"
    )
    ax.set_xlabel("logit of OOS rank  (< 0: IS winner is below the OOS median)")
    ax.set_ylabel("number of splits")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def is_vs_oos(out: StudyOutput, path: Path) -> None:
    is_sr = out.pbo.is_best_perf * ANNUALIZE
    oos_sr = out.pbo.oos_best_perf * ANNUALIZE
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(is_sr, oos_sr, s=8, color=BLUE, alpha=0.15, linewidths=0)
    slope, intercept = np.polyfit(is_sr, oos_sr, 1)
    xs = np.linspace(is_sr.min(), is_sr.max(), 50)
    ax.plot(xs, intercept + slope * xs, color=ORANGE, linewidth=2, label=f"OLS slope {slope:.2f}")
    lim = [min(is_sr.min(), oos_sr.min()), max(is_sr.max(), oos_sr.max())]
    ax.plot(lim, lim, color=GRAY, linewidth=1, linestyle="--", label="no degradation (y = x)")
    ax.set_title("Selected trial: in-sample vs out-of-sample Sharpe")
    ax.set_xlabel("IS Sharpe of the IS winner (annualized)")
    ax.set_ylabel("OOS Sharpe of same trial")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cumulative(out: StudyOutput, path: Path) -> None:
    idx = out.wf.oos_returns.index
    series = {
        "hindsight: full-sample best trial": (out.matrix[out.summary["selected"]["trial"]], ORANGE),
        "walk-forward: trailing-10y winner": (out.wf.oos_returns, BLUE),
        "equal weight of all trials": (out.matrix.loc[idx].mean(axis=1), AQUA),
    }
    fig, ax = plt.subplots(figsize=(7, 3.8))
    dates = idx.to_timestamp()
    for label, (r, color) in series.items():
        wealth = np.cumprod(1 + r.loc[idx].to_numpy())
        ax.plot(dates, wealth, color=color, linewidth=2, label=label)
    ax.set_yscale("log")
    ax.set_title("Growth of $1 over the walk-forward test months (net, log scale)")
    ax.set_ylabel("wealth")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def permutation_hist(out: StudyOutput, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(out.null_max_sharpes, bins=25, color=GRAY, edgecolor="#fcfcfb", linewidth=1)
    s = out.summary
    _vline(ax, s["selected"]["sharpe"], ORANGE, "observed best", 0.95)
    ax.set_title(
        f"Best-of-{s['sample']['n_trials']} Sharpe on month-shuffled data "
        f"({s['permutation_null']['n_permutations']} shuffles)"
    )
    ax.set_xlabel("annualized Sharpe of the best trial")
    ax.set_ylabel("number of shuffles")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


FIGURES = {
    "sharpe_distribution.png": sharpe_distribution,
    "pbo_logits.png": pbo_logits,
    "is_vs_oos.png": is_vs_oos,
    "cumulative.png": cumulative,
    "permutation_null.png": permutation_hist,
}


def make_figures(out: StudyOutput, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fn in FIGURES.items():
        fn(out, out_dir / name)
        paths.append(out_dir / name)
    return paths
