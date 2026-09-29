"""Run the full study on Kenneth French data and write everything under results/.

Usage:
    python scripts/run_experiment.py                  # downloads data on first run
    python scripts/run_experiment.py --permutations 20  # quicker, noisier null
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

from btoverfit import data
from btoverfit.plots import make_figures
from btoverfit.study import StudyConfig, run_study

DATA_START = "1963-07"  # first month of the Fama-French 5-factor file; earlier data = warm-up
EVAL_START = "1969-07"  # first month in which all 49 industries have returns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=data.DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--permutations", type=int, default=100)
    args = parser.parse_args()

    industries, vintage = data.load("industries49", args.cache)
    ff3, _ = data.load("ff3", args.cache)
    ff5, _ = data.load("ff5", args.cache)

    # Industry returns in excess of the 1-month T-bill: every strategy is then either
    # dollar-neutral or T-bill funded, and all P&L is an excess return.
    months = industries.index.intersection(ff3.index)
    excess = industries.loc[months].sub(ff3.loc[months, "RF"], axis=0).loc[DATA_START:]

    config = StudyConfig(eval_start=EVAL_START, n_permutations=args.permutations)
    started = time.perf_counter()
    out = run_study(excess, ff3, ff5, config)
    elapsed = time.perf_counter() - started

    out.summary["meta"] = {
        "data_source": "Kenneth R. French Data Library, "
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
        "datasets": list(data.DATASETS.values()),
        "data_vintage": vintage,
        "data_start_incl_warmup": DATA_START,
        "runtime_seconds": round(elapsed, 1),
        "hardware": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("numpy", "pandas", "scipy", "statsmodels")},
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(out.summary, indent=2) + "\n")
    out.trials.to_csv(args.out / "trials.csv", index=False, float_format="%.6f")
    out.wf.selections.to_csv(args.out / "walk_forward_selections.csv", index=False)
    make_figures(out, args.out / "figures")

    s = out.summary
    print(f"trials: {s['sample']['n_trials']}, months: {s['sample']['n_months']}")
    print(f"selected: {s['selected']['trial']}  Sharpe {s['selected']['sharpe']:.2f}")
    print(f"DSR {s['selected']['deflated_sharpe']:.3f}  PBO {s['pbo']['pbo']:.3f}")
    print(f"walk-forward OOS Sharpe {s['walk_forward']['oos']['sharpe']:.2f}")
    print(f"done in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
