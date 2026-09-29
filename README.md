# backtest-overfit

How much of a good backtest is luck? This repo backtests 118 parameterizations of four
textbook strategy families (time-series momentum, cross-sectional momentum, short-term
reversal, moving-average crossovers) on the 49 Kenneth French industry portfolios,
picks the best one, and then runs the standard overfitting diagnostics on that choice:
walk-forward selection, Combinatorially Symmetric Cross-Validation (Probability of
Backtest Overfitting), the Deflated Sharpe Ratio, a month-shuffling permutation null,
block-bootstrap confidence intervals, and Fama-French alphas with Newey-West t-stats.
The goal is a direct comparison between the Sharpe ratio a backtest reports and the one
you should expect to get.

Report page: https://adleraaa.github.io/backtest-overfit/

## Finding

The backtest winner is industry momentum (12-month lookback, top/bottom 20%, 12-month
volatility window), with a net Sharpe ratio of 0.68 over 1969-07 to 2026-08. The signal
is real: on month-shuffled data the best of the same 118 trials averaged 0.25 and never
went above 0.57 in 200 shuffles (p = 0.005), and PBO is only 0.02. The *size* of that
Sharpe ratio is still inflated by selection. The same selection rule, run walk-forward
(pick the trailing 10-year winner each year), earned 0.36 out of sample, compared with
0.57 for the hindsight winner over the same months. The winner's own Sharpe ratio also
fell from 0.90 in the first half of the sample to 0.46 in the second. The Deflated Sharpe
Ratio does not give a stable answer here. It ranges from 0.01 to 0.995 depending on which
variance of trial Sharpe ratios you plug in (see Design decisions).

## Results

All numbers come from `results/summary.json`, `results/trials.csv` and
`results/walk_forward_selections.csv`, written by `scripts/run_experiment.py`.
Returns are monthly and net of 10 bps per unit of turnover. Every trial is scaled to a
10% annualized volatility target. The evaluation window is 686 months (1969-07 to
2026-08), and data from 1963-07 onward is used as signal warm-up. Data vintage: "202608
CRSP database". The full run (200 permutations) took 621 s on an Intel
i9-14900HX laptop.

**Selected strategy vs. what to expect**

| Measure | Value |
|---|---|
| Selected trial | `xsmom(lookback=12,skip=0,quantile=0.2)\|vol12` |
| Net Sharpe, full sample (in-sample) | 0.68 (95% block-bootstrap CI 0.41 to 0.95) |
| Ann. return / vol / max drawdown | 8.1% / 11.9% / -34.7% |
| Sharpe 1969-07 to 1998-01 / 1998-02 to 2026-08 | 0.90 / 0.46 |
| Best-of-grid Sharpe on shuffled months: mean / 95th pct / max (200 shuffles) | 0.25 / 0.42 / 0.57 |
| Permutation p-value | 0.005 (= 1/201, the smallest possible with 200 shuffles) |
| PBO (CSCV, 16 blocks, 12,870 splits) | 0.023 |
| CSCV median Sharpe of the IS winner: IS / OOS | 0.66 / 0.54 |
| Walk-forward OOS Sharpe (10y train, 1y test, 566 months) | 0.36 (95% CI 0.05 to 0.68) |
| Hindsight winner over the same 566 months | 0.57 |
| Equal weight of all 118 trials, same months | 0.26 |
| Probabilistic Sharpe Ratio vs 0 | 1.000 |
| Deflated Sharpe, cross-trial Var[SR] (E[max SR] = 0.99) | 0.011 |
| Deflated Sharpe, no-signal Var[SR] from shuffles (E[max SR] = 0.33) | 0.995 |

**Alpha (OLS, Newey-West t-stats, 6 lags for the full sample and 5 for walk-forward)**

| Series | FF3 alpha / yr | t | FF5 alpha / yr | t |
|---|---|---|---|---|
| Selected (hindsight) | 9.6% | 5.66 | 9.5% | 5.17 |
| Walk-forward OOS | 4.4% | 2.46 | 3.8% | 2.04 |

**By family** (PBO computed within the family only)

| Family | Trials | Best Sharpe | Median Sharpe | PBO |
|---|---|---|---|---|
| Cross-sectional momentum | 48 | 0.68 | 0.38 | 0.02 |
| Time-series momentum | 14 | 0.39 | 0.21 | 0.32 |
| Moving-average crossover | 38 | 0.36 | 0.22 | 0.34 |
| Short-term reversal | 18 | -0.49 | -0.65 | 0.06 |

Other results:

- Transaction costs: the selected trial's Sharpe is 0.76 at 0 bps, 0.68 at 10 bps, 0.56
  at 25 bps and 0.37 at 50 bps.
- Deflated Sharpe with the paper's cross-trial variance, as a function of the number of
  trials N: 1.000 (N=1), 0.980 (4), 0.719 (10), 0.204 (30), 0.011 (118).
- Industry "reversal" loses money in every variant (Sharpe -0.49 to -0.88). At the
  industry level, the recent-return effect is continuation, the opposite of the one-month
  reversal known for individual stocks. This matches Moskowitz and Grinblatt (1999).
- Walk-forward picked 18 distinct trials across 48 windows: cross-sectional momentum 75%
  of the time, MA crossover 23%, time-series momentum 2%.

![Sharpe distribution](results/figures/sharpe_distribution.png)
![Permutation null](results/figures/permutation_null.png)
![Walk-forward vs hindsight](results/figures/cumulative.png)
![CSCV logits](results/figures/pbo_logits.png)
![IS vs OOS](results/figures/is_vs_oos.png)

The IS-vs-OOS scatter has a negative slope (-0.67). This is partly mechanical in CSCV:
the in-sample and out-of-sample halves are complements of one fixed sample, so a trial
that looks unusually good in one half must look worse in the other.

## Architecture

```
data.py        download + cache French library zips, parse the monthly table
   |
strategies.py  returns (T x N) -> target weights; row t uses data up to t only
   |
backtest.py    hold weights of t during t+1; vol targeting; costs on turnover
   |
grid.py        59 signal specs x 2 vol windows = 118 trials
   |
study.py       run grid -> trial return matrix -> select in-sample best, then:
   |             metrics.py  Sharpe, PSR, DSR, E[max SR], block bootstrap CI
   |             overfit.py  CSCV/PBO, walk-forward selection
   |             factors.py  FF3/FF5 alpha, Newey-West
   |             permutation null (re-run grid on month-shuffled data)
   |
plots.py / report.py   PNG figures, static HTML report for GitHub Pages
```

## Quickstart / Reproduce

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

pytest                             # 106 tests, synthetic data only, no network
ruff check . && ruff format --check .

python scripts/run_experiment.py   # downloads ~0.5 MB into data/raw/, writes results/
python -m btoverfit.report         # builds site/index.html from results/
```

Use `--permutations 20` for a faster run. This only changes the permutation-null rows and
the DSR computed with the null variance.

## Design decisions

- **Excess returns everywhere.** Industry returns minus the 1-month T-bill are used for
  both signals and P&L. Long-short and T-bill-funded positions are then on the same
  footing, and a Sharpe ratio is just mean/std.
- **Look-ahead is tested, not assumed.** `tests/test_no_lookahead.py` recomputes every
  grid signal on data truncated at month t and requires the same weights. A deliberately
  leaky signal is included to show that the check catches it. Another test uses an
  alternating return series, where a backtester without the one-month lag would show
  a perfect record.
- **Volatility targeting uses only past unscaled P&L.** Warm-up months are NaN, not
  zero. Zeros would shrink the volatility estimate and push every strategy to the
  leverage cap (5x gross).
- **CSCV from block sums.** Each of the 12,870 splits is a sum of per-block sums and
  sums of squares, so all splits cost two matrix products. A slow loop implementation in
  the tests checks it exactly.
- **Two Var[SR] inputs for the DSR.** The paper's estimator uses the spread of Sharpe
  ratios across trials. Here that spread mostly reflects real differences between
  families (reversal is strongly negative), which gives E[max SR] = 0.99. Using the
  no-signal sampling variance from shuffled data gives E[max SR] = 0.33. The empirical
  best-of-grid under the null averaged 0.25, below 0.33 because the trials are
  correlated. Both DSRs are reported rather than picking the one that flatters the
  result.
- **Month-shuffling null.** Permuting whole months keeps each month's cross-section
  (correlations, fat tails) and removes all time-series structure. The null therefore
  asks "could this grid find a Sharpe this high with no predictability at all?" and not
  whether the result is due to a known factor. The factor question is handled by the
  alpha regressions.

## Limitations

- Transaction costs are a flat bps charge on turnover. They ignore weight drift within
  the month, borrowing costs for shorts, and the fact that industry portfolios are not
  directly tradable (an ETF implementation would have tracking error and fees).
- The grid (118 trials) is small and was chosen by me. A real research process usually
  tries more variants, and those extra variants are never logged, which is exactly what
  PBO and DSR are meant to penalize. The reported DSR and PBO describe this grid only.
- The strategy families are well known and published. Selecting momentum in 2026 uses
  knowledge from after 1969, which no in-sample test can remove. The walk-forward result
  only removes the parameter-selection part of that look-ahead.
- The bootstrap CI assumes stationarity within 12-month blocks. The large drop between
  sample halves suggests the process is not stationary.
- The permutation p-value has a floor of 1/201 with 200 shuffles.
- The last walk-forward window is partial (2026-07 to 2026-08).

## Data

Kenneth R. French Data Library,
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
(49 Industry Portfolios, Fama/French 3 Factors, Fama/French 5 Factors (2x3); monthly,
value-weighted). The library does not publish a redistribution license, so no raw data
is committed. `src/btoverfit/data.py` downloads and caches the files, and only derived
statistics are committed under `results/`.

## References

- Bailey, D., Borwein, J., Lopez de Prado, M., Zhu, Q. (2015). The Probability of Backtest
  Overfitting. *Journal of Computational Finance* 20(4).
- Bailey, D., Lopez de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio
  Management* 40(5).
- Bailey, D., Lopez de Prado, M. (2012). The Sharpe Ratio Efficient Frontier. *Journal of
  Risk* 15(2).
- Moskowitz, T., Grinblatt, M. (1999). Do Industries Explain Momentum? *Journal of Finance*
  54(4).
- Newey, W., West, K. (1994). Automatic Lag Selection in Covariance Matrix Estimation.
  *Review of Economic Studies* 61(4).

## License

MIT, copyright 2026 Yunlong Lu. See [LICENSE](LICENSE).
