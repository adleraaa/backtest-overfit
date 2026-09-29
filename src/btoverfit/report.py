"""Build the static GitHub Pages report from committed results (stdlib only, no network).

Usage: python -m btoverfit.report [--results results] [--out site]
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from html import escape
from pathlib import Path

REPO_URL = "https://github.com/adleraaa/backtest-overfit"


def f2(x: float) -> str:
    return f"{x:.2f}"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def table(headers: list[str], rows: list[list[str]], numeric_from: int = 1) -> str:
    head = "".join(
        f'<th class="{"num" if i >= numeric_from else ""}">{escape(h)}</th>'
        for i, h in enumerate(headers)
    )
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="{"num" if i >= numeric_from else ""}">{escape(c)}</td>'
            for i, c in enumerate(r)
        )
        + "</tr>"
        for r in rows
    )
    return (
        f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def stat(label: str, value: str, note: str) -> str:
    return (
        f'<div class="stat"><div class="stat-label">{escape(label)}</div>'
        f'<div class="stat-value">{escape(value)}</div>'
        f'<div class="stat-note">{escape(note)}</div></div>'
    )


CAPTIONS = {
    "sharpe_distribution.png": "Distribution of net Sharpe ratios across the grid.",
    "permutation_null.png": "Best-of-grid Sharpe on month-shuffled data vs the observed best.",
    "pbo_logits.png": "Logit of the out-of-sample rank of the in-sample winner across CSCV splits.",
    "is_vs_oos.png": "In-sample vs out-of-sample Sharpe of the selected trial across CSCV splits.",
    "cumulative.png": "Growth of $1 over the walk-forward test months.",
}


def figure(src: str) -> str:
    caption = CAPTIONS[src]
    return (
        f'<figure><img src="figures/{src}" alt="{escape(caption)}" loading="lazy">'
        f"<figcaption>{escape(caption)}</figcaption></figure>"
    )


CSS = """
:root{--bg:#fcfcfb;--surface:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#7a7974;
--line:#e4e3de;--accent:#2a78d6;--code:#f3f2ee}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#141413;
--surface:#1d1d1c;--ink:#f5f5f2;--ink2:#c3c2b7;--muted:#98978f;--line:#34332f;
--accent:#6aa5ef;--code:#262624}}
:root[data-theme="dark"]{--bg:#141413;--surface:#1d1d1c;--ink:#f5f5f2;--ink2:#c3c2b7;
--muted:#98978f;--line:#34332f;--accent:#6aa5ef;--code:#262624}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 "IBM Plex Sans",
system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:920px;margin:0 auto;padding:40px 16px 64px}
h1{font-size:1.9rem;line-height:1.2;margin:0 0 8px;letter-spacing:-0.01em}
h2{font-size:1.25rem;margin:48px 0 12px;padding-top:16px;border-top:1px solid var(--line)}
p{color:var(--ink2);margin:0 0 12px}
a{color:var(--accent)}
.lede{font-size:1.08rem;max-width:68ch}
.meta{color:var(--muted);font-size:.85rem}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;
margin:28px 0}
.stat{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.stat-label{font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.stat-value{font:600 1.7rem/1.2 "IBM Plex Mono",ui-monospace,monospace;margin:4px 0}
.stat-note{font-size:.82rem;color:var(--ink2)}
figure{margin:20px 0;background:#fcfcfb;border:1px solid var(--line);border-radius:10px;
padding:10px}
figure img{width:100%;height:auto;display:block}
figcaption{color:#52514e;font-size:.85rem;padding:6px 4px 0}
.table-wrap{overflow-x:auto;margin:12px 0 20px;border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:.88rem;background:var(--surface)}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--muted);font-weight:600;font-size:.78rem;text-transform:uppercase;
letter-spacing:.03em}
tr:last-child td{border-bottom:0}
td.num,th.num{text-align:right;font-family:"IBM Plex Mono",ui-monospace,monospace}
code{background:var(--code);padding:1px 5px;border-radius:4px;font-size:.88em;
overflow-wrap:anywhere}
ul{color:var(--ink2);padding-left:20px;overflow-wrap:anywhere}
"""


def build(results: Path, out: Path) -> Path:
    s = json.loads((results / "summary.json").read_text())
    with (results / "trials.csv").open() as fh:
        trials = list(csv.DictReader(fh))

    sel, pbo, wf, null = s["selected"], s["pbo"], s["walk_forward"], s["permutation_null"]
    sample, meta = s["sample"], s["meta"]

    stats_html = "".join(
        [
            stat("Selected, in-sample", f2(sel["sharpe"]), "annualized net Sharpe"),
            stat("Walk-forward OOS", f2(wf["oos"]["sharpe"]), "same selection rule, live"),
            stat("PBO (CSCV)", f2(pbo["pbo"]), f"{pbo['n_splits']:,} splits"),
            stat(
                "Deflated Sharpe",
                f"{sel['deflated_sharpe']:.3f} / {sel['deflated_sharpe_null_var']:.3f}",
                "cross-trial Var[SR] / null Var[SR]",
            ),
            stat(
                "Permutation p-value",
                f"{null['p_value']:.3f}",
                f"{null['n_permutations']} shuffles",
            ),
        ]
    )

    family_rows = [
        [name, str(f["n_trials"]), f2(f["best_sharpe"]), f2(f["median_sharpe"]), f2(f["pbo"])]
        for name, f in s["families"].items()
    ]
    top_rows = [
        [
            t["trial"],
            f2(float(t["sharpe"])),
            pct(float(t["ann_return"])),
            pct(float(t["ann_vol"])),
            pct(float(t["max_drawdown"])),
            f"{float(t['ann_turnover']):.1f}",
        ]
        for t in trials[:10]
    ]
    alpha_rows = []
    for label, block in (("Selected (hindsight)", sel["alpha"]), ("Walk-forward OOS", wf["alpha"])):
        for model in ("ff3", "ff5"):
            a = block[model]
            alpha_rows.append(
                [label, model.upper(), pct(a["alpha_annual"]), f2(a["alpha_t"]), str(a["nw_lags"])]
            )
    comparison_rows = [
        ["Walk-forward OOS", f2(wf["oos"]["sharpe"]), pct(wf["oos"]["ann_return"])],
        [
            "Hindsight best, same months",
            f2(wf["hindsight_best_same_months"]["sharpe"]),
            pct(wf["hindsight_best_same_months"]["ann_return"]),
        ],
        [
            "Equal weight of all trials",
            f2(wf["equal_weight_all_trials_same_months"]["sharpe"]),
            pct(wf["equal_weight_all_trials_same_months"]["ann_return"]),
        ],
    ]
    cost_rows = [[f"{float(k):.0f} bps", f2(v)] for k, v in sel["sharpe_by_cost_bps"].items()]
    dsr_rows = [[n, f"{v:.3f}"] for n, v in sel["deflated_sharpe_by_n_trials"].items()]
    lo, hi = sel["sharpe_ci95"]
    wlo, whi = wf["oos"]["sharpe_ci95"]

    body = f"""
<main>
<p class="meta">backtest-overfit &middot; <a href="{REPO_URL}">source on GitHub</a></p>
<h1>How much of a good backtest is luck?</h1>
<p class="lede">{sample["n_trials"]} parameterizations of four textbook strategy families were
backtested on the 49 Kenneth French industry portfolios ({sample["eval_start"]} to
{sample["eval_end"]}, {sample["n_months"]} months, {sample["cost_bps"]:.0f} bps costs,
{pct(sample["vol_target"])} volatility target). The best one is then put through the standard
overfitting diagnostics. All numbers on this page are read from
<code>results/summary.json</code>.</p>
<div class="stats">{stats_html}</div>

<h2>The selected strategy</h2>
<p><code>{escape(sel["trial"])}</code> has the highest full-sample net Sharpe ratio,
{f2(sel["sharpe"])} (95% block-bootstrap CI {f2(lo)} to {f2(hi)}). Its Sharpe was
{f2(sel["sharpe_first_half"])} before {escape(sel["halves_split_month"])} and
{f2(sel["sharpe_second_half"])} after. The Deflated Sharpe Ratio depends heavily on the
assumed variance of Sharpe ratios across trials: with the cross-trial variance used in the
paper (std {f2(sel["trial_sharpe_std_ann"])} annualized, expected best-of-{sel["n_trials"]}
Sharpe {f2(sel["expected_max_sharpe_ann"])}) it is {sel["deflated_sharpe"]:.3f}; with the
no-signal variance estimated from shuffled data (std {f2(sel["null_sharpe_std_ann"])},
expected best {f2(sel["expected_max_sharpe_null_var_ann"])}) it is
{sel["deflated_sharpe_null_var"]:.3f}.</p>
{figure("sharpe_distribution.png")}
{figure("permutation_null.png")}
{table(["Trials assumed (N)", "DSR"], dsr_rows)}
{table(["Cost per unit turnover", "Sharpe"], cost_rows)}

<h2>Probability of Backtest Overfitting</h2>
<p>CSCV splits the {pbo["rows_used"]} months into {pbo["n_blocks"]} blocks and evaluates all
{pbo["n_splits"]:,} half/half splits. The in-sample winner lands below the out-of-sample median
in {pct(pbo["pbo"])} of splits and loses money out of sample in {pct(pbo["prob_oos_loss"])}.
Median Sharpe of the winner: {f2(pbo["median_is_sharpe_ann"])} in sample,
{f2(pbo["median_oos_sharpe_ann"])} out of sample.</p>
{figure("pbo_logits.png")}
{figure("is_vs_oos.png")}
{table(["Family", "Trials", "Best Sharpe", "Median Sharpe", "PBO"], family_rows)}

<h2>Walk-forward selection</h2>
<p>Every {wf["test_months"]} months, pick the trial with the best Sharpe over the trailing
{wf["train_months"]} months and hold it. Over {wf["n_windows"]} windows this chose
{wf["distinct_trials_selected"]} distinct trials.</p>
{figure("cumulative.png")}
{table(["Series (same months)", "Sharpe", "Ann. return"], comparison_rows)}
<p>Walk-forward OOS Sharpe 95% bootstrap CI: {f2(wlo)} to {f2(whi)}.</p>

<h2>Factor-adjusted alpha</h2>
<p>OLS on Fama-French factors with Newey-West standard errors.</p>
{table(["Series", "Model", "Alpha (ann.)", "t (NW)", "Lags"], alpha_rows, numeric_from=2)}

<h2>Top 10 trials</h2>
{table(["Trial", "Sharpe", "Ann. ret", "Ann. vol", "Max DD", "Turnover/yr"], top_rows)}

<h2>Data and methods</h2>
<ul>
<li>Data: {escape(meta["data_source"])} ({escape(meta["data_vintage"])}). Raw files are not
redistributed; the repository contains a download script.</li>
<li>Signals use information up to month t and are traded in month t+1; costs are charged
per unit of turnover; positions are scaled to a volatility target using trailing realized
volatility.</li>
<li>Bailey, Borwein, L&oacute;pez de Prado, Zhu (2015), The Probability of Backtest
Overfitting. Bailey &amp; L&oacute;pez de Prado (2014), The Deflated Sharpe Ratio.</li>
</ul>
</main>"""

    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Backtest Overfitting Report</title>"
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&'
        'family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">'
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(results / "figures", out / "figures", dirs_exist_ok=True)
    (out / "index.html").write_text(html, encoding="utf-8")
    return out / "index.html"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("site"))
    args = parser.parse_args()
    print(build(args.results, args.out))


if __name__ == "__main__":
    main()
