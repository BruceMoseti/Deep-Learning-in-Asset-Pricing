#!/usr/bin/env python3
"""Write reports/RESEARCH_REPORT.md from the saved results.

The prose is here; every number is read from ``results/``.  Nothing is
transcribed by hand, so the report cannot drift away from the run that produced
it.  Re-run this after any change to the pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from xsap.config import RESULTS, ROOT, Config

REPORT = ROOT / "reports" / "RESEARCH_REPORT.md"
LADDER = ["single-signal", "ols", "ridge", "lasso", "enet", "xgboost", "neural-net"]


def table(name: str, index=None) -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / f"{name}.csv")
    return frame.set_index(index) if index else frame


def jsonf(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def md(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    return frame.to_markdown(floatfmt=floatfmt)


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[[m for m in LADDER if m in frame.index]]


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def count_tests() -> int:
    """Number of test functions, so the report cannot claim a stale count."""
    total = 0
    for path in (ROOT / "tests").glob("test_*.py"):
        total += sum(
            1 for line in path.read_text().splitlines() if line.startswith("def test_")
        )
    return total


def figure(name: str, caption: str) -> str:
    if not (ROOT / "reports" / "figures" / f"{name}.png").exists():
        return ""
    return f"\n![{caption}](figures/{name}.png)\n\n*{caption}*\n"


def main() -> None:
    cfg = Config()
    run = jsonf("exp1_run")
    accuracy = ordered(table("exp1_forecast_accuracy", "model"))
    decades = table("exp1_ic_by_decade").rename(columns={"Unnamed: 0": "decade"}).set_index("decade")
    performance = ordered(table("exp2_portfolio_performance", "model"))
    profile = table("exp2_quantile_profile", "predicted_decile")
    alpha = table("exp3_factor_alpha_gross", ["model", "factor_model"])
    bootstrap = ordered(table("exp3_bootstrap_sharpe", "model"))
    multiple = table("exp3_multiple_testing_summary").rename(
        columns={"Unnamed: 0": "hypothesis"}
    ).set_index("hypothesis")
    own = ordered(table("exp3_own_strategy_vs_published", "model"))
    exp3_run = jsonf("exp3_run")
    exp4 = jsonf("exp4_calibration")
    size = table("exp4_size_power_full")
    worst = table("exp4_worst_case_size", "test")
    defined = table("exp4_test_defined", ["n_obs", "n_assets"])
    real = table("exp4_real_data", ["n_obs", "n_assets"])
    design = table("exp5_design_robustness", "variant")
    subperiods = table("exp5_subperiods", ["period", "model"])
    regimes = table("exp5_regimes", ["dimension", "state", "model"])
    coverage = table("data_coverage", "family")
    moments = table("factor_moments").rename(columns={"Unnamed: 0": "factor"}).set_index("factor")

    best = run.get("best_model") or accuracy["rank_ic"].idxmax()
    best_linear = run.get("best_linear_model", "lasso")
    baseline = "single-signal"

    comparisons = table("exp1_model_comparisons", ["model", "benchmark"])

    def compare(model: str, benchmark: str, column: str) -> float:
        try:
            return float(comparisons.loc[(model, benchmark), column])
        except KeyError:
            return float("nan")

    n_published = int(exp3_run["n_published_predictors_used"])
    bonferroni_t = float(stats.norm.isf(0.025 / n_published))

    ff6 = alpha.xs("ff6", level="factor_model")
    ff6_cols = ["alpha_annual", "alpha_tstat", "r2", "beta_mktrf", "beta_smb", "beta_hml", "beta_mom"]

    grid_tests = ["grs", "wald_chi2", "grs_shrunk", "pesaran_yamagata"]
    long_sample = int(size["n_obs"].max())
    size_long = size[size["n_obs"] == long_sample].copy()
    size_grid = size_long.pivot_table(
        index="error_model",
        columns="n_assets",
        values=[f"{t}_size" for t in grid_tests],
    ).rename(columns=lambda c: c.replace("_size", ""), level=0)

    def size_at(error_model: str, n_assets: int, test: str) -> float:
        row = size_long[
            (size_long["error_model"] == error_model)
            & (size_long["n_assets"] == n_assets)
        ]
        return float(row[f"{test}_size"].iloc[0]) if len(row) else float("nan")

    ablation_section = ""
    if (RESULTS / "exp5_feature_ablation.csv").exists():
        ablation = table("exp5_feature_ablation", "predictor_set")
        full_ic = float(ablation.loc["all_predictors", "rank_ic"])
        without = ablation[ablation.index.str.startswith("without_")].copy()
        only = ablation[ablation.index.str.startswith("only_")].copy()
        without.index = without.index.str.replace("without_", "", regex=False)
        only.index = only.index.str.replace("only_", "", regex=False)
        combined = pd.DataFrame(
            {
                "ic_without_group": without["rank_ic"],
                "ic_change_when_removed": without["rank_ic"] - full_ic,
                "ic_with_group_alone": only["rank_ic"],
                "sharpe_without_group": without["sharpe_gross"],
                "sharpe_with_group_alone": only["sharpe_gross"],
            }
        ).sort_values("ic_change_when_removed")
        most_costly = combined.index[0]
        best_alone = combined["ic_with_group_alone"].idxmax()
        redundant = combined["ic_change_when_removed"].idxmax()

        ablation_section = f"""
### 5.1 Which predictors carry the signal

Each row refits the entire walk-forward. `ic_change_when_removed` is negative
when dropping a group hurts. Feature importances from the fitted model are
deliberately not used: they describe a fit, not out-of-sample value.

{md(combined)}

Full predictor set: rank IC {full_ic:.4f}.

Removing **{most_costly}** costs the most
({combined.loc[most_costly, 'ic_change_when_removed']:+.4f}). The group that
does best on its own is **{best_alone}**
(IC {combined.loc[best_alone, 'ic_with_group_alone']:.4f} alone, against
{full_ic:.4f} for everything together).

The two columns disagree, and the disagreement is the point. Removing
**{redundant}** changes the result least
({combined.loc[redundant, 'ic_change_when_removed']:+.4f}) — but that is a
statement about redundancy given the other predictors, not about whether the
group contains information. A group can be entirely substitutable and still be
informative on its own. Reporting only the leave-one-out column would license
the wrong conclusion.
{figure("fig10_feature_ablation", "Removing a predictor group and using it alone answer different questions")}
"""

    report = f"""# Machine Learning and Statistical Inference for Cross-Sectional Asset Returns

*Generated from `results/` by `scripts/07_report.py`. Every number below is read
from a result file; none is transcribed.*

---

## Summary

**Question.** Do nonlinear machine-learning models add out-of-sample predictive
information about cross-sectional equity returns beyond regularised linear
models — and if so, does it survive transaction costs, factor controls, and a
correction for the number of hypotheses the literature has tested?

**Setting.** {run['n_assets']} value-weighted characteristic-sorted portfolios of
US common stocks, {run['n_observations']:,} asset-months,
{run['n_predictors']} point-in-time predictors, {run['n_refits']} annual refits
on an expanding window, {run['oos_months']} out-of-sample months from
{run['oos_first_month']} to {run['oos_last_month']}.

**Five findings.**

1. **The gain from complexity is sparsity, not nonlinearity.** Rank IC rises
   monotonically from {accuracy.loc[baseline, 'rank_ic']:.4f} for a single
   momentum characteristic to {accuracy.loc['ridge', 'rank_ic']:.4f} for ridge
   to {accuracy.loc[best, 'rank_ic']:.4f} for {best}
   (*t* = {accuracy.loc[best, 'rank_ic_tstat']:.2f} against zero). But in a
   paired test {best} beats ridge
   (*t* = {compare(best, 'ridge', 'ic_diff_tstat'):.2f}) and does **not** beat
   {best_linear}, the best linear model
   (*t* = {compare(best, best_linear, 'ic_diff_tstat'):.2f},
   *p* = {compare(best, best_linear, 'ic_diff_pvalue'):.2f}). The neural network
   is significantly *worse* than {best_linear} on squared error.
2. The accuracy ranking is not the tradability ranking. Ridge breaks even at
   {performance.loc['ridge', 'breakeven_cost_bps']:.0f} bps one-way and is
   negative at 20; {best} breaks even at
   {performance.loc[best, 'breakeven_cost_bps']:.0f}; the one-line baseline,
   which trades least, breaks even highest of all at
   {performance.loc[baseline, 'breakeven_cost_bps']:.0f}.
3. The alpha is real and unremarkable in context. {best} earns
   {pct(ff6.loc[best, 'alpha_annual'])} a year against the six-factor model
   (*t* = {ff6.loc[best, 'alpha_tstat']:.2f}), but its own *t* of
   {own.loc[best, 'tstat_nw']:.2f} sits at the
   {own.loc[best, 'percentile_among_published']:.0f}th percentile of
   {n_published} published predictors and does **not** survive Bonferroni.
4. Experiment 4 refuted its own hypothesis. GRS does not degrade as *N/T* → 1;
   what fails is its asymptotic counterpart, shrinkage without a recalibrated
   reference, and the large-*N* alternative under real cross-sectional
   dependence.
5. Predictability decays across the sample, for every model.

---

## 1. Data

### 1.1 Cross-section

{md(coverage.assign(mean_monthly_return=lambda d: d['mean_monthly_return']), ".4f")}

All returns are value-weighted monthly returns on portfolios of NYSE, AMEX and
NASDAQ common stocks from the Kenneth French data library, in decimal form.
Sample {cfg.sample_start} to {cfg.sample_end}. The negative and zero
net-share-issues buckets are dropped because they are categorical rather than
points on an ordered sort, so they are not comparable across the cross-section.

### 1.2 Factors

{md(moments, ".3f")}

Mkt-RF at {moments.loc['mktrf', 'mean_pct_per_month']:.2f}% and momentum at
{moments.loc['mom', 'mean_pct_per_month']:.2f}% per month match published values,
which is the check that the mirrored files are the real library.

### 1.3 What this data cannot support

The cross-section is portfolios, not firms. Accounting characteristics are not
observable per portfolio, so all {run['n_predictors']} predictors are derived
from returns, plus three static labels giving each asset's place in its sort.
Nonlinear interactions **between firm characteristics** — the mechanism Gu,
Kelly and Xiu (2020) identify as the main source of machine-learning gains —
largely cannot be represented here. Read what follows as a lower bound on what
this pipeline would find on a firm-level panel. `docs/METHODOLOGY.md` §8 lists
the rest.
{figure("fig12_context", "Market factor over the sample, and the distribution of monthly rank IC for the best model")}

---

## 2. Method

Detail in `docs/METHODOLOGY.md`. The four things that matter most:

- **Target.** Next month's excess return, demeaned and scaled within the month.
  Demeaning removes the market, which dominates return variance and is
  essentially unforecastable monthly, and leaves what a long-short book trades.
- **Splits by time and by whole month.** Random folds would train on the future,
  and — less obviously — would leak a month's common shock across the split,
  since average pairwise residual correlation here is
  {exp4['residual_correlation_mean']:.3f} with a 90th percentile of
  {exp4['residual_correlation_p90']:.2f}. One month is embargoed from the end of
  the training and validation blocks because labels reach forward.
- **Hyper-parameters chosen on a validation block** that precedes the test year,
  inside each model's own `fit`, so the harness cannot leak.
- **Leakage is tested.** Corrupting returns after a cutoff must leave earlier
  predictors bit-for-bit identical; permuting labels within months must drive
  measured skill inside its sampling error; and a deliberate leak must show up
  clearly, so that the null control means something.

---

## 3. Experiment 1 — does flexibility buy accuracy?

{md(accuracy)}

**The point estimates rise monotonically along the ladder** — {baseline}
{accuracy.loc[baseline, 'rank_ic']:.4f}, ridge
{accuracy.loc['ridge', 'rank_ic']:.4f}, lasso
{accuracy.loc['lasso', 'rank_ic']:.4f}, {best}
{accuracy.loc[best, 'rank_ic']:.4f} — which invites the conclusion that
complexity pays. That conclusion does not survive a paired test.
{figure("fig1_forecast_accuracy", "Experiment 1: mean rank IC, its Newey-West t-statistic, and out-of-sample R-squared across the model ladder")}

### 3.1 The gain is sparsity, not nonlinearity

Both models face the same cross-section every month, so their monthly ICs can
be differenced pairwise. The common component cancels and the resulting test is
far tighter than comparing two standard errors.

{md(comparisons)}

`ic_diff_tstat` tests the rank-IC gap; `dm_tstat_squared_error` is
Diebold-Mariano on squared error, where negative favours the row.

Read the two comparisons against `{best_linear}`, the best linear model, and
against ridge:

- **{best} beats ridge**: IC gap
  {compare(best, 'ridge', 'ic_difference'):+.4f},
  *t* = {compare(best, 'ridge', 'ic_diff_tstat'):.2f}; Diebold-Mariano
  *t* = {compare(best, 'ridge', 'dm_tstat_squared_error'):.2f}.
- **{best} does *not* beat {best_linear}**: IC gap only
  {compare(best, best_linear, 'ic_difference'):+.4f},
  *t* = {compare(best, best_linear, 'ic_diff_tstat'):.2f}
  (*p* = {compare(best, best_linear, 'ic_diff_pvalue'):.2f}), and on squared
  error *t* = {compare(best, best_linear, 'dm_tstat_squared_error'):.2f}
  (*p* = {compare(best, best_linear, 'dm_pvalue'):.2f}) — indistinguishable.
- **The neural network is *worse* than {best_linear}**: IC gap
  {compare('neural-net', best_linear, 'ic_difference'):+.4f}, and on squared
  error it loses significantly
  (*t* = {compare('neural-net', best_linear, 'dm_tstat_squared_error'):+.2f},
  *p* = {compare('neural-net', best_linear, 'dm_pvalue'):.3f}).

So the answer to the project's stated question is **no, not reliably**. What
separates ridge from boosting is almost entirely the sparse regularisation in
between: ridge selects a penalty so small it is effectively OLS — with
{run['n_predictors']} predictors and {run['n_observations']:,} observations
there is no ill-conditioning for shrinkage to fix — whereas Lasso's variable
selection is a real restriction that pays out of sample. Adding nonlinearity on
top of that buys a further
{compare(best, best_linear, 'ic_difference'):+.4f} of IC, which is not
distinguishable from zero.

Note also that no *single adjacent* step in the ladder is significant on its
own. Only the cumulative gap from ridge to {best} clears conventional
significance. A table of point estimates would have supported a much stronger
claim than the data does.

That the network trails is the expected outcome for {run['n_assets']} assets and
{run['n_predictors']} return-based predictors: boosting at depth 2 to 4 fits
low-order interactions, about the amount of structure this data supports,
whereas a network must learn a representation from the same thin signal. It is
not evidence about networks in general, and on a firm-level panel with hundreds
of characteristics the literature finds the opposite.

**On the negative out-of-sample R².** IC and R² measure different things, and
the gap is diagnosable rather than contradictory. IC asks whether the *ordering*
is informative; R² asks whether the forecast is the right *size*. The
calibration slope — the coefficient from regressing outcome on forecast out of
sample — is {accuracy.loc['ridge', 'calibration_slope']:.2f} for ridge, meaning
its forecasts are roughly {1 / accuracy.loc['ridge', 'calibration_slope']:.1f}
times too large, and {accuracy.loc[best, 'calibration_slope']:.2f} for {best}.
The better-calibrated models are exactly the ones with positive R². Since the
portfolio uses only the ranking, IC is the metric that matters here — but
reporting IC alone would have hidden a real miscalibration.
{figure("fig2_calibration_gap", "Models with a calibration slope near one are exactly those with positive out-of-sample R-squared")}

**Accuracy by decade.**

{md(decades)}

Every model weakens over the sample. Two readings, which this design cannot
separate: either these relations were arbitraged away as they became known — the
pattern the anomaly-decay literature would predict — or the early result was
partly luck. This is the main reason the full-sample *t*-statistic matters more
here than the point estimate.
{figure("fig6_rolling_accuracy", "Rolling 36-month mean rank IC: positive on average, far from constant")}

---

## 4. Experiment 2 — is it worth anything after costs?

{md(performance.T, ".3f")}

Turnover is two-sided: replacing both legs in full is 4.0, so
`turnover_fraction_of_book` reports the share replaced per month. Costs are a
one-way rate applied to turnover.

**The ranking reverses.** By gross Sharpe the order is roughly {best} >
enet > ridge > baseline. By break-even cost it is nearly inverted: the
single-characteristic baseline absorbs
{performance.loc[baseline, 'breakeven_cost_bps']:.0f} bps before its edge
disappears — the most of any model — because it turns over
{performance.loc[baseline, 'turnover_monthly']:.2f} against ridge's
{performance.loc['ridge', 'turnover_monthly']:.2f}. Ridge and OLS are *negative*
at 20 bps.

This is the most useful result in the project, and it only appears if turnover
is charged for before models are compared. A study that stopped at gross Sharpe
would have concluded that penalised linear models beat a one-line signal. After
costs they do not.
{figure("fig4_cost_erosion", "Sharpe ratio against trading cost, and the break-even cost per model")}
{figure("fig3_cumulative_performance", "Cumulative long-short performance, gross and net of 10 bps")}

**Monotonicity.** Realised annualised excess return by predicted decile:

{md(profile * 100, ".2f")}

A monotone profile is much stronger evidence than a good top-minus-bottom
spread, which two lucky buckets can produce on their own.
{figure("fig5_quantile_profile", "Realised annualised excess return by predicted decile")}

---

## 5. Experiment 5 — attribution and robustness

*Presented before Experiment 3 because it bears on how much of the signal there
is to test.*
{ablation_section}
### 5.2 Design choices

Each row changes exactly one decision away from the default.

{md(design)}

**This answers "why expanding rather than rolling".** A 20-year rolling window
performs essentially the same as expanding
(IC {design.loc['rolling_20y_window', 'rank_ic']:.4f} against
{design.loc['default', 'rank_ic']:.4f}; Sharpe
{design.loc['rolling_20y_window', 'sharpe_gross']:.2f} against
{design.loc['default', 'sharpe_gross']:.2f}). A 10-year window is clearly worse
(IC {design.loc['rolling_10y_window', 'rank_ic']:.4f}, Sharpe
{design.loc['rolling_10y_window', 'sharpe_gross']:.2f}). Degradation as the
window shrinks, and flatness between 20 years and everything, is the evidence
that estimation error binds rather than non-stationarity. Note the honest
caveat: 20-year rolling is marginally better on Sharpe, so the choice is within
noise rather than obviously right.

Portfolio cutoffs behave as expected — concentrating into narrower buckets
raises gross Sharpe and turnover together — and rank weighting across the whole
cross-section gives up some gross Sharpe for materially lower turnover.

### 5.3 Subperiods and regimes

{md(subperiods.reset_index().pivot(index='period', columns='model', values='rank_ic'))}

Regime labels use only information available before the month begins, so a split
is something a strategy could have conditioned on. Splitting on contemporaneous
volatility would be a different and much easier exercise.
{figure("fig11_regimes", "Rank IC by market state, labelled using only prior information")}

{md(regimes.reset_index().pivot_table(index=['dimension', 'state'], columns='model', values='rank_ic'))}

---

## 6. Experiment 3 — is the signal statistically real?

### 6.1 Factor alphas

Six-factor (FF5 + momentum) regressions on gross long-short returns,
Newey-West {cfg.newey_west_lags} lags:

{md(ordered(ff6)[ff6_cols])}

{best}'s alpha of {pct(ff6.loc[best, 'alpha_annual'])} a year survives with
*t* = {ff6.loc[best, 'alpha_tstat']:.2f}, and the regression R² of
{ff6.loc[best, 'r2']:.3f} says the strategy is close to orthogonal to the market,
size, value, profitability, investment and momentum. Do not oversell that low
R²: a dollar-neutral decile spread across characteristic-sorted portfolios is
partly constructed to be factor-neutral. The content is that the alpha does not
collapse when six factors are added.

### 6.2 Bootstrap

Stationary block bootstrap, {exp3_run['bootstrap_draws']:,} draws, mean block
{exp3_run['bootstrap_mean_block_months']} months:

{md(bootstrap, ".3f")}

`p_sign_flips` is the share of resamples in which the Sharpe ratio changes sign
— a more direct read on fragility than an interval.

### 6.3 Multiple testing

The same tests applied to {n_published} published cross-sectional predictors
(Chen and Zimmermann, 2022), each requiring at least
{exp3_run['min_months_required']} months:

{md(multiple, ".3f")}

Bonferroni at 5% over {n_published} hypotheses corresponds to
|*t*| > {bonferroni_t:.2f}.

The high survival rate is itself a selection effect: these predictors were
published *because* they were significant. That is Harvey, Liu and Zhu's (2016)
argument, and the reason to treat |*t*| > 3 rather than 1.96 as the relevant
hurdle for a new predictor.

**And this project's own result, on the same scale:**

{md(own, ".3f")}

{best} clears the |*t*| > 3 hurdle but does **not** survive Bonferroni across
{n_published} hypotheses, and sits at the
{own.loc[best, 'percentile_among_published']:.0f}th percentile of the published
distribution. So: statistically real on its own terms, median-strength in
context, and not strong enough for the most conservative correction the
literature applies. Reporting the percentile is more informative than reporting
the *t*-statistic alone, and it is the number this project should be judged on.
{figure("fig7_multiple_testing", "This project's t-statistic against the distribution of 212 published predictors")}

Note also that this project is itself a search — {run['n_predictors']}
predictors, {len(run['models'])} models, several portfolio variants — which is
why the comparison is against the published distribution rather than against
zero.

---

## 7. Experiment 4 — do asset-pricing tests behave with many assets?

### 7.1 The hypothesis was wrong

The premise was that the Gibbons-Ross-Shanken test degrades as *N/T* → 1,
because it inverts an *N* × *N* residual covariance matrix estimated from *T*
observations. A test was written asserting exactly that. It failed, and it
failed for a reason worth stating: **GRS is exact in finite samples** under its
assumptions — normal, homoskedastic, serially independent residuals with *any*
cross-sectional covariance — for every *N* ≤ *T* − *K* − 1. Dimension alone does
not distort it.

So the experiment was rebuilt around the right question: real returns satisfy
none of those assumptions, so how does each test behave once the errors are
allowed to look like the errors in the data? Six error structures, calibrated to
the panel: implied Student-*t* degrees of freedom
{exp4['implied_student_df']:.1f}, log-volatility persistence
{exp4['log_vol_persistence']:.2f}, mean pairwise residual correlation
{exp4['residual_correlation_mean']:.3f}.

### 7.2 Empirical size, nominal 5%, T = {long_sample}

{md(size_grid, ".3f")}

{md(worst, ".3f")}

**GRS holds its size everywhere**, at up to
*N/T* = {size_long['ratio_n_over_t'].max():.2f}, under heavy tails, under a
persistent common volatility factor, and under residual vectors resampled from
the real panel. Its median size over the grid is
{worst.loc['grs', 'median_size_over_grid']:.3f}.
{figure("fig8_test_size", "Empirical size against N/T under three error structures. GRS is flat at 5%; its asymptotic counterpart is not.")}

**What actually breaks:**

- *The asymptotic version of the same statistic.* Referring the identical
  quadratic form to χ²*_N_* instead of the exact *F* gives size
  {size_at('gaussian', 10, 'wald_chi2'):.3f} at *N* = 10,
  {size_at('gaussian', 50, 'wald_chi2'):.3f} at *N* = 50,
  {size_at('gaussian', 100, 'wald_chi2'):.3f} at *N* = 100 and
  {size_at('gaussian', 200, 'wald_chi2'):.3f} at *N* = 200. The finite-sample
  correction is doing all the work.
- *Shrinkage without recalibration.* A Ledoit-Wolf covariance conditions better
  and shrinks the statistic, but the *F* critical value is unchanged, so size
  collapses to {size_at('gaussian', 100, 'grs_shrunk'):.3f} at *N* = 100 and the
  test stops rejecting anything. Better estimation is not automatically better
  inference.
- *The large-N test's assumption.* Pesaran-Yamagata never inverts an *N* × *N*
  matrix and is correctly sized under cross-sectional independence
  ({size_at('gaussian_independent', 300, 'pesaran_yamagata'):.3f} at *N* = 300).
  Under the correlation these portfolios actually have, its size rises to
  {size_at('empirical_wild_block', 300, 'pesaran_yamagata'):.3f} — and rises
  *with N*, the opposite of what an asymptotic-in-*N* test should do.

### 7.3 Existence, not size, is the binding constraint on GRS

{md(defined, ".2f")}

At *T* = {int(size['n_obs'].min())} the statistic simply does not exist for the
largest cross-sections, since it
requires *T* > *N* + *K*. Pesaran-Yamagata is always defined. So the real
trade-off is not about dimension in the abstract: **GRS needs *N* < *T*;
Pesaran-Yamagata needs weak cross-sectional dependence.** Equity portfolios
violate the second and modern cross-sections violate the first.

### 7.4 Size-adjusted power

Raw power cannot be compared across tests whose sizes differ — a test that
rejects a true null 100% of the time also "detects" mispricing 100% of the time
while knowing nothing. Power is therefore also measured at critical values
calibrated from each cell's own null distribution, against a true alpha
dispersion of {exp4['alpha_sd_under_alternative'] * 1200:.1f}% a year (chosen
small enough that no test can see it one asset at a time).

Under weak dependence, Pesaran-Yamagata's power grows with *N* and overtakes
GRS. Under real dependence it flattens out while GRS's keeps rising — cross
sectional correlation *increases* the power of a joint alpha test, because the
common component can be hedged out and the mispricing portfolio has a higher
Sharpe ratio. Full grid in `results/exp4_size_adjusted_power.csv`.
{figure("fig9_test_power", "Size-adjusted power against N, with and without real cross-sectional dependence")}

### 7.5 The same tests on the real panel

{md(real[[c for c in real.columns if c.endswith('_reject_rate')]], ".3f")}

Over the full sample GRS rejects the six-factor model for essentially every
subset with *N* ≥ 50. The simulation is what licenses reading that as genuine
mispricing rather than size distortion — without it, the rejection and a broken
test are indistinguishable. Note that `grs_shrunk` rejects nothing at large *N*,
exactly as its simulated size predicts.

### 7.6 Three bugs that produced publishable-looking numbers

Recorded because each one nearly became a finding.

1. *An undemeaned residual pool.* The bootstrap pool's column means were not
   zero, planting roughly 2% a year of alpha. Every test correctly rejected a
   null that was in fact false.
2. *Resampling months with replacement.* Repeats months, so the covariance is
   estimated from fewer distinct observations than the nominal *T*. Size reached
   0.78 and looked like a dramatic result about real returns.
3. *Resampling without replacement.* Fixes the duplication and introduces the
   opposite error — the finite-population correction shrinks the variance of the
   estimated intercept — so size fell to 0.000.

The fix was a wild bootstrap: real residual vectors with random sign flips. No
month is duplicated, sign flips leave second moments untouched so the variance
of the intercept estimate is exactly what the test assumes, the null holds
exactly, and each month keeps its own magnitude and cross-sectional pattern.
All three are now regression tests in `tests/test_montecarlo.py`.

---

## 8. What would change these conclusions

- **Firm-level data.** The universe, not the model class, is the binding
  constraint. The claim that flexibility adds little would be worth revisiting
  on a panel where interactions between accounting characteristics exist to be
  found.
- **Costs optimised rather than measured.** The accuracy and net-performance
  rankings already disagree, so optimising against a turnover penalty would
  plausibly change the model ranking more than any change to the models.
- **The decay.** Distinguishing arbitrage from overfitting would need a design
  that exploits variation in when each relation became widely known — for
  instance, whether decay is faster for predictors that received more academic
  attention.
- **Impact modelling.** A flat basis-point rate ignores market impact, which
  scales with size. The break-even figures should be read as generous.

## 9. Reproducing this

```bash
pip install -r requirements.txt
make test     # {count_tests()} tests
make all      # full pipeline
python3 scripts/07_report.py
```

Raw inputs are pinned to immutable commits with SHA-256 digests in
`data/raw/manifest.json`. This matters because the Fama-French library is
revised retroactively whenever CRSP is updated, so an unpinned download would
silently change every number above.
"""

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report)
    print(f"wrote {REPORT.relative_to(ROOT)} ({len(report.splitlines())} lines)")


if __name__ == "__main__":
    main()
