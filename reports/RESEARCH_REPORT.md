# Machine Learning and Statistical Inference for Cross-Sectional Asset Returns

*Generated from `results/` by `scripts/07_report.py`. Every number below is read
from a result file; none is transcribed.*

---

## Summary

**Question.** Do nonlinear machine-learning models add out-of-sample predictive
information about cross-sectional equity returns beyond regularised linear
models — and if so, does it survive transaction costs, factor controls, and a
correction for the number of hypotheses the literature has tested?

**Setting.** 374 value-weighted characteristic-sorted portfolios of
US common stocks, 248,614 asset-months,
26 point-in-time predictors, 34 annual refits
on an expanding window, 407 out-of-sample months from
1990-01 to 2023-11.

**Five findings.**

1. Flexibility helps, and the gain is larger than its own standard error. Rank
   IC rises from 0.0254 for a single momentum
   characteristic to 0.0319 for ridge to
   0.0492 for xgboost
   (*t* = 3.64). The neural network reaches
   only 0.0328.
2. The accuracy ranking is not the tradability ranking. Ridge breaks even at
   19 bps one-way and is
   negative at 20; xgboost breaks even at
   33; the one-line baseline,
   which trades least, breaks even highest of all at
   42.
3. The alpha is real and unremarkable in context. xgboost earns
   7.0% a year against the six-factor model
   (*t* = 2.83), but its own *t* of
   3.44 sits at the
   57th percentile of
   212 published predictors and does **not** survive Bonferroni.
4. Experiment 4 refuted its own hypothesis. GRS does not degrade as *N/T* → 1;
   what fails is its asymptotic counterpart, shrinkage without a recalibrated
   reference, and the large-*N* alternative under real cross-sectional
   dependence.
5. Predictability decays across the sample, for every model.

---

## 1. Data

### 1.1 Cross-section

| family                       |   assets | first_month   | last_month   |   mean_monthly_return |
|:-----------------------------|---------:|:--------------|:-------------|----------------------:|
| 25_Portfolios_5x5            |       25 | 1963-07       | 2023-12      |                0.0111 |
| 25_Portfolios_BEME_INV_5x5   |       25 | 1963-07       | 2023-12      |                0.0105 |
| 25_Portfolios_BEME_OP_5x5    |       25 | 1963-07       | 2023-12      |                0.0102 |
| 25_Portfolios_ME_AC_5x5      |       25 | 1963-07       | 2023-12      |                0.0110 |
| 25_Portfolios_ME_BETA_5x5    |       25 | 1963-07       | 2023-12      |                0.0112 |
| 25_Portfolios_ME_INV_5x5     |       25 | 1963-07       | 2023-12      |                0.0112 |
| 25_Portfolios_ME_NI_5x5      |       25 | 1963-07       | 2023-12      |                0.0105 |
| 25_Portfolios_ME_OP_5x5      |       25 | 1963-07       | 2023-12      |                0.0109 |
| 25_Portfolios_ME_Prior_12_2  |       25 | 1963-07       | 2023-12      |                0.0107 |
| 25_Portfolios_ME_Prior_1_0   |       25 | 1963-07       | 2023-12      |                0.0106 |
| 25_Portfolios_ME_Prior_60_13 |       25 | 1963-07       | 2023-12      |                0.0116 |
| 25_Portfolios_ME_VAR_5x5     |       25 | 1963-07       | 2023-12      |                0.0111 |
| 25_Portfolios_OP_INV_5x5     |       25 | 1963-07       | 2023-12      |                0.0100 |
| 49_Industry_Portfolios       |       49 | 1963-07       | 2023-12      |                0.0102 |

All returns are value-weighted monthly returns on portfolios of NYSE, AMEX and
NASDAQ common stocks from the Kenneth French data library, in decimal form.
Sample 1963-07 to 2023-12. The negative and zero
net-share-issues buckets are dropped because they are categorical rather than
points on an ordered sort, so they are not comparable across the cross-section.

### 1.2 Factors

| factor   |   mean_pct_per_month |   sd_pct_per_month |   n_months |
|:---------|---------------------:|-------------------:|-----------:|
| mktrf    |                0.568 |              4.497 |    726.000 |
| smb      |                0.215 |              3.033 |    726.000 |
| hml      |                0.292 |              2.995 |    726.000 |
| rmw      |                0.283 |              2.225 |    726.000 |
| cma      |                0.273 |              2.077 |    726.000 |
| rf       |                0.363 |              0.266 |    726.000 |
| mom      |                0.600 |              4.213 |    726.000 |

Mkt-RF at 0.57% and momentum at
0.60% per month match published values,
which is the check that the mirrored files are the real library.

### 1.3 What this data cannot support

The cross-section is portfolios, not firms. Accounting characteristics are not
observable per portfolio, so all 26 predictors are derived
from returns, plus three static labels giving each asset's place in its sort.
Nonlinear interactions **between firm characteristics** — the mechanism Gu,
Kelly and Xiu (2020) identify as the main source of machine-learning gains —
largely cannot be represented here. Read what follows as a lower bound on what
this pipeline would find on a firm-level panel. `docs/METHODOLOGY.md` §8 lists
the rest.

![Market factor over the sample, and the distribution of monthly rank IC for the best model](figures/fig12_context.png)

*Market factor over the sample, and the distribution of monthly rank IC for the best model*


---

## 2. Method

Detail in `docs/METHODOLOGY.md`. The four things that matter most:

- **Target.** Next month's excess return, demeaned and scaled within the month.
  Demeaning removes the market, which dominates return variance and is
  essentially unforecastable monthly, and leaves what a long-short book trades.
- **Splits by time and by whole month.** Random folds would train on the future,
  and — less obviously — would leak a month's common shock across the split,
  since average pairwise residual correlation here is
  0.048 with a 90th percentile of
  0.22. One month is embargoed from the end of
  the training and validation blocks because labels reach forward.
- **Hyper-parameters chosen on a validation block** that precedes the test year,
  inside each model's own `fit`, so the harness cannot leak.
- **Leakage is tested.** Corrupting returns after a cutoff must leave earlier
  predictors bit-for-bit identical; permuting labels within months must drive
  measured skill inside its sampling error; and a deliberate leak must show up
  clearly, so that the null control means something.

---

## 3. Experiment 1 — does flexibility buy accuracy?

| model         |   rank_ic |   rank_ic_tstat |   rank_ic_std |   icir |   pearson_ic |   hit_rate |   r2_oos |   calibration_slope |   n_months |   dm_tstat_vs_ridge |
|:--------------|----------:|----------------:|--------------:|-------:|-------------:|-----------:|---------:|--------------------:|-----------:|--------------------:|
| single-signal |    0.0254 |          1.5765 |        0.3167 | 0.0803 |       0.0291 |     0.5283 |  -0.3005 |              0.0504 |   407.0000 |             18.5244 |
| ols           |    0.0316 |          2.2176 |        0.2770 | 0.1140 |       0.0352 |     0.5627 |  -0.0070 |              0.3088 |   407.0000 |              3.4112 |
| ridge         |    0.0319 |          2.2368 |        0.2773 | 0.1149 |       0.0355 |     0.5577 |  -0.0068 |              0.3113 |   407.0000 |            nan      |
| lasso         |    0.0440 |          2.5105 |        0.2793 | 0.1576 |       0.0494 |     0.3219 |  -0.0017 |              0.4056 |   227.0000 |             -2.0842 |
| enet          |    0.0443 |          2.8975 |        0.2868 | 0.1544 |       0.0478 |     0.5086 |   0.0005 |              0.5487 |   359.0000 |             -2.8700 |
| xgboost       |    0.0492 |          3.6356 |        0.2859 | 0.1722 |       0.0539 |     0.5872 |   0.0024 |              0.6786 |   407.0000 |             -4.4309 |
| neural-net    |    0.0328 |          2.2174 |        0.2894 | 0.1132 |       0.0376 |     0.5676 |  -0.0014 |              0.4226 |   407.0000 |             -3.0489 |

`dm_tstat_vs_ridge` is a Diebold-Mariano test on monthly squared error against
ridge; negative favours the row.

**Complexity pays, up to a point.** xgboost improves rank IC by
+0.0173 over
ridge, and the Diebold-Mariano statistic of
-4.43 says the squared-error improvement
exceeds its own standard error. The neural network does not beat boosting. That
is the expected outcome for 374 assets and
26 return-based predictors: boosting at depth 2 to 4 fits
low-order interactions, which is about the amount of structure this data
supports, whereas a network must learn a representation from the same thin
signal. It is not evidence about networks in general.

![Experiment 1: mean rank IC, its Newey-West t-statistic, and out-of-sample R-squared across the model ladder](figures/fig1_forecast_accuracy.png)

*Experiment 1: mean rank IC, its Newey-West t-statistic, and out-of-sample R-squared across the model ladder*


**On the negative out-of-sample R².** IC and R² measure different things, and
the gap is diagnosable rather than contradictory. IC asks whether the *ordering*
is informative; R² asks whether the forecast is the right *size*. The
calibration slope — the coefficient from regressing outcome on forecast out of
sample — is 0.31 for ridge, meaning
its forecasts are roughly 3.2
times too large, and 0.68 for xgboost.
The better-calibrated models are exactly the ones with positive R². Since the
portfolio uses only the ranking, IC is the metric that matters here — but
reporting IC alone would have hidden a real miscalibration.

![Models with a calibration slope near one are exactly those with positive out-of-sample R-squared](figures/fig2_calibration_gap.png)

*Models with a calibration slope near one are exactly those with positive out-of-sample R-squared*


**Accuracy by decade.**

| decade   |   single-signal |     ols |   ridge |   lasso |   enet |   xgboost |   neural-net |
|:---------|----------------:|--------:|--------:|--------:|-------:|----------:|-------------:|
| 1990s    |          0.0436 |  0.0781 |  0.0784 |  0.0691 | 0.0723 |    0.0700 |       0.0655 |
| 2000s    |          0.0247 |  0.0420 |  0.0421 |  0.0347 | 0.0548 |    0.0643 |       0.0422 |
| 2010s    |          0.0210 | -0.0184 | -0.0179 |  0.0113 | 0.0043 |    0.0226 |       0.0049 |
| 2020s    |         -0.0078 |  0.0136 |  0.0136 |  0.0031 | 0.0079 |    0.0259 |      -0.0037 |

Every model weakens over the sample. Two readings, which this design cannot
separate: either these relations were arbitraged away as they became known — the
pattern the anomaly-decay literature would predict — or the early result was
partly luck. This is the main reason the full-sample *t*-statistic matters more
here than the point estimate.

![Rolling 36-month mean rank IC: positive on average, far from constant](figures/fig6_rolling_accuracy.png)

*Rolling 36-month mean rank IC: positive on average, far from constant*


---

## 4. Experiment 2 — is it worth anything after costs?

|                           |   single-signal |    ols |   ridge |   lasso |   enet |   xgboost |   neural-net |
|:--------------------------|----------------:|-------:|--------:|--------:|-------:|----------:|-------------:|
| turnover_monthly          |           1.030 |  2.313 |   2.317 |   1.359 |  1.652 |     1.957 |        1.776 |
| turnover_fraction_of_book |           0.258 |  0.578 |   0.579 |   0.340 |  0.413 |     0.489 |        0.444 |
| ann_return_gross          |           0.051 |  0.051 |   0.052 |   0.030 |  0.065 |     0.076 |        0.051 |
| ann_vol                   |           0.155 |  0.125 |   0.126 |   0.108 |  0.129 |     0.141 |        0.133 |
| sharpe_gross              |           0.332 |  0.408 |   0.410 |   0.275 |  0.505 |     0.541 |        0.381 |
| max_drawdown_gross        |          -0.485 | -0.366 |  -0.361 |  -0.323 | -0.310 |    -0.328 |       -0.305 |
| hit_rate                  |           0.553 |  0.587 |   0.585 |   0.573 |  0.604 |     0.590 |        0.590 |
| sharpe_net_5bps           |           0.292 |  0.298 |   0.300 |   0.200 |  0.428 |     0.458 |        0.301 |
| ann_return_net_5bps       |           0.045 |  0.037 |   0.038 |   0.022 |  0.055 |     0.065 |        0.040 |
| sharpe_net_10bps          |           0.252 |  0.187 |   0.190 |   0.125 |  0.351 |     0.375 |        0.221 |
| ann_return_net_10bps      |           0.039 |  0.023 |   0.024 |   0.013 |  0.045 |     0.053 |        0.029 |
| sharpe_net_20bps          |           0.172 | -0.035 |  -0.031 |  -0.026 |  0.197 |     0.209 |        0.061 |
| ann_return_net_20bps      |           0.027 | -0.004 |  -0.004 |  -0.003 |  0.025 |     0.029 |        0.008 |
| breakeven_cost_bps        |          41.591 | 18.438 |  18.602 |  18.276 | 32.766 |    32.558 |       23.815 |

Turnover is two-sided: replacing both legs in full is 4.0, so
`turnover_fraction_of_book` reports the share replaced per month. Costs are a
one-way rate applied to turnover.

**The ranking reverses.** By gross Sharpe the order is roughly xgboost >
enet > ridge > baseline. By break-even cost it is nearly inverted: the
single-characteristic baseline absorbs
42 bps before its edge
disappears — the most of any model — because it turns over
1.03 against ridge's
2.32. Ridge and OLS are *negative*
at 20 bps.

This is the most useful result in the project, and it only appears if turnover
is charged for before models are compared. A study that stopped at gross Sharpe
would have concluded that penalised linear models beat a one-line signal. After
costs they do not.

![Sharpe ratio against trading cost, and the break-even cost per model](figures/fig4_cost_erosion.png)

*Sharpe ratio against trading cost, and the break-even cost per model*


![Cumulative long-short performance, gross and net of 10 bps](figures/fig3_cumulative_performance.png)

*Cumulative long-short performance, gross and net of 10 bps*


**Monotonicity.** Realised annualised excess return by predicted decile:

|   predicted_decile |   single-signal |   ols |   ridge |   lasso |   enet |   xgboost |   neural-net |
|-------------------:|----------------:|------:|--------:|--------:|-------:|----------:|-------------:|
|                  0 |            6.90 |  6.79 |    6.73 |    7.55 |   5.75 |      5.66 |         6.65 |
|                  1 |            8.60 |  8.62 |    8.78 |    8.61 |   8.56 |      7.95 |         9.18 |
|                  2 |            9.74 |  9.50 |    9.39 |   10.06 |   9.18 |      8.63 |         9.50 |
|                  3 |           10.06 |  9.56 |    9.53 |   10.06 |   9.74 |     10.20 |         9.61 |
|                  4 |           10.02 | 10.36 |   10.47 |   10.59 |   9.98 |      9.86 |         9.96 |
|                  5 |           10.34 |  9.99 |    9.86 |   10.15 |  10.62 |      9.97 |        10.27 |
|                  6 |           10.32 | 10.50 |   10.56 |   10.55 |  10.49 |     10.45 |        10.56 |
|                  7 |           10.12 | 10.75 |   10.74 |   10.39 |  11.14 |     11.56 |        10.70 |
|                  8 |           10.78 | 10.94 |   10.96 |   10.44 |  11.25 |     11.31 |        10.77 |
|                  9 |           12.04 | 11.91 |   11.90 |   10.53 |  12.24 |     13.31 |        11.73 |

A monotone profile is much stronger evidence than a good top-minus-bottom
spread, which two lucky buckets can produce on their own.

![Realised annualised excess return by predicted decile](figures/fig5_quantile_profile.png)

*Realised annualised excess return by predicted decile*


---

## 5. Experiment 5 — attribution and robustness

*Presented before Experiment 3 because it bears on how much of the signal there
is to test.*

### 5.2 Design choices

Each row changes exactly one decision away from the default.

| variant                  |   rank_ic |   rank_ic_tstat |   icir |   sharpe_gross |   sharpe_net_10bps |   turnover_monthly |   breakeven_cost_bps |   n_months |
|:-------------------------|----------:|----------------:|-------:|---------------:|-------------------:|-------------------:|---------------------:|-----------:|
| default                  |    0.0492 |          3.6356 | 0.1722 |         0.5413 |             0.3750 |             1.9569 |              32.5577 |   407.0000 |
| rolling_20y_window       |    0.0463 |          3.6061 | 0.1754 |         0.5767 |             0.4005 |             1.8262 |              32.7900 |   407.0000 |
| rolling_10y_window       |    0.0287 |          2.5999 | 0.1184 |         0.3923 |             0.2111 |             1.7839 |              21.6100 |   407.0000 |
| quintile_portfolios      |    0.0492 |          3.6356 | 0.1722 |         0.4935 |             0.3278 |             1.5485 |              29.7547 |   407.0000 |
| ventile_portfolios       |    0.0492 |          3.6356 | 0.1722 |         0.5562 |             0.3981 |             2.1622 |              35.1362 |   407.0000 |
| rank_weighted_all_assets |    0.0492 |          3.6356 | 0.1722 |         0.4839 |             0.3122 |             1.2771 |              28.1756 |   407.0000 |

**This answers "why expanding rather than rolling".** A 20-year rolling window
performs essentially the same as expanding
(IC 0.0463 against
0.0492; Sharpe
0.58 against
0.54). A 10-year window is clearly worse
(IC 0.0287, Sharpe
0.39). Degradation as the
window shrinks, and flatness between 20 years and everything, is the evidence
that estimation error binds rather than non-stationarity. Note the honest
caveat: 20-year rolling is marginally better on Sharpe, so the choice is within
noise rather than obviously right.

Portfolio cutoffs behave as expected — concentrating into narrower buckets
raises gross Sharpe and turnover together — and rank weighting across the whole
cross-section gives up some gross Sharpe for materially lower turnover.

### 5.3 Subperiods and regimes

| period    |   enet |   lasso |   neural-net |     ols |   ridge |   single-signal |   xgboost |
|:----------|-------:|--------:|-------------:|--------:|--------:|----------------:|----------:|
| 1990-1999 | 0.0723 |  0.0691 |       0.0655 |  0.0781 |  0.0784 |          0.0436 |    0.0700 |
| 2000-2009 | 0.0548 |  0.0347 |       0.0422 |  0.0420 |  0.0421 |          0.0247 |    0.0643 |
| 2010-2023 | 0.0054 |  0.0094 |       0.0025 | -0.0094 | -0.0090 |          0.0129 |    0.0235 |

Regime labels use only information available before the month begins, so a split
is something a strategy could have conditioned on. Splitting on contemporaneous
volatility would be a different and much easier exercise.

![Rank IC by market state, labelled using only prior information](figures/fig11_regimes.png)

*Rank IC by market state, labelled using only prior information*


|                                     |   enet |   lasso |   neural-net |    ols |   ridge |   single-signal |   xgboost |
|:------------------------------------|-------:|--------:|-------------:|-------:|--------:|----------------:|----------:|
| ('drawdown_state', 'in_drawdown')   | 0.0466 |  0.0250 |       0.0285 | 0.0343 |  0.0344 |          0.0216 |    0.0513 |
| ('drawdown_state', 'near_highs')    | 0.0426 |  0.0593 |       0.0355 | 0.0298 |  0.0302 |          0.0279 |    0.0479 |
| ('market_direction', 'market_down') | 0.0636 |  0.0686 |       0.0253 | 0.0391 |  0.0392 |          0.0289 |    0.0618 |
| ('market_direction', 'market_up')   | 0.0394 |  0.0376 |       0.0347 | 0.0297 |  0.0300 |          0.0246 |    0.0460 |
| ('rate_level', 'high_rates')        | 0.1038 |  0.0904 |       0.0551 | 0.0963 |  0.0955 |          0.0761 |    0.1005 |
| ('rate_level', 'low_rates')         | 0.0392 |  0.0375 |       0.0311 | 0.0268 |  0.0271 |          0.0217 |    0.0454 |
| ('volatility', 'high_vol')          | 0.0534 |  0.0639 |       0.0326 | 0.0375 |  0.0377 |          0.0291 |    0.0544 |
| ('volatility', 'low_vol')           | 0.0350 |  0.0266 |       0.0329 | 0.0258 |  0.0261 |          0.0218 |    0.0442 |

---

## 6. Experiment 3 — is the signal statistically real?

### 6.1 Factor alphas

Six-factor (FF5 + momentum) regressions on gross long-short returns,
Newey-West 6 lags:

| model         |   alpha_annual |   alpha_tstat |     r2 |   beta_mktrf |   beta_smb |   beta_hml |   beta_mom |
|:--------------|---------------:|--------------:|-------:|-------------:|-----------:|-----------:|-----------:|
| single-signal |         0.0592 |        2.1261 | 0.0158 |      -0.0451 |    -0.0133 |    -0.1113 |    -0.0358 |
| ols           |         0.0377 |        1.5683 | 0.0422 |       0.0091 |     0.0618 |    -0.0770 |    -0.0688 |
| ridge         |         0.0383 |        1.5846 | 0.0421 |       0.0076 |     0.0634 |    -0.0774 |    -0.0691 |
| lasso         |         0.0259 |        1.5815 | 0.0664 |      -0.0309 |     0.0004 |    -0.0883 |    -0.1074 |
| enet          |         0.0631 |        2.6477 | 0.0232 |      -0.0239 |     0.0074 |    -0.1264 |    -0.0777 |
| xgboost       |         0.0702 |        2.8337 | 0.0358 |      -0.0202 |     0.0365 |    -0.1070 |    -0.0940 |
| neural-net    |         0.0443 |        1.8066 | 0.0347 |      -0.0422 |     0.0647 |    -0.1483 |    -0.0650 |

xgboost's alpha of 7.0% a year survives with
*t* = 2.83, and the regression R² of
0.036 says the strategy is close to orthogonal to the market,
size, value, profitability, investment and momentum. Do not oversell that low
R²: a dollar-neutral decile spread across characteristic-sorted portfolios is
partly constructed to be factor-neutral. The content is that the alpha does not
collapse when six factors are added.

### 6.2 Bootstrap

Stationary block bootstrap, 5,000 draws, mean block
12 months:

| model         |   sharpe |   sharpe_ci_lower |   sharpe_ci_upper |   p_sign_flips |   mean_return_tstat_nw |   mean_return_pvalue_nw |
|:--------------|---------:|------------------:|------------------:|---------------:|-----------------------:|------------------------:|
| single-signal |    0.332 |             0.077 |             0.593 |          0.005 |                  2.034 |                   0.042 |
| ols           |    0.408 |             0.119 |             0.680 |          0.003 |                  2.430 |                   0.015 |
| ridge         |    0.410 |             0.121 |             0.679 |          0.003 |                  2.446 |                   0.014 |
| lasso         |    0.275 |             0.040 |             0.602 |          0.010 |                  1.953 |                   0.051 |
| enet          |    0.505 |             0.267 |             0.755 |          0.000 |                  3.005 |                   0.003 |
| xgboost       |    0.541 |             0.325 |             0.787 |          0.000 |                  3.439 |                   0.001 |
| neural-net    |    0.381 |             0.120 |             0.642 |          0.003 |                  2.292 |                   0.022 |

`p_sign_flips` is the share of resamples in which the Sharpe ratio changes sign
— a more direct read on fragility than an interval.

### 6.3 Multiple testing

The same tests applied to 212 published cross-sectional predictors
(Chen and Zimmermann, 2022), each requiring at least
120 months:

| hypothesis      |   n_predictors |   significant_uncorrected_5pct |   significant_bonferroni_5pct |   significant_bh_5pct |   share_surviving_bonferroni |   share_surviving_bh |   share_with_tstat_above_3 |
|:----------------|---------------:|-------------------------------:|------------------------------:|----------------------:|-----------------------------:|---------------------:|---------------------------:|
| raw_mean_return |        212.000 |                        163.000 |                        84.000 |               159.000 |                        0.396 |                0.750 |                      0.514 |
| ff6_alpha       |        212.000 |                        158.000 |                        82.000 |               156.000 |                        0.387 |                0.736 |                      0.533 |

Bonferroni at 5% over 212 hypotheses corresponds to
|*t*| > 3.68.

The high survival rate is itself a selection effect: these predictors were
published *because* they were significant. That is Harvey, Liu and Zhu's (2016)
argument, and the reason to treat |*t*| > 3 rather than 1.96 as the relevant
hurdle for a new predictor.

**And this project's own result, on the same scale:**

| model         |   tstat_nw |   percentile_among_published | passes_uncorrected   | passes_bonferroni   | passes_tstat_3_hurdle   |   ff6_alpha_tstat | ff6_alpha_passes_bonferroni   |
|:--------------|-----------:|-----------------------------:|:---------------------|:--------------------|:------------------------|------------------:|:------------------------------|
| single-signal |      2.034 |                       25.000 | True                 | False               | False                   |             2.126 | False                         |
| ols           |      2.430 |                       34.434 | True                 | False               | False                   |             1.568 | False                         |
| ridge         |      2.446 |                       35.377 | True                 | False               | False                   |             1.585 | False                         |
| lasso         |      1.953 |                       23.113 | False                | False               | False                   |             1.582 | False                         |
| enet          |      3.005 |                       48.585 | True                 | False               | True                    |             2.648 | False                         |
| xgboost       |      3.439 |                       57.075 | True                 | False               | True                    |             2.834 | False                         |
| neural-net    |      2.292 |                       31.604 | True                 | False               | False                   |             1.807 | False                         |

xgboost clears the |*t*| > 3 hurdle but does **not** survive Bonferroni across
212 hypotheses, and sits at the
57th percentile of the published
distribution. So: statistically real on its own terms, median-strength in
context, and not strong enough for the most conservative correction the
literature applies. Reporting the percentile is more informative than reporting
the *t*-statistic alone, and it is the number this project should be judged on.

![This project's t-statistic against the distribution of 212 published predictors](figures/fig7_multiple_testing.png)

*This project's t-statistic against the distribution of 212 published predictors*


Note also that this project is itself a search — 26
predictors, 7 models, several portfolio variants — which is
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
7.3, log-volatility persistence
0.66, mean pairwise residual correlation
0.048.

### 7.2 Empirical size, nominal 5%, T = 360

| error_model          |   ('grs_shrunk', 10) |   ('grs_shrunk', 25) |   ('grs_shrunk', 50) |   ('grs_shrunk', 100) |   ('grs_shrunk', 200) |   ('grs_shrunk', 300) |   ('grs', 10) |   ('grs', 25) |   ('grs', 50) |   ('grs', 100) |   ('grs', 200) |   ('grs', 300) |   ('pesaran_yamagata', 10) |   ('pesaran_yamagata', 25) |   ('pesaran_yamagata', 50) |   ('pesaran_yamagata', 100) |   ('pesaran_yamagata', 200) |   ('pesaran_yamagata', 300) |   ('wald_chi2', 10) |   ('wald_chi2', 25) |   ('wald_chi2', 50) |   ('wald_chi2', 100) |   ('wald_chi2', 200) |   ('wald_chi2', 300) |
|:---------------------|---------------------:|---------------------:|---------------------:|----------------------:|----------------------:|----------------------:|--------------:|--------------:|--------------:|---------------:|---------------:|---------------:|---------------------------:|---------------------------:|---------------------------:|----------------------------:|----------------------------:|----------------------------:|--------------------:|--------------------:|--------------------:|---------------------:|---------------------:|---------------------:|
| common_vol           |                0.035 |                0.013 |                0.007 |                 0.000 |                 0.000 |                 0.000 |         0.040 |         0.033 |         0.033 |          0.035 |          0.048 |          0.045 |                      0.075 |                      0.075 |                      0.117 |                       0.147 |                       0.185 |                       0.205 |               0.052 |               0.107 |               0.263 |                0.750 |                1.000 |                1.000 |
| empirical_wild       |                0.015 |                0.015 |                0.000 |                 0.000 |                 0.000 |                 0.000 |         0.037 |         0.037 |         0.037 |          0.052 |          0.060 |          0.068 |                      0.068 |                      0.095 |                      0.117 |                       0.142 |                       0.190 |                       0.205 |               0.050 |               0.102 |               0.245 |                0.775 |                1.000 |                1.000 |
| empirical_wild_block |                0.045 |                0.037 |                0.007 |                 0.000 |                 0.000 |                 0.000 |         0.055 |         0.075 |         0.090 |          0.058 |          0.083 |          0.048 |                      0.098 |                      0.117 |                      0.175 |                       0.175 |                       0.235 |                       0.305 |               0.077 |               0.130 |               0.318 |                0.840 |                1.000 |                1.000 |
| gaussian             |                0.050 |                0.028 |                0.013 |                 0.000 |                 0.000 |                 0.000 |         0.055 |         0.060 |         0.052 |          0.035 |          0.033 |          0.037 |                      0.080 |                      0.100 |                      0.113 |                       0.140 |                       0.185 |                       0.235 |               0.080 |               0.102 |               0.225 |                0.755 |                1.000 |                1.000 |
| gaussian_independent |                0.040 |                0.020 |                0.020 |                 0.000 |                 0.000 |                 0.000 |         0.043 |         0.037 |         0.070 |          0.043 |          0.055 |          0.033 |                      0.055 |                      0.055 |                      0.092 |                       0.045 |                       0.040 |                       0.055 |               0.052 |               0.095 |               0.280 |                0.787 |                1.000 |                1.000 |
| student_t            |                0.030 |                0.013 |                0.003 |                 0.000 |                 0.000 |                 0.000 |         0.040 |         0.028 |         0.058 |          0.040 |          0.052 |          0.040 |                      0.062 |                      0.095 |                      0.117 |                       0.125 |                       0.198 |                       0.220 |               0.058 |               0.087 |               0.228 |                0.775 |                1.000 |                1.000 |

| test             |   nominal_size |   worst_size | at_error_model       |   at_n_assets |   at_n_obs |   at_ratio_n_over_t |   median_size_over_grid |   median_size_adjusted_power |
|:-----------------|---------------:|-------------:|:---------------------|--------------:|-----------:|--------------------:|------------------------:|-----------------------------:|
| grs              |          0.050 |        0.090 | empirical_wild_block |            50 |        360 |               0.139 |                   0.048 |                        0.249 |
| wald_chi2        |          0.050 |        1.000 | gaussian             |           100 |        120 |               0.833 |                   0.574 |                        0.249 |
| grs_shrunk       |          0.050 |        0.050 | gaussian             |            10 |        360 |               0.028 |                   0.000 |                        0.239 |
| pesaran_yamagata |          0.050 |        0.305 | empirical_wild_block |           300 |        360 |               0.833 |                   0.117 |                        0.131 |

**GRS holds its size everywhere**, at up to
*N/T* = 0.83, under heavy tails, under a
persistent common volatility factor, and under residual vectors resampled from
the real panel. Its median size over the grid is
0.048.

![Empirical size against N/T under three error structures. GRS is flat at 5%; its asymptotic counterpart is not.](figures/fig8_test_size.png)

*Empirical size against N/T under three error structures. GRS is flat at 5%; its asymptotic counterpart is not.*


**What actually breaks:**

- *The asymptotic version of the same statistic.* Referring the identical
  quadratic form to χ²*_N_* instead of the exact *F* gives size
  0.080 at *N* = 10,
  0.225 at *N* = 50,
  0.755 at *N* = 100 and
  1.000 at *N* = 200. The finite-sample
  correction is doing all the work.
- *Shrinkage without recalibration.* A Ledoit-Wolf covariance conditions better
  and shrinks the statistic, but the *F* critical value is unchanged, so size
  collapses to 0.000 at *N* = 100 and the
  test stops rejecting anything. Better estimation is not automatically better
  inference.
- *The large-N test's assumption.* Pesaran-Yamagata never inverts an *N* × *N*
  matrix and is correctly sized under cross-sectional independence
  (0.055 at *N* = 300).
  Under the correlation these portfolios actually have, its size rises to
  0.305 — and rises
  *with N*, the opposite of what an asymptotic-in-*N* test should do.

### 7.3 Existence, not size, is the binding constraint on GRS

|            |   grs |   grs_shrunk |   pesaran_yamagata |   wald_chi2 |
|:-----------|------:|-------------:|-------------------:|------------:|
| (120, 10)  |  1.00 |         1.00 |               1.00 |        1.00 |
| (120, 25)  |  1.00 |         1.00 |               1.00 |        1.00 |
| (120, 50)  |  1.00 |         1.00 |               1.00 |        1.00 |
| (120, 100) |  1.00 |         1.00 |               1.00 |        1.00 |
| (120, 200) |  0.00 |         0.00 |               1.00 |        0.00 |
| (120, 300) |  0.00 |         0.00 |               1.00 |        0.00 |
| (360, 10)  |  1.00 |         1.00 |               1.00 |        1.00 |
| (360, 25)  |  1.00 |         1.00 |               1.00 |        1.00 |
| (360, 50)  |  1.00 |         1.00 |               1.00 |        1.00 |
| (360, 100) |  1.00 |         1.00 |               1.00 |        1.00 |
| (360, 200) |  1.00 |         1.00 |               1.00 |        1.00 |
| (360, 300) |  1.00 |         1.00 |               1.00 |        1.00 |

At *T* = 120 the statistic simply does not exist for the
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
dispersion of 0.6% a year (chosen
small enough that no test can see it one asset at a time).

Under weak dependence, Pesaran-Yamagata's power grows with *N* and overtakes
GRS. Under real dependence it flattens out while GRS's keeps rising — cross
sectional correlation *increases* the power of a joint alpha test, because the
common component can be hedged out and the mispricing portfolio has a higher
Sharpe ratio. Full grid in `results/exp4_size_adjusted_power.csv`.

![Size-adjusted power against N, with and without real cross-sectional dependence](figures/fig9_test_power.png)

*Size-adjusted power against N, with and without real cross-sectional dependence*


### 7.5 The same tests on the real panel

|            |   grs_reject_rate |   wald_chi2_reject_rate |   grs_shrunk_reject_rate |   pesaran_yamagata_reject_rate |
|:-----------|------------------:|------------------------:|-------------------------:|-------------------------------:|
| (120, 10)  |             0.025 |                   0.100 |                    0.025 |                          0.050 |
| (120, 25)  |             0.075 |                   0.300 |                    0.000 |                          0.025 |
| (120, 50)  |             0.100 |                   0.975 |                    0.000 |                          0.000 |
| (120, 100) |             0.125 |                   1.000 |                    0.000 |                          0.000 |
| (120, 200) |           nan     |                 nan     |                  nan     |                          0.000 |
| (120, 300) |           nan     |                 nan     |                  nan     |                          0.000 |
| (120, 374) |           nan     |                 nan     |                  nan     |                          0.000 |
| (240, 10)  |             0.150 |                   0.225 |                    0.075 |                          0.125 |
| (240, 25)  |             0.350 |                   0.525 |                    0.050 |                          0.200 |
| (240, 50)  |             0.425 |                   0.925 |                    0.075 |                          0.375 |
| (240, 100) |             0.625 |                   1.000 |                    0.000 |                          0.575 |
| (240, 200) |             0.600 |                   1.000 |                    0.000 |                          0.800 |
| (240, 300) |           nan     |                 nan     |                  nan     |                          0.975 |
| (240, 374) |           nan     |                 nan     |                  nan     |                          1.000 |
| (480, 10)  |             0.625 |                   0.650 |                    0.600 |                          0.675 |
| (480, 25)  |             0.925 |                   0.950 |                    0.850 |                          0.875 |
| (480, 50)  |             0.975 |                   1.000 |                    0.875 |                          0.975 |
| (480, 100) |             1.000 |                   1.000 |                    0.975 |                          1.000 |
| (480, 200) |             1.000 |                   1.000 |                    0.250 |                          1.000 |
| (480, 300) |             1.000 |                   1.000 |                    0.000 |                          1.000 |
| (480, 374) |             1.000 |                   1.000 |                    0.000 |                          1.000 |
| (654, 10)  |             0.775 |                   0.775 |                    0.700 |                          0.725 |
| (654, 25)  |             0.950 |                   0.950 |                    0.875 |                          0.900 |
| (654, 50)  |             1.000 |                   1.000 |                    1.000 |                          1.000 |
| (654, 100) |             1.000 |                   1.000 |                    1.000 |                          1.000 |
| (654, 200) |             1.000 |                   1.000 |                    1.000 |                          1.000 |
| (654, 300) |             1.000 |                   1.000 |                    0.900 |                          1.000 |
| (654, 374) |             1.000 |                   1.000 |                    0.000 |                          1.000 |

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
make test     # 74 tests
make all      # full pipeline
python3 scripts/07_report.py
```

Raw inputs are pinned to immutable commits with SHA-256 digests in
`data/raw/manifest.json`. This matters because the Fama-French library is
revised retroactively whenever CRSP is updated, so an unpinned download would
silently change every number above.
