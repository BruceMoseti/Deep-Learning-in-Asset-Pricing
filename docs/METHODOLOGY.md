# Methodology

Every choice below changes a reported number. Each one states what was chosen,
why, and what the alternative would have done — measured where that was
possible rather than asserted.

---

## 1. What is being predicted

**Target.** The next month's excess return, cross-sectionally demeaned and
scaled by the cross-sectional standard deviation within that month.

**Why relative rather than absolute.** The variance of a stock's monthly return
is dominated by the market, and the market's monthly return is close to
unforecastable. A model asked to predict absolute returns spends its capacity
on that component and is graded mostly on it. Demeaning within a month removes
it and leaves the quantity a long-short portfolio actually trades: which assets
outperform the rest of the cross-section. It also means the forecast is
automatically dollar-neutral in construction, so no market-timing bet is
smuggled into a result that claims to be about the cross-section.

**Why standardised.** Cross-sectional return dispersion varies by a factor of
several between calm and turbulent months. Without scaling, the loss function
is dominated by a handful of high-dispersion months — the fit is effectively
estimated on 2008 and a few of its neighbours. Scaling makes each month
contribute comparably.

**What this costs.** The model no longer forecasts a return in percent, so its
output is not directly a dollar expectation, and out-of-sample \(R^2\) is
measured against a standardised target rather than a raw one. Both are stated
where they matter. `--target rank` runs the whole pipeline on a rank-transformed
target instead, as a robustness check.

---

## 2. Predictors

Twenty-six predictors in nine groups, all computed from returns realised up to
and including the formation month: multi-horizon momentum, short- and long-term
reversal, several volatility measures, market beta and comovement, coskewness,
higher moments, drawdown and trend, annual seasonality, return persistence, and
three static labels describing the asset's place in its sort.

**Rank transform.** Each predictor is mapped to \([-1, 1]\) by its
cross-sectional rank within the month, following Gu, Kelly and Xiu (2020). This
is scale-free, insensitive to the heavy tails and level shifts that a rolling
z-score would propagate forward, and — importantly — uses only information from
within that one month, so it cannot leak. Missing values map to 0, the centre of
the cross-section.

**No accounting predictors.** The cross-section here is characteristic-sorted
portfolios, not individual firms, so book-to-market, profitability, accruals and
the rest are not observable per asset. They enter only through the static sort
labels. This is the main substantive limitation of the study and Section 8
covers it.

---

## 3. Point-in-time discipline

A row keyed `(month = t, asset = i)` holds predictors computed from returns
through month `t` and the excess return of month `t+1` as its label. A portfolio
formed at the close of `t` earns that label.

This is enforced by tests, not by inspection (`tests/test_no_lookahead.py`):

- **Future-perturbation test.** Returns after a cutoff date are replaced with
  noise; every predictor dated on or before the cutoff must be bit-for-bit
  unchanged. This is the definition of point-in-time, checked directly.
- **Label alignment.** For 200 randomly drawn rows, the stored label must equal
  the asset's excess return in the following month.
- **Shuffled-label control.** Permuting labels within each month must drive
  measured skill inside its own sampling error. If information were leaking from
  label to predictor anywhere in the pipeline, this control would still show
  skill.
- **Leak-sensitivity control.** A deliberate leak must produce an obviously
  larger information coefficient, which establishes that the metric is capable
  of detecting one — without this, a null control result means nothing.
- **Within-month transform test.** Perturbing one month's cross-section must not
  move any other month's ranks.

---

## 4. Validation scheme

**Expanding window, refit once per calendar year.** For test year `Y`:
validation is the five calendar years before `Y`, training is everything before
that, and the training block grows by one year per step.

**Why not k-fold cross-validation.** Two independent reasons, either sufficient.
Random splits train on months that follow the months they test on. And returns
within a month are strongly cross-correlated, so placing some assets from month
`t` in the training fold and others in the test fold leaks that month's common
shock across the split — even if the dates were respected.

**Why expanding rather than rolling.** The relations are weak and the monthly
cross-section is noisy, so estimation error, not non-stationarity, is expected
to bind; discarding old data makes that worse. This was tested rather than
assumed (`results/exp5_design_robustness.csv`): a 20-year rolling window gives
essentially the same result (rank IC 0.046 versus 0.049, Sharpe 0.58 versus
0.54), while a 10-year window is materially worse (IC 0.029, Sharpe 0.39). The
degradation at ten years is the evidence that estimation error dominates.

**Embargo.** A row dated `t` carries the return of `t+1`, so without care the
last training row's label falls inside the validation block and the last
validation row's label falls inside the test year. One month is dropped from the
end of the training and validation blocks. The gap is small but it is exactly
the kind of one-month leak that quietly flatters a hyper-parameter choice.

**Hyper-parameters.** Selected inside each model's `fit` on the validation
block, by mean squared error, and never using the test year. Model-specific
tuning lives with the model, so the harness cannot leak by construction.

---

## 5. Models

A ladder of increasing flexibility, so the question "does complexity pay?" has
an answer at every step rather than only at the ends.

| Model | Tuned on validation |
| --- | --- |
| Single characteristic (12-month momentum rank) | sign only, on the training block |
| OLS | nothing |
| Ridge, Lasso, Elastic Net | penalty (and mixing weight) |
| XGBoost | depth, shrinkage, early-stopped tree count |
| Neural network | architecture, weight decay; then ensembled over 3 seeds |

**Why a single-characteristic baseline.** Without it, "XGBoost achieves a rank
IC of 0.049" has no scale. It turns out to matter: the baseline has the *highest*
break-even cost of any model, because it trades least.

**Why the network is ensembled.** A single network's out-of-sample forecast on
data this noisy is dominated by initialisation variance. Averaging over seeds is
standard for this reason and is not a way of quietly running more models.

---

## 6. Portfolio construction and costs

**Weights.** Decile long-short, equal-weighted within each leg, `+1` notional
long and `-1` short. Gross exposure 2, net 0. The reported return is the
long-short spread per dollar of each leg — not a return on posted margin, which
would be a leverage assumption presented as performance.

**Turnover.** Two-sided, \(\sum_i |w_{i,t} - w_{i,t-1}|\). Replacing both legs
in full gives 4, since each leg is both exited and entered. Positions are not
drifted between rebalances, which slightly overstates turnover — the direction
an honest cost estimate should err in.

**Costs.** Turnover times a one-way rate, at 0, 5, 10 and 20 basis points. The
headline number is the **break-even cost**: the rate at which the average gross
return is exactly consumed. It converts a Sharpe ratio into a statement about
how much slippage the signal can absorb, which is the question that decides
whether a signal is tradable.

**Why ranks feed the portfolio.** Each model's forecast is converted to a
within-month rank before weights are formed. Differences in forecast *scale*
come from how hard a model shrinks, not from what it knows, and would otherwise
change the portfolio comparison for no informational reason.

---

## 7. Inference

**Newey-West, 6 lags,** for the mean of any monthly series. Monthly strategy
returns and IC series are serially dependent; an i.i.d. t-statistic overstates
precision. Verified in tests to inflate the standard error of a persistent
series and to reduce to the sample variance at zero lags.

**Factor regressions** against CAPM, Fama-French five-factor, and five-factor
plus momentum, with Newey-West standard errors. The intercept is what the factor
model cannot explain — the number that separates a discovery from a repackaging
of premia already for sale.

**Stationary block bootstrap** (Politis-Romano, mean block 12 months, 5000
draws) for the Sharpe ratio, which is not a sample mean and so has no clean
analytic standard error. Reported alongside the fraction of resamples in which
the sign flips, which is a more direct read on fragility than a confidence
interval. Interval coverage is verified by a 120-trial coverage study, not by
checking that one interval happens to bracket the truth.

**Multiple testing.** Bonferroni and Benjamini-Hochberg over 212 published
predictors, plus the Harvey-Liu-Zhu \(|t| > 3\) hurdle. The point of applying
this to the published cross-section rather than only to this project's own
strategy is that it puts the strategy's t-statistic on a scale: its percentile
among predictors that have already been through the same filter.

---

## 8. What this design cannot do

Stated plainly, because these are the first things a reader should ask.

**The cross-section is portfolios, not firms.** 374 characteristic-sorted
portfolios of US stocks, which is a standard test-asset universe, but it is not
a firm-level panel. Nonlinear interactions *between firm characteristics* — the
mechanism Gu, Kelly and Xiu identify as the main source of machine-learning
gains — largely cannot be represented here, because accounting characteristics
are not observable per asset. The result is therefore a lower bound on what the
same pipeline would find on firm data, and a statement about a different (and
easier to trade) universe. `src/xsap/data/` is structured so a firm-level panel
drops in behind the same interface; see the README.

**Portfolio returns are cleaner than firm returns.** No delistings, no
survivorship decisions, no microcap illiquidity, no shorting constraints on
individual names. Transaction costs on these portfolios would in reality be
higher than a flat basis-point rate implies, because the underlying rebalancing
is itself costly. The break-even figures should be read as generous.

**Costs are modelled, not measured.** A flat one-way rate ignores market impact,
which scales with size, and spread variation across time and across assets. The
cost scenarios are a sensitivity analysis, not a fill simulation.

**Data vintage.** The Fama-French library is revised retroactively whenever CRSP
is updated, so results depend on the vintage. Every file is pinned to an
immutable commit and its SHA-256 recorded in `data/raw/manifest.json`.

**Researcher degrees of freedom in this project itself.** Twenty-six predictors,
seven models, one universe, one target definition, several portfolio variants.
That is a search, even though each individual choice was made for a stated
reason. This is why Experiment 3 compares the result to the distribution of
*published* predictors instead of to zero, and why the Bonferroni verdict is
reported even though it is unflattering.
