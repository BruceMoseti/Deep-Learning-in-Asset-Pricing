# Discussion

Objections to this study, and what the data says about them. Every figure
traces to a file in `results/`; re-derive them after changing the pipeline
rather than trusting this document.

---

## Validation

### Random cross-validation would use the data more efficiently

It would also invalidate the result, for two independent reasons.

Random folds train on months that come after the months they test on, so the
model is told what happened next.

Less obviously, returns within a month are strongly cross-correlated. A split
that respected dates but mixed assets within a month would still leak: put half
of March 1998's assets in training and half in test, and the model learns March
1998's common shock from the training half and is graded on the test half. In
this panel the average pairwise residual correlation is 0.048 after removing six
factors, with the 90th percentile at 0.22, so that common component is not
negligible.

The split is therefore by time and by whole months. And because a row dated `t`
carries the return of `t+1`, one month is dropped from the end of the training
and validation blocks; otherwise the last training label sits inside the
validation window and the last validation label sits inside the test year.

### An expanding window mixes regimes; a rolling window would be safer

The two concerns pull in opposite directions. Expanding windows risk fitting a
market that no longer exists; rolling windows discard history, and these
relations are weak enough that estimation error is already the binding
constraint.

The measurement decides it (`results/exp5_design_robustness.csv`). A 20-year
rolling window performs essentially the same as expanding — rank IC 0.046
against 0.049, Sharpe 0.58 against 0.54. A 10-year window is clearly worse: IC
0.029, Sharpe 0.39. Performance degrades as the window shrinks and is flat
between 20 years and the full history, which is what one would expect if
estimation error dominates and there is little exploitable non-stationarity at
this horizon.

The honest caveat is that 20-year rolling is marginally better on Sharpe. The
choice is within noise, not obviously right.

### Predicting cross-sectional ranks rather than returns discards information

The information it discards is the market return, which dominates the variance
of a stock's monthly return and is close to unforecastable at that horizon.
Regressing raw returns on characteristics spends most of the fitted variance,
and most of the loss function, on a component about which the model has no view.
Demeaning within the month removes it and leaves what a long-short book trades.

It also makes the forecast dollar-neutral by construction, so a market-timing
bet cannot be reported as a cross-sectional finding.

The cost is real: the output is no longer a return in percent, so it is not
directly a dollar expectation, and out-of-sample R² is measured against a
standardised target. Both are reported, and `--target rank` re-runs the pipeline
on a rank-transformed target as a robustness check.

### Point-in-time discipline is asserted rather than demonstrated

It is tested. `tests/test_no_lookahead.py`:

- Replaces every return after a cutoff date with noise; every predictor dated on
  or before the cutoff must come out bit-for-bit identical.
- Checks, for 200 random rows, that the stored label equals that asset's excess
  return in the following month.
- Permutes labels within each month and requires measured skill to fall inside
  its own sampling error. If anything leaked from label to predictor anywhere in
  the pipeline, this control would still show skill.
- Introduces a deliberate leak and requires it to produce an obviously larger
  information coefficient. Without this, a null control result would only
  establish that the metric is blind.

---

## Results

### A negative out-of-sample R² contradicts a positive information coefficient

They measure different things, and the gap is diagnosable rather than
contradictory. The information coefficient asks whether the ordering carries
information. R² asks whether the forecast is the right size, and squared
error punishes over-scaling hard.

A calibration regression separates the two: regress the outcome on the forecast
out of sample and read the slope (`results/exp1_forecast_accuracy.csv`,
`reports/figures/fig2_calibration_gap.png`). Ridge's slope is 0.31, so its
forecasts are roughly three times too large. Gradient boosting's is 0.67 and its
R²(OOS) is positive. The better-calibrated models are precisely those with
positive R².

The portfolio uses only the ranking, so the information coefficient is the
operative metric here — but reporting it alone would have concealed a genuine
miscalibration.

### The nonlinear models won, so complexity pays

The point estimates support that reading and a paired test does not.

Rank IC rises monotonically: 0.025 for a single momentum characteristic, 0.032
for ridge, 0.042 for lasso, 0.049 for gradient boosting. Because every model
faces the same cross-section each month, their monthly ICs can be differenced
pairwise; the common component cancels and the resulting test is far tighter
than comparing two standard errors (`results/exp1_model_comparisons.csv`):

| Comparison | IC gap | *t* | Squared-error *p* |
| --- | --- | --- | --- |
| Boosting vs ridge | +0.018 | 2.4 | 0.000 |
| Boosting vs elastic net | +0.006 | 1.5 | 0.77 |
| Network vs elastic net | -0.012 | -1.5 | 0.009 |

Boosting beats ridge. It does not beat elastic net, the best linear model. The
neural network is significantly worse than elastic net on squared error. And no
adjacent rung of the ladder is significant at all — the only significant
adjacent comparison is the network losing to boosting.

So almost the whole apparent gain from complexity is the sparse regularisation
in the middle of the ladder, not the nonlinearity on top. Ridge selects a
penalty small enough to reproduce OLS, which is unsurprising with 26 predictors
and 249,000 observations: there is no ill-conditioning for shrinkage to fix.
Variable selection is a real restriction and it pays out of sample.

Stated carefully: nonlinear models improved average accuracy, but the
improvement over the best regularised linear model is not statistically
distinguishable, and the additional flexibility of a neural network hurt.

### The neural network was under-trained or badly specified

Possibly, but the more likely explanation is the data. Roughly 250,000
asset-months, 26 predictors, and a signal explaining a fraction of a percent of
variance. Boosting at depth 2 to 4 fits shallow, low-order interactions, which
is about the structure this panel supports. A network must learn its own
representation from the same thin signal, and initialisation variance is large
relative to that signal — which is why forecasts are averaged over seeds, and
why a single network would have been misleading in either direction.

On a firm-level panel with hundreds of characteristics, Gu, Kelly and Xiu (2020)
find networks perform best. This cross-section is 374 portfolios with
return-based predictors only, so most of the high-order interaction structure
that would favour a network is not present to be found. The result is about this
data, not about networks.

### The best model is whichever has the highest information coefficient

The ranking changes twice depending on the criterion.

By raw accuracy, gradient boosting (IC 0.049). By accuracy net of its own
standard error, elastic net — boosting's advantage over it is not
distinguishable, and elastic net is the simpler model. By tradability, neither
of the two most accurate linear models: ridge and OLS reach gross Sharpe ratios
around 0.41 but turn over 2.3 times the book per month, break even at about 18
basis points one-way, and turn negative by 20.

The single-characteristic baseline — 12-month momentum — has the highest
break-even cost of anything fitted here, about 42 basis points, because it
trades least, despite having nearly the lowest gross Sharpe. That reversal only
appears if turnover is charged for before models are ranked.

### Transaction costs were assumed away

They were varied, and the variation is reported as the headline. Boosting: gross
Sharpe 0.54, 0.37 at 10 basis points, 0.21 at 20. Ridge: 0.41 gross, 0.19 at 10,
negative at 20.

The number worth quoting is the break-even cost, because a Sharpe ratio at an
assumed cost level buries the assumption. Break-even turns it into a tractable
question: can a monthly rebalance of this book be executed inside 32 basis
points one-way?

What is still missing: market impact, which scales with size, and optimisation
against a turnover penalty rather than measurement of turnover after the fact.
The second would plausibly matter more than any change to the models, since the
gross-accuracy and net-of-cost rankings already disagree.

### The performance is a repackaging of known factors

Gradient boosting's gross long-short return carries a six-factor alpha of 7.0% a
year with t = 2.83, and the factor-regression R² is only 0.036 — market
beta is -0.02, and the strategy is close to orthogonal to size, value,
profitability, investment and momentum.

That low R² should not be oversold. A dollar-neutral decile spread across
characteristic-sorted portfolios is partly constructed to be factor-neutral. The
substantive content is that alpha does not collapse when the six factors are
added.

### One strategy tested once tells you nothing about how many were tried

Correct, which is why the multiple-testing analysis is applied to the published
cross-section rather than only to this strategy.

The same tests were run across 212 published cross-sectional predictors from the
Chen-Zimmermann dataset. Of those, 163 are significant uncorrected at 5%, 159
survive Benjamini-Hochberg, only 84 survive Bonferroni, and 51% clear the
Harvey-Liu-Zhu |t| > 3 hurdle.

Placed on that scale, boosting's t = 3.43 clears the |t| > 3 hurdle but
does not survive a Bonferroni correction across 212 hypotheses, and sits at
roughly the 57th percentile of the published distribution. Statistically real on
its own terms; median-strength in context; not strong enough for the most
conservative correction the literature applies.

One subtlety: the high survival rate among published predictors is itself a
selection effect, since they were published because they were significant. That
is the Harvey-Liu-Zhu argument, and the reason to treat 3.0 rather than 1.96 as
the relevant hurdle for a new predictor.

This project is also a search — 26 predictors, seven models, several portfolio
variants — which is why the comparison is against the published distribution
rather than against zero.

### The model is a black box; there is no way to know what it learned

The ablation refits the entire walk-forward with each predictor group removed,
and again with each group alone. Feature importances are deliberately not used:
they describe a fit, not out-of-sample value.

The trend and drawdown group — distance from the trailing high, max drawdown,
trailing Sharpe — reaches rank IC 0.044 on its own, against 0.049 for all 26
predictors. The signal is concentrated rather than assembled from many weak
pieces.

The two ablation directions disagree, and computing only one would mislead.
Removing momentum costs almost nothing (-0.0020), which a leave-one-out
study alone would read as momentum containing no information. But momentum on
its own delivers 0.0228, second only to trend: it is substitutable, not
uninformative, because the other predictors already span most of what it knows.
Higher moments is the genuine null case — cheap to remove and useless alone.

There is also a cost dimension no importance measure can see. Dropping the
reversal group costs 0.0061 of IC but raises the break-even cost from 32 to 49
basis points, because reversal is what drives the turnover. A group can be
informative and not worth trading.

### The signal decays, so it was probably never there

It does decay: by decade the pattern is strongest in the 1990s and much weaker
in the 2010s, for every model. Two readings are consistent with that, and this
design cannot separate them. Either the relations were arbitraged away as they
became known — the pattern the anomaly-decay literature predicts — or the early
result was partly luck. The second possibility is why the full-sample
*t*-statistic matters more here than the point estimate.

---

## Experiment 4

### Asset-pricing tests must degrade as the number of assets grows

This was the study's own initial hypothesis, and it is wrong. The intuition —
that GRS inverts an *N* × *N* covariance matrix estimated from *T*
observations, so it must degrade as *N/T* → 1 — does not survive simulation.
A test asserting it failed.

GRS is exact in finite samples under its assumptions: normal, homoskedastic,
serially independent residuals with any cross-sectional covariance, for every
*N* ≤ *T* − *K* − 1. It holds its nominal 5% at *N/T* = 0.83, and its size
further survives heavy tails (*t* with 7.3 degrees of freedom, calibrated to
the data), a persistent common volatility factor, and residual vectors
resampled from the real panel. Median size across the whole grid is 0.050.

What does break is different, and more informative:

- **The asymptotic version of the same statistic.** Referring the identical
  quadratic form to χ²(N) rather than the exact *F* gives a rejection
  rate of about 10% at N = 10, 28% at N = 50, 75% at N = 100, and
  100% at N ≥ 200 with T = 360. Same data, same statistic; only the
  reference distribution differs. The finite-sample correction does all the
  work.
- **Shrinkage without recalibration.** A Ledoit-Wolf covariance conditions
  better and shrinks the statistic, but the *F* critical value is unchanged,
  so size collapses to zero and the test rejects nothing. Better estimation is
  not automatically better inference.
- **The large-*N* test's assumption.** Pesaran-Yamagata never inverts an
  *N* × *N* matrix and stays defined when *N* > *T*, and it is correctly
  sized under cross-sectional independence. Under the correlation actually
  present in these portfolios its size reaches 0.28 — and rises with *N*,
  the opposite of what an asymptotic-in-*N* test should do.

The trade-off is therefore not about dimension. GRS needs *N* < *T*;
Pesaran-Yamagata needs weak cross-sectional dependence. Equity portfolios
violate the second and modern cross-sections violate the first.

### The empirical error model is doing the work

It was, twice, before being fixed — and both times it measured the resampling
scheme rather than the data.

Resampling months with replacement repeats months, so the residual covariance
is estimated from fewer distinct observations than the nominal *T*. Size
reached 0.78, which is large enough to look like a finding about real returns.
Sampling without replacement removed the duplication and introduced the
opposite error: the finite-population correction shrinks the variance of the
estimated intercept, and size fell to 0.000.

The scheme now used is a wild bootstrap — real residual vectors with randomly
flipped signs. Every month appears at most once, so nothing is duplicated; sign
flips leave the second moments untouched, so the variance of the intercept
estimate is exactly what the test assumes; the null holds exactly because each
flip is mean zero; and each month keeps its own magnitude and cross-sectional
pattern. Under it, GRS holds its nominal size.

An earlier bug of the same family: the residual pool was not demeaned, so
resampling injected an alpha of about 2% a year and every test correctly
rejected a null that was in fact false. All three are regression tests in
`tests/test_montecarlo.py`.

---

## Scope

### 374 portfolios is a small cross-section for machine learning

Yes, and this is the study's main limitation. It is a standard test-asset
universe, but nonlinear interactions between firm characteristics — the
mechanism most often credited for machine-learning gains in this literature —
largely cannot be represented, because accounting characteristics are not
observable per portfolio. The results are best read as a lower bound on what
this pipeline would find on firm-level data.

CRSP and Compustat are licensed and were not available. `src/xsap/data/`
isolates the universe behind a single function returning a long panel, so a
firm-level panel from Open Source Asset Pricing, Gu-Kelly-Xiu or
Jensen-Kelly-Pedersen substitutes without touching the rest of the pipeline;
adding the accounting groups to `FEATURE_GROUPS` extends the ablation
automatically.

### Portfolio returns are cleaner than the returns anyone actually trades

They are. No delistings, no survivorship decisions, no microcap illiquidity, no
name-level shorting constraints. Real transaction costs on these portfolios
would exceed a flat basis-point rate, because the underlying rebalancing is
itself costly. The break-even figures should be read as generous.

### What would change the conclusions

In order of expected effect: firm-level data, because the universe rather than
the model class is the binding constraint. Then optimisation against a turnover
penalty, since the accuracy and net-performance rankings already disagree. Then
a design capable of separating arbitrage from overfitting in the decay — for
instance, testing whether decay is faster for predictors that received more
academic attention.
