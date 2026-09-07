# Defending this project

Answers to the questions that decide whether a project is yours. Every number
here traces to a file in `results/`; if you change the pipeline, re-derive them
rather than trusting this document.

The general shape of a good answer: state the choice, state the reason, state
what the alternative did when you tried it, and state what you still do not
know. The last part is what separates a researcher from someone presenting a
backtest.

---

## The four questions you must not fumble

### "Why not just use cross-validation?"

Two independent reasons, and you should give both because each alone is fatal.

First, random folds train on months that come *after* the months they test on.
The model is told what happened next.

Second — and this is the one most people miss — returns within a month are
strongly cross-correlated. Even a split that respected dates but mixed assets
within a month would leak: put half of March 1998's assets in training and half
in test, and the model learns March 1998's common shock from the training half
and is graded on the test half. In this panel the average pairwise residual
correlation is 0.048 after removing six factors, with the 90th percentile at
0.22, so the common component is real.

So the split is by time *and* by whole months. And because a row dated `t`
carries the return of `t+1`, one month is dropped from the end of the training
and validation blocks — otherwise the last training label sits inside the
validation window and the last validation label sits inside the test year.

### "Why an expanding window rather than a rolling one?"

Because I tested it rather than guessing. The concern with expanding windows is
non-stationarity: old data may describe a market that no longer exists. The
concern with rolling windows is estimation error: these relations are weak, so
throwing away history makes an already-noisy estimate noisier.

The measurement (`results/exp5_design_robustness.csv`) says estimation error
binds. A 20-year rolling window performs essentially the same as expanding
(rank IC 0.046 versus 0.049; Sharpe 0.58 versus 0.54 — if anything slightly
better on Sharpe). A 10-year window is clearly worse (IC 0.029, Sharpe 0.39).
The fact that performance degrades when the window shrinks, and is flat between
20 years and everything, is the evidence: there is no useful non-stationarity to
exploit at this horizon, and there is a real cost to having less data.

Honest caveat to offer unprompted: 20-year rolling is a little better on Sharpe,
so the choice is not free — it is within noise, not obviously right.

### "Why predict rankings instead of returns?"

Because the return I would be predicting is mostly the market, and the market's
next month is essentially unforecastable. If I regress raw returns on
characteristics, most of the fitted variance and most of the loss function are
about a component I have no view on. Demeaning within the month removes it and
leaves what a long-short book actually trades.

It also makes the forecast dollar-neutral by construction, so I cannot
accidentally report a market-timing bet as a cross-sectional finding.

The cost, which you should raise yourself: the output is no longer a return in
percent, so it is not directly a dollar expectation, and out-of-sample \(R^2\)
is measured against a standardised target. Both are reported.

### "How do you know there's no look-ahead bias?"

Not by having been careful — by testing it. `tests/test_no_lookahead.py`:

- Replace every return after a cutoff date with noise. Every predictor dated on
  or before the cutoff must come out bit-for-bit identical. That is the
  definition of point-in-time and it is checked directly rather than reasoned
  about.
- For 200 random rows, the stored label must equal that asset's excess return in
  the following month.
- Permute labels within each month: measured skill must fall inside its own
  sampling error. If anything leaked from label to predictor anywhere in the
  pipeline, this control would still show skill.
- And the control that makes the previous one meaningful: a deliberate leak must
  produce an obviously larger IC. Otherwise "the control came out at zero" only
  shows the metric is blind.

---

## On the results

### "Your \(R^2_{OOS}\) is negative but your IC is positive. Which is it?"

Both, and the two are not in conflict — they measure different things and the
gap is diagnosable.

IC is rank correlation: does the ordering carry information? \(R^2_{OOS}\) is
squared error: is the forecast the right *size*? A model can order the
cross-section correctly and still be badly scaled, and mean squared error
punishes over-scaling hard.

I diagnosed it with a calibration regression — regress the outcome on the
forecast out of sample and look at the slope
(`results/exp1_forecast_accuracy.csv`, `reports/figures/fig2_calibration_gap.png`).
Ridge's slope is about 0.31, meaning its forecasts are roughly three times too
large; XGBoost's is about 0.68 and its \(R^2_{OOS}\) is positive. The models
that are better calibrated are exactly the ones with positive \(R^2_{OOS}\).

Which matters for me? IC, because the portfolio uses only the ranking. But
reporting only IC would have hidden a real miscalibration, so I report both.

### "So did the nonlinear models win?" — the most important answer here

No, not reliably, and this is the finding I would lead with.

The point estimates look like a clean win for complexity: rank IC of 0.025 for a
single momentum characteristic, 0.032 for ridge, 0.042 for lasso, 0.049 for
gradient boosting. Read that table and you would conclude nonlinearity pays.

It does not survive a paired test. Both models see the same cross-section every
month, so I difference their monthly ICs — the common component cancels and the
test is far tighter than comparing two standard errors
(`results/exp1_model_comparisons.csv`):

- Boosting versus ridge: IC gap \(+0.018\), \(t = 2.4\); Diebold-Mariano on
  squared error \(t = -4.4\). Significant.
- Boosting versus **elastic net**, the best linear model: gap \(+0.006\),
  \(t = 1.5\), \(p = 0.14\); on squared error \(p = 0.77\).
  **Indistinguishable.**
- The neural network versus elastic net: significantly *worse* on squared error
  (\(p = 0.009\)).
- And no *adjacent* rung of the ladder is significant at all. The only
  significant adjacent comparison is the neural network losing to boosting.

So almost the whole apparent gain from "complexity" is the sparse
regularisation in between, not the nonlinearity on top. Ridge selects a penalty
so small it is effectively OLS — with 26 predictors and 249,000 observations
there is no ill-conditioning for shrinkage to fix — whereas the sparse variable selection in Lasso and
elastic net is a real restriction that pays out of sample.

And no single adjacent step in the ladder is significant on its own. Only the
cumulative ridge-to-boosting gap clears conventional significance.

The conclusion I would state: *nonlinear models improved average accuracy, but
the improvement over the best regularised linear model was not statistically
distinguishable, and the extra flexibility of a neural network actively hurt.
The reliable gain came from variable selection.*

### "Why did the network lose?"

Sample size, and I would not generalise from it. Roughly 250,000 asset-months,
26 predictors, and a signal explaining a fraction of a percent of variance.
Boosting at depth 2 to 4 fits shallow, low-order interactions — about the
structure this data supports. A network must learn its own representation from
the same thin signal, and here initialisation variance is large relative to it,
which is why I ensemble over seeds and why a single network would have been
misleading in either direction.

On a firm-level panel with hundreds of characteristics, Gu, Kelly and Xiu find
networks do best. My cross-section is 374 portfolios with return-based
predictors only, so most of the high-order interaction structure that would
favour a network is not present to be found. That is a statement about my data,
not about networks.

### "Which is the best model?"

Depends on the question, and the ranking changes twice.

By raw accuracy, XGBoost (rank IC 0.049). By accuracy *net of its own standard
error*, lasso — because boosting's advantage over it is not distinguishable, and
lasso is the simpler model. By tradability, neither: ridge and OLS have gross
Sharpe ratios around 0.41 but turn over 2.3 times the book per month, so they
break even at about 18 basis points and go *negative* by 20.

And the single-characteristic baseline — 12-month momentum, one line of code —
has the **highest** break-even cost of anything I fit, about 42 basis points,
because it turns over least, despite having nearly the lowest gross Sharpe. That
reversal is the most useful thing in Experiment 2, and it only appears if you
charge for turnover before ranking models.

If someone forced me to pick one for a real book, it would be lasso or elastic
net: within noise of boosting on accuracy, better break-even cost than
ridge, and a model whose behaviour I can fully explain.

### "How much survived transaction costs?"

XGBoost: gross Sharpe 0.54, 0.37 at 10 basis points, 0.21 at 20. Ridge: 0.41
gross, 0.19 at 10, **negative** at 20.

The number I would quote is the break-even cost, because a Sharpe ratio at an
assumed cost level buries the assumption. Break-even converts it into a question
you can argue about with a trader: can we execute a monthly rebalance of a
374-portfolio book inside 33 basis points one-way?

What I would change with more time: model impact rather than a flat rate, and
optimise the portfolio against a turnover penalty instead of measuring turnover
after the fact. The second would likely matter more than any model change, since
the model rankings by gross accuracy and by net-of-cost performance already
disagree.

### "How much alpha survived factor controls?"

XGBoost's gross long-short return carries a six-factor alpha of about 7.0% a
year with \(t = 2.83\), and the factor regression \(R^2\) is only 0.036 — the
strategy is close to orthogonal to the market, size, value, profitability,
investment and momentum. Market beta is about \(-0.02\).

Do not oversell that. The low \(R^2\) partly reflects the fact that a
dollar-neutral decile spread across characteristic-sorted portfolios is
constructed to be roughly factor-neutral. The interesting content is that alpha
does not collapse when the six factors are added.

### "How did you handle multiple testing?"

Two ways, and the second is the one worth talking about.

First, on the strategy itself: Newey-West standard errors, and a stationary
block bootstrap of the Sharpe ratio, giving a 95% interval of roughly
\([0.33, 0.79]\) with no resample flipping sign.

Second, and more to the point: I ran the same tests across 212 published
cross-sectional predictors from the Chen-Zimmermann open-source dataset. Of
those, 163 are significant uncorrected at 5%, 159 survive
Benjamini-Hochberg, but only 84 survive Bonferroni, and 51% clear the
Harvey-Liu-Zhu \(|t| > 3\) hurdle.

Then I put my own result on that scale. XGBoost's \(t = 3.44\) clears the
\(|t| > 3\) hurdle but does **not** survive a Bonferroni correction across 212
hypotheses, and it sits at roughly the **57th percentile** of the published
distribution. So: statistically real on its own terms, median-strength in
context, and not strong enough to survive the most conservative correction the
literature applies.

Say that plainly. A candidate who reports the 57th percentile is more credible
than one who reports \(t = 3.44\) and stops.

One subtlety worth raising: the high survival rate among published predictors is
itself a selection effect — they were published *because* they were significant.
That is the Harvey-Liu-Zhu argument, and it is a reason to treat 3.0 rather than
1.96 as the relevant hurdle.

### "Did any group of predictors matter more than the others?"

Yes, and the ablation gave a result I did not expect. I refit the whole
walk-forward with each predictor group removed, and again with each group
alone. The trend and drawdown group — distance from the trailing high, max
drawdown, trailing Sharpe — reaches rank IC 0.044 **on its own**, against 0.049
for all 26 predictors. The signal is concentrated, not assembled from many weak
pieces.

The more useful lesson is that the two ablation directions disagree, and only
computing both keeps you honest. Removing momentum costs almost nothing
(\(-0.0020\)) — a leave-one-out study alone would call it uninformative. But
momentum on its own delivers 0.0228, second only to trend. It is *substitutable*,
not uninformative; the other predictors already span most of what it knows.
Higher moments is the genuine null case: cheap to remove *and* useless alone.

There is also a cost dimension that importance scores cannot see. Dropping the
reversal group costs 0.0061 of IC but raises the break-even cost from 32 to 49
basis points, because reversal is what drives the turnover. A group can be
informative and still not worth trading.

### "What surprised you?"

Experiment 4, and it reversed my prior.

I set out to show that asset-pricing tests break down as the number of assets
grows relative to the length of the sample — the intuition being that GRS
inverts an \(N \times N\) covariance matrix estimated from \(T\) observations,
so it must degrade as \(N/T \to 1\). I wrote a test asserting exactly that. It
failed.

GRS is *exact in finite samples* under its assumptions — normal, homoskedastic,
serially independent residuals with **any** cross-sectional covariance — for
every \(N \le T - K - 1\). I confirmed it holds its nominal 5% at \(N/T = 0.83\),
and further that its size survives heavy tails (\(t\) with 7.3 degrees of
freedom, calibrated to the data), a persistent common volatility factor, and
residual vectors resampled from the real panel.

What actually breaks is different, and more interesting:

- **The asymptotic version of the same statistic.** Referring the identical
  quadratic form to \(\chi^2_N\) instead of the exact \(F\) gives a rejection
  rate of about 10% at \(N = 10\), 28% at \(N = 50\), 75% at \(N = 100\), and
  **100%** at \(N \ge 200\) with \(T = 360\). Same data, same statistic, only
  the reference distribution differs -- the finite-sample correction is doing
  all the work.
- **Shrinkage without recalibration.** A Ledoit-Wolf covariance conditions
  better and shrinks the statistic, but the \(F\) critical value is unchanged,
  so size collapses to zero and the test stops rejecting anything. Better
  estimation is not automatically better inference.
- **The large-\(N\) test's assumption.** Pesaran-Yamagata never inverts an
  \(N \times N\) matrix and stays defined when \(N > T\), and it is correctly
  sized under cross-sectional independence. Under the correlation actually
  present in these portfolios its size reaches 28% — and it rises
  *with \(N\)*, which is the opposite of what an asymptotic-in-\(N\) test should
  do.

So the real trade-off is not about dimension. GRS needs \(N < T\);
Pesaran-Yamagata needs weak cross-sectional dependence. Equity portfolios
violate the second, and modern cross-sections violate the first.

### "Anything else that surprised you?"

Yes, and it is a debugging story rather than a finding.

My first two attempts at the empirical error model in Experiment 4 measured my
own resampling scheme rather than the data.

Resampling months **with** replacement repeats months, so the residual
covariance is estimated from fewer distinct observations than the nominal \(T\).
Size reached 78% and I nearly wrote it up as a finding about real returns.
Sampling **without** replacement fixed that and introduced the opposite error:
the finite-population correction shrinks the variance of the estimated
intercept, so size fell to 0.000.

The fix was a wild bootstrap — take real residual vectors and flip their signs.
Every month appears at most once, so nothing is duplicated; sign flips leave the
second moments untouched, so the variance of the intercept estimate is exactly
what the test assumes; the null holds exactly because each flip is mean zero;
and each month keeps its own magnitude and cross-sectional pattern. With that,
GRS holds its nominal size.

There is also an earlier bug of the same family: the residual pool was not
demeaned, so resampling injected an alpha of about 2% a year and every test
correctly rejected a null that was in fact false. All three are now regression
tests in `tests/test_montecarlo.py`.

The lesson I would offer: when a simulation produces a dramatic result,
the first hypothesis should be that the simulation is wrong.

### "Did you find any bugs in your own pipeline?"

One that changed a reported number, and it is worth volunteering.

Lasso selected a penalty that zeroed every coefficient in 15 of 34 test years,
so it produced a flat forecast in 180 of 407 out-of-sample months. My IC
function returned `NaN` for those months and dropped them — so Lasso's average
IC was computed only over the months in which it *chose* to have a view. That is
selection on the model's own confidence, and it lifted its apparent IC from
0.025 to 0.044, making it look better than OLS.

Worse, the portfolio code broke ties by row order, so a flat score was sorted by
asset name, split into deciles, and traded as a real book — crediting the model
with the return of a portfolio built from no information.

A flat forecast now scores zero IC and holds no position. Both are regression
tests.

---

## Questions to have an answer ready for

**"Isn't 374 portfolios a very small cross-section for machine learning?"**
Yes. That is the main limitation and I would lead with it. It is a standard
test-asset universe, but nonlinear interactions *between firm characteristics* —
the mechanism most often credited for machine-learning gains in this literature
— largely cannot be represented, because accounting characteristics are not
observable per portfolio. So my result is a lower bound on what the same
pipeline would find on firm-level data, and the data layer is built so a
firm-level panel drops in behind the same interface.

**"Why not individual stocks?"** Because CRSP and Compustat are licensed and I
did not have access. I would rather run a defensible study on a smaller real
universe than a firm-level study on data I could not verify. The pipeline is
written so the universe is a swap.

**"Your IC declines over the sample. Doesn't that kill it?"** It declines
substantially: by decade the pattern is strongest in the 1990s and much weaker
in the 2010s. Two readings, and I cannot separate them with this data. Either
these relations have been arbitraged away as they became known — which is what
you would expect, and consistent with the broader literature on anomaly decay —
or the early result was partly luck. The second reading is why the full-sample
\(t\)-statistic matters more to me than the point estimate.

**"What would you do next?"** In order: get firm-level data, because the
universe is the binding constraint, not the models. Then optimise against a
turnover penalty rather than measuring turnover after the fact, since the
accuracy and net-performance rankings already disagree. Then look at whether the
decay is arbitrage or overfitting, probably by testing whether decay is faster
for predictors that received more academic attention.

**"How long did this take and what did you write yourself?"** Answer honestly.
If you used an assistant, say so, and then demonstrate ownership by explaining a
decision it would not have made for you — the embargo, the Lasso scoring bug, or
why the Experiment 4 hypothesis had to be abandoned.
