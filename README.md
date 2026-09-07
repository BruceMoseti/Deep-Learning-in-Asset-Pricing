# Machine Learning and Statistical Inference for Cross-Sectional Asset Returns

**Research question.** Do nonlinear machine-learning models provide incremental
out-of-sample predictive information about cross-sectional equity returns
relative to regularised linear models — and does any predictability that
survives remain economically meaningful after transaction costs, and
statistically meaningful after controlling for known factors and for the number
of hypotheses tested?

Five experiments, each answering a question the previous one raises:

| | Question | Verdict |
| --- | --- | --- |
| 1 | Does model flexibility buy out-of-sample accuracy? | Only through sparsity; nonlinearity adds nothing distinguishable |
| 2 | Is that accuracy worth anything after costs? | Only for the low-turnover models |
| 3 | Is the signal statistically real, and real *in context*? | Real on its own terms; median-strength among published predictors |
| 4 | Do asset-pricing tests behave when there are many assets? | Not in the way usually assumed |
| 5 | Where does the signal come from and when does it fail? | Concentrated, and decaying over the sample |

---

## Headline findings

**The gain from complexity is sparsity, not nonlinearity — and that is the
answer to the research question.** Rank information coefficient rises
monotonically along the ladder: 0.025 for a single momentum characteristic,
0.032 for ridge, 0.042 for lasso, **0.049 for gradient boosting**
(\(t = 3.6\) against zero) over 407 out-of-sample months. Taken at face value
that looks like complexity paying. It does not survive a paired test. Boosting
beats ridge (IC gap \(+0.018\), \(t = 2.4\); Diebold-Mariano \(t = -4.4\)) but is
**statistically indistinguishable from lasso** (gap \(+0.007\), \(t = 1.2\),
\(p = 0.23\); on squared error \(p = 0.93\)), and the neural network is
significantly *worse* than lasso. Ridge selects a penalty so small it is
effectively OLS — with 26 predictors and 249,000 observations there is no
ill-conditioning for shrinkage to fix — so what separates the ends of the ladder
is Lasso's variable selection, not the nonlinearity layered on top. No single
adjacent step is significant on its own.

**The accuracy ranking is not the tradability ranking.** Ridge and OLS turn over
2.3 times the book per month; they break even at about 18 basis points one-way
and are **negative** at 20. Gradient boosting breaks even at about 33. And the
single-characteristic baseline — one line of code, lowest turnover of anything
fit — has the **highest** break-even cost, about 42 basis points. Charging for
turnover before ranking models reverses the ordering.

**The signal is real, and unremarkable in context.** Boosting's long-short
return carries a six-factor alpha of about 7.0% a year (\(t = 2.83\)) with a
factor-regression \(R^2\) of only 0.036. Its own \(t\)-statistic of 3.44 clears
the Harvey-Liu-Zhu \(|t| > 3\) hurdle but does **not** survive a Bonferroni
correction across the 212 published predictors it is measured against — where it
sits at roughly the 57th percentile.

**Experiment 4 reversed its own hypothesis.** The premise was that the
Gibbons-Ross-Shanken test degrades as \(N/T \to 1\) because it inverts an
\(N \times N\) covariance matrix. It does not: GRS is exact in finite samples
under its assumptions for any \(N \le T - K - 1\), and holds its nominal 5% at
\(N/T = 0.83\) even with heavy tails, a persistent common volatility factor, and
residuals resampled from the real panel. What does break is the *asymptotic*
version of the same statistic (rejection rate 5% → 100% as \(N\) goes 10 → 200),
shrinkage paired with an unadjusted reference distribution (size collapses to
zero), and the large-\(N\) alternative, whose size rises to 31% under the
cross-sectional dependence real portfolios actually have — and rises *with*
\(N\).

**Predictability declines across the sample.** Mean IC by decade is strongest in
the 1990s and much weaker in the 2010s, for every model. Either these relations
have been arbitraged away as they became known, or the early result was partly
luck. This design cannot separate the two, and the report says so.

---

## What is real data and what is not

The single most important thing to know before reading any number here.

| Component | Source | Real? |
| --- | --- | --- |
| Cross-section (374 assets × 726 months, 1963-07 to 2023-12) | Kenneth French characteristic-sorted portfolio library | **Real** |
| Factor controls (FF5 + momentum) | Kenneth French | **Real** |
| Multiple-testing study (212 predictors) | Chen-Zimmermann Open Source Asset Pricing | **Real** |
| Experiment 4 error structures | Calibrated to the real panel's residuals | Simulated **by design** |
| Firm-level accounting characteristics | — | **Not used** |

**The cross-section is portfolios, not individual firms.** It is 374
value-weighted portfolios of US common stocks, sorted on size, book-to-market,
profitability, investment, momentum, short- and long-term reversal, beta,
variance, accruals, net share issues, and industry. This is a standard
test-asset universe, and the returns are real, but it is not a firm-level panel.

**What that costs.** Accounting characteristics are not observable per
portfolio, so the 26 predictors are all derived from returns (momentum at
several horizons, reversal, volatility, beta and comovement, coskewness, higher
moments, drawdown, seasonality, persistence) plus three static labels giving
each asset's position in its sort. Nonlinear interactions *between firm
characteristics* — the mechanism Gu, Kelly and Xiu (2020) identify as the main
source of machine-learning gains — largely cannot be represented. **Read the
results as a lower bound** on what this pipeline would find on firm-level data.

**Running it on firm-level data.** `src/xsap/data/` isolates the universe behind
one function returning a long panel of `(month, asset, ret, ...)`. Point
`load_portfolio_panel` at a CRSP/Compustat, Open Source Asset Pricing, or
Jensen-Kelly-Pedersen firm panel and the rest of the pipeline is unchanged;
extend `FEATURE_GROUPS` in `src/xsap/features.py` with the accounting groups and
the ablation picks them up automatically.

**Reproducibility.** Every raw file is pinned to an immutable git commit and its
SHA-256 recorded in `data/raw/manifest.json` on first download. This matters
more than it sounds: the Fama-French library is revised retroactively whenever
CRSP is updated, so an unpinned download silently changes results. The mirrors
are used because the primary hosts are not reachable from every network;
`results/data_appendix.csv` records the upstream source of each file, and
`results/factor_moments.csv` is the check that the mirror is the real thing
(Mkt-RF and UMD print at 0.57% and 0.60% per month, matching published values).

---

## Running it

```bash
pip install -r requirements.txt
make test        # 70 tests, ~15s -- run this first
make fast        # end-to-end smoke run, ~15 min, same code path
make all         # full run, ~90 min on 8 cores
```

Individual stages: `make data exp1 exp2 exp3 exp4 exp5 figures`. Outputs land in
`results/` (CSV and Parquet), `reports/tables/` (Markdown), and
`reports/figures/` (PNG). The full write-up is
[`reports/RESEARCH_REPORT.md`](reports/RESEARCH_REPORT.md).

---

## Layout

```
src/xsap/
  config.py        every number that affects a result
  data/            pinned downloads, French/OSAP parsers, universe assembly
  features.py      point-in-time predictors and cross-sectional transforms
  models.py        the ladder: single signal -> OLS -> penalised -> boosting -> network
  walkforward.py   expanding-window splits with an embargo
  metrics.py       IC, ICIR, out-of-sample R2, calibration slope, Diebold-Mariano
  portfolio.py     decile long-short, turnover, costs, break-even cost
  inference.py     Newey-West, factor alphas, block bootstrap, GRS, Bonferroni/BH
  montecarlo.py    Experiment 4: six error structures, four tests
  robustness.py    ablations, regimes, subperiods, alternative design choices
scripts/           00_fetch_data ... 06_figures, one per experiment
tests/             70 tests; test_no_lookahead.py is the important one
docs/
  METHODOLOGY.md     every choice, its reason, and what the alternative did
  INTERVIEW_NOTES.md the questions this project has to survive
```

---

## Design decisions worth knowing about

Full detail in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md); the four that most
often go wrong:

**No random cross-validation.** Random folds train on months after the months
they test on. Worse, returns within a month are cross-correlated (mean pairwise
residual correlation 0.048 after six factors, 90th percentile 0.22), so even a
date-respecting split that mixed assets within a month would leak that month's
common shock. The split is by time *and* by whole months.

**A one-month embargo.** A row dated `t` carries the return of `t+1`, so the
last training row's label would otherwise fall inside the validation block, and
the last validation row's label inside the test year. One month is dropped from
the end of each.

**Expanding window, because it was tested.** A 20-year rolling window gives
essentially the same result (IC 0.046 vs 0.049); a 10-year window is clearly
worse (0.029). Degradation as the window shrinks, and flatness beyond 20 years,
is the evidence that estimation error binds rather than non-stationarity.

**Leakage is tested, not asserted.** Corrupting all returns after a cutoff must
leave every earlier predictor bit-for-bit identical; permuting labels within
months must drive measured skill inside its own sampling error; and a
*deliberate* leak must show up clearly, so that the null control means
something.

---

## Bugs found and fixed, since they changed reported numbers

Kept in the README because a project without a list like this has probably not
been checked.

**Lasso's IC was selected on the model's own confidence.** Lasso chose a penalty
that zeroed every coefficient in 15 of 34 test years, producing a flat forecast
in 180 of 407 months. Those months returned `NaN` and were dropped, so its
average IC was taken only over months where it chose to have a view — inflating
it from 0.025 to 0.044 and making it appear to beat OLS. Separately, the
portfolio code broke ties by row order, so a flat score was sorted by asset
name, cut into deciles, and traded as a real book. A flat forecast now scores
zero IC and holds no position.

**Two residual bootstraps in Experiment 4 measured the resampling scheme.**
Sampling months with replacement repeats months, so the covariance is estimated
from fewer distinct observations than the nominal \(T\); size reached 78%.
Sampling without replacement introduced the opposite error via the
finite-population correction; size fell to 0.000. Replaced with a wild bootstrap
(real residual vectors, random sign flips), which duplicates nothing, leaves
second moments untouched, and imposes the null exactly.

**An undemeaned residual pool injected a fake alpha.** Before that, the
bootstrap pool's column means were not zero, planting roughly 2% a year of
alpha; every test correctly rejected a null that was in fact false.

All three are regression tests.

---

## References

Chen, A. and Zimmermann, T. (2022). Open source cross-sectional asset pricing.
*Critical Finance Review*.
Gibbons, M., Ross, S. and Shanken, J. (1989). A test of the efficiency of a
given portfolio. *Econometrica*.
Gu, S., Kelly, B. and Xiu, D. (2020). Empirical asset pricing via machine
learning. *Review of Financial Studies*.
Harvey, C., Liu, Y. and Zhu, H. (2016). ... and the cross-section of expected
returns. *Review of Financial Studies*.
Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for
large-dimensional covariance matrices. *Journal of Multivariate Analysis*.
Pesaran, M. H. and Yamagata, T. (2012). Testing CAPM with a large number of
assets. *IZA Discussion Paper*.
Politis, D. and Romano, J. (1994). The stationary bootstrap. *JASA*.
