# Machine Learning and Statistical Inference for Cross-Sectional Asset Returns

Do nonlinear machine-learning models add out-of-sample predictive information
about cross-sectional equity returns beyond regularised linear models — and if
so, does it survive transaction costs, factor controls, and a correction for the
number of hypotheses the literature has tested?

Five experiments on 374 characteristic-sorted portfolios of US common stocks,
1963–2023, with 407 out-of-sample months from an expanding-window walk-forward.

| | Question | Finding |
| --- | --- | --- |
| 1 | Does model flexibility buy out-of-sample accuracy? | Only through sparsity; nonlinearity adds nothing distinguishable |
| 2 | Is that accuracy worth anything after costs? | Only for the low-turnover models |
| 3 | Is the signal statistically real, and real in context? | Real on its own terms; median-strength among published predictors |
| 4 | Do asset-pricing tests behave when there are many assets? | Not in the way usually assumed |
| 5 | Where does the signal come from, and when does it fail? | Concentrated in trend measures, and decaying |

Full write-up: [`reports/RESEARCH_REPORT.md`](reports/RESEARCH_REPORT.md).
Method: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).
Objections and robustness: [`docs/DISCUSSION.md`](docs/DISCUSSION.md).

---

## Results

**The gain from complexity is sparsity, not nonlinearity.** Rank information
coefficient rises monotonically along the model ladder — 0.025 for a single
momentum characteristic, 0.032 ridge, 0.042 lasso, 0.049 gradient boosting
(t = 3.6 against zero). A paired test on the monthly IC differences tells a
different story: boosting beats ridge (gap +0.018, t = 2.4) but is
indistinguishable from elastic net, the best linear model (gap +0.006,
t = 1.5, p = 0.14; on squared error p = 0.77), and the neural network
is significantly worse than elastic net. No single adjacent step in the ladder
is significant. Ridge selects a penalty small enough to reproduce OLS, which is
expected with 26 predictors and 249,000 observations, so what separates the ends
of the ladder is variable selection rather than the nonlinearity above it.

![Differences in mean rank IC with 95% intervals](reports/figures/fig13_model_comparisons.png)

**Accuracy and tradability rank models differently.** Ridge and OLS turn over
2.3 times the book per month, break even at about 18 basis points one-way, and
turn negative at 20. Boosting breaks even at 32. The single-characteristic
baseline, which trades least and has nearly the lowest gross Sharpe, breaks even
highest of all at 42.

**The alpha survives factor controls and does not survive Bonferroni.** Boosting
earns a six-factor alpha of 7.0% a year (t = 2.83) with a factor-regression
R² of 0.036. Measured against 212 published predictors from the
Chen-Zimmermann dataset — of which 163 are significant uncorrected, 159 survive
Benjamini-Hochberg and 84 survive Bonferroni — its t = 3.43 clears the
Harvey-Liu-Zhu |t| > 3 hurdle, fails Bonferroni, and sits at the 57th
percentile of the published distribution.

**Dimension is not what breaks asset-pricing tests.** Contrary to this study's
own initial hypothesis, the Gibbons-Ross-Shanken test does not degrade as
*N/T* → 1. It is exact in finite samples for any *N* ≤ *T* − *K* − 1, and holds
its nominal 5% at *N/T* = 0.83 even under heavy tails, a persistent common
volatility factor, and residuals resampled from the real panel — median size
0.050 across the grid. What breaks instead is the asymptotic version of the same
statistic (5% → 100% as *N* rises from 10 to 200), shrinkage paired with an
unadjusted reference distribution (median size 0.000, so it rejects nothing),
and the large-*N* alternative, whose size reaches 0.28 under real
cross-sectional dependence and rises with *N* rather than falling. GRS needs
*N* < *T*; Pesaran-Yamagata needs weak cross-sectional dependence; equity panels
violate the second and large cross-sections violate the first.

![Empirical size against N/T under three error structures](reports/figures/fig8_test_size.png)

**Predictability is concentrated and decaying.** The trend and drawdown group
alone reaches IC 0.044 of the 0.049 available from all 26 predictors. Momentum
is substitutable rather than uninformative — removing it costs 0.0020 while it
delivers 0.0228 alone. Dropping reversal costs 0.0061 of IC but raises the
break-even cost from 32 to 49 basis points. Mean IC by decade is strongest in
the 1990s and much weaker in the 2010s for every model; this design cannot
separate arbitrage from luck.

---

## Data

| Component | Source |
| --- | --- |
| Cross-section: 374 value-weighted characteristic-sorted portfolios, 1963-07 to 2023-12 | Kenneth French data library |
| Factor controls: FF5 + momentum, monthly | Kenneth French data library |
| Multiple-testing sample: 212 published predictors | Chen and Zimmermann (2022), Open Source Asset Pricing |
| Experiment 4 error structures | Calibrated to this panel's residuals |

The cross-section is **portfolios, not individual firms**. Sorts are on size,
book-to-market, profitability, investment, momentum, short- and long-term
reversal, beta, variance, accruals, net share issues, and industry. Accounting
characteristics are not observable per portfolio, so all 26 predictors are
derived from returns, plus three static labels giving each asset's position in
its sort. Nonlinear interactions *between firm characteristics* — the mechanism
most often credited for machine-learning gains in this literature — therefore
largely cannot be represented, and the results should be read as a lower bound
on what this pipeline would find on a firm-level panel.

`src/xsap/data/` isolates the universe behind one function returning a long
panel of `(month, asset, ret, ...)`. A CRSP/Compustat, Open Source Asset Pricing
or Jensen-Kelly-Pedersen firm panel substitutes without touching the rest of the
pipeline; adding the accounting groups to `FEATURE_GROUPS` extends the ablation
automatically.

Every raw file is pinned to an immutable commit with its SHA-256 recorded in
`data/raw/manifest.json`. The Fama-French library is revised retroactively
whenever CRSP is updated, so an unpinned download silently changes results.
`results/factor_moments.csv` is the check that the data is what it claims to be:
Mkt-RF and UMD print at 0.57% and 0.60% per month.

---

## Running it

```bash
make setup       # pip install -e ".[dev]"
make test        # 98 tests, ~15s
make fast        # end-to-end smoke run on the same code path, ~15 min
make all         # full run, ~45 min on 8 cores
```

`requirements.txt` pins the exact versions used to produce the reported numbers;
`pyproject.toml` carries looser floors for installation.

Stages run individually with `make data exp1 exp2 exp3 exp4 exp5 figures
report`. Outputs land in `results/` (CSV and Parquet), `reports/tables/`
(Markdown) and `reports/figures/` (PNG). `scripts/07_report.py` reads every
number in the report from `results/`, so the report cannot drift from the run
that produced it.

---

## Layout

```
src/xsap/
  config.py        every number that affects a result
  data/            pinned downloads, French/OSAP parsers, universe assembly
  features.py      point-in-time predictors and cross-sectional transforms
  models.py        single signal -> OLS -> penalised -> boosting -> network
  walkforward.py   expanding-window splits with a one-month embargo
  metrics.py       IC, ICIR, out-of-sample R2, calibration slope, paired tests
  portfolio.py     decile long-short, turnover, costs, break-even cost
  inference.py     Newey-West, factor alphas, block bootstrap, GRS, Bonferroni/BH
  montecarlo.py    six error structures, four tests of the joint alpha null
  robustness.py    ablations, regimes, subperiods, alternative design choices
scripts/           00_fetch_data ... 07_report, one per experiment
tests/             98 tests
docs/              METHODOLOGY.md, DISCUSSION.md
reports/           RESEARCH_REPORT.md and 13 figures, generated from results/
```

---

## Method

Detail in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). The four choices that
most affect the results:

**Target.** Next month's excess return, demeaned and scaled within the month.
Demeaning removes the market, which dominates return variance and is close to
unforecastable monthly, and leaves what a long-short book trades.

**Splits by time and by whole months.** Random folds would train on the future
and, less obviously, would leak a month's common shock across the split — mean
pairwise residual correlation here is 0.048 after six factors, with a 90th
percentile of 0.22. One month is embargoed from the end of the training and
validation blocks because labels reach forward.

**Expanding window, tested against the alternative.** A 20-year rolling window
gives IC 0.046 against 0.049; a 10-year window gives 0.029. Degradation as the
window shrinks, and flatness beyond 20 years, is the evidence that estimation
error binds rather than non-stationarity.

**Leakage tested rather than asserted.** `tests/test_no_lookahead.py` corrupts
every return after a cutoff and requires each earlier predictor to be
bit-for-bit identical; permutes labels within months and requires measured skill
to fall inside its sampling error; and introduces a deliberate leak to establish
that the metric would detect one.

---

## Limitations

The cross-section is portfolios rather than firms, and accounting
characteristics are absent. Portfolio returns carry no delistings, survivorship
decisions, microcap illiquidity or name-level shorting constraints. Costs are
modelled as a flat one-way rate, which ignores market impact, so the break-even
figures are generous. And this project is itself a search — 26 predictors, seven
models, several portfolio variants — which is why Experiment 3 compares the
result against the distribution of published predictors rather than against
zero. [`docs/DISCUSSION.md`](docs/DISCUSSION.md) takes each of these in turn.

---

## References

Chen, A. and Zimmermann, T. (2022). Open source cross-sectional asset pricing.
*Critical Finance Review* 11(2).
Gibbons, M., Ross, S. and Shanken, J. (1989). A test of the efficiency of a
given portfolio. *Econometrica* 57(5).
Gu, S., Kelly, B. and Xiu, D. (2020). Empirical asset pricing via machine
learning. *Review of Financial Studies* 33(5).
Harvey, C., Liu, Y. and Zhu, H. (2016). … and the cross-section of expected
returns. *Review of Financial Studies* 29(1).
Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for
large-dimensional covariance matrices. *Journal of Multivariate Analysis* 88(2).
Pesaran, M. H. and Yamagata, T. (2012). Testing CAPM with a large number of
assets. IZA Discussion Paper 6469.
Politis, D. and Romano, J. (1994). The stationary bootstrap.
*Journal of the American Statistical Association* 89(428).
