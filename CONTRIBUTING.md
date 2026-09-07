# Working in this repository

Empirical research code has a failure mode that ordinary software does not: a
bug usually produces a plausible number rather than an exception. Everything
below exists because of that.

## A result that has not been tested for leakage is not a result

Any change to `features.py`, `walkforward.py`, or the timing convention must be
accompanied by `pytest tests/test_no_lookahead.py`.

- A row keyed `(month = t, asset = i)` holds predictors computed only from
  returns through `t`, and the excess return of `t+1` as its label.
- Cross-sectional transforms operate within a single month. A transform that
  touches two months is a leak unless proven otherwise.
- Training and validation blocks are embargoed by one month because labels
  reach forward. Do not remove the `[:-1]` slices in `expanding_splits`
  without replacing the guarantee.

## When a simulation produces a dramatic result, suspect the simulation

This has happened three times here, and all three mistakes produced
publishable-looking numbers:

- An undemeaned residual pool planted a false alpha of about 2% a year.
- Resampling months with replacement pushed a test's size to 0.78.
- Resampling without replacement pushed it to 0.000.

Before concluding that a standard method fails, check that the method's
assumptions are met by the data-generating process you wrote. Add a regression
test for the diagnosis, not just for the fix.

## Justify a design choice by measuring the alternative

"Expanding window because estimation error dominates" is an assertion.
"Expanding window; 20-year rolling gives 0.046 against 0.049, 10-year gives
0.029" is a justification. If a choice matters, put the alternative in
`robustness.py` and report both.

## Report the unflattering number

The Bonferroni verdict, the negative out-of-sample \(R^2\), the 57th
percentile, the decade-by-decade decay, the one-line baseline having the best
break-even cost. Do not drop a result because it weakens a headline.

## Numbers live in one place

Anything affecting a reported result belongs in `src/xsap/config.py`. Tables are
written by `save_table`, and `scripts/07_report.py` reads every number in the
report from `results/`. Never hand-transcribe a number into Markdown that a
script could emit.

## Comments explain constraints, not mechanics

Comments here record why a non-obvious choice was made, usually because a
simpler alternative was tried and failed. Do not narrate what the next line
does.

## Before opening a pull request

```bash
make test
make fast     # end-to-end on the same code path, ~15 min
```
