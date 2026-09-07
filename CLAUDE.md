# CLAUDE.md

Behavioural guidelines for working in this repository. General principles first,
then the rules specific to this project.

---

## 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

## 3. Surgical Changes

Touch only what you must. Clean up only your own mess.

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
- Remove imports/variables/functions that *your* changes made unused.

Every changed line should trace directly to the request.

## 4. Goal-Driven Execution

Define success criteria. Loop until verified.

- "Add validation" → "write tests for invalid inputs, then make them pass"
- "Fix the bug" → "write a test that reproduces it, then make it pass"
- "Refactor X" → "ensure tests pass before and after"

For multi-step tasks, state a brief plan with a verification step per item.

---

## Project-specific rules

This is empirical research code. A bug here does not throw an exception, it
produces a plausible number. The rules below exist because of that.

### A result that has not been tested for leakage is not a result

Any change to `features.py`, `walkforward.py`, or the timing convention must be
accompanied by `pytest tests/test_no_lookahead.py`. That file is the reason the
rest of the numbers can be believed. In particular:

- A row keyed `(month = t, asset = i)` holds predictors computed only from
  returns through `t`, and the excess return of `t+1` as its label. Never
  weaken this.
- Cross-sectional transforms operate within a single month. A transform that
  touches two months is a leak unless proven otherwise.
- Training and validation blocks are embargoed by one month because labels
  reach forward. Do not remove the `[:-1]` slices in `expanding_splits` without
  replacing the guarantee.

### When a simulation produces a dramatic result, suspect the simulation

This has already happened three times in this repository, and all three
mistakes produced *publishable-looking* numbers:

- An undemeaned residual pool planted a fake alpha of ~2% a year.
- Resampling months with replacement pushed a test's size to 78%.
- Resampling without replacement pushed it to 0.000.

Before reporting that a standard method fails, check that the method's
assumptions are actually being met by the data-generating process you wrote.
Add a regression test for the diagnosis, not just the fix.

### Report the unflattering number

The Bonferroni verdict, the negative out-of-sample \(R^2\), the 57th percentile,
the decade-by-decade decay, the fact that a one-line baseline has the best
break-even cost. These are in the report because omitting them would make the
project worse, not better. Do not quietly drop a result because it weakens the
headline.

### Justify design choices by measuring the alternative

"Expanding window because estimation error dominates" is an assertion.
"Expanding window; 20-year rolling gives 0.046 versus 0.049, 10-year gives
0.029" is a justification. If a choice matters, put the alternative in
`robustness.py` and report both.

### Numbers live in one place

Anything that affects a reported result belongs in `src/xsap/config.py`. Tables
are written by `save_table` so the report cannot drift from the run. Never
hand-transcribe a number into Markdown that a script could emit.

### Comments explain constraints, not mechanics

Existing comments in this repository record *why* a non-obvious choice was made,
usually because a simpler alternative was tried and failed. Match that. Do not
add comments narrating what the next line does.
