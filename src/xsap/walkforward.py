"""Expanding-window walk-forward evaluation.

Why not k-fold cross-validation
-------------------------------
Random splits break this problem in two ways.  First, they train on months that
come after the months they test on, so the model is told about the future.
Second, returns within a month are strongly cross-correlated, so putting some
assets from month ``t`` in the training fold and others in the test fold leaks
that month's common shock across the split.  Either alone is enough to make
measured accuracy meaningless.  So the split is by time, and by whole months.

Why expanding rather than rolling
---------------------------------
The relations being estimated are weak and the monthly cross-section is noisy,
so estimation error is the binding constraint; discarding old data to chase
non-stationarity makes that worse.  An expanding window is the default here and
a rolling window is reported as a robustness check, which is the only honest way
to make the claim -- the choice is an empirical question, not a matter of taste.

Embargo
-------
A row dated month ``t`` carries the return of month ``t+1``.  Without care, the
last training row's label falls inside the validation block and the last
validation row's label falls inside the test block.  One month is dropped from
the end of the training and validation blocks to prevent that.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from xsap.config import Config
from xsap.features import FEATURE_NAMES, _to_unit_interval


@dataclass(frozen=True)
class Split:
    test_year: int
    train: pd.PeriodIndex
    valid: pd.PeriodIndex
    test: pd.PeriodIndex

    def describe(self) -> str:
        return (
            f"test {self.test_year}: "
            f"train {self.train[0]}..{self.train[-1]} ({len(self.train)}m), "
            f"valid {self.valid[0]}..{self.valid[-1]} ({len(self.valid)}m), "
            f"test {self.test[0]}..{self.test[-1]} ({len(self.test)}m)"
        )


def expanding_splits(
    months: pd.PeriodIndex, cfg: Config | None = None, *, rolling_years: int | None = None
) -> list[Split]:
    """Build one split per test year.

    ``rolling_years`` caps the training block at that many years, turning the
    scheme into a rolling window for the robustness check.
    """
    cfg = cfg or Config()
    schedule = cfg.split
    months = pd.PeriodIndex(sorted(set(months)), freq="M")
    first_year = int(months[0].year)

    splits: list[Split] = []
    for test_year in range(schedule.first_test_year, schedule.last_test_year + 1):
        valid_start = test_year - schedule.validation_years
        train_end_year = valid_start - 1
        if train_end_year - first_year + 1 < schedule.min_train_years:
            continue

        train_start = first_year
        if rolling_years is not None:
            train_start = max(first_year, train_end_year - rolling_years + 1)

        train = months[
            (months.year >= train_start) & (months.year <= train_end_year)
        ][:-1]
        valid = months[
            (months.year >= valid_start) & (months.year <= test_year - 1)
        ][:-1]
        test = months[months.year == test_year]
        if len(train) == 0 or len(valid) == 0 or len(test) == 0:
            continue
        splits.append(Split(test_year, train, valid, test))
    return splits


def _block(data: pd.DataFrame, target: pd.Series, months, features):
    mask = data.index.get_level_values("month").isin(months)
    return data.loc[mask, list(features)].to_numpy(dtype=float), target[mask].to_numpy(
        dtype=float
    )


def run_walkforward(
    data: pd.DataFrame,
    target: pd.Series,
    models: list,
    cfg: Config | None = None,
    features: tuple[str, ...] = FEATURE_NAMES,
    *,
    rolling_years: int | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit each model once per test year and collect out-of-sample forecasts.

    Returns ``(predictions, diagnostics)``.  ``predictions`` is indexed by
    ``(month, asset)`` with one column per model plus the realised target and
    excess return; ``diagnostics`` records the chosen hyper-parameters and fit
    time for every model-year, which is what makes the run auditable after the
    fact.
    """
    cfg = cfg or Config()
    months = pd.PeriodIndex(data.index.get_level_values("month").unique(), freq="M")
    splits = expanding_splits(months, cfg, rolling_years=rolling_years)
    if not splits:
        raise ValueError("no walk-forward splits; check sample and split settings")

    frames, diagnostics = [], []
    for split in splits:
        x_train, y_train = _block(data, target, split.train, features)
        x_valid, y_valid = _block(data, target, split.valid, features)
        test_mask = data.index.get_level_values("month").isin(split.test)
        x_test = data.loc[test_mask, list(features)].to_numpy(dtype=float)

        block = pd.DataFrame(index=data.index[test_mask])
        for template in models:
            model = copy.deepcopy(template)
            started = time.perf_counter()
            model.fit(x_train, y_train, x_valid, y_valid, feature_names=features)
            block[model.name] = model.predict(x_test)
            diagnostics.append(
                {
                    "test_year": split.test_year,
                    "model": model.name,
                    "n_train": len(y_train),
                    "n_valid": len(y_valid),
                    "n_test": len(x_test),
                    "fit_seconds": round(time.perf_counter() - started, 2),
                    "hyperparameters": str(getattr(model, "best", {})),
                }
            )
        frames.append(block)
        if verbose:
            print(f"  {split.describe()}", flush=True)

    predictions = pd.concat(frames).sort_index()
    predictions["y"] = target.reindex(predictions.index)
    predictions["fwd_excess_ret"] = data["fwd_excess_ret"].reindex(predictions.index)
    return predictions, pd.DataFrame(diagnostics)


def standardise_predictions(predictions: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Convert each model's forecast to a within-month cross-sectional rank.

    Portfolio construction only needs the ordering, and ranking puts every model
    on the same footing so that differences in forecast scale -- which come from
    how hard each model shrinks, not from what it knows -- cannot change the
    portfolio comparison.
    """
    out = predictions.copy()
    for model in models:
        grouped = out.groupby("month")[model]
        out[model + "_rank"] = _to_unit_interval(
            grouped.rank(), grouped.transform("count")
        )
    return out
