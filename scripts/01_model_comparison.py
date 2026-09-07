#!/usr/bin/env python3
"""Experiment 1 -- does added model flexibility buy out-of-sample accuracy?

Fits the ladder from a single standardised characteristic through to a small
neural network, one refit per test year on an expanding window, and compares
forecast accuracy out of sample.  The comparison is by information coefficient
(what the portfolio actually uses), by out-of-sample R-squared against a zero
forecast, and by a Diebold-Mariano test against the best linear model so that
"better" is backed by a standard error rather than by a ranking.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from xsap.artifacts import dataset, save_json, save_series, save_table
from xsap.config import Config
from xsap.features import FEATURE_NAMES
from xsap.metrics import diebold_mariano, monthly_ic, summarise_forecasts
from xsap.models import default_models
from xsap.walkforward import expanding_splits, run_walkforward

BENCHMARK = "ridge"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="linear models only, for a quick end-to-end check",
    )
    args = parser.parse_args()

    cfg = Config()
    data, target = dataset(cfg)
    print(
        f"dataset: {len(data):,} asset-months, "
        f"{data.index.get_level_values('asset').nunique()} assets, "
        f"{len(FEATURE_NAMES)} predictors"
    )

    models = default_models()
    if args.fast:
        models = [m for m in models if m.name in {"single-signal", "ols", "ridge"}]

    splits = expanding_splits(
        pd.PeriodIndex(data.index.get_level_values("month").unique(), freq="M"), cfg
    )
    print(f"walk-forward: {len(splits)} annual refits, {splits[0].describe()}")

    predictions, diagnostics = run_walkforward(data, target, models, cfg)
    save_series(predictions, "predictions")
    save_table(diagnostics.set_index(["test_year", "model"]), "walkforward_diagnostics")

    names = [m.name for m in models]
    rows = {name: summarise_forecasts(predictions[name], predictions["y"]) for name in names}
    accuracy = pd.DataFrame(rows).T
    accuracy.index.name = "model"

    # Is any accuracy gain over the best linear model larger than its own noise?
    for name in names:
        if name == BENCHMARK:
            accuracy.loc[name, "dm_tstat_vs_ridge"] = float("nan")
            continue
        test = diebold_mariano(
            predictions[name], predictions[BENCHMARK], predictions["y"], cfg.newey_west_lags
        )
        accuracy.loc[name, "dm_tstat_vs_ridge"] = test["tstat"]
    save_table(accuracy, "exp1_forecast_accuracy")

    ic_series = pd.concat(
        {name: monthly_ic(predictions[name], predictions["y"]) for name in names}, axis=1
    )
    save_series(ic_series, "ic_monthly")

    # Accuracy by decade: an average over 34 years can hide a signal that only
    # ever worked in one part of the sample.
    by_decade = {}
    decade = (ic_series.index.year // 10) * 10
    for label, block in ic_series.groupby(decade):
        by_decade[f"{label}s"] = block.mean()
    save_table(pd.DataFrame(by_decade).T, "exp1_ic_by_decade")

    save_json(
        {
            "n_observations": int(len(data)),
            "n_assets": int(data.index.get_level_values("asset").nunique()),
            "n_predictors": len(FEATURE_NAMES),
            "n_refits": len(splits),
            "oos_first_month": str(predictions.index.get_level_values("month").min()),
            "oos_last_month": str(predictions.index.get_level_values("month").max()),
            "oos_months": int(predictions.index.get_level_values("month").nunique()),
            "models": names,
        },
        "exp1_run",
    )
    print("\n" + accuracy.to_string(float_format=lambda v: f"{v: .4f}"))


if __name__ == "__main__":
    main()
