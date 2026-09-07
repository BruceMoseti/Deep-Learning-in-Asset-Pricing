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
from xsap.metrics import (
    diebold_mariano,
    ic_difference_test,
    monthly_ic,
    summarise_forecasts,
)
from xsap.models import default_models
from xsap.walkforward import expanding_splits, run_walkforward

BENCHMARK = "ridge"
LINEAR = ("ols", "ridge", "lasso", "enet")


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

    # Two separate comparisons, because "does complexity pay?" hides two
    # questions.  Against ridge, any gain mixes better regularisation with
    # nonlinearity.  Against the *best* linear model, only the nonlinearity is
    # left -- which is the question actually being asked.
    best_linear = max(
        (n for n in names if n in LINEAR), key=lambda n: rows[n]["rank_ic"], default=None
    )
    for name in names:
        for benchmark, column in (
            (BENCHMARK, "dm_tstat_vs_ridge"),
            (best_linear, "dm_tstat_vs_best_linear"),
        ):
            if benchmark is None or name == benchmark:
                accuracy.loc[name, column] = float("nan")
                continue
            test = diebold_mariano(
                predictions[name],
                predictions[benchmark],
                predictions["y"],
                cfg.newey_west_lags,
            )
            accuracy.loc[name, column] = test["tstat"]
    save_table(accuracy, "exp1_forecast_accuracy")

    # Pairwise tests, because a table of point estimates invites the reader to
    # rank models by eye and call the ordering a result.  Adjacent rungs of the
    # ladder answer "did this step help?"; the contrasts against the best
    # linear model answer "did nonlinearity help, once regularisation is held
    # fixed?", which is the question the project actually poses.
    pairs = [(a, b) for a, b in zip(names[1:], names[:-1])]
    for candidate in (best_linear, BENCHMARK, "single-signal"):
        for name in names:
            if candidate and name != candidate and (name, candidate) not in pairs:
                pairs.append((name, candidate))

    comparisons = []
    for a, b in pairs:
        if a not in names or b not in names:
            continue
        paired = ic_difference_test(
            predictions[a], predictions[b], predictions["y"], cfg.newey_west_lags
        )
        squared = diebold_mariano(
            predictions[a], predictions[b], predictions["y"], cfg.newey_west_lags
        )
        comparisons.append(
            {
                "model": a,
                "benchmark": b,
                "ic_model": paired["ic_a"],
                "ic_benchmark": paired["ic_b"],
                "ic_difference": paired["ic_difference"],
                "ic_diff_tstat": paired["tstat"],
                "ic_diff_pvalue": paired["pvalue"],
                "dm_tstat_squared_error": squared["tstat"],
                "dm_pvalue": squared["pvalue"],
            }
        )
    save_table(
        pd.DataFrame(comparisons).set_index(["model", "benchmark"]),
        "exp1_model_comparisons",
    )

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
            "best_linear_model": best_linear,
            "best_model": max(names, key=lambda n: rows[n]["rank_ic"]),
        },
        "exp1_run",
    )
    print("\n" + accuracy.to_string(float_format=lambda v: f"{v: .4f}"))


if __name__ == "__main__":
    main()
