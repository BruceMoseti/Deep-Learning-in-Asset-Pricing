#!/usr/bin/env python3
"""Experiment 5 -- where does the signal come from, and when does it stop working?

Runs the ablation, regime, subperiod and design-choice analyses on the model
that won Experiment 1.  The ablation refits the whole walk-forward with each
group of predictors removed, and again with each group alone, rather than
reading feature importances off the fitted model: importances describe a fit,
not out-of-sample value.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from xsap.artifacts import (
    dataset,
    factors,
    load_series,
    read_json,
    save_json,
    save_table,
)
from xsap.config import Config
from xsap.models import GradientBoosted, Linear
from xsap.robustness import (
    define_regimes,
    design_robustness,
    feature_group_ablation,
    regime_analysis,
    subperiod_analysis,
)

MODELS = {"xgboost": GradientBoosted, "ridge": lambda: Linear(kind="ridge")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="xgboost", choices=sorted(MODELS))
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    data, target = dataset(cfg)
    ff = factors(cfg)
    model = MODELS[args.model]()

    predictions = load_series("predictions")
    gross = load_series("strategy_returns_gross")
    models = read_json("exp1_run")["models"]

    print("regime analysis")
    regimes = define_regimes(ff)
    save_table(regime_analysis(predictions, gross, regimes, models, cfg), "exp5_regimes")
    save_table(
        regimes.dropna().apply(pd.Series.value_counts).fillna(0).astype(int),
        "exp5_regime_counts",
    )

    print("subperiod analysis")
    save_table(subperiod_analysis(predictions, gross, models), "exp5_subperiods")

    print(f"design robustness ({args.model})")
    save_table(design_robustness(data, target, model, cfg), "exp5_design_robustness")

    if not args.skip_ablation:
        print(f"feature-group ablation ({args.model}); this refits the walk-forward")
        ablation = feature_group_ablation(data, target, model, cfg)
        save_table(ablation, "exp5_feature_ablation")
        print("\n" + ablation.to_string(float_format=lambda v: f"{v: .4f}"))

    save_json({"ablation_model": args.model}, "exp5_run")


if __name__ == "__main__":
    main()
