#!/usr/bin/env python3
"""Experiment 3 -- is the performance real, or is it a known premium and luck?

Three separate ways for the result to fail, tested separately.

1. The strategy may be a repackaging of premia that are already for sale.
   Tested by regressing its returns on CAPM, the Fama-French five factors, and
   those five plus momentum, with Newey-West standard errors, and reading the
   intercept.
2. The Sharpe ratio may not be distinguishable from zero once serial
   dependence and heavy tails are accounted for.  Tested with a stationary
   block bootstrap, reporting the fraction of resamples in which the sign
   flips.
3. The whole exercise may be one of many that were tried.  Tested on 212
   published cross-sectional predictors, asking how many survive Bonferroni
   and Benjamini-Hochberg corrections -- the same discipline this project's own
   result has to submit to.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from xsap.artifacts import factors, load_series, read_json, save_json, save_table
from xsap.config import Config
from xsap.data import load_anomaly_returns
from xsap.inference import (
    CAPM,
    FF5,
    FF6,
    bootstrap_statistic,
    factor_alpha,
    mean_tstat,
    multiple_testing_summary,
)


def sharpe(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(12)) if sd > 0 else np.nan


def factor_regressions(returns: pd.DataFrame, ff: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for name in returns.columns:
        for label, model in (("capm", CAPM), ("ff5", FF5), ("ff6", FF6)):
            result = factor_alpha(returns[name], ff, model, cfg.newey_west_lags)
            rows.append(
                {
                    "model": name,
                    "factor_model": label,
                    "alpha_monthly": result["alpha"],
                    "alpha_annual": result["alpha_annual"],
                    "alpha_tstat": result["alpha_tstat"],
                    "r2": result["r2"],
                    **{f"beta_{k}": v for k, v in result["betas"].items()},
                }
            )
    return pd.DataFrame(rows).set_index(["model", "factor_model"])


def main() -> None:
    cfg = Config()
    ff = factors(cfg)
    models = read_json("exp1_run")["models"]
    gross = load_series("strategy_returns_gross")
    net10 = load_series("strategy_returns_net_10bps")
    gross.index = pd.PeriodIndex(gross.index, freq="M")
    net10.index = pd.PeriodIndex(net10.index, freq="M")

    print("factor regressions on gross returns")
    save_table(factor_regressions(gross, ff, cfg), "exp3_factor_alpha_gross")
    print("factor regressions on returns net of 10bps")
    save_table(factor_regressions(net10, ff, cfg), "exp3_factor_alpha_net10")

    print("stationary bootstrap of the Sharpe ratio")
    rows = {}
    for name in models:
        series = gross[name].dropna()
        boot = bootstrap_statistic(
            series.to_numpy(),
            sharpe,
            n_draws=cfg.bootstrap_draws,
            mean_block=cfg.block_length,
            seed=cfg.seed,
        )
        test = mean_tstat(series.to_numpy(), cfg.newey_west_lags)
        rows[name] = {
            "sharpe": boot["point"],
            "sharpe_ci_lower": boot["ci_lower"],
            "sharpe_ci_upper": boot["ci_upper"],
            "p_sign_flips": boot["p_wrong_sign"],
            "mean_return_tstat_nw": test.tstat,
            "mean_return_pvalue_nw": test.pvalue,
        }
    bootstrap_table = pd.DataFrame(rows).T
    bootstrap_table.index.name = "model"
    save_table(bootstrap_table, "exp3_bootstrap_sharpe")

    # --- Multiple testing across 212 published predictors.
    print("multiple testing across published predictors")
    anomalies = load_anomaly_returns(cfg)
    # Require a usable sample per predictor; ragged coverage is inherent to the
    # data, since each predictor starts when the inputs it needs exist.
    usable = anomalies.loc[:, anomalies.notna().sum() >= 120]

    records = []
    for name in usable.columns:
        series = usable[name].dropna()
        test = mean_tstat(series.to_numpy(), cfg.newey_west_lags)
        alpha = factor_alpha(series, ff, FF6, cfg.newey_west_lags)
        records.append(
            {
                "predictor": name,
                "n_months": test.n_obs,
                "mean_monthly": test.mean,
                "tstat_nw": test.tstat,
                "pvalue_nw": test.pvalue,
                "ff6_alpha_monthly": alpha["alpha"],
                "ff6_alpha_tstat": alpha["alpha_tstat"],
                "ff6_alpha_pvalue": alpha["alpha_pvalue"],
            }
        )
    per_predictor = pd.DataFrame(records).set_index("predictor")

    summary_rows = {}
    for label, column in (("raw_mean_return", "pvalue_nw"), ("ff6_alpha", "ff6_alpha_pvalue")):
        decisions = multiple_testing_summary(
            per_predictor[column].to_numpy(), per_predictor.index
        )
        per_predictor[f"{label}_bonferroni"] = decisions["bonferroni"].to_numpy()
        per_predictor[f"{label}_bh"] = decisions["benjamini_hochberg"].to_numpy()
        n = len(per_predictor)
        summary_rows[label] = {
            "n_predictors": n,
            "significant_uncorrected_5pct": int(decisions["uncorrected"].sum()),
            "significant_bonferroni_5pct": int(decisions["bonferroni"].sum()),
            "significant_bh_5pct": int(decisions["benjamini_hochberg"].sum()),
            "share_surviving_bonferroni": float(decisions["bonferroni"].mean()),
            "share_surviving_bh": float(decisions["benjamini_hochberg"].mean()),
            # Harvey, Liu and Zhu (2016) argue the appropriate hurdle for a
            # newly proposed predictor is around 3.0 rather than 1.96.
            "share_with_tstat_above_3": float(
                (per_predictor[column.replace("pvalue", "tstat")].abs() > 3.0).mean()
            ),
        }
    save_table(pd.DataFrame(summary_rows).T, "exp3_multiple_testing_summary")
    save_table(
        per_predictor.sort_values("tstat_nw", key=abs, ascending=False).head(40),
        "exp3_predictors_top40",
    )
    save_table(per_predictor, "exp3_predictors_all")

    # Where does this project's own strategy sit in that distribution?  The
    # honest comparison is against the whole population of published
    # predictors, not against zero.
    own = {}
    for name in models:
        series = gross[name].dropna()
        test = mean_tstat(series.to_numpy(), cfg.newey_west_lags)
        alpha = factor_alpha(series, ff, FF6, cfg.newey_west_lags)
        n = len(per_predictor)
        own[name] = {
            "tstat_nw": test.tstat,
            "percentile_among_published": float(
                (per_predictor["tstat_nw"] < test.tstat).mean() * 100.0
            ),
            "passes_uncorrected": bool(test.pvalue < 0.05),
            "passes_bonferroni": bool(test.pvalue < 0.05 / n),
            "passes_tstat_3_hurdle": bool(abs(test.tstat) > 3.0),
            "ff6_alpha_tstat": alpha["alpha_tstat"],
            "ff6_alpha_passes_bonferroni": bool(alpha["alpha_pvalue"] < 0.05 / n),
        }
    own_table = pd.DataFrame(own).T
    own_table.index.name = "model"
    save_table(own_table, "exp3_own_strategy_vs_published")

    save_json(
        {
            "n_published_predictors_available": int(anomalies.shape[1]),
            "n_published_predictors_used": int(usable.shape[1]),
            "min_months_required": 120,
            "bonferroni_alpha": 0.05 / int(usable.shape[1]),
            "newey_west_lags": cfg.newey_west_lags,
            "bootstrap_draws": cfg.bootstrap_draws,
            "bootstrap_mean_block_months": cfg.block_length,
        },
        "exp3_run",
    )

    print("\n" + pd.DataFrame(summary_rows).T.to_string(float_format=lambda v: f"{v: .3f}"))
    print("\n" + own_table.to_string(float_format=lambda v: f"{v: .3f}"))


if __name__ == "__main__":
    main()
