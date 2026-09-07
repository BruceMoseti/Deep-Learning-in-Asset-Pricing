#!/usr/bin/env python3
"""Experiment 2 -- is the forecast accuracy worth anything after costs?

Turns each model's out-of-sample forecasts into a monthly decile long-short
portfolio and charges for the trading.  A statistically detectable information
coefficient and a tradable strategy are different claims, and the gap between
them is the point of this experiment, so the headline number reported here is
the break-even cost: the one-way rate in basis points at which the strategy's
average gross return is exactly consumed by turnover.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from xsap.artifacts import load_series, read_json, save_series, save_table
from xsap.config import Config
from xsap.portfolio import backtest, cost_breakeven, performance, quantile_profile
from xsap.walkforward import standardise_predictions


def main() -> None:
    cfg = Config()
    predictions = load_series("predictions")
    models = read_json("exp1_run")["models"]
    scored = standardise_predictions(predictions, models)

    rows, monthly, profiles = {}, {}, {}
    for name in models:
        result = backtest(
            scored[name + "_rank"],
            scored["fwd_excess_ret"],
            n_quantiles=cfg.portfolio.n_quantiles,
            weighting=cfg.portfolio.weighting,
            cost_bps=cfg.portfolio.cost_bps,
        )
        monthly[name] = result

        row = {"turnover_monthly": float(result["turnover"].mean())}
        row["turnover_fraction_of_book"] = row["turnover_monthly"] / 4.0
        gross = performance(result["gross"])
        row["ann_return_gross"] = gross["ann_return_arith"]
        row["ann_vol"] = gross["ann_vol"]
        row["sharpe_gross"] = gross["sharpe"]
        row["max_drawdown_gross"] = gross["max_drawdown"]
        row["hit_rate"] = gross["hit_rate"]
        for bps in cfg.portfolio.cost_bps:
            if bps == 0:
                continue
            net = performance(result[f"net_{bps:g}bps"])
            row[f"sharpe_net_{bps:g}bps"] = net["sharpe"]
            row[f"ann_return_net_{bps:g}bps"] = net["ann_return_arith"]
        row["breakeven_cost_bps"] = cost_breakeven(result["gross"], result["turnover"])
        rows[name] = row

        profiles[name] = quantile_profile(
            scored[name + "_rank"], scored["fwd_excess_ret"], cfg.portfolio.n_quantiles
        )

    table = pd.DataFrame(rows).T
    table.index.name = "model"
    save_table(table, "exp2_portfolio_performance")

    returns = pd.concat({name: m["gross"] for name, m in monthly.items()}, axis=1)
    turnover = pd.concat({name: m["turnover"] for name, m in monthly.items()}, axis=1)
    save_series(returns, "strategy_returns_gross")
    save_series(turnover, "strategy_turnover")
    for bps in cfg.portfolio.cost_bps:
        net = pd.concat(
            {name: m[f"net_{bps:g}bps"] for name, m in monthly.items()}, axis=1
        )
        save_series(net, f"strategy_returns_net_{bps:g}bps")

    profile = pd.DataFrame(profiles)
    profile.index.name = "predicted_decile"
    save_table(profile, "exp2_quantile_profile")

    # How much of the gross Sharpe each cost level removes, in one place.
    erosion = pd.DataFrame(
        {
            f"{bps:g}bps": table[f"sharpe_net_{bps:g}bps"] / table["sharpe_gross"]
            for bps in cfg.portfolio.cost_bps
            if bps > 0
        }
    )
    erosion.insert(0, "sharpe_gross", table["sharpe_gross"])
    erosion.index.name = "model"
    save_table(erosion, "exp2_cost_erosion")

    columns = [
        "sharpe_gross",
        "sharpe_net_5bps",
        "sharpe_net_10bps",
        "sharpe_net_20bps",
        "turnover_monthly",
        "breakeven_cost_bps",
    ]
    print("\n" + table[columns].to_string(float_format=lambda v: f"{v: .3f}"))


if __name__ == "__main__":
    main()
