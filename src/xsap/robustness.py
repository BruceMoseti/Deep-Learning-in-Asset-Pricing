"""Where the result comes from, and when it stops working.

Three questions that a single headline number cannot answer.  Which groups of
predictors carry the signal, established by removing each group and refitting
rather than by reading a feature-importance chart -- importance scores describe
a fitted model, not out-of-sample value, and a group can look important while
contributing nothing that survives.  Whether the result depends on choices that
were made for a reason but could have been made differently.  And whether it
holds in the market states that matter, rather than only on average.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xsap.config import Config
from xsap.features import FEATURE_GROUPS, FEATURE_NAMES
from xsap.metrics import summarise_forecasts
from xsap.portfolio import backtest, cost_breakeven, performance
from xsap.walkforward import run_walkforward


def _evaluate(predictions: pd.DataFrame, name: str, cfg: Config) -> dict:
    """Accuracy and portfolio statistics for one column of forecasts."""
    stats = summarise_forecasts(predictions[name], predictions["y"], cfg.newey_west_lags)
    ranked = 2.0 * predictions.groupby("month")[name].rank(pct=True) - 1.0
    result = backtest(
        ranked,
        predictions["fwd_excess_ret"],
        cfg.portfolio.n_quantiles,
        cfg.portfolio.weighting,
        cfg.portfolio.cost_bps,
    )
    gross = performance(result["gross"])
    net = performance(result["net_10bps"])
    return {
        "rank_ic": stats["rank_ic"],
        "rank_ic_tstat": stats["rank_ic_tstat"],
        "icir": stats["icir"],
        "sharpe_gross": gross["sharpe"],
        "sharpe_net_10bps": net["sharpe"],
        "turnover_monthly": float(result["turnover"].mean()),
        "breakeven_cost_bps": cost_breakeven(result["gross"], result["turnover"]),
        "n_months": stats["n_months"],
    }


def feature_group_ablation(
    data: pd.DataFrame, target: pd.Series, model, cfg: Config | None = None
) -> pd.DataFrame:
    """Refit the model with one group of predictors removed at a time.

    Also refits with each group *alone*, because the two answers differ in a
    way that matters: a group can be redundant when the others are present
    (small drop when removed) and still be the only one that works on its own.
    """
    cfg = cfg or Config()
    rows = {}

    full, _ = run_walkforward(data, target, [model], cfg, FEATURE_NAMES, verbose=False)
    rows["all_predictors"] = _evaluate(full, model.name, cfg)
    baseline = rows["all_predictors"]["rank_ic"]

    for group, members in FEATURE_GROUPS.items():
        remaining = tuple(f for f in FEATURE_NAMES if f not in members)
        without, _ = run_walkforward(
            data, target, [model], cfg, remaining, verbose=False
        )
        rows[f"without_{group}"] = _evaluate(without, model.name, cfg)
        rows[f"without_{group}"]["ic_change_vs_all"] = (
            rows[f"without_{group}"]["rank_ic"] - baseline
        )

        only, _ = run_walkforward(data, target, [model], cfg, members, verbose=False)
        rows[f"only_{group}"] = _evaluate(only, model.name, cfg)
        rows[f"only_{group}"]["ic_change_vs_all"] = (
            rows[f"only_{group}"]["rank_ic"] - baseline
        )

    table = pd.DataFrame(rows).T
    table.index.name = "predictor_set"
    return table


def define_regimes(factors: pd.DataFrame, lookback: int = 12) -> pd.DataFrame:
    """Label each month by market conditions known at its start.

    Every label uses only data through the previous month, so a regime split is
    something a strategy could actually have conditioned on.  Splitting on
    contemporaneous or future volatility would be a different and much easier
    exercise.
    """
    market = factors["mktrf"]
    realised_vol = market.rolling(lookback, min_periods=lookback).std().shift(1)
    trailing = market.rolling(lookback, min_periods=lookback).sum().shift(1)
    level = np.exp(np.log1p(market).cumsum())
    drawdown = (level / level.cummax() - 1.0).shift(1)
    rate = factors["rf"].rolling(lookback, min_periods=lookback).mean().shift(1)

    return pd.DataFrame(
        {
            "volatility": np.where(
                realised_vol > realised_vol.expanding(60).median(), "high_vol", "low_vol"
            ),
            "market_direction": np.where(trailing > 0, "market_up", "market_down"),
            "drawdown_state": np.where(drawdown < -0.10, "in_drawdown", "near_highs"),
            "rate_level": np.where(
                rate > rate.expanding(60).median(), "high_rates", "low_rates"
            ),
        },
        index=factors.index,
    ).where(realised_vol.notna())


def regime_analysis(
    predictions: pd.DataFrame,
    strategy_returns: pd.DataFrame,
    regimes: pd.DataFrame,
    models: list[str],
    cfg: Config | None = None,
) -> pd.DataFrame:
    """Accuracy and Sharpe within each regime, for each model."""
    cfg = cfg or Config()
    from xsap.metrics import monthly_ic

    ic = pd.concat(
        {name: monthly_ic(predictions[name], predictions["y"]) for name in models},
        axis=1,
    )
    ic.index = pd.PeriodIndex(ic.index, freq="M")
    returns = strategy_returns.copy()
    returns.index = pd.PeriodIndex(returns.index, freq="M")

    rows = []
    for dimension in regimes.columns:
        labels = regimes[dimension].reindex(ic.index)
        for state, block in labels.groupby(labels):
            if len(block) < 24:
                continue
            for name in models:
                window = returns.loc[block.index, name].dropna()
                rows.append(
                    {
                        "dimension": dimension,
                        "state": state,
                        "model": name,
                        "n_months": len(block),
                        "rank_ic": float(ic.loc[block.index, name].mean()),
                        "sharpe_gross": performance(window).get("sharpe", np.nan),
                    }
                )
    return pd.DataFrame(rows).set_index(["dimension", "state", "model"]).sort_index()


def design_robustness(
    data: pd.DataFrame,
    target: pd.Series,
    model,
    cfg: Config | None = None,
) -> pd.DataFrame:
    """Re-run the whole pipeline under alternative design choices.

    Each row changes exactly one decision away from the default so that a
    change in the result can be attributed to it.
    """
    cfg = cfg or Config()
    variants: dict[str, tuple[Config, int | None]] = {"default": (cfg, None)}

    rolling = Config(
        sample_start=cfg.sample_start,
        sample_end=cfg.sample_end,
        target=cfg.target,
        newey_west_lags=cfg.newey_west_lags,
    )
    variants["rolling_20y_window"] = (rolling, 20)
    variants["rolling_10y_window"] = (rolling, 10)

    rows = {}
    for label, (variant_cfg, rolling_years) in variants.items():
        predictions, _ = run_walkforward(
            data,
            target,
            [model],
            variant_cfg,
            FEATURE_NAMES,
            rolling_years=rolling_years,
            verbose=False,
        )
        rows[label] = _evaluate(predictions, model.name, variant_cfg)

    # Portfolio-side choices reuse the default forecasts, since they change how
    # a forecast is traded rather than how it is produced.
    default, _ = run_walkforward(data, target, [model], cfg, FEATURE_NAMES, verbose=False)
    ranked = 2.0 * default.groupby("month")[model.name].rank(pct=True) - 1.0
    for label, quantiles, weighting in (
        ("quintile_portfolios", 5, "equal"),
        ("ventile_portfolios", 20, "equal"),
        ("rank_weighted_all_assets", cfg.portfolio.n_quantiles, "rank"),
    ):
        result = backtest(
            ranked,
            default["fwd_excess_ret"],
            quantiles,
            weighting,
            cfg.portfolio.cost_bps,
        )
        gross = performance(result["gross"])
        rows[label] = {
            "rank_ic": rows["default"]["rank_ic"],
            "rank_ic_tstat": rows["default"]["rank_ic_tstat"],
            "icir": rows["default"]["icir"],
            "sharpe_gross": gross["sharpe"],
            "sharpe_net_10bps": performance(result["net_10bps"])["sharpe"],
            "turnover_monthly": float(result["turnover"].mean()),
            "breakeven_cost_bps": cost_breakeven(result["gross"], result["turnover"]),
            "n_months": gross["n_months"],
        }

    table = pd.DataFrame(rows).T
    table.index.name = "variant"
    return table


def subperiod_analysis(
    predictions: pd.DataFrame,
    strategy_returns: pd.DataFrame,
    models: list[str],
    breakpoints: tuple[int, ...] = (2000, 2010),
) -> pd.DataFrame:
    """Split the out-of-sample period at fixed dates and re-measure.

    Fixed calendar breakpoints chosen in advance, not where the series happens
    to change: picking the split after seeing the data is how a stable result
    is made to look unstable, or the reverse.
    """
    from xsap.metrics import monthly_ic

    ic = pd.concat(
        {name: monthly_ic(predictions[name], predictions["y"]) for name in models},
        axis=1,
    )
    ic.index = pd.PeriodIndex(ic.index, freq="M")
    returns = strategy_returns.copy()
    returns.index = pd.PeriodIndex(returns.index, freq="M")

    edges = (ic.index.year.min(), *breakpoints, ic.index.year.max() + 1)
    rows = []
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (ic.index.year >= start) & (ic.index.year < end)
        if mask.sum() < 24:
            continue
        for name in models:
            rows.append(
                {
                    "period": f"{start}-{end - 1}",
                    "model": name,
                    "n_months": int(mask.sum()),
                    "rank_ic": float(ic.loc[mask, name].mean()),
                    "sharpe_gross": performance(returns.loc[mask, name]).get(
                        "sharpe", np.nan
                    ),
                }
            )
    return pd.DataFrame(rows).set_index(["period", "model"]).sort_index()
