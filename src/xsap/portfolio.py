"""Turning forecasts into portfolios, and charging for the trading.

Weight convention
-----------------
A long-short book holds ``+1`` of notional in the long leg and ``-1`` in the
short leg, so gross exposure is ``2`` and net exposure is ``0``.  The reported
return is therefore the return per dollar long plus per dollar short -- the
usual academic long-short spread -- and *not* a return on margin posted, which
would be a leverage assumption dressed up as performance.

Turnover convention
-------------------
Two-sided: ``turnover_t = sum_i |w_it - w_i,t-1|``.  Replacing both legs in
full gives ``4``, since each leg is both exited and entered.  Costs are
``turnover_t`` times the one-way rate, which is the natural pairing: every unit
of notional traded pays the rate once.  Positions are not drifted between
rebalances, so this slightly overstates turnover, which is the direction an
honest cost estimate should err in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12


def quantile_weights(
    scores: pd.Series, n_quantiles: int = 10, weighting: str = "equal"
) -> pd.Series:
    """Dollar-neutral long-short weights from a cross-sectional score.

    ``equal`` weights the extreme quantiles equally; ``rank`` spreads exposure
    across the whole cross-section in proportion to the demeaned score, which
    trades less aggressively at the tails.
    """
    if weighting == "rank":
        centred = scores - scores.groupby("month").transform("mean")
        gross = centred.abs().groupby("month").transform("sum")
        return (2.0 * centred / gross.where(gross > 0)).fillna(0.0).rename("weight")

    if weighting != "equal":
        raise ValueError(f"unknown weighting: {weighting}")

    def _month(block: pd.Series) -> pd.Series:
        # A constant score carries no ranking information, so the book is empty.
        # Without this the tie-break below would sort assets by name and build a
        # real portfolio out of nothing, which a penalised model that has
        # shrunk every coefficient to zero would silently be credited for.
        if len(block) < 2 * n_quantiles or block.nunique() < 2:
            return pd.Series(0.0, index=block.index)
        # ``rank(method='first')`` keeps buckets balanced when scores tie.
        buckets = pd.qcut(
            block.rank(method="first"), n_quantiles, labels=False, duplicates="drop"
        )
        weights = pd.Series(0.0, index=block.index)
        top, bottom = buckets == buckets.max(), buckets == 0
        if top.sum() and bottom.sum():
            weights[top] = 1.0 / top.sum()
            weights[bottom] = -1.0 / bottom.sum()
        return weights

    return (
        scores.groupby("month", group_keys=False).apply(_month).rename("weight")
    )


def backtest(
    scores: pd.Series,
    forward_returns: pd.Series,
    n_quantiles: int = 10,
    weighting: str = "equal",
    cost_bps: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0),
) -> pd.DataFrame:
    """Monthly gross return, turnover and net returns for one score.

    ``scores`` and ``forward_returns`` share the ``(month, asset)`` index; the
    forward return is realised in the month after the score is observed, so the
    output is dated by the *formation* month.
    """
    aligned = pd.concat(
        [scores.rename("score"), forward_returns.rename("fwd")], axis=1
    ).dropna()
    weights = quantile_weights(aligned["score"], n_quantiles, weighting)

    gross = (weights * aligned["fwd"]).groupby("month").sum().rename("gross")

    wide = weights.unstack("asset").fillna(0.0).sort_index()
    turnover = wide.diff().abs().sum(axis=1)
    # The first rebalance builds the book from cash; charge for it.
    turnover.iloc[0] = wide.iloc[0].abs().sum()
    turnover = turnover.rename("turnover")

    out = pd.concat([gross, turnover], axis=1)
    for bps in cost_bps:
        out[f"net_{bps:g}bps"] = out["gross"] - out["turnover"] * bps / 10_000.0
    return out


def performance(returns: pd.Series, periods: int = MONTHS) -> dict:
    """Risk and return summary for a monthly return series."""
    r = returns.dropna()
    n = len(r)
    if n < 2:
        return {}
    mean, vol = float(r.mean()), float(r.std(ddof=1))
    ann_vol = vol * np.sqrt(periods)
    cumulative = float(np.expm1(np.log1p(r).sum()))
    ann_geometric = (1.0 + cumulative) ** (periods / n) - 1.0

    level = np.exp(np.log1p(r).cumsum())
    drawdown = level / np.maximum.accumulate(level) - 1.0
    return {
        "mean_monthly": mean,
        "ann_return_arith": mean * periods,
        "ann_return_geom": float(ann_geometric),
        "ann_vol": float(ann_vol),
        "sharpe": float(mean / vol * np.sqrt(periods)) if vol > 0 else np.nan,
        "max_drawdown": float(drawdown.min()),
        "calmar": (
            float(ann_geometric / abs(drawdown.min())) if drawdown.min() < 0 else np.nan
        ),
        "hit_rate": float((r > 0).mean()),
        "skew": float(r.skew()),
        "worst_month": float(r.min()),
        "best_month": float(r.max()),
        "n_months": n,
    }


def cost_breakeven(gross: pd.Series, turnover: pd.Series) -> float:
    """One-way cost in basis points at which mean gross return goes to zero.

    A single number for how much slippage a signal can absorb before it stops
    being a strategy.
    """
    mean_turnover = float(turnover.mean())
    if mean_turnover <= 0:
        return float("inf")
    return float(gross.mean() / mean_turnover * 10_000.0)


def quantile_profile(
    scores: pd.Series, forward_returns: pd.Series, n_quantiles: int = 10
) -> pd.Series:
    """Mean forward return by predicted quantile, annualised.

    A monotone profile is much stronger evidence than a good top-minus-bottom
    spread, which two lucky buckets can produce on their own.
    """
    aligned = pd.concat(
        [scores.rename("score"), forward_returns.rename("fwd")], axis=1
    ).dropna()

    def _bucket(block: pd.Series) -> pd.Series:
        if block.nunique() < 2:
            return pd.Series(np.nan, index=block.index)
        return pd.qcut(
            block.rank(method="first"), n_quantiles, labels=False, duplicates="drop"
        )

    buckets = aligned.groupby("month", group_keys=False)["score"].apply(_bucket)
    return (
        aligned["fwd"].groupby(buckets).mean().mul(MONTHS).rename("ann_return")
    )
