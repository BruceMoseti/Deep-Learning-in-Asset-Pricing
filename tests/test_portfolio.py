from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xsap.portfolio import (
    backtest,
    cost_breakeven,
    performance,
    quantile_profile,
    quantile_weights,
)


def _scores(n_months=48, n_assets=100, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.period_range("2000-01", periods=n_months, freq="M")
    assets = [f"a{i:03d}" for i in range(n_assets)]
    index = pd.MultiIndex.from_product([months, assets], names=["month", "asset"])
    return pd.Series(rng.normal(size=len(index)), index=index)


def test_long_short_weights_are_dollar_neutral_with_gross_two():
    weights = quantile_weights(_scores(), n_quantiles=10)
    by_month = weights.groupby("month")
    assert by_month.sum().abs().max() == pytest.approx(0.0, abs=1e-12)
    assert by_month.apply(lambda w: w.abs().sum()).min() == pytest.approx(2.0)


def test_only_the_extreme_deciles_are_held():
    weights = quantile_weights(_scores(n_assets=100), n_quantiles=10)
    held = weights[weights != 0].groupby("month").size()
    assert held.unique().tolist() == [20]


def test_rank_weighting_spreads_exposure_across_the_cross_section():
    weights = quantile_weights(_scores(), weighting="rank")
    assert (weights != 0).groupby("month").sum().min() > 50
    assert weights.groupby("month").sum().abs().max() == pytest.approx(0.0, abs=1e-12)


def test_a_static_book_has_zero_turnover_after_it_is_built():
    """Turnover must measure trading, not merely holding."""
    months = pd.period_range("2000-01", periods=24, freq="M")
    assets = [f"a{i:02d}" for i in range(40)]
    index = pd.MultiIndex.from_product([months, assets], names=["month", "asset"])
    constant = pd.Series(np.tile(np.arange(40, dtype=float), len(months)), index=index)
    returns = pd.Series(0.0, index=index)

    result = backtest(constant, returns, cost_bps=(10.0,))
    assert result["turnover"].iloc[0] == pytest.approx(2.0)
    assert result["turnover"].iloc[1:].abs().max() == pytest.approx(0.0, abs=1e-12)


def test_fully_replacing_both_legs_costs_four_units_of_turnover():
    """Exiting and re-entering both legs trades 4x notional, so charge for 4x."""
    months = pd.period_range("2000-01", periods=2, freq="M")
    assets = [f"a{i:02d}" for i in range(20)]
    index = pd.MultiIndex.from_product([months, assets], names=["month", "asset"])
    flipping = pd.Series(
        np.concatenate([np.arange(20, dtype=float), np.arange(20, dtype=float)[::-1]]),
        index=index,
    )
    result = backtest(
        flipping, pd.Series(0.0, index=index), n_quantiles=10, cost_bps=(10.0,)
    )
    assert result["turnover"].iloc[1] == pytest.approx(4.0)


def test_costs_scale_linearly_with_the_rate():
    scores = _scores(seed=1)
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0.0, 0.05, len(scores)), index=scores.index)
    result = backtest(scores, returns, cost_bps=(0.0, 5.0, 10.0, 20.0))

    drag5 = (result["gross"] - result["net_5bps"]).mean()
    drag20 = (result["gross"] - result["net_20bps"]).mean()
    assert result["net_0bps"].equals(result["gross"])
    assert drag20 == pytest.approx(4.0 * drag5)


def test_breakeven_cost_makes_the_net_mean_zero():
    scores = _scores(seed=3)
    rng = np.random.default_rng(4)
    # Give the score real predictive content so the gross mean is positive.
    returns = 0.02 * scores + pd.Series(
        rng.normal(0.0, 0.04, len(scores)), index=scores.index
    )
    result = backtest(scores, returns, cost_bps=(0.0,))
    breakeven = cost_breakeven(result["gross"], result["turnover"])
    net = result["gross"] - result["turnover"] * breakeven / 10_000.0
    assert net.mean() == pytest.approx(0.0, abs=1e-12)


def test_a_constant_score_holds_nothing():
    """A model with no view must hold no position, not an alphabetical portfolio.

    ``rank(method='first')`` breaks ties by row order, so without an explicit
    guard a flat score would be split into deciles by asset name and the
    resulting return credited to the model.  A penalised model that has shrunk
    every coefficient to zero produces exactly this input.
    """
    months = pd.period_range("2000-01", periods=12, freq="M")
    assets = [f"a{i:02d}" for i in range(60)]
    index = pd.MultiIndex.from_product([months, assets], names=["month", "asset"])
    flat = pd.Series(0.0, index=index)
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0.0, 0.05, len(index)), index=index)

    weights = quantile_weights(flat, n_quantiles=10)
    assert weights.abs().max() == pytest.approx(0.0)

    result = backtest(flat, returns, cost_bps=(10.0,))
    assert result["gross"].abs().max() == pytest.approx(0.0)
    assert result["turnover"].abs().max() == pytest.approx(0.0)


def test_a_partly_flat_score_is_only_traded_when_it_has_a_view():
    """Months with a view are traded; months without are held in cash."""
    months = pd.period_range("2000-01", periods=4, freq="M")
    assets = [f"a{i:02d}" for i in range(40)]
    index = pd.MultiIndex.from_product([months, assets], names=["month", "asset"])
    scores = pd.Series(0.0, index=index)
    scores.loc[months[1]] = np.arange(40, dtype=float)
    scores.loc[months[2]] = np.arange(40, dtype=float)

    result = backtest(scores, pd.Series(0.0, index=index), cost_bps=(10.0,))
    # Enter in the second month, hold through the third, exit in the fourth.
    assert result["turnover"].to_numpy() == pytest.approx([0.0, 2.0, 0.0, 2.0])


def test_performance_matches_hand_computed_values():
    returns = pd.Series([0.01] * 24, index=pd.period_range("2000-01", periods=24, freq="M"))
    stats = performance(returns)
    assert stats["ann_return_geom"] == pytest.approx(1.01**12 - 1.0)
    assert stats["ann_vol"] == pytest.approx(0.0)
    assert stats["max_drawdown"] == pytest.approx(0.0)
    assert stats["hit_rate"] == 1.0


def test_max_drawdown_is_measured_peak_to_trough():
    returns = pd.Series(
        [0.10, -0.20, -0.20, 0.30],
        index=pd.period_range("2000-01", periods=4, freq="M"),
    )
    trough = 1.10 * 0.80 * 0.80
    assert performance(returns)["max_drawdown"] == pytest.approx(trough / 1.10 - 1.0)


def test_quantile_profile_is_monotone_for_an_informative_score():
    scores = _scores(n_months=120, seed=5)
    rng = np.random.default_rng(6)
    returns = 0.01 * scores + pd.Series(
        rng.normal(0.0, 0.03, len(scores)), index=scores.index
    )
    profile = quantile_profile(scores, returns)
    assert profile.is_monotonic_increasing
    assert profile.iloc[-1] > profile.iloc[0]
