"""Point-in-time predictors built from the return panel.

Timing convention, which every downstream module relies on
----------------------------------------------------------
A row keyed ``(month=t, asset=i)`` holds

* predictors computed **only** from returns realised in months <= t, and
* ``fwd_excess_ret``, the excess return of asset ``i`` over month ``t+1``.

So a portfolio formed at the close of month ``t`` from these predictors earns
``fwd_excess_ret``.  Nothing in a row dated ``t`` is a function of a return
after ``t``, except the label itself.  ``tests/test_no_lookahead.py`` checks
this by construction rather than by inspection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xsap.config import Config

# Predictor groups.  Used for the ablation study and to organise the report.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "momentum": ("mom_2_6", "mom_2_12", "mom_7_12"),
    "reversal": ("ret_1m", "mom_13_36"),
    "volatility": ("vol_6", "vol_12", "vol_36", "dnvol_36", "volofvol_36"),
    "comovement": ("beta_36", "beta_60", "idiovol_36", "corr_36", "coskew_36"),
    "higher_moments": ("skew_36", "kurt_36"),
    "trend": ("maxdd_36", "dist_high_36", "sharpe_12", "sharpe_36"),
    "seasonality": ("seas_20y",),
    "persistence": ("ac1_36",),
    "static": ("sort1", "sort2", "is_industry"),
}

FEATURE_NAMES: tuple[str, ...] = tuple(
    name for names in FEATURE_GROUPS.values() for name in names
)


def _wide(panel: pd.DataFrame, value: str = "ret") -> pd.DataFrame:
    wide = panel.pivot(index="month", columns="asset", values=value).sort_index()
    wide.columns.name = None
    return wide


def _cum_return(excess: pd.DataFrame, start_lag: int, end_lag: int) -> pd.DataFrame:
    """Compound returns over months ``t-start_lag .. t-end_lag`` inclusive."""
    window = start_lag - end_lag + 1
    log1p = np.log1p(excess)
    rolled = log1p.rolling(window, min_periods=window).sum().shift(end_lag)
    return np.expm1(rolled)


def _rolling_windows(values: np.ndarray, window: int) -> np.ndarray:
    """(T, N) -> (T, N, window) with NaN padding for the first window-1 rows."""
    n_time, n_asset = values.shape
    out = np.full((n_time, n_asset, window), np.nan)
    view = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)
    out[window - 1 :] = view
    return out


def _drawdown_features(excess: pd.DataFrame, window: int = 36) -> pd.DataFrame:
    """Trailing max drawdown and distance from the trailing high."""
    windows = _rolling_windows(excess.to_numpy(dtype=float), window)
    level = np.cumprod(1.0 + windows, axis=2)
    running_max = np.maximum.accumulate(level, axis=2)
    drawdown = level / running_max - 1.0

    # A window containing any NaN return makes the whole path meaningless.
    invalid = np.isnan(windows).any(axis=2)
    maxdd = np.where(invalid, np.nan, np.min(np.nan_to_num(drawdown, nan=0.0), axis=2))
    dist_high = np.where(invalid, np.nan, drawdown[:, :, -1])

    idx, cols = excess.index, excess.columns
    return pd.DataFrame(maxdd, index=idx, columns=cols), pd.DataFrame(
        dist_high, index=idx, columns=cols
    )


def _seasonality(excess: pd.DataFrame, years: int = 20, min_obs: int = 5) -> pd.DataFrame:
    """Mean same-calendar-month excess return over the previous ``years`` years.

    The annual seasonality documented by Heston and Sadka (2008).  Lags are
    strictly positive multiples of 12, so month ``t`` itself is excluded.
    """
    total = pd.DataFrame(0.0, index=excess.index, columns=excess.columns)
    count = pd.DataFrame(0.0, index=excess.index, columns=excess.columns)
    for k in range(1, years + 1):
        lagged = excess.shift(12 * k)
        total = total.add(lagged.fillna(0.0))
        count = count.add(lagged.notna().astype(float))
    mean = total / count.where(count > 0)
    return mean.where(count >= min_obs)


def build_features(
    panel: pd.DataFrame, factors: pd.DataFrame, cfg: Config | None = None
) -> pd.DataFrame:
    """Return a tidy frame of predictors and next-month labels.

    ``panel`` is the long return panel from :mod:`xsap.data.universe`;
    ``factors`` supplies the risk-free rate and the market factor.
    """
    cfg = cfg or Config()
    returns = _wide(panel)
    rf = factors["rf"].reindex(returns.index)
    market = factors["mktrf"].reindex(returns.index)
    excess = returns.sub(rf, axis=0)

    feats: dict[str, pd.DataFrame] = {}

    feats["ret_1m"] = excess
    feats["mom_2_6"] = _cum_return(excess, 5, 1)
    feats["mom_2_12"] = _cum_return(excess, 11, 1)
    feats["mom_7_12"] = _cum_return(excess, 11, 6)
    feats["mom_13_36"] = _cum_return(excess, 35, 12)

    for window in (6, 12, 36):
        feats[f"vol_{window}"] = excess.rolling(window, min_periods=window).std()
    downside = excess.clip(upper=0.0)
    feats["dnvol_36"] = np.sqrt(
        (downside**2).rolling(36, min_periods=36).mean()
    )
    feats["volofvol_36"] = feats["vol_6"].rolling(36, min_periods=36).std()

    for window in (36, 60):
        mean_x = excess.rolling(window, min_periods=window).mean()
        mean_m = market.rolling(window, min_periods=window).mean()
        cov = (
            excess.mul(market, axis=0).rolling(window, min_periods=window).mean()
            - mean_x.mul(mean_m, axis=0)
        )
        var_m = market.rolling(window, min_periods=window).var(ddof=0)
        feats[f"beta_{window}"] = cov.div(var_m, axis=0)
        if window == 36:
            var_x = excess.rolling(window, min_periods=window).var(ddof=0)
            beta = feats["beta_36"]
            resid_var = (var_x - beta.pow(2).mul(var_m, axis=0)).clip(lower=0.0)
            feats["idiovol_36"] = np.sqrt(resid_var)
            feats["corr_36"] = cov.div(np.sqrt(var_x).mul(np.sqrt(var_m), axis=0))

            # Harvey and Siddique (2000) coskewness, normalised by sd(x)*var(m).
            e_xmm = (
                excess.mul(market.pow(2), axis=0)
                .rolling(window, min_periods=window)
                .mean()
            )
            e_xm = excess.mul(market, axis=0).rolling(window, min_periods=window).mean()
            numer = (
                e_xmm
                - mean_x.mul(var_m, axis=0)
                + mean_x.mul(mean_m.pow(2), axis=0)
                - 2.0 * e_xm.mul(mean_m, axis=0)
            )
            feats["coskew_36"] = numer.div(
                np.sqrt(var_x).mul(var_m, axis=0)
            )

    feats["skew_36"] = excess.rolling(36, min_periods=36).skew()
    feats["kurt_36"] = excess.rolling(36, min_periods=36).kurt()

    feats["maxdd_36"], feats["dist_high_36"] = _drawdown_features(excess)
    for window in (12, 36):
        mean_x = excess.rolling(window, min_periods=window).mean()
        std_x = excess.rolling(window, min_periods=window).std()
        feats[f"sharpe_{window}"] = mean_x / std_x

    feats["seas_20y"] = _seasonality(excess)

    lag1 = excess.shift(1)
    mean_x = excess.rolling(36, min_periods=36).mean()
    mean_l = lag1.rolling(36, min_periods=36).mean()
    cov_l = (excess * lag1).rolling(36, min_periods=36).mean() - mean_x * mean_l
    feats["ac1_36"] = cov_l / (
        excess.rolling(36, min_periods=36).std(ddof=0)
        * lag1.rolling(36, min_periods=36).std(ddof=0)
    )

    long = pd.concat(
        {name: frame.stack(future_stack=True) for name, frame in feats.items()}, axis=1
    )
    long.index.names = ["month", "asset"]

    static = (
        panel.drop_duplicates("asset")
        .set_index("asset")[["family", "sort1", "sort2", "is_industry"]]
    )
    out = long.join(static, on="asset")
    out["fwd_excess_ret"] = excess.shift(-1).stack(future_stack=True)

    # An asset needs enough history for the longest-lookback predictor.
    have_history = out[
        [c for c in FEATURE_NAMES if c not in FEATURE_GROUPS["static"]]
    ].notna().all(axis=1)
    out = out[have_history & out["fwd_excess_ret"].notna()]

    counts = out.groupby("month").size()
    keep = counts[counts >= cfg.min_assets_per_month].index
    return out[out.index.get_level_values("month").isin(keep)].sort_index()


def _to_unit_interval(ranks: pd.Series, counts: pd.Series) -> pd.Series:
    """Map ranks 1..n onto [-1, 1] with mean zero.

    ``rank(pct=True)`` divides by ``n``, which puts the lowest asset at ``1/n``
    rather than at zero and leaves the transformed cross-section with a mean of
    ``+1/n`` instead of zero.  Dividing by ``n - 1`` after subtracting one puts
    the extremes exactly at the endpoints and centres the cross-section.
    """
    spread = (counts - 1.0).where(counts > 1)
    return 2.0 * (ranks - 1.0) / spread - 1.0


def rank_normalise(frame: pd.DataFrame, columns=FEATURE_NAMES) -> pd.DataFrame:
    """Map each predictor to [-1, 1] by its cross-sectional rank within a month.

    This is the transform used by Gu, Kelly and Xiu (2020).  It is applied one
    month at a time, so it uses no information from any other date, and it makes
    the predictors scale-free and insensitive to the heavy tails and level
    shifts that a rolling z-score would propagate.  Missing values map to 0, the
    centre of the cross-section.
    """
    out = frame.copy()
    grouped = out.groupby("month", group_keys=False)
    for column in columns:
        ranks = grouped[column].rank()
        counts = grouped[column].transform("count")
        out[column] = _to_unit_interval(ranks, counts).fillna(0.0)
    return out


def make_target(frame: pd.DataFrame, mode: str = "zscore") -> pd.Series:
    """Cross-sectionally standardised next-month excess return.

    Demeaning within a month strips out the market return, which dominates the
    variance of raw returns and is essentially unforecastable at the monthly
    horizon.  What is left is exactly the quantity the strategy trades: relative
    performance within the cross-section.
    """
    y = frame["fwd_excess_ret"]
    grouped = y.groupby("month")
    if mode == "zscore":
        return ((y - grouped.transform("mean")) / grouped.transform("std")).rename("y")
    if mode == "rank":
        return _to_unit_interval(grouped.rank(), grouped.transform("count")).rename("y")
    raise ValueError(f"unknown target mode: {mode}")
