"""Forecast-accuracy measures for cross-sectional predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from xsap.inference import mean_tstat, newey_west_variance


def monthly_ic(
    predictions: pd.Series, targets: pd.Series, method: str = "spearman"
) -> pd.Series:
    """Cross-sectional correlation between forecast and outcome, month by month.

    Reported per month rather than pooled because pooling would let months with
    a wide cross-sectional return dispersion dominate, and because the
    month-to-month series is what determines whether the average is reliable.
    """
    frame = pd.concat([predictions.rename("p"), targets.rename("y")], axis=1).dropna()

    def _corr(block: pd.DataFrame) -> float:
        if len(block) < 5 or block["y"].nunique() < 2:
            return np.nan
        if block["p"].nunique() < 2:
            # A penalised model sometimes shrinks every coefficient to zero and
            # so declines to rank the cross-section at all.  That is a forecast
            # of "no view", worth exactly zero, and it is scored as zero.
            # Dropping these months instead would average the model's IC over
            # only the months in which it chose to have an opinion, which
            # selects on the model's own confidence and flatters it: Lasso does
            # this in 44% of months here, and dropping them raised its apparent
            # IC from 0.025 to 0.044.
            return 0.0
        if method == "spearman":
            return stats.spearmanr(block["p"], block["y"]).statistic
        return float(np.corrcoef(block["p"], block["y"])[0, 1])

    return frame.groupby("month").apply(_corr, include_groups=False).rename(f"ic_{method}")


def r2_oos(predictions: pd.Series, targets: pd.Series) -> float:
    """Out-of-sample R-squared against a zero forecast.

    The target is already demeaned within each month, so zero -- "no view on
    any asset" -- is the honest benchmark.  Using the *realised* cross-sectional
    mean of the test period as the benchmark instead would hand the model
    information it could not have had, and inflates this number.
    """
    frame = pd.concat([predictions.rename("p"), targets.rename("y")], axis=1).dropna()
    sse = float(((frame["y"] - frame["p"]) ** 2).sum())
    sst = float((frame["y"] ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else float("nan")


def calibration_slope(predictions: pd.Series, targets: pd.Series) -> float:
    """Slope from regressing the outcome on the forecast, pooled out of sample.

    A slope near one means the forecast is on the right scale; a slope well
    below one means the forecast is directionally useful but too large, which
    is how a model can have a positive information coefficient and a negative
    out-of-sample R-squared at the same time.  Reporting this makes that
    distinction visible instead of leaving it as an apparent contradiction.
    """
    frame = pd.concat([predictions.rename("p"), targets.rename("y")], axis=1).dropna()
    var = float(frame["p"].var())
    if var <= 0:
        return float("nan")
    return float(frame[["p", "y"]].cov().iloc[0, 1] / var)


def summarise_forecasts(
    predictions: pd.Series, targets: pd.Series, nw_lags: int = 6
) -> dict:
    """Headline accuracy statistics for one model."""
    rank_ic = monthly_ic(predictions, targets, "spearman")
    pearson_ic = monthly_ic(predictions, targets, "pearson")
    test = mean_tstat(rank_ic, lags=nw_lags)
    return {
        "rank_ic": float(rank_ic.mean()),
        "rank_ic_tstat": test.tstat,
        "rank_ic_std": float(rank_ic.std()),
        # Information ratio of the IC series: how consistent the signal is,
        # independent of how large it is.
        "icir": float(rank_ic.mean() / rank_ic.std()) if rank_ic.std() > 0 else np.nan,
        "pearson_ic": float(pearson_ic.mean()),
        "hit_rate": float((rank_ic > 0).mean()),
        "r2_oos": r2_oos(predictions, targets),
        "calibration_slope": calibration_slope(predictions, targets),
        "n_months": int(rank_ic.notna().sum()),
    }


def ic_difference_test(
    predictions_a: pd.Series,
    predictions_b: pd.Series,
    targets: pd.Series,
    lags: int = 6,
) -> dict:
    """Paired test that two forecasts have the same mean information coefficient.

    The Diebold-Mariano test compares squared error, but the portfolio only uses
    the ranking, so the headline metric here is the information coefficient.
    Its month-by-month difference is paired -- both models face the same
    cross-section each month -- which removes the common component and gives a
    far tighter test than comparing two standard errors.
    """
    ic_a = monthly_ic(predictions_a, targets)
    ic_b = monthly_ic(predictions_b, targets)
    difference = (ic_a - ic_b).dropna()
    test = mean_tstat(difference.to_numpy(), lags=lags)
    return {
        "ic_a": float(ic_a.mean()),
        "ic_b": float(ic_b.mean()),
        "ic_difference": test.mean,
        "tstat": test.tstat,
        "pvalue": test.pvalue,
        "n_months": test.n_obs,
    }


def diebold_mariano(
    predictions_a: pd.Series,
    predictions_b: pd.Series,
    targets: pd.Series,
    lags: int = 6,
) -> dict:
    """Test of equal squared-error accuracy between two forecasts.

    Comparing two models by whose average IC is larger says nothing about
    whether the gap is distinguishable from noise.  This puts a standard error
    on the difference, using the monthly loss differential so the serial
    dependence in relative performance is accounted for.
    """
    frame = pd.concat(
        [predictions_a.rename("a"), predictions_b.rename("b"), targets.rename("y")],
        axis=1,
    ).dropna()
    loss_a = (frame["y"] - frame["a"]) ** 2
    loss_b = (frame["y"] - frame["b"]) ** 2
    monthly = (loss_a - loss_b).groupby("month").mean()
    n = len(monthly)
    lrv = newey_west_variance(monthly.to_numpy(), lags)
    stderr = float(np.sqrt(lrv / n)) if n > 2 else float("nan")
    diff = float(monthly.mean())
    tstat = diff / stderr if stderr > 0 else float("nan")
    return {
        "loss_diff": diff,
        "tstat": tstat,
        "pvalue": float(2.0 * stats.norm.sf(abs(tstat))) if np.isfinite(tstat) else np.nan,
        "favours": "b" if diff > 0 else "a",
        "n_months": n,
    }
