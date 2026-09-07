"""Statistical inference for return series.

Monthly strategy returns are serially dependent and heavy tailed, so an
i.i.d. t-statistic overstates precision.  Everything here is built to give a
defensible standard error under those conditions: Newey-West for the
autocorrelation, a stationary bootstrap when the statistic is not a mean, GRS
for the joint hypothesis that a set of alphas is zero, and Bonferroni /
Benjamini-Hochberg when many hypotheses are tested at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def newey_west_variance(x: np.ndarray, lags: int) -> float:
    """Bartlett-kernel long-run variance of a series, for the variance of its mean."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    centred = x - x.mean()
    gamma0 = float(centred @ centred) / n
    total = gamma0
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = float(centred[lag:] @ centred[:-lag]) / n
        total += 2.0 * weight * gamma
    return max(total, 0.0)


@dataclass(frozen=True)
class MeanTest:
    mean: float
    stderr: float
    tstat: float
    pvalue: float
    n_obs: int


def mean_tstat(x, lags: int = 6) -> MeanTest:
    """Newey-West t-test that the mean of ``x`` is zero."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return MeanTest(float("nan"), float("nan"), float("nan"), float("nan"), n)
    lrv = newey_west_variance(x, lags)
    stderr = float(np.sqrt(lrv / n))
    mean = float(x.mean())
    tstat = mean / stderr if stderr > 0 else float("nan")
    pvalue = float(2.0 * stats.norm.sf(abs(tstat))) if np.isfinite(tstat) else float("nan")
    return MeanTest(mean, stderr, tstat, pvalue, n)


def ols_newey_west(y, x, lags: int = 6, add_constant: bool = True) -> dict:
    """OLS with Newey-West standard errors.  ``x`` is (T, K), ``y`` is (T,)."""
    y = np.asarray(y, dtype=float)
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[0] != len(y):
        x = x.T
    if add_constant:
        x = np.column_stack([np.ones(len(y)), x])

    keep = np.isfinite(y) & np.isfinite(x).all(axis=1)
    y, x = y[keep], x[keep]
    n, k = x.shape

    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta

    # Bartlett-weighted meat matrix.
    scores = x * resid[:, None]
    meat = scores.T @ scores
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        block = scores[lag:].T @ scores[:-lag]
        meat += weight * (block + block.T)
    cov = xtx_inv @ meat @ xtx_inv

    stderr = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = beta / stderr
    tss = float(((y - y.mean()) ** 2).sum())
    return {
        "params": beta,
        "stderr": stderr,
        "tstat": tstat,
        "pvalue": 2.0 * stats.norm.sf(np.abs(tstat)),
        "r2": 1.0 - float((resid**2).sum()) / tss if tss > 0 else float("nan"),
        "n_obs": n,
        "resid": resid,
    }


FF5 = ("mktrf", "smb", "hml", "rmw", "cma")
FF6 = (*FF5, "mom")
CAPM = ("mktrf",)


def factor_alpha(returns: pd.Series, factors: pd.DataFrame, model=FF6, lags: int = 6) -> dict:
    """Regress a strategy's return on a factor model and report the intercept.

    The intercept is the part of average performance the factor model cannot
    explain.  It is the number that decides whether a strategy has found
    something, or has rediscovered a premium that is already for sale.
    """
    aligned = pd.concat([returns.rename("y"), factors[list(model)]], axis=1).dropna()
    fit = ols_newey_west(aligned["y"].to_numpy(), aligned[list(model)].to_numpy(), lags=lags)
    return {
        "alpha": float(fit["params"][0]),
        "alpha_tstat": float(fit["tstat"][0]),
        "alpha_pvalue": float(fit["pvalue"][0]),
        "alpha_annual": float(fit["params"][0]) * 12.0,
        "betas": dict(zip(model, fit["params"][1:])),
        "beta_tstats": dict(zip(model, fit["tstat"][1:])),
        "r2": fit["r2"],
        "n_obs": fit["n_obs"],
        "model": "+".join(model),
    }


def stationary_bootstrap_indices(n: int, mean_block: int, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap: geometric blocks, wrapping around.

    Resampling in blocks preserves the serial dependence of monthly returns,
    which an i.i.d. bootstrap would destroy and so would understate the
    sampling variability of a Sharpe ratio.
    """
    restart = rng.random(n) < 1.0 / mean_block
    restart[0] = True
    starts = np.flatnonzero(restart)
    position = np.arange(n)
    segment = np.searchsorted(starts, position, side="right") - 1
    origin = rng.integers(0, n, size=len(starts))
    return (origin[segment] + position - starts[segment]) % n


def bootstrap_statistic(
    x, statistic, n_draws: int = 5000, mean_block: int = 12, seed: int = 0
) -> dict:
    """Stationary-bootstrap distribution of ``statistic`` applied to ``x``."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws)
    for b in range(n_draws):
        draws[b] = statistic(x[stationary_bootstrap_indices(len(x), mean_block, rng)])
    point = float(statistic(x))
    return {
        "point": point,
        "mean": float(draws.mean()),
        "std": float(draws.std(ddof=1)),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
        # Share of resamples in which the statistic flips sign, a direct read on
        # how fragile the result is to the particular history we observed.
        "p_wrong_sign": float(np.mean(np.sign(draws) != np.sign(point))),
        "draws": draws,
    }


def grs_test(returns: pd.DataFrame, factors: pd.DataFrame) -> dict:
    """Gibbons, Ross and Shanken (1989) test that all alphas are jointly zero.

    Requires ``T > N + K``.  The whole point of Experiment 4 is that this
    requirement is not merely technical: as ``N/T`` rises the test's actual
    rejection rate under the null drifts far above its nominal level.
    """
    aligned = pd.concat([returns, factors], axis=1).dropna()
    r = aligned[returns.columns].to_numpy(dtype=float)
    f = aligned[factors.columns].to_numpy(dtype=float)
    n_obs, n_assets = r.shape
    n_factors = f.shape[1]
    dof = n_obs - n_assets - n_factors
    if dof <= 0:
        return {
            "statistic": float("nan"),
            "pvalue": float("nan"),
            "n_obs": n_obs,
            "n_assets": n_assets,
            "note": "T <= N + K, statistic undefined",
        }

    design = np.column_stack([np.ones(n_obs), f])
    coef = np.linalg.lstsq(design, r, rcond=None)[0]
    alpha = coef[0]
    resid = r - design @ coef
    sigma = resid.T @ resid / n_obs
    mu_f = f.mean(axis=0)
    omega = np.cov(f, rowvar=False, bias=True).reshape(n_factors, n_factors)

    sharpe_f = float(mu_f @ np.linalg.pinv(omega) @ mu_f)
    quad = float(alpha @ np.linalg.pinv(sigma) @ alpha)
    statistic = (dof / n_assets) * quad / (1.0 + sharpe_f)
    return {
        "statistic": float(statistic),
        "pvalue": float(stats.f.sf(statistic, n_assets, dof)),
        "n_obs": n_obs,
        "n_assets": n_assets,
        "n_factors": n_factors,
        "mean_abs_alpha": float(np.mean(np.abs(alpha))),
    }


def benjamini_hochberg(pvalues, alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg step-up procedure controlling the false discovery rate."""
    p = np.asarray(pvalues, dtype=float)
    finite = np.isfinite(p)
    n = int(finite.sum())
    order = np.argsort(np.where(finite, p, np.inf))
    ranked = p[order][:n]
    thresholds = alpha * np.arange(1, n + 1) / n
    passing = np.nonzero(ranked <= thresholds)[0]
    cutoff = ranked[passing.max()] if passing.size else 0.0
    rejected = finite & (p <= cutoff)
    return {
        "n_tests": n,
        "cutoff": float(cutoff),
        "n_rejected": int(rejected.sum()),
        "rejected": rejected,
    }


def multiple_testing_summary(pvalues, names, alpha: float = 0.05) -> pd.DataFrame:
    """Discoveries under no correction, Bonferroni, and Benjamini-Hochberg."""
    p = np.asarray(pvalues, dtype=float)
    n = int(np.isfinite(p).sum())
    bh = benjamini_hochberg(p, alpha)
    return pd.DataFrame(
        {
            "name": list(names),
            "pvalue": p,
            "uncorrected": p <= alpha,
            "bonferroni": p <= alpha / n,
            "benjamini_hochberg": bh["rejected"],
        }
    )
