"""Assemble the asset-month return panel and the factor series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xsap.config import Config
from xsap.data import sources
from xsap.data.french import read_factors, read_french_csv, sort_ranks

# Portfolio sets whose columns are industries rather than ordered sorts.
_INDUSTRY_SETS = frozenset({"17_Industry_Portfolios", "49_Industry_Portfolios"})


def load_portfolio_panel(cfg: Config | None = None) -> pd.DataFrame:
    """Return a long panel of monthly total returns on the test-asset universe.

    Columns: ``month``, ``asset``, ``family``, ``ret``, ``sort1``, ``sort2``,
    ``is_industry``.  Returns are decimal (not percent).

    Portfolios whose label is not an ordered quintile in both dimensions are
    dropped: the net-issues set contains ``NegNI``/``ZeroNI`` buckets that are
    categorical, not points on a sort, so they are not comparable across the
    cross-section.
    """
    cfg = cfg or Config()
    frames = []
    for source in sources.PORTFOLIO_SETS:
        wide = read_french_csv(sources.fetch(source))
        industry = source.key in _INDUSTRY_SETS

        keep, ranks = [], {}
        for column in wide.columns:
            s1, s2 = sort_ranks(column)
            if industry:
                keep.append(column)
                ranks[column] = (np.nan, np.nan)
            elif not (np.isnan(s1) or np.isnan(s2)):
                keep.append(column)
                ranks[column] = (s1, s2)

        long = (
            wide[keep]
            .stack()
            .rename("ret")
            .reset_index()
            .rename(columns={"level_1": "portfolio"})
        )
        long["family"] = source.key
        long["asset"] = source.key + ":" + long["portfolio"]
        long["sort1"] = long["portfolio"].map(lambda c: ranks[c][0])
        long["sort2"] = long["portfolio"].map(lambda c: ranks[c][1])
        long["is_industry"] = float(industry)
        frames.append(long.drop(columns="portfolio"))

    panel = pd.concat(frames, ignore_index=True)
    panel["ret"] = panel["ret"] / 100.0

    start = pd.Period(cfg.sample_start, freq="M")
    end = pd.Period(cfg.sample_end, freq="M")
    panel = panel[(panel["month"] >= start) & (panel["month"] <= end)]
    return panel.sort_values(["month", "asset"], ignore_index=True)


def load_factors(cfg: Config | None = None) -> pd.DataFrame:
    """Return monthly factor returns in decimal: mktrf, smb, hml, rmw, cma, rf, mom."""
    cfg = cfg or Config()
    ff5 = read_factors(
        sources.fetch(sources.FF5),
        rename={
            "Mkt-RF": "mktrf",
            "SMB": "smb",
            "HML": "hml",
            "RMW": "rmw",
            "CMA": "cma",
            "RF": "rf",
        },
    )
    mom = read_factors(sources.fetch(sources.MOM), rename={"Mom": "mom"})
    factors = ff5.join(mom[["mom"]], how="left")

    start = pd.Period(cfg.sample_start, freq="M")
    end = pd.Period(cfg.sample_end, freq="M")
    return factors.loc[(factors.index >= start) & (factors.index <= end)]


def load_anomaly_returns(cfg: Config | None = None) -> pd.DataFrame:
    """Return monthly long-short returns of published predictors, in decimal.

    One column per predictor.  Coverage is ragged: predictors start when the
    data they need becomes available, so columns have different first dates.
    """
    cfg = cfg or Config()
    raw = pd.read_csv(sources.fetch(sources.ANOMALIES))
    month = pd.PeriodIndex(pd.to_datetime(raw["date"]), freq="M")
    wide = raw.drop(columns="date").set_axis(month, axis=0).sort_index() / 100.0
    wide.index.name = "month"

    start = pd.Period(cfg.sample_start, freq="M")
    end = pd.Period(cfg.sample_end, freq="M")
    return wide.loc[(wide.index >= start) & (wide.index <= end)]
