#!/usr/bin/env python3
"""Download the raw data and print its provenance.

Run this first.  It writes ``data/raw/manifest.json``, which records the URL,
byte count and SHA-256 of every file, so a later run can prove it used the
same data vintage.
"""

from __future__ import annotations

import argparse

import pandas as pd

from xsap.artifacts import save_table
from xsap.config import Config, ensure_dirs
from xsap.data import load_anomaly_returns, load_factors, load_portfolio_panel, sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-download everything")
    args = parser.parse_args()

    ensure_dirs()
    print("fetching raw data")
    paths = sources.fetch_all(refresh=args.refresh)
    for key, path in paths.items():
        print(f"  {key:32s} {path.stat().st_size / 1e6:7.2f} MB")

    cfg = Config()
    panel = load_portfolio_panel(cfg)
    factors = load_factors(cfg)
    anomalies = load_anomaly_returns(cfg)

    print(
        f"\npanel: {panel['asset'].nunique()} assets x "
        f"{panel['month'].nunique()} months "
        f"({panel['month'].min()} to {panel['month'].max()})"
    )
    print(f"factors: {list(factors.columns)}")
    print(f"anomalies: {anomalies.shape[1]} published predictors")

    # A data appendix a reader can check against the published sources.
    appendix = pd.DataFrame(
        [
            {
                "source": source.key,
                "kind": source.kind,
                "note": source.note,
                "upstream": source.upstream,
            }
            for source in sources.ALL_SOURCES
        ]
    ).set_index("source")
    save_table(appendix, "data_appendix")

    coverage = (
        panel.groupby("family")
        .agg(
            assets=("asset", "nunique"),
            first_month=("month", "min"),
            last_month=("month", "max"),
            mean_monthly_return=("ret", "mean"),
        )
        .sort_index()
    )
    save_table(coverage, "data_coverage")

    moments = pd.DataFrame(
        {
            "mean_pct_per_month": factors.mean() * 100.0,
            "sd_pct_per_month": factors.std() * 100.0,
            "n_months": factors.notna().sum(),
        }
    )
    save_table(moments, "factor_moments")
    print(
        "\nFactor means are the check that the mirror is the real thing: "
        "Mkt-RF and UMD should print near 0.57 and 0.60 percent per month."
    )


if __name__ == "__main__":
    main()
