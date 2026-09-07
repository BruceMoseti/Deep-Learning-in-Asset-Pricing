"""Caching of the built dataset and writing of result tables.

Every table is written twice: a CSV under ``results/`` for downstream code and
a Markdown copy under ``reports/tables/`` so the research report can include
numbers that were actually produced by a run rather than transcribed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from xsap.config import DATA_PROC, RESULTS, TABLES, Config, ensure_dirs
from xsap.features import build_features, make_target, rank_normalise

_DATASET = DATA_PROC / "dataset.parquet"


def dataset(cfg: Config | None = None, *, refresh: bool = False):
    """Return ``(features, target)`` for the real portfolio universe, cached."""
    cfg = cfg or Config()
    ensure_dirs()
    if _DATASET.exists() and not refresh:
        data = pd.read_parquet(_DATASET)
    else:
        from xsap.data import load_factors, load_portfolio_panel

        data = rank_normalise(build_features(load_portfolio_panel(cfg), load_factors(cfg), cfg))
        data.to_parquet(_DATASET)
    return data, make_target(data, cfg.target)


def factors(cfg: Config | None = None) -> pd.DataFrame:
    from xsap.data import load_factors

    return load_factors(cfg or Config())


def save_table(frame: pd.DataFrame, name: str, *, float_format: str = "%.4f") -> None:
    ensure_dirs()
    frame.to_csv(RESULTS / f"{name}.csv", float_format=float_format)
    (TABLES / f"{name}.md").write_text(frame.to_markdown(floatfmt=".4f") + "\n")
    print(f"  wrote {name}: {frame.shape[0]}x{frame.shape[1]}")


def save_json(payload: dict, name: str) -> None:
    ensure_dirs()
    (RESULTS / f"{name}.json").write_text(
        json.dumps(payload, indent=2, default=str, sort_keys=True) + "\n"
    )
    print(f"  wrote {name}.json")


def save_series(frame: pd.DataFrame, name: str) -> None:
    ensure_dirs()
    frame.to_parquet(RESULTS / f"{name}.parquet")
    print(f"  wrote {name}.parquet: {frame.shape[0]}x{frame.shape[1]}")


def load_series(name: str) -> pd.DataFrame:
    path = RESULTS / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run the earlier scripts first")
    return pd.read_parquet(path)


def read_json(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def exists(name: str, suffix: str = "parquet") -> bool:
    return (RESULTS / f"{name}.{suffix}").exists()
