"""Parsers for Kenneth French data-library CSV files.

The library ships one file per portfolio set.  A file contains several stacked
blocks (value-weighted monthly returns, equal-weighted monthly returns, annual
returns, number of firms, average firm size, ...) separated by blank lines,
preceded by a free-text description.  We always want the *first* monthly block,
which is value-weighted monthly returns.

Mirrored copies are often already reduced to that block with a ``Date`` header.
Both layouts are handled.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

# The library encodes "not available" as these sentinels rather than as blanks.
MISSING = (-99.99, -999.0, -99.0)

_YYYYMM = re.compile(r"^\s*(\d{6})\s*$")


def _monthly_block(text: str) -> tuple[list[str], list[list[str]]]:
    """Return (header, rows) for the first block indexed by YYYYMM."""
    header: list[str] | None = None
    rows: list[list[str]] = []
    for raw in io.StringIO(text):
        line = raw.rstrip("\n").rstrip("\r")
        if not line.strip():
            if rows:
                break
            continue
        fields = [f.strip() for f in line.split(",")]
        if _YYYYMM.match(fields[0]):
            if header is None:
                raise ValueError("found data rows before any header row")
            rows.append(fields)
        elif rows:
            break
        else:
            # Candidate header: first field empty or "Date", and at least one
            # further field that is not numeric.
            rest = fields[1:]
            if fields[0].lower() in {"", "date"} and rest and not _all_numeric(rest):
                header = rest
    if header is None or not rows:
        raise ValueError("no monthly return block found")
    return header, rows


def _all_numeric(fields: list[str]) -> bool:
    for f in fields:
        if f == "":
            continue
        try:
            float(f)
        except ValueError:
            return False
    return True


def read_french_csv(path) -> pd.DataFrame:
    """Read a French-library file into a month x portfolio frame of percents."""
    with open(path, encoding="utf-8-sig", errors="replace") as handle:
        text = handle.read()
    header, rows = _monthly_block(text)

    index = pd.PeriodIndex([r[0] for r in rows], freq="M")
    width = len(header)
    data = np.full((len(rows), width), np.nan)
    for i, row in enumerate(rows):
        for j, field in enumerate(row[1 : width + 1]):
            if field:
                data[i, j] = float(field)

    frame = pd.DataFrame(data, index=index, columns=[h.strip() for h in header])
    frame = frame.mask(frame.isin(MISSING))
    frame.index.name = "month"
    return frame


_SPECIAL = {"SMALL": 1, "BIG": 5}


def _bucket(token: str) -> int | None:
    """Map one sort label to a quintile 1..5, or None if it is not a quintile.

    Labels look like ``ME3``, ``BM2``, ``LoOP``, ``HiPRIOR``, ``SMALL``, ``BIG``,
    and, for the net-issues set, ``NegNI`` / ``ZeroNI`` which are not quintiles.
    """
    if token in _SPECIAL:
        return _SPECIAL[token]
    if token.startswith(("Neg", "Zero")):
        return None
    if token.startswith("Lo"):
        return 1
    if token.startswith("Hi"):
        return 5
    match = re.search(r"(\d)$", token)
    return int(match.group(1)) if match else None


def sort_ranks(column: str) -> tuple[float, float]:
    """Return the two sort dimensions of a portfolio label, scaled to [0, 1].

    ``nan`` marks a dimension that is not an ordered quintile (industries, and
    the negative/zero net-issues buckets).
    """
    tokens = column.split()
    if len(tokens) != 2:
        return (np.nan, np.nan)
    buckets = [_bucket(t) for t in tokens]
    scaled = (np.nan if b is None else (b - 1) / 4.0 for b in buckets)
    return tuple(scaled)  # type: ignore[return-value]


def read_factors(path, rename: dict[str, str] | None = None) -> pd.DataFrame:
    """Read a factor file and convert percent per month to decimal returns."""
    frame = read_french_csv(path) / 100.0
    if rename:
        frame = frame.rename(columns=rename)
    return frame
