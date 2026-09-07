"""Registry of the raw data files used by the project, with pinned provenance.

Every file is fetched from a git mirror pinned to an immutable commit SHA and
verified against a SHA-256 recorded in ``data/raw/manifest.json`` on first
download.  ``upstream`` records where the file originally came from so a reader
can re-derive it from the primary source.

Why mirrors rather than the primary sites: the primary hosts (Dartmouth for the
Fama-French library, and the Open Source Asset Pricing distribution) are not
reachable from every network, and pinning to a commit SHA additionally freezes
the *vintage* of the data.  The Fama-French library is retroactively revised
whenever CRSP is updated, so an unpinned download makes results irreproducible.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from xsap.config import DATA_RAW

_FRENCH_LIB = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"
)

# Immutable commit SHAs of the mirror repositories.
_TFRP = "a5ba25ace97100bb4a5ac26339f72ea43d05c4c6"  # a91quaini/reproduceTFRP
_SPS = "4115328241b298ede63f3d9b4797c94a61fe6932"  # a91quaini/SparsePortfolioSelection
_IFRP = "164e809f15c7667b2e1d4aeef93cd2ede3a5082b"  # a91quaini/intrinsicFRP


def _raw(repo: str, sha: str, path: str) -> str:
    quoted = urllib.parse.quote(path)
    return f"https://raw.githubusercontent.com/{repo}/{sha}/{quoted}"


@dataclass(frozen=True)
class Source:
    key: str
    filename: str
    url: str
    upstream: str
    kind: str  # "portfolios" | "factors" | "anomalies"
    # Free-text description of what one column is, used in the data appendix.
    note: str = ""


def _french_portfolios(name: str, note: str, ext: str = "CSV") -> Source:
    return Source(
        key=name,
        filename=f"{name}.csv",
        url=_raw("a91quaini/reproduceTFRP", _TFRP, f"data-raw/{name}.{ext}"),
        upstream=_FRENCH_LIB,
        kind="portfolios",
        note=note,
    )


# --- Cross-section: value-weighted monthly returns on characteristic-sorted
# --- portfolios of US common stocks (NYSE/AMEX/NASDAQ).
PORTFOLIO_SETS: tuple[Source, ...] = (
    _french_portfolios("49_Industry_Portfolios", "SIC-based industry, 49 groups"),
    _french_portfolios("25_Portfolios_5x5", "size x book-to-market, 5x5"),
    _french_portfolios("25_Portfolios_ME_OP_5x5", "size x operating profitability"),
    _french_portfolios("25_Portfolios_ME_INV_5x5", "size x investment"),
    _french_portfolios("25_Portfolios_ME_Prior_12_2", "size x momentum (t-12,t-2)"),
    _french_portfolios("25_Portfolios_ME_Prior_1_0", "size x short-term reversal"),
    _french_portfolios("25_Portfolios_ME_Prior_60_13", "size x long-term reversal"),
    _french_portfolios("25_Portfolios_ME_BETA_5x5", "size x market beta", ext="csv"),
    _french_portfolios("25_Portfolios_ME_VAR_5x5", "size x return variance", ext="csv"),
    _french_portfolios("25_Portfolios_ME_AC_5x5", "size x accruals", ext="csv"),
    _french_portfolios("25_Portfolios_ME_NI_5x5", "size x net share issues", ext="csv"),
    _french_portfolios("25_Portfolios_BEME_OP_5x5", "book-to-market x profitability"),
    _french_portfolios("25_Portfolios_BEME_INV_5x5", "book-to-market x investment"),
    _french_portfolios("25_Portfolios_OP_INV_5x5", "profitability x investment"),
)

FF5 = Source(
    key="ff5",
    filename="F-F_Research_Data_5_Factors_2x3.csv",
    url=_raw(
        "a91quaini/SparsePortfolioSelection",
        _SPS,
        "data-raw/managed_portfolios_monthly/F-F_Research_Data_5_Factors_2x3.csv",
    ),
    upstream=_FRENCH_LIB,
    kind="factors",
    note="Mkt-RF, SMB, HML, RMW, CMA, RF; percent per month",
)

MOM = Source(
    key="mom",
    filename="F-F_Momentum_Factor.csv",
    url=_raw("a91quaini/intrinsicFRP", _IFRP, "data-raw/F-F_Momentum_Factor.CSV"),
    upstream=_FRENCH_LIB,
    kind="factors",
    note="Mom (UMD); percent per month",
)

# --- 200+ published cross-sectional predictors, each as a monthly long-short
# --- portfolio return.  Used for the multiple-testing study.
ANOMALIES = Source(
    key="osap",
    filename="PredictorLSretWide.csv",
    url=_raw(
        "a91quaini/SparsePortfolioSelection",
        _SPS,
        "data-raw/anomalies/PredictorLSretWide.csv",
    ),
    upstream="https://www.openassetpricing.com  (Chen and Zimmermann, 2022)",
    kind="anomalies",
    note="one column per published predictor; long-short decile return in percent",
)

ALL_SOURCES: tuple[Source, ...] = (*PORTFOLIO_SETS, FF5, MOM, ANOMALIES)

_MANIFEST = DATA_RAW / "manifest.json"


def _load_manifest() -> dict[str, dict[str, str]]:
    if _MANIFEST.exists():
        return json.loads(_MANIFEST.read_text())
    return {}


def _save_manifest(manifest: dict[str, dict[str, str]]) -> None:
    _MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    _MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(source: Source, *, refresh: bool = False):
    """Download ``source`` into the raw-data cache and return its path.

    Raises if a cached file's digest no longer matches the manifest, which is
    the signal that the pinned data changed underneath us.
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    path = DATA_RAW / source.filename
    manifest = _load_manifest()

    if path.exists() and not refresh:
        recorded = manifest.get(source.key, {}).get("sha256")
        if recorded and sha256(path) != recorded:
            raise RuntimeError(
                f"{path.name} does not match the recorded digest; delete it and refetch"
            )
        return path

    try:
        with urllib.request.urlopen(source.url, timeout=120) as resp:
            payload = resp.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"could not fetch {source.key} from {source.url}") from exc

    path.write_bytes(payload)
    manifest[source.key] = {
        "filename": source.filename,
        "url": source.url,
        "upstream": source.upstream,
        "bytes": str(len(payload)),
        "sha256": sha256(path),
    }
    _save_manifest(manifest)
    return path


def fetch_all(*, refresh: bool = False) -> dict[str, object]:
    return {s.key: fetch(s, refresh=refresh) for s in ALL_SOURCES}
