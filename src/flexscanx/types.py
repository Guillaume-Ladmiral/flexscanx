"""Type definitions for flexscanx scan results using TypedDict for structural typing."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

__all__ = [
    "RatioValue",
    "RatioCI",
    "ScanClusterBase",
    "ScanClusterDict",
    "SeriesEntryResultBase",
    "SeriesEntryResultDict",
    "ScanResultBase",
    "ScanResultDict",
    "SeriesEntryBothResultBase",
    "SeriesEntryBothResultDict",
    "BothResultBase",
    "BothResultDict",
    "SeriesResultBase",
]

RatioValue = float | None
RatioCI = tuple[float, float] | None


class ScanClusterBase(TypedDict):
    """Required fields for a detected cluster."""

    area: list[int]  # Result area IDs
    n_case: float  # observed count in cluster
    expected: float  # expected count (POISSON) or population (BINOMIAL)
    stats: float  # test statistic (LLR)
    rank: int  # Monte Carlo rank among simulations; 1 when Monte Carlo is disabled
    type: Literal["HOT", "COLD"]  # cluster direction


class ScanClusterDict(ScanClusterBase, total=False):
    """Full result for a single detected cluster (required + optional fields).

    Raw scan results use canonical ratio keys:
    - 'Rate_Ratio' for POISSON
    - 'Risk_Ratio' for BINOMIAL

    Python summary/report helpers may derive a display-only 'RR' column separately.
    """

    name: list[str]  # Area names (if provided)
    Rate_Ratio: RatioValue  # Rate ratio (POISSON only); may be None for degenerate/empty outputs
    Rate_Ratio_CI: RatioCI  # Reporting CI for Rate_Ratio (POISSON model only)
    Risk_Ratio: RatioValue  # Risk ratio (BINOMIAL only); may be None for degenerate/empty outputs
    Risk_Ratio_CI: RatioCI  # Reporting CI for Risk_Ratio (BINOMIAL model only)
    ratio_smoothing_applied: Literal["NONE", "HALDANE", "EB"]
    pval: float | None  # Monte Carlo p-value when requested
    # Completed Monte Carlo replicates behind THIS cluster's pval: pval == rank / (n + 1).
    # Present on secondary clusters.  Equals scan_settings["simcount_actual"] unless
    # secondary_null="conditional" is combined with mc_early_stop=True, when each
    # secondary's conditional null stops independently.
    simcount_actual: int
    max_dist: float  # Maximum distance within cluster
    from_area: int  # Result area ID at one endpoint of max distance
    to_area: int  # Result area ID at other endpoint


class SeriesEntryResultBase(TypedDict):
    """Required fields for one single-direction entry in ``scan_series().per_time``."""

    area: list[int]  # Primary cluster result area IDs
    n_case: float  # Primary cluster observed count
    expected: float  # Primary cluster expected count or population
    stats: float  # Primary cluster test statistic
    rank: int  # Monte Carlo rank among simulations; 1 when Monte Carlo is disabled
    type: Literal["HOT", "COLD"]  # Scan direction


class SeriesEntryResultDict(SeriesEntryResultBase, total=False):
    """Full per-time single-direction result (same payload as ``scan()`` minus ``scan_settings``).

    Raw scan results use canonical ratio keys:
    - 'Rate_Ratio' for POISSON
    - 'Risk_Ratio' for BINOMIAL

    Python summary/report helpers may derive a display-only 'RR' column separately.
    """

    name: list[str]  # Primary cluster area names
    Rate_Ratio: RatioValue  # Rate ratio (POISSON only); may be None for degenerate/empty outputs
    Rate_Ratio_CI: RatioCI  # Reporting CI for Rate_Ratio (POISSON model only)
    Risk_Ratio: RatioValue  # Risk ratio (BINOMIAL only); may be None for degenerate/empty outputs
    Risk_Ratio_CI: RatioCI  # Reporting CI for Risk_Ratio (BINOMIAL model only)
    ratio_smoothing_applied: Literal["NONE", "HALDANE", "EB"]
    pval: float | None  # Primary cluster Monte Carlo p-value
    pval_hot: float | None  # BOTH envelope primary: HOT directional one-sided p-value
    pval_cold: float | None  # BOTH envelope primary: COLD directional one-sided p-value
    max_dist: float  # Primary cluster max distance
    from_area: int  # Primary cluster distance endpoint area ID
    to_area: int
    secondaries: list[ScanClusterDict]  # Secondary clusters


class ScanResultBase(SeriesEntryResultBase):
    """Required fields for a top-level single-direction ``scan()`` result."""

    scan_settings: dict[str, Any]  # Requested and scalar actual simulation provenance


class ScanResultDict(ScanResultBase, total=False):
    """Full top-level single-direction scan result (includes ``scan_settings``)."""

    name: list[str]  # Primary cluster area names
    Rate_Ratio: RatioValue  # Rate ratio (POISSON only); may be None for degenerate/empty outputs
    Rate_Ratio_CI: RatioCI  # Reporting CI for Rate_Ratio (POISSON model only)
    Risk_Ratio: RatioValue  # Risk ratio (BINOMIAL only); may be None for degenerate/empty outputs
    Risk_Ratio_CI: RatioCI  # Reporting CI for Risk_Ratio (BINOMIAL model only)
    ratio_smoothing_applied: Literal["NONE", "HALDANE", "EB"]
    pval: float | None  # Primary cluster Monte Carlo p-value
    pval_hot: float | None  # BOTH envelope primary: HOT directional one-sided p-value
    pval_cold: float | None  # BOTH envelope primary: COLD directional one-sided p-value
    max_dist: float  # Primary cluster max distance
    from_area: int  # Primary cluster distance endpoint area ID
    to_area: int
    secondaries: list[ScanClusterDict]  # Secondary clusters


class SeriesEntryBothResultBase(TypedDict):
    """Required fields for one ``clustertype='BOTH'`` entry in ``scan_series().per_time``."""

    type: Literal["BOTH"]  # Always "BOTH"
    primary_type: Literal["HOT", "COLD"]  # Which direction had higher stats
    primary: SeriesEntryResultDict  # The primary result (HOT or COLD)
    hot: SeriesEntryResultDict  # HOT scan result
    cold: SeriesEntryResultDict  # COLD scan result


class SeriesEntryBothResultDict(SeriesEntryBothResultBase, total=False):
    """Full per-time BOTH result (same payload as ``scan(..., clustertype='BOTH')`` minus
    settings)."""


class BothResultBase(TypedDict):
    """Required fields for a top-level ``scan(..., clustertype='BOTH')`` result."""

    type: Literal["BOTH"]  # Always "BOTH"
    primary_type: Literal["HOT", "COLD"]  # Which direction had higher stats
    primary: ScanResultDict  # The primary result (HOT or COLD)
    hot: ScanResultDict  # HOT scan result
    cold: ScanResultDict  # COLD scan result
    scan_settings: dict[str, Any]  # Requested and scalar actual simulation provenance


class BothResultDict(BothResultBase, total=False):
    """Full top-level BOTH result."""


class SeriesResultBase(TypedDict):
    """Required fields for a time-series scan result."""

    per_time: list[SeriesEntryResultDict | SeriesEntryBothResultDict]  # Results per time slice
    times: list[object]  # Time labels (integer indices if not provided by caller)
    scan_settings: dict[str, Any]  # simcount_actual is a per_time-aligned list in series provenance
