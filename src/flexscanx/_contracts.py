from __future__ import annotations

from typing import Any

DIRECTION_REQUIRED_KEYS = frozenset({"area", "n_case", "expected", "stats", "rank", "type"})
BOTH_REQUIRED_KEYS = frozenset({"type", "primary_type", "primary", "hot", "cold"})
SERIES_REQUIRED_KEYS = frozenset({"per_time", "times", "scan_settings"})
VALID_DIRECTIONS = frozenset({"HOT", "COLD"})


def _ensure_dict(obj: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError(f"{where} must be a dict")
    return obj


def _ensure_required_keys(d: dict[str, Any], required: frozenset[str], *, where: str) -> None:
    missing = sorted(required - set(d))
    if missing:
        raise ValueError(f"{where} is missing required keys: {missing}")


def validate_direction_result_contract(
    result: Any, *, require_scan_settings: bool = False, where: str = "result"
) -> dict[str, Any]:
    """Validate single-direction result envelope."""
    d = _ensure_dict(result, where=where)
    _ensure_required_keys(d, DIRECTION_REQUIRED_KEYS, where=where)

    dtype = str(d.get("type", "")).upper()
    if dtype not in VALID_DIRECTIONS:
        raise ValueError(f"{where}.type must be HOT or COLD, got {d.get('type')!r}")

    if require_scan_settings and "scan_settings" not in d:
        raise ValueError(f"{where} must include 'scan_settings'")
    if "scan_settings" in d and not isinstance(d["scan_settings"], dict):
        raise ValueError(f"{where}.scan_settings must be a dict when present")

    if not isinstance(d.get("area"), list):
        raise ValueError(f"{where}.area must be a list")
    if "secondaries" in d:
        secs = d["secondaries"]
        if not isinstance(secs, list):
            raise ValueError(f"{where}.secondaries must be a list when present")
        for i, sec in enumerate(secs):
            sec_d = _ensure_dict(sec, where=f"{where}.secondaries[{i}]")
            _ensure_required_keys(sec_d, DIRECTION_REQUIRED_KEYS, where=f"{where}.secondaries[{i}]")
            sec_type = str(sec_d.get("type", "")).upper()
            if sec_type not in VALID_DIRECTIONS:
                raise ValueError(
                    f"{where}.secondaries[{i}].type must be HOT or COLD, got {sec_d.get('type')!r}"
                )
    return d


def validate_both_result_contract(
    result: Any, *, require_scan_settings: bool = False, where: str = "result"
) -> dict[str, Any]:
    """Validate BOTH result envelope."""
    d = _ensure_dict(result, where=where)
    _ensure_required_keys(d, BOTH_REQUIRED_KEYS, where=where)

    if str(d.get("type", "")).upper() != "BOTH":
        raise ValueError(f"{where}.type must be 'BOTH', got {d.get('type')!r}")

    primary_type = str(d.get("primary_type", "")).upper()
    if primary_type not in VALID_DIRECTIONS:
        raise ValueError(f"{where}.primary_type must be HOT or COLD, got {d.get('primary_type')!r}")

    validate_direction_result_contract(d["hot"], require_scan_settings=False, where=f"{where}.hot")
    validate_direction_result_contract(
        d["cold"], require_scan_settings=False, where=f"{where}.cold"
    )
    primary = validate_direction_result_contract(
        d["primary"], require_scan_settings=False, where=f"{where}.primary"
    )
    primary_dir = str(primary.get("type", "")).upper()
    if primary_dir != primary_type:
        raise ValueError(
            f"{where}.primary.type ({primary_dir}) must match {where}.primary_type ({primary_type})"
        )

    if require_scan_settings and "scan_settings" not in d:
        raise ValueError(f"{where} must include 'scan_settings'")
    if "scan_settings" in d and not isinstance(d["scan_settings"], dict):
        raise ValueError(f"{where}.scan_settings must be a dict when present")
    return d


def validate_scan_result_contract(
    result: Any, *, require_scan_settings: bool = False, where: str = "result"
) -> dict[str, Any]:
    """Validate scan() envelope (single direction or BOTH)."""
    d = _ensure_dict(result, where=where)
    if str(d.get("type", "")).upper() == "BOTH":
        return validate_both_result_contract(
            d, require_scan_settings=require_scan_settings, where=where
        )
    return validate_direction_result_contract(
        d, require_scan_settings=require_scan_settings, where=where
    )


def validate_series_entry_contract(result: Any, *, where: str = "result") -> dict[str, Any]:
    """Validate one entry from ``scan_series().per_time``.

    Per-time entries intentionally omit ``scan_settings``; the shared provenance lives on
    the series-level envelope.
    """
    return validate_scan_result_contract(result, require_scan_settings=False, where=where)


def validate_series_result_contract(result: Any, *, where: str = "result") -> dict[str, Any]:
    """Validate scan_series() envelope."""
    d = _ensure_dict(result, where=where)
    _ensure_required_keys(d, SERIES_REQUIRED_KEYS, where=where)

    per_time = d.get("per_time")
    if not isinstance(per_time, list):
        raise ValueError(f"{where}.per_time must be a list")
    times = d.get("times")
    if not isinstance(times, list):
        raise ValueError(f"{where}.times must be a list")
    if len(times) != len(per_time):
        raise ValueError(
            f"{where}.times length ({len(times)}) must equal "
            f"{where}.per_time length ({len(per_time)})"
        )
    if not isinstance(d.get("scan_settings"), dict):
        raise ValueError(f"{where}.scan_settings must be a dict")

    for i, entry in enumerate(per_time):
        validate_series_entry_contract(entry, where=f"{where}.per_time[{i}]")
    return d
