"""Mutation-sensitive contracts for audit test integrity repairs."""

from __future__ import annotations

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent


def _source(name: str) -> str:
    return (TESTS / name).read_text(encoding="utf-8")


def test_calibration_cache_is_redirected_for_every_test() -> None:
    source = _source("conftest.py")
    assert "def isolate_calibration_cache" in source
    assert 'monkeypatch.setattr(validation, "_CACHE_FILE"' in source


def test_calibration_cache_redirection_is_active_in_this_process(tmp_path: Path) -> None:
    """The behavioural half of the guard above: inside a test, the public
    ``calibrate_llr_threshold`` resolves its cache path from ``validation``,
    and that path points into pytest's temporary directory, not the user's cache
    (2026-09-07 adjudication of the static test-integrity guards)."""
    import flexscanx
    import flexscanx._calibration as calibration
    import flexscanx.validation as validation

    assert flexscanx.calibrate_llr_threshold.__module__ == "flexscanx.validation"
    active = Path(validation._CACHE_FILE)
    assert active.is_relative_to(tmp_path.parent.parent), active
    assert str(active) != calibration._CACHE_FILE
    assert not active.is_relative_to(Path.home() / ".cache"), active


def test_consensus_secondary_regression_uses_a_distinct_fake_secondary() -> None:
    source = _source("test_regressions_audit_fixes_v010.py")
    assert '"secondaries": [{"area": [11]}]' in source
    assert 'result["freq"][10] == 0.0' in source


def test_vacuous_assertion_patterns_are_removed() -> None:
    assert "test_warm_start_performance" not in _source("test_audit_v001.py")
    assert "assert warm_dur >= 0.0" not in _source("test_audit_v001.py")
    assert "p is None or" not in _source("test_scan_manhattan.py")
    assert "or np.isnan(moran_I)" not in _source("test_binomial_phase2.py")
    assert "not _chain_csr," not in _source("test_audit_remediation_v001b.py")
    assert "REQUIRED_SCAN_RESULT_KEYS <= set(public)" in _source("test_result_surface_contracts.py")


def test_seed_regression_compares_equal_and_distinct_streams() -> None:
    source = _source("test_boundaries_and_mc.py")
    assert 'same_seed["rank"] == a["rank"]' in source
    assert '(b["rank"], b["pval"]) != (a["rank"], a["pval"])' in source


def test_test_suite_has_no_process_global_numpy_rng_calls() -> None:
    """Every test module (not only the two repaired in 2026-08) must draw from a
    local ``np.random.default_rng`` generator, never the process-global state."""
    offenders: list[str] = []
    names = sorted(path.name for path in TESTS.glob("*.py"))
    assert len(names) >= 100, names
    for name in names:
        tree = ast.parse(_source(name), filename=name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = node.func.value
            if isinstance(owner, ast.Attribute) and isinstance(owner.value, ast.Name):
                if (
                    owner.value.id == "np"
                    and owner.attr == "random"
                    and node.func.attr != "default_rng"
                ):
                    offenders.append(f"{name}:{node.lineno}:{node.func.attr}")
    assert offenders == []


def test_minor_test_hygiene_patterns_are_removed() -> None:
    assert "NamedTemporaryFile" not in _source("test_rr_key_fix.py")
    assert "pytest.raises((ImportError, TypeError))" not in _source(
        "test_audit_remediation_v001b.py"
    )
