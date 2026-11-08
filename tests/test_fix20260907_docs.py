"""Executable contract tests for the documentation repairs of the 2026-09-07 run.

FXA-175 (names default), FXA-176 (hot/cold optionality), FXA-177 (cap-warning promise)
and FXA-178 (documented submodules declare ``__all__``).
"""

from __future__ import annotations

import importlib
import inspect
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

import flexscanx as flx

REPO_ROOT = Path(__file__).resolve().parents[1]
API_REFERENCE = REPO_ROOT / "API_REFERENCE.md"


def _path_graph(n: int) -> tuple[np.ndarray, tuple[list[int], list[int]]]:
    coords = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    row_ptr, col_idx = [0], []
    for i in range(n):
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < n]
        col_idx.extend(neighbours)
        row_ptr.append(len(col_idx))
    return coords, (row_ptr, col_idx)


def _irregular_path(n: int) -> tuple[np.ndarray, tuple[list[int], list[int]]]:
    """A path graph with strictly distinct pairwise distances (no KNN ties)."""
    spacing = 1.0 + 0.37 * np.arange(n - 1)
    xs = np.concatenate([[0.0], np.cumsum(spacing)])
    coords = np.column_stack([xs, np.zeros(n)])
    distances = sorted(abs(xs[i] - xs[j]) for i in range(n) for j in range(i + 1, n))
    assert len(set(distances)) == len(distances), "fixture must be tie-free"
    row_ptr, col_idx = [0], []
    for i in range(n):
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < n]
        col_idx.extend(neighbours)
        row_ptr.append(len(col_idx))
    return coords, (row_ptr, col_idx)


def _both_result_block() -> str:
    text = API_REFERENCE.read_text(encoding="utf-8")
    start = text.index("### BothResult")
    code_start = text.index("```python", start)
    code_end = text.index("```", code_start + len("```python"))
    return text[code_start:code_end]


# ---------------------------------------------------------------------------
# FXA-175: names defaults to None and no 'name' key is emitted
# ---------------------------------------------------------------------------


def test_reference_states_that_names_default_emits_no_name_key() -> None:
    text = API_REFERENCE.read_text(encoding="utf-8")
    assert "Area identifiers (default: 1-based integers)" not in text
    assert "no `name` key is emitted" in text


def test_scan_without_names_emits_no_name_key_and_with_names_does() -> None:
    coords, adjacency = _path_graph(4)
    plain = flx.scan(
        coords=coords, adjacency=adjacency, observed=[1, 4, 1, 1], expected=[1.0] * 4, simcount=0
    )
    assert "name" not in plain
    named = flx.scan(
        coords=coords,
        adjacency=adjacency,
        observed=[1, 4, 1, 1],
        expected=[1.0] * 4,
        simcount=0,
        names=["a", "b", "c", "d"],
    )
    assert named["name"] == ["b"]


# ---------------------------------------------------------------------------
# FXA-176: hot and cold are never optional in a BOTH result
# ---------------------------------------------------------------------------


def test_reference_both_result_block_declares_hot_and_cold_as_required_mappings() -> None:
    block = _both_result_block()
    assert re.search(r"'hot':\s*ScanResult,", block), block
    assert re.search(r"'cold':\s*ScanResult,", block), block
    assert "| None" not in block


def test_both_result_always_carries_hot_and_cold_dicts() -> None:
    coords, adjacency = _path_graph(4)
    both = flx.scan(
        coords=coords,
        adjacency=adjacency,
        observed=[1, 4, 1, 1],
        expected=[1.0] * 4,
        simcount=0,
        clustertype="BOTH",
    )
    assert isinstance(both["hot"], dict) and isinstance(both["cold"], dict)


# ---------------------------------------------------------------------------
# FXA-177: the cap warning fires only when clustersize was set explicitly
# ---------------------------------------------------------------------------


def test_scan_docstring_and_reference_qualify_the_cap_warning() -> None:
    doc = inspect.getdoc(flx.scan) or ""
    assert "Capped to K with a warning if" not in doc
    assert "only when *clustersize* was passed explicitly" in doc
    text = API_REFERENCE.read_text(encoding="utf-8")
    assert "a warning is emitted only when `clustersize` was passed explicitly" in text


def test_default_clustersize_is_capped_silently_and_explicit_one_warns() -> None:
    coords, adjacency = _irregular_path(4)  # tie-free, so only the cap could warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = flx.scan(
            coords=coords,
            adjacency=adjacency,
            observed=[1, 4, 1, 1],
            expected=[1.0] * 4,
            K=2,
            simcount=0,
        )
    assert result["scan_settings"]["clustersize"] == 2
    with pytest.warns(UserWarning, match="capping clustersize"):
        flx.scan(
            coords=coords,
            adjacency=adjacency,
            observed=[1, 4, 1, 1],
            expected=[1.0] * 4,
            K=2,
            clustersize=15,
            simcount=0,
        )


# ---------------------------------------------------------------------------
# FXA-178: documented public submodules declare an explicit __all__
# ---------------------------------------------------------------------------

DOCUMENTED_SUBMODULES = ("plotting", "report", "summary", "graph", "helpers", "types", "validation")
_TYPING_NOISE = {
    "Any",
    "cast",
    "Sequence",
    "Iterable",
    "Optional",
    "Union",
    "TYPE_CHECKING",
    "TypedDict",
    "Literal",
}


@pytest.mark.parametrize("name", DOCUMENTED_SUBMODULES)
def test_documented_submodule_declares_a_resolvable_all(name: str) -> None:
    module = importlib.import_module(f"flexscanx.{name}")
    exported = getattr(module, "__all__", None)
    assert isinstance(exported, (list, tuple)) and exported, f"flexscanx.{name} has no __all__"
    for symbol in exported:
        assert hasattr(module, symbol), f"flexscanx.{name}.__all__ names missing {symbol}"
        assert not symbol.startswith("_"), symbol
    assert not (set(exported) & _TYPING_NOISE)
    assert "validate_integer_parameter" not in exported


# ---------------------------------------------------------------------------
# FXA-189: the reproducibility claim is scoped to thread counts and identical builds
# ---------------------------------------------------------------------------


def test_migration_note_does_not_claim_cross_platform_bitwise_reproducibility() -> None:
    text = (REPO_ROOT / "docs" / "MIGRATION_FROM_RFLEXSCAN.md").read_text(encoding="utf-8")
    assert "identically across thread counts and platforms" not in text
    assert "across threads/platforms" not in text
    assert "identical builds only" in text
    assert "std::poisson_distribution" in text


# ---------------------------------------------------------------------------
# FXA-192: the alias sampler draws a different null sample for the same seed
# ---------------------------------------------------------------------------


def test_reference_states_that_the_alias_sampler_changes_the_null_sample() -> None:
    reference = (REPO_ROOT / "API_REFERENCE.md").read_text(encoding="utf-8")
    advanced = (REPO_ROOT / "docs" / "ADVANCED_USAGE.md").read_text(encoding="utf-8")
    alias_bullets = [
        line for line in reference.splitlines() if line.startswith("- **`mc_use_alias`**")
    ]
    assert len(alias_bullets) == 2, alias_bullets
    for bullet in alias_bullets:
        assert "different" in bullet and "Monte Carlo error" in bullet, bullet
    tips = reference[reference.index("FLEXSCANX_MC_ALIAS") :]
    assert "mc_alias_env_override" in tips.splitlines()[0]
    advanced_line = next(line for line in advanced.splitlines() if "mc_use_alias=True" in line)
    assert "Monte Carlo error" in advanced_line and "mc_alias_env_override" in advanced_line


def _alias_scan(model: str, alias: bool, seed: int = 7) -> dict:
    n = 14
    gaps = 1.0 + 0.173 * np.arange(n, dtype=float)
    coords = np.column_stack([np.cumsum(gaps), np.zeros(n)])
    row_ptr = [0]
    col_idx: list[int] = []
    for i in range(n):
        if i > 0:
            col_idx.append(i - 1)
        if i < n - 1:
            col_idx.append(i + 1)
        row_ptr.append(len(col_idx))
    rng = np.random.default_rng(1)
    expected = rng.uniform(4.0, 9.0, size=n)
    observed = rng.poisson(expected).astype(int)
    observed[5:8] += 2
    kwargs: dict = dict(
        coords=coords,
        adjacency=(row_ptr, col_idx),
        observed=observed,
        model=model,
        scanmethod="FLEXIBLE",
        stattype="ORIGINAL",
        clustertype="HOT",
        clustersize=4,
        K=6,
        simcount=999,
        seed=seed,
        mc_use_alias=alias,
    )
    if model == "POISSON":
        kwargs["expected"] = expected
    else:
        kwargs["population"] = (expected * 30).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return flx.scan(**kwargs)


def test_alias_sampler_is_a_distinct_replayable_null_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FLEXSCANX_MC_ALIAS", raising=False)
    default = _alias_scan("POISSON", alias=False)
    alias = _alias_scan("POISSON", alias=True)
    assert list(default["area"]) == list(alias["area"]) and default["stats"] == alias["stats"]
    assert default["rank"] != alias["rank"], "the documented difference must be observable"
    assert _alias_scan("POISSON", alias=True)["rank"] == alias["rank"], "alias path replays"
    assert default["scan_settings"]["mc_use_alias"] is False
    assert alias["scan_settings"]["mc_use_alias"] is True
    assert alias["scan_settings"]["mc_alias_env_override"] is False
    monkeypatch.setenv("FLEXSCANX_MC_ALIAS", "1")
    forced = _alias_scan("POISSON", alias=False)
    assert forced["rank"] == alias["rank"]
    assert forced["scan_settings"]["mc_alias_env_override"] is True


def test_alias_sampler_is_inert_for_the_binomial_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLEXSCANX_MC_ALIAS", raising=False)
    default = _alias_scan("BINOMIAL", alias=False)
    alias = _alias_scan("BINOMIAL", alias=True)
    assert (default["rank"], default["pval"]) == (alias["rank"], alias["pval"])


# ---------------------------------------------------------------------------
# FXA-199: the reference states what both_parallel does to n_jobs_centers
# ---------------------------------------------------------------------------


def test_reference_states_that_both_parallel_forces_one_centre_thread_per_direction() -> None:
    reference = (REPO_ROOT / "API_REFERENCE.md").read_text(encoding="utf-8")
    bullets = [line for line in reference.splitlines() if line.startswith("- **`both_parallel`**")]
    assert len(bullets) == 1, bullets  # scan_series() shares scan()'s parameter list
    for bullet in bullets:
        assert "n_jobs_centers" in bullet and "= 1" in bullet, bullet
    docstring = flx.scan.__doc__ or ""
    assert "Forces *n_jobs_centers* = 1" in docstring
