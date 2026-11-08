"""Regression tests for public scan contracts and provenance (2026-08-16)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import get_args, get_type_hints

import numpy as np
import pytest

import flexscanx as flx
import flexscanx.api as api
from flexscanx._scan_internal import _CRN_DEFAULT_SEED
from flexscanx._utils import _validate_csr_adjacency
from flexscanx.types import SeriesResultBase


def _line_inputs() -> dict:
    return {
        "coords": np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        "adjacency": ([0, 1, 3, 4], [1, 0, 2, 1]),
        "observed": [4, 1, 1],
        "expected": [2.0, 2.0, 2.0],
        "K": 3,
        "clustersize": 2,
    }


def test_default_scan_seed_is_recorded_and_legally_replayable() -> None:
    kwargs = _line_inputs()
    default = flx.scan(**kwargs, simcount=19, seed=None)
    replay = flx.scan(**kwargs, simcount=19, seed=_CRN_DEFAULT_SEED)

    assert default["rank"] == replay["rank"]
    assert default["pval"] == replay["pval"]
    assert default["scan_settings"]["seed_requested"] is None
    assert default["scan_settings"]["seed"] == _CRN_DEFAULT_SEED
    assert default["scan_settings"]["seed_effective"] == _CRN_DEFAULT_SEED

    series = flx.scan_series(
        **{
            **kwargs,
            "observed": [kwargs["observed"]],
            "expected": [kwargs["expected"]],
        },
        simcount=0,
        seed=None,
        crn=True,
    )
    assert series["scan_settings"]["seed"] == _CRN_DEFAULT_SEED
    assert series["scan_settings"]["seed_requested"] is None
    assert series["scan_settings"]["crn"] is True


def test_reused_handle_rejects_incompatible_static_arguments() -> None:
    kwargs = _line_inputs()
    handle = flx.build_static(kwargs["coords"], kwargs["adjacency"], K=3)
    dynamic = {key: kwargs[key] for key in ("observed", "expected", "clustersize")}

    with pytest.raises(ValueError, match="handle.*K"):
        flx.scan(handle=handle, K=2, simcount=0, **dynamic)
    with pytest.raises(ValueError, match="handle.*clusterradius"):
        flx.scan(handle=handle, K=3, clusterradius=0.5, simcount=0, **dynamic)
    with pytest.raises(ValueError, match="handle.*knn_mode"):
        flx.scan(handle=handle, K=3, knn_mode="K_FIXED", simcount=0, **dynamic)


def test_geodataframe_adapter_rejects_handle_before_building_an_internal_graph() -> None:
    with pytest.raises(TypeError, match="handle.*built internally"):
        api.scan_from_geodataframe(object(), id_col="id", handle=object())


def test_package_generated_edgeless_csr_crosses_the_static_boundary() -> None:
    adjacency = flx.csr_from_dense(np.zeros((3, 3), dtype=int))
    assert adjacency == ([0, 0, 0, 0], [])
    assert _validate_csr_adjacency(adjacency, 3) == adjacency

    handle = flx.build_static(np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]), adjacency, K=3)
    assert flx.describe_static(handle)["N"] == 3


def test_scan_provenance_covers_effective_result_affecting_controls(monkeypatch) -> None:
    monkeypatch.setenv("FLEXSCANX_MC_ALIAS", "1")
    result = flx.scan(**_line_inputs(), simcount=0, seed=7)
    settings = result["scan_settings"]
    required = {
        "model",
        "scanmethod",
        "stattype",
        "clustertype",
        "clustersize",
        "clusterradius",
        "ralpha",
        "K",
        "secondary",
        "seed",
        "seed_requested",
        "seed_effective",
        "n_jobs_centers",
        "simcount",
        "simcount_requested",
        "rantype",
        "mc_early_stop",
        "mc_alpha",
        "mc_progress_every",
        "debug_prune",
        "enable_center_ub_gating",
        "cache_candidate_llr",
        "enable_frame_ub_prune",
        "scheduler_backend",
        "knn_method",
        "knn_mode",
        "knn_refill_force",
        "both_parallel",
        "mc_use_alias",
        "mc_use_alias_requested",
        "ratio_smoothing",
        "secondary_p_stop",
        "secondary_stats_stop",
        "secondary_null",
    }
    assert required <= settings.keys()
    assert settings["mc_use_alias"] is True
    assert settings["mc_use_alias_requested"] is False
    assert settings["mc_alias_env_override"] is True
    assert settings["seed_requested"] == 7
    assert settings["seed_effective"] == 7

    encoded = json.dumps(result, allow_nan=False)
    assert json.loads(encoded)["scan_settings"]["clusterradius"] is None

    both = flx.scan(**_line_inputs(), clustertype="BOTH", simcount=0, seed=7)
    both_decoded = json.loads(json.dumps(both, allow_nan=False))
    assert both_decoded["scan_settings"]["clusterradius"] is None

    series = flx.scan_series(
        **{
            **_line_inputs(),
            "observed": [_line_inputs()["observed"]],
            "expected": [_line_inputs()["expected"]],
        },
        clustertype="BOTH",
        simcount=0,
        seed=7,
    )
    series_decoded = json.loads(json.dumps(series, allow_nan=False))
    assert series_decoded["scan_settings"]["clusterradius"] is None


def test_both_primary_never_aliases_a_directional_branch_without_monte_carlo() -> None:
    result = flx.scan(**_line_inputs(), clustertype="BOTH", simcount=0)
    branch = result["hot"] if result["primary_type"] == "HOT" else result["cold"]

    assert all(result["primary"][key] == value for key, value in branch.items())
    assert result["primary"] is not branch
    result["primary"]["area"].append(999)
    assert 999 not in branch["area"]


def test_public_annotations_match_accepted_arraylike_and_optional_values() -> None:
    build_static_hints = get_type_hints(api.build_static)
    assert type(None) in get_args(build_static_hints["K"])

    series_hints = get_type_hints(api.scan_series)
    assert series_hints["observed"] != np.ndarray
    assert get_type_hints(SeriesResultBase)["times"] == list[object]

    stub = (Path(api.__file__).with_name("_core.pyi")).read_text(encoding="utf-8")
    assert "class CoreScanClusterBase(TypedDict):" in stub
    assert "class CoreScanClusterDict(CoreScanClusterBase, total=False):" in stub

    assert inspect.signature(api.scan_series).parameters["observed"].annotation != "np.ndarray"
