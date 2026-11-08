"""Internal utilities for sequence analysis.

Copyright (c) 2025 flexscanx contributors
Portions adapted from traminex (MIT License)
"""

from __future__ import annotations

import numpy as np


def _extract_clusters_for_timepoint(t_result: object) -> list[tuple[str, list[int]]]:
    """Normalize one time-slice result into (type, area_ids) cluster tuples.

    Supports:
    - single-direction result: {'area': [...], 'type': 'HOT'|'COLD', 'secondaries': [...]}
    - BOTH envelope: {'hot': {...}, 'cold': {...}, ...}
    - list of cluster dicts
    """

    out: list[tuple[str, list[int]]] = []

    def _append_cluster(cluster: object, *, default_type: str | None = None) -> None:
        if not isinstance(cluster, dict):
            return

        ctype = str(cluster.get("type", default_type or "")).upper()
        if ctype not in ("HOT", "COLD"):
            return

        areas_raw = cluster.get("area")
        if areas_raw is None:
            return
        try:
            areas = [int(v) for v in areas_raw]
        except (TypeError, ValueError):
            areas = []
        if areas:
            out.append((ctype, areas))

        secondaries = cluster.get("secondaries")
        if isinstance(secondaries, list):
            for sec in secondaries:
                if not isinstance(sec, dict):
                    continue
                sec_type = str(sec.get("type", ctype)).upper()
                if sec_type not in ("HOT", "COLD"):
                    continue
                sec_area_raw = sec.get("area")
                if sec_area_raw is None:
                    continue
                try:
                    sec_areas = [int(v) for v in sec_area_raw]
                except (TypeError, ValueError):
                    sec_areas = []
                if sec_areas:
                    out.append((sec_type, sec_areas))

    if isinstance(t_result, dict):
        # Single-direction result.
        if "area" in t_result:
            _append_cluster(t_result, default_type=str(t_result.get("type", "")).upper())
        # BOTH envelope.
        if "hot" in t_result:
            _append_cluster(t_result.get("hot"), default_type="HOT")
        if "cold" in t_result:
            _append_cluster(t_result.get("cold"), default_type="COLD")
    elif isinstance(t_result, (list, tuple)):
        for cluster in t_result:
            _append_cluster(cluster)

    return out


def extract_sequences(
    result: dict, area_ids: list[int] | None = None, include_type: str = "both"
) -> tuple[np.ndarray, list[int]]:
    """Extract cluster membership sequences from scan_series result.

    Converts temporal cluster detection results into a sequence matrix
    where each row represents an area and each column represents a time point.

    Parameters
    ----------
    result : dict
        Output from scan_series() containing 'per_time' key.
        Each time point can be:
        - a single-direction cluster dict with 'area' and 'type'
        - a BOTH envelope with 'hot' and 'cold' sub-dicts
        - optionally including 'secondaries' lists
    area_ids : list of int, optional
        Specific area IDs to extract. If None, extracts all areas that appear
        in at least one cluster across all time points.
    include_type : str, default='both'
        Which cluster types to encode:
        - 'hot': code HOT=1, others=0
        - 'cold': code COLD=-1, others=0
        - 'both': code HOT=1, COLD=-1, none=0

    Returns
    -------
    sequences : ndarray of shape (N, T)
        Membership matrix where:
        - rows correspond to areas (N areas)
        - columns correspond to time points (T time points)
        - values encode cluster membership:
            - 1 if area was in HOT cluster at that time (if include_type in ['hot', 'both'])
            - -1 if area was in COLD cluster at that time (if include_type in ['cold', 'both'])
            - 0 if area was not in any cluster
    area_ids_out : list of int
        Area IDs corresponding to rows of sequences matrix (sorted)

    Examples
    --------
    >>> # After running scan_series
    >>> result = flx.scan_series(observed=obs_TN, expected=exp_TN, ...)
    >>> seqs, ids = extract_sequences(result, include_type='both')
    >>> print(seqs.shape)  # (N_areas, T_timepoints)
    >>> print(seqs[0])  # First area's trajectory: [1, 1, 0, -1, 0, ...]

    >>> # Extract only specific areas
    >>> seqs, ids = extract_sequences(result, area_ids=[10, 20, 30])
    >>> assert list(ids) == [10, 20, 30]

    >>> # Track only HOT clusters
    >>> seqs_hot, ids = extract_sequences(result, include_type='hot')
    >>> # Now seqs_hot has only 0 (not in HOT) and 1 (in HOT)

    Notes
    -----
    - Areas that never appear in any cluster are included with all-zero sequences
      if they are in the area_ids list, otherwise they are omitted.
    - The function supports scan_series outputs for both single-direction and
      clustertype='BOTH' runs.
    - If a time point has no significant clusters, the corresponding column will
      be all zeros.

    Adapted from PROJECT_PY_TRAMINEX sequence handling utilities.
    """
    include_type = include_type.lower()
    if include_type not in ("hot", "cold", "both"):
        raise ValueError(f"include_type must be 'hot', 'cold', or 'both', got {include_type!r}")

    # Extract per-time cluster information
    per_time = result.get("per_time", [])
    if not per_time:
        raise ValueError("result must contain 'per_time' key with cluster information")

    T = len(per_time)  # number of time points

    # Collect all unique area IDs if not specified
    if area_ids is None:
        all_areas = set()
        for t_result in per_time:
            for _, areas in _extract_clusters_for_timepoint(t_result):
                all_areas.update(areas)
        if not all_areas:
            raise ValueError("No areas found in any clusters across all time points")
        requested_area_ids = list(all_areas)
    else:
        requested_area_ids = list(area_ids)

    normalized_area_ids: list[int] = []
    seen_area_ids: set[int] = set()
    for position, value in enumerate(requested_area_ids):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"area_ids[{position}] must be a positive integer, got {value!r}")
        area_id = int(value)
        if area_id < 1:
            raise ValueError(f"area_ids[{position}] must be >= 1, got {area_id}")
        if area_id in seen_area_ids:
            raise ValueError(f"area_ids must be unique; duplicate value {area_id}")
        seen_area_ids.add(area_id)
        normalized_area_ids.append(area_id)
    area_ids = sorted(normalized_area_ids)

    N = len(area_ids)
    area_to_idx = {aid: i for i, aid in enumerate(area_ids)}

    # Initialize sequence matrix (N areas × T time points)
    sequences = np.zeros((N, T), dtype=np.int8)

    # Accumulate directional flags before encoding. This makes a HOT/COLD
    # conflict neutral regardless of list-form input ordering.
    direction_flags = np.zeros((N, T), dtype=np.int8)
    for t, t_result in enumerate(per_time):
        for cluster_type, cluster_areas in _extract_clusters_for_timepoint(t_result):
            if cluster_type == "HOT":
                flag = 1
            elif cluster_type == "COLD":
                flag = 2
            else:
                continue

            for area_id in cluster_areas:
                idx = area_to_idx.get(area_id)
                if idx is None:
                    continue
                direction_flags[idx, t] |= flag

    if include_type == "hot":
        sequences[(direction_flags & 1) != 0] = 1
    elif include_type == "cold":
        sequences[(direction_flags & 2) != 0] = -1
    else:
        sequences[direction_flags == 1] = 1
        sequences[direction_flags == 2] = -1

    return sequences, area_ids
