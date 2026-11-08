"""GATE-4: two-sided (BOTH) RESTRICTED scan must screen each tail at ralpha/2.

Per Tango (2008, Jpn. J. Biometrics 29(2):75-95, p.81): for the two-sided
restricted statistic, delete the direction indicator and "replace alpha1 with
alpha1/2 in equation (6)" -- a Bonferroni split of the per-region significance
level across the two tails. flexscanx builds BOTH as an envelope of two one-sided
RESTRICTED sub-scans, so each sub-scan must screen at ralpha/2.

Audit reference:
archive/reports/2026/rflexscan_flaw_audit_2026-06-15/findings/02_gating.md (GATE-4),
archive/reports/2026/rflexscan_flaw_audit_2026-06-15/ISSUES_AND_FIXES.md (A5).
Before the fix both engines screened BOTH at
the full ralpha (too permissive by 2x on the per-region level).

Two checks:
  * metadata -- the two sub-scans of a BOTH/RESTRICTED scan report ralpha/2, while
    the merged envelope still reports the user-configured ralpha; ORIGINAL is
    untouched (it has no per-region gate).
  * behaviour -- a centre whose one-sided middle-p sits between ralpha/2 and
    ralpha is admitted by a one-sided RESTRICTED scan but screened out of BOTH,
    and re-admitted to BOTH only when ralpha is doubled.
"""

import numpy as np

import flexscanx as flx


def _grid3_csr():
    """3x3 rook-adjacency grid: coords + (row_ptr, col_idx) CSR adjacency."""
    n = 3
    N = n * n
    coords = np.array([[i % n, i // n] for i in range(N)], dtype=float)
    adj = {i: [] for i in range(N)}
    for i in range(N):
        r, c = i // n, i % n
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                adj[i].append(nr * n + nc)
    row_ptr, col_idx = [0], []
    for i in range(N):
        col_idx.extend(sorted(adj[i]))
        row_ptr.append(len(col_idx))
    return coords, (row_ptr, col_idx), N


def test_both_restricted_subscans_screen_at_half_ralpha_metadata():
    """The two BOTH sub-scans run at ralpha/2; the envelope reports ralpha."""
    coords, adjacency, N = _grid3_csr()
    observed = np.full(N, 3, dtype=int)
    observed[4] = 6
    expected = np.full(N, 3.0)

    res = flx.scan(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        clustertype="BOTH",
        stattype="RESTRICTED",
        ralpha=0.2,
        clustersize=3,
        K=8,
        simcount=0,
        n_jobs_centers=1,
    )

    assert res["type"] == "BOTH"
    # Each one-sided sub-scan applied (and reports) the Bonferroni-split level.
    assert res["hot"]["scan_settings"]["ralpha"] == 0.1
    assert res["cold"]["scan_settings"]["ralpha"] == 0.1
    # The merged envelope keeps the user-configured two-sided level.
    assert res["scan_settings"]["ralpha"] == 0.2


def test_both_original_is_unaffected_metadata():
    """ORIGINAL has no per-region gate -> ralpha is not halved (no misleading meta)."""
    coords, adjacency, N = _grid3_csr()
    observed = np.full(N, 3, dtype=int)
    observed[4] = 6
    expected = np.full(N, 3.0)

    res = flx.scan(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        clustertype="BOTH",
        stattype="ORIGINAL",
        ralpha=0.2,
        clustersize=3,
        K=8,
        simcount=0,
        n_jobs_centers=1,
    )
    assert res["hot"]["scan_settings"]["ralpha"] == 0.2
    assert res["cold"]["scan_settings"]["ralpha"] == 0.2
    assert res["scan_settings"]["ralpha"] == 0.2


def _areas(result):
    return set(int(a) for a in (result.get("area") or []))


def test_both_alpha_split_changes_admissibility_behaviour():
    """End-to-end: a centre with HOT middle-p in (ralpha/2, ralpha) is admitted by
    a one-sided HOT RESTRICTED scan but screened out of BOTH (which now gates at
    ralpha/2), and re-admitted to BOTH when ralpha is doubled.

    Centre area 4 (1-based id 5): observed=5, expected=3 -> Poisson HOT middle-p
    P(X>=6)+0.5 P(X=5) for Pois(3) = 0.0839 + 0.5*0.1008 ~= 0.134, which lies
    between 0.10 (= 0.2/2) and 0.20. All other areas are at expectation (obs=exp),
    so they are not HOT candidates and fail the COLD gate -> the BOTH result is
    governed solely by whether the centre clears its HOT screen.
    """
    coords, adjacency, N = _grid3_csr()
    observed = np.full(N, 3, dtype=int)
    observed[4] = 5  # HOT middle-p ~ 0.134
    expected = np.full(N, 3.0)
    center_id = 5  # 1-based id of node index 4

    common = dict(
        coords=coords,
        adjacency=adjacency,
        observed=observed,
        expected=expected,
        model="POISSON",
        scanmethod="FLEXIBLE",
        stattype="RESTRICTED",
        clustersize=1,  # singletons only -> reported iff centre clears the gate
        K=8,
        simcount=0,
        n_jobs_centers=1,
    )

    # One-sided HOT at ralpha=0.2: 0.134 < 0.2 -> centre admitted and reported.
    hot = flx.scan(clustertype="HOT", ralpha=0.2, **common)
    assert center_id in _areas(hot), "centre should pass one-sided HOT gate at 0.2"

    # BOTH at ralpha=0.2 -> per-tail 0.10: 0.134 > 0.10 -> centre screened out.
    both_narrow = flx.scan(clustertype="BOTH", ralpha=0.2, **common)
    assert center_id not in _areas(
        both_narrow["hot"]
    ), "centre with HOT middle-p 0.134 must be screened out of BOTH at ralpha/2=0.10"

    # BOTH at ralpha=0.4 -> per-tail 0.20: 0.134 < 0.20 -> centre re-admitted.
    both_wide = flx.scan(clustertype="BOTH", ralpha=0.4, **common)
    assert center_id in _areas(
        both_wide["hot"]
    ), "centre should be re-admitted to BOTH when ralpha/2 rises to 0.20"
