"""Regression pins for the 2026-06-15 rflexscan-vs-flexscanx flaw audit.

Each test class locks in the *paper-correct* behaviour for a flaw where the audit
found rflexscan deviates from the method papers and **flexscanx is already
correct**
(``archive/reports/2026/rflexscan_flaw_audit_2026-06-15/RFLEXSCAN_FLAWS_REPORT_2026-06-15.md``
§2/§4;
``archive/reports/2026/rflexscan_flaw_audit_2026-06-15/ISSUES_AND_FIXES.md``
actions A1-A4, A6, A11, A13-A17). The
assertions are the audit's own numeric counterexamples, not values re-derived from
intuition — so if a pin ever fails, flexscanx has regressed *toward* rflexscan's
bug (a latent defect), and the fix is to restore the paper value, never to weaken
the assertion (the Phase-1b contingency in the remediation plan).

Reference probabilities are computed in closed form with :mod:`math` (small
counts, no underflow) so the suite carries no SciPy dependency; the large-``n``
CC-3 references are hard-coded SciPy/log-space values with their provenance noted.

The style mirrors ``tests/test_gate4_both_alpha_split.py`` (small coords + CSR
adjacency driving ``flexscanx.scan``); direct middle-p assertions use the compiled
bindings exposed at ``flexscanx._core``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import flexscanx as flx
import flexscanx._core as core

# --- audit numeric anchors (RFLEXSCAN_FLAWS_REPORT_2026-06-15.md) -------------

# LLR-5 / MC-2: the finite cold log-likelihood ratio of a zero-in-window window.
# Poisson nZ=0, mZ=2, nG=10, mG=10 -> 10*log(10/8) = 10*log(1.25).
_LLR5_POISSON = 2.231436  # report §2 / A3; exact 2.2314355131420975
# Binomial c=0, p=20, C=10, N=100 (four-cell ratio minus L0).
_LLR5_BINOMIAL = 2.366684  # report §2 / A3; exact 2.366684438629875

# GATE-1: Poisson lower-tail middle-p at n for xi=2 (the "at n", not "n-1", form).
_GATE1_POISSON_XI = 2.0
_GATE1_POISSON_MIDP = {0: 0.067668, 1: 0.270671, 2: 0.541341, 3: 0.766900}

# GATE-3: the xi below which a zero-case Poisson cold region must FAIL the alpha1
# screen (0.5*exp(-xi) crosses 0.20 at xi ~ 0.9163).
_GATE3_XI_CROSSOVER = 0.9163
_RALPHA_DEFAULT = 0.20


# --- graph builders (mirror the style template) ------------------------------


def _permute_csr(
    adjacency: tuple[list[int], list[int]], order: np.ndarray
) -> tuple[list[int], list[int]]:
    """Relabel a CSR adjacency so that new row ``i`` is old row ``order[i]``."""
    row_ptr, col_idx = adjacency
    n = len(row_ptr) - 1
    inverse = np.empty(n, dtype=int)
    inverse[np.asarray(order)] = np.arange(n)
    new_row_ptr, new_col_idx = [0], []
    for new_i in range(n):
        old_i = int(order[new_i])
        neighbours = sorted(int(inverse[j]) for j in col_idx[row_ptr[old_i] : row_ptr[old_i + 1]])
        new_col_idx.extend(neighbours)
        new_row_ptr.append(len(new_col_idx))
    return new_row_ptr, new_col_idx


def _chain(n: int) -> tuple[np.ndarray, tuple[list[int], list[int]]]:
    """An ``n``-node line graph: coords on the x-axis + CSR rook adjacency."""
    coords = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    row_ptr, col_idx = [0], []
    for i in range(n):
        if i > 0:
            col_idx.append(i - 1)
        if i < n - 1:
            col_idx.append(i + 1)
        row_ptr.append(len(col_idx))
    return coords, (row_ptr, col_idx)


def _grid(n: int) -> tuple[np.ndarray, tuple[list[int], list[int]], int]:
    """An ``n`` x ``n`` rook-adjacency grid: coords + CSR adjacency + node count."""
    total = n * n
    coords = np.array([[i % n, i // n] for i in range(total)], dtype=float)
    row_ptr, col_idx = [0], []
    for i in range(total):
        r, c = i // n, i % n
        nbrs = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                nbrs.append(nr * n + nc)
        col_idx.extend(sorted(nbrs))
        row_ptr.append(len(col_idx))
    return coords, (row_ptr, col_idx), total


def _areas(result: dict) -> list[int]:
    """Sorted 1-based area ids of a (sub-)result's primary cluster."""
    return sorted(int(a) for a in (result.get("area") or []))


# --- closed-form reference probabilities (no SciPy dependency) ----------------


def _poisson_pmf(k: int, mu: float) -> float:
    return math.exp(-mu) * mu**k / math.factorial(k)


def _poisson_cdf(k: int, mu: float) -> float:
    return math.fsum(_poisson_pmf(j, mu) for j in range(k + 1))


def _poisson_midp_lower(k: int, mu: float) -> float:
    """DEF-MIDP-COLD: F(k) - 0.5 f(k), evaluated AT k."""
    return _poisson_cdf(k, mu) - 0.5 * _poisson_pmf(k, mu)


def _binom_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * p**k * (1.0 - p) ** (n - k)


def _binom_cdf(k: int, n: int, p: float) -> float:
    return math.fsum(_binom_pmf(j, n, p) for j in range(k + 1))


def _binom_midp_lower(k: int, n: int, p: float) -> float:
    return _binom_cdf(k, n, p) - 0.5 * _binom_pmf(k, n, p)


class TestLLR5_MC2_zero_case_cold:
    """LLR-5 / MC-2 (A3 / A11): a zero-in-window COLD/BOTH window is the strongest
    cold spot, evaluates to a FINITE log-likelihood ratio (not ``NaN``), is
    selectable as the MLC, and participates in the Monte-Carlo null.

    rflexscan computes ``0*log0 = NaN`` and silently drops the window from both the
    observed search and every replicate maximum; flexscanx guards the in-window
    term (``llr_poisson.hpp:35``) so the window survives.
    """

    def test_poisson_zero_case_cold_llr_is_finite_paper_value(self) -> None:
        coords, adjacency = _chain(4)
        observed = [0, 0, 5, 5]
        expected = [1.0, 1.0, 4.0, 4.0]  # calibrated: sum == sum(observed) == 10
        for scanmethod in ("FLEXIBLE", "CIRCULAR"):
            res = flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                expected=expected,
                model="POISSON",
                scanmethod=scanmethod,
                stattype="ORIGINAL",
                clustertype="COLD",
                clustersize=2,
                K=3,
                simcount=0,
                n_jobs_centers=1,
            )
            assert _areas(res) == [1, 2], f"{scanmethod}: zero-case blob must be MLC"
            assert math.isfinite(res["stats"])
            assert res["stats"] == pytest.approx(_LLR5_POISSON, abs=1e-6)

    def test_binomial_zero_case_cold_llr_is_finite_paper_value(self) -> None:
        coords, adjacency = _chain(4)
        observed = [0, 0, 5, 5]
        population = [10.0, 10.0, 40.0, 40.0]  # c=0,p=20 in C=10,N=100
        for scanmethod in ("FLEXIBLE", "CIRCULAR"):
            res = flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                population=population,
                model="BINOMIAL",
                scanmethod=scanmethod,
                stattype="ORIGINAL",
                clustertype="COLD",
                clustersize=2,
                K=3,
                simcount=0,
                n_jobs_centers=1,
            )
            assert _areas(res) == [1, 2], f"{scanmethod}: zero-case blob must be MLC"
            assert math.isfinite(res["stats"])
            assert res["stats"] == pytest.approx(_LLR5_BINOMIAL, abs=1e-6)

    def test_zero_case_blob_is_mlc_and_has_finite_pvalue_across_matrix(self) -> None:
        """The zero-case cold blob is the MLC and yields a finite Monte-Carlo
        p-value for COLD and the COLD half of BOTH, both models, both scanmethods
        -- proving the window is not NaN-dropped from the null."""
        cases = {
            "POISSON": dict(expected=[1.0, 1.0, 4.0, 4.0]),
            "BINOMIAL": dict(population=[10.0, 10.0, 40.0, 40.0]),
        }
        coords, adjacency = _chain(4)
        observed = [0, 0, 5, 5]
        for model, payload in cases.items():
            for scanmethod in ("FLEXIBLE", "CIRCULAR"):
                for clustertype in ("COLD", "BOTH"):
                    res = flx.scan(
                        coords=coords,
                        adjacency=adjacency,
                        observed=observed,
                        model=model,
                        scanmethod=scanmethod,
                        stattype="ORIGINAL",
                        clustertype=clustertype,
                        clustersize=2,
                        K=3,
                        simcount=99,
                        seed=12345,
                        n_jobs_centers=1,
                        **payload,
                    )
                    cold = res["cold"] if clustertype == "BOTH" else res
                    label = f"{model}/{scanmethod}/{clustertype}"
                    assert _areas(cold) == [1, 2], f"{label}: cold MLC must be blob"
                    assert math.isfinite(cold["stats"]), f"{label}: finite LLR"
                    pval = cold["pval"]
                    assert (
                        pval is not None and 0.0 < float(pval) <= 1.0
                    ), f"{label}: zero-case cold blob must get a finite p-value"


class TestGATE1_cold_midp_at_n:
    """GATE-1 (A4): the lower-tail middle-p is evaluated AT n (``F(n) - 0.5 f(n)``),
    not at ``n-1``. rflexscan computes ``1 - Ppfm(n-1)``, over-admitting cold nodes
    by one count.
    """

    def test_poisson_lower_midp_matches_reported_values_at_n(self) -> None:
        for n, expected in _GATE1_POISSON_MIDP.items():
            got = core.poisson_middle_p_lower(n, _GATE1_POISSON_XI)
            assert got == pytest.approx(expected, abs=1e-6), f"xi=2, n={n}"
            assert got == pytest.approx(_poisson_midp_lower(n, _GATE1_POISSON_XI), abs=1e-12)

    def test_poisson_lower_midp_is_at_n_not_n_minus_one(self) -> None:
        """The "at n" value must differ from the buggy "n-1" form by
        ``0.5*(f(n)+f(n-1))`` -- a clearly non-zero gap for n >= 1."""
        for mu in (0.7, 2.0, 5.0):
            for n in range(1, 6):
                at_n = core.poisson_middle_p_lower(n, mu)
                at_n_minus_1 = _poisson_midp_lower(n - 1, mu)
                gap = 0.5 * (_poisson_pmf(n, mu) + _poisson_pmf(n - 1, mu))
                assert at_n == pytest.approx(_poisson_midp_lower(n, mu), abs=1e-12)
                assert abs(at_n - at_n_minus_1) == pytest.approx(gap, abs=1e-12)
                assert abs(at_n - at_n_minus_1) > 1e-6

    def test_binomial_lower_midp_is_at_n_not_n_minus_one(self) -> None:
        for n, p in ((5, 0.1), (20, 0.3), (10, 0.5)):
            for k in range(1, n):
                at_n = core.binomial_middle_p_lower(k, n, p)
                assert at_n == pytest.approx(_binom_midp_lower(k, n, p), abs=1e-12)
                if k >= 1:
                    gap = 0.5 * (_binom_pmf(k, n, p) + _binom_pmf(k - 1, n, p))
                    at_n_minus_1 = _binom_midp_lower(k - 1, n, p)
                    assert abs(at_n - at_n_minus_1) == pytest.approx(gap, abs=1e-12)


class TestGATE2_pure_cold_center:
    """GATE-2 (A1): a pure-cold cluster -- every member low-rate, no hot-significant
    seed -- is detected by RESTRICTED COLD and by the COLD half of BOTH. rflexscan
    gates the scan *center* on the HOT tail regardless of direction, so a genuinely
    cold seed fails and the whole cold cluster is never enumerated.
    """

    @staticmethod
    def _pure_cold_data() -> tuple[np.ndarray, tuple[list[int], list[int]], list[int], list[float]]:
        coords, adjacency = _chain(6)
        observed = [0, 0, 3, 3, 3, 3]  # every member <= expected -> no hot seed
        expected = [3.0] * 6
        return coords, adjacency, observed, expected

    def test_restricted_cold_detects_pure_cold_cluster(self) -> None:
        coords, adjacency, observed, expected = self._pure_cold_data()
        res = flx.scan(
            coords=coords,
            adjacency=adjacency,
            observed=observed,
            expected=expected,
            model="POISSON",
            scanmethod="FLEXIBLE",
            stattype="RESTRICTED",
            clustertype="COLD",
            ralpha=_RALPHA_DEFAULT,
            clustersize=2,
            K=4,
            simcount=0,
            n_jobs_centers=1,
        )
        detected = _areas(res)
        assert detected == [1, 2], "pure-cold blob must be the RESTRICTED COLD MLC"
        # Every detected member is genuinely cold (observed < expected): there is no
        # hot seed that rflexscan's HOT center gate could have latched onto.
        for area_id in detected:
            assert observed[area_id - 1] < expected[area_id - 1]

    def test_both_cold_half_detects_pure_cold_cluster(self) -> None:
        coords, adjacency, observed, expected = self._pure_cold_data()
        res = flx.scan(
            coords=coords,
            adjacency=adjacency,
            observed=observed,
            expected=expected,
            model="POISSON",
            scanmethod="FLEXIBLE",
            stattype="RESTRICTED",
            clustertype="BOTH",
            ralpha=_RALPHA_DEFAULT,
            clustersize=2,
            K=4,
            simcount=0,
            n_jobs_centers=1,
        )
        assert res["type"] == "BOTH"
        assert _areas(res["cold"]) == [1, 2], "COLD half of BOTH must find the blob"


class TestGATE3_midp_in_unit_interval:
    """GATE-3 (A2): the lower-tail middle-p stays in [0, 1] at n=0; rflexscan's
    ``1 - Ppfm(-1) < 0`` makes every zero-count region auto-pass the COLD gate. A
    zero-case region with xi < ~0.916 must instead be REJECTED at alpha1 = 0.20.
    """

    def test_poisson_zero_count_midp_in_unit_interval(self) -> None:
        for xi in (0.01, 0.3, 0.5, 0.9163, 1.0, 2.0, 5.0, 50.0, 200.0):
            lo = core.poisson_middle_p_lower(0, xi)
            hi = core.poisson_middle_p_upper(0, xi)
            assert 0.0 <= lo <= 1.0, f"poisson lower midp(0, {xi}) out of [0,1]"
            assert 0.0 <= hi <= 1.0, f"poisson upper midp(0, {xi}) out of [0,1]"

    def test_binomial_zero_count_midp_in_unit_interval(self) -> None:
        for n, p in ((5, 0.1), (20, 0.3), (50, 0.5), (100, 0.9), (1000, 0.999)):
            lo = core.binomial_middle_p_lower(0, n, p)
            hi = core.binomial_middle_p_upper(0, n, p)
            assert 0.0 <= lo <= 1.0, f"binom lower midp(0, {n}, {p}) out of [0,1]"
            assert 0.0 <= hi <= 1.0, f"binom upper midp(0, {n}, {p}) out of [0,1]"

    def test_zero_count_region_below_crossover_fails_alpha1_screen(self) -> None:
        """For xi < ~0.916 the cold screen p_i(0) >= alpha1, so the region is
        rejected (the paper value 0.5*exp(-0.3) = 0.370409, not rflexscan's
        -0.370409)."""
        assert core.poisson_middle_p_lower(0, 0.3) == pytest.approx(0.370409, abs=1e-6)
        for xi in (0.30, 0.50, 0.90):
            assert xi < _GATE3_XI_CROSSOVER
            assert core.poisson_middle_p_lower(0, xi) >= _RALPHA_DEFAULT
        # Above the crossover the gate correctly admits (it discriminates, it does
        # not auto-pass everything as rflexscan's negative value would).
        for xi in (1.5, 3.0):
            assert core.poisson_middle_p_lower(0, xi) < _RALPHA_DEFAULT
        # Binomial twin: bn=5, p=0.1 -> 0.295245 >= alpha1 (rejected).
        assert core.binomial_middle_p_lower(0, 5, 0.1) == pytest.approx(0.295245, abs=1e-6)
        assert core.binomial_middle_p_lower(0, 5, 0.1) >= _RALPHA_DEFAULT

    def test_zero_case_region_rejected_only_under_restricted_gate(self) -> None:
        """Behavioural: a 0-case xi<0.916 region is found by ORIGINAL COLD but
        screened out by RESTRICTED COLD -- the screen does the rejecting."""
        coords, adjacency = _chain(4)
        observed = [0, 0, 5, 5]
        expected = [0.5, 0.5, 4.0, 4.0]  # cold singletons have small calibrated xi
        common = dict(
            coords=coords,
            adjacency=adjacency,
            observed=observed,
            expected=expected,
            model="POISSON",
            scanmethod="FLEXIBLE",
            clustertype="COLD",
            ralpha=_RALPHA_DEFAULT,
            clustersize=1,
            K=3,
            simcount=0,
            n_jobs_centers=1,
        )
        assert _areas(flx.scan(stattype="ORIGINAL", **common)) != []
        assert _areas(flx.scan(stattype="RESTRICTED", **common)) == []


class TestENUM3_binomial_radius:
    """ENUM-3 (A6): BINOMIAL honours a finite ``clusterradius`` exactly as POISSON
    does. rflexscan omits the radius guard from its four Binomial scan functions,
    so a finite radius is silently void for ``model='BINOMIAL'``.
    """

    @staticmethod
    def _run(model: str, scanmethod: str, radius: float) -> tuple[tuple[int, ...], float]:
        coords, adjacency = _chain(8)
        observed = [1, 1, 1, 1, 8, 8, 8, 8]  # hot run on nodes 4..7
        payload = dict(expected=[1.0] * 8) if model == "POISSON" else dict(population=[10.0] * 8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                model=model,
                scanmethod=scanmethod,
                stattype="ORIGINAL",
                clustertype="HOT",
                clusterradius=radius,
                clustersize=8,
                K=8,
                simcount=0,
                n_jobs_centers=1,
                **payload,
            )
        return tuple(_areas(res)), float(res["max_dist"])

    def test_binomial_radius_matches_poisson_path(self) -> None:
        for scanmethod in ("FLEXIBLE", "CIRCULAR"):
            for radius in (float("inf"), 1.0, 0.5):
                binom = self._run("BINOMIAL", scanmethod, radius)
                poisson = self._run("POISSON", scanmethod, radius)
                assert binom == poisson, f"{scanmethod} R={radius}: BINOMIAL != POISSON"

    def test_binomial_radius_cap_binds(self) -> None:
        """A finite radius shrinks the BINOMIAL cluster (it is not ignored), and the
        cluster diameter never exceeds 2*radius (every member within radius of the
        centre)."""
        for scanmethod in ("FLEXIBLE", "CIRCULAR"):
            unbounded, _ = self._run("BINOMIAL", scanmethod, float("inf"))
            mid_area, mid_diam = self._run("BINOMIAL", scanmethod, 1.0)
            tight_area, tight_diam = self._run("BINOMIAL", scanmethod, 0.5)
            assert len(unbounded) > len(mid_area) > len(tight_area)
            assert len(tight_area) == 1, "radius 0.5 on unit spacing -> singleton"
            assert mid_diam <= 2.0 * 1.0 + 1e-9
            assert tight_diam <= 2.0 * 0.5 + 1e-9


class TestMC4_binomial_null:
    """MC-4 (A13): the BINOMIAL model accepts only the independent-binomial null.
    rflexscan defaults BINOMIAL to a clamped MULTINOMIAL draw (the wrong reference
    null); flexscanx rejects MULTINOMIAL/POISSON rantypes for BINOMIAL.
    """

    @staticmethod
    def _binomial_scan(rantype: str | None) -> dict:
        coords, adjacency = _chain(4)
        return flx.scan(
            coords=coords,
            adjacency=adjacency,
            observed=[0, 1, 5, 5],
            population=[10.0] * 4,
            model="BINOMIAL",
            scanmethod="CIRCULAR",
            clustertype="HOT",
            clustersize=2,
            K=3,
            simcount=0,
            n_jobs_centers=1,
            rantype=rantype,
        )

    def test_multinomial_and_poisson_rantype_rejected(self) -> None:
        for rantype in ("MULTINOMIAL", "POISSON"):
            with pytest.raises(ValueError, match="rantype"):
                self._binomial_scan(rantype)

    def test_binomial_default_null_is_independent_binomial(self) -> None:
        # The default (rantype=None) and the explicit BINOMIAL null both run.
        for rantype in (None, "BINOMIAL"):
            res = self._binomial_scan(rantype)
            assert isinstance(res, dict) and "area" in res


class TestNUM2_projected_coords:
    """NUM-2 (A14): flexscanx uses projected Cartesian KNN with an index tie-break
    (deterministic under distance ties) and warns on a geographic CRS -- it has no
    inaccurate lat/lon great-circle path. rflexscan's ``R*acos(dot)`` loses
    precision and its unstable ``qsort`` reorders tied neighbours.
    """

    def test_ensure_projected_warns_on_geographic_crs(self) -> None:
        class _GeographicCRS:
            is_geographic = True

            def __str__(self) -> str:
                return "EPSG:4326"

        class _GeoFrame:
            crs = _GeographicCRS()

        with pytest.warns(UserWarning, match="geographic CRS"):
            flx.ensure_projected(_GeoFrame())

    def test_ensure_projected_silent_on_projected_crs(self) -> None:
        class _ProjectedCRS:
            is_geographic = False

            def __str__(self) -> str:
                return "EPSG:3857"

        class _ProjFrame:
            crs = _ProjectedCRS()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            flx.ensure_projected(_ProjFrame())  # must not warn

    def test_knn_fixed_selection_is_tie_stable(self) -> None:
        """On an integer grid (many exact distance ties) the deterministic K_FIXED
        top-K (distance^2, then index tie-break) yields identical results across
        repeated runs."""
        coords, adjacency, total = _grid(4)
        observed = np.full(total, 3, dtype=int)
        observed[5] = 10
        observed[6] = 10
        expected = np.full(total, 3.0)
        common = dict(
            coords=coords,
            adjacency=adjacency,
            observed=observed,
            expected=expected,
            model="POISSON",
            scanmethod="CIRCULAR",
            stattype="ORIGINAL",
            clustertype="HOT",
            clustersize=4,
            K=8,
            simcount=0,
            n_jobs_centers=1,
            knn_mode="K_FIXED",
        )
        first = flx.scan(**common)
        second = flx.scan(**common)
        assert _areas(first) == _areas(second)
        assert first["stats"] == pytest.approx(second["stats"], abs=0.0)

    def test_tie_at_the_universe_boundary_is_announced_for_every_labelling(self) -> None:
        """D4 / FXA-187: on the integer grid the K-limited universe boundary falls inside a
        tied distance group, so the index tie-break makes the admissible windows depend
        on the row order.  Each labelling is itself reproducible, and each one emits the
        tie warning so the dependence is discoverable."""
        coords, adjacency, total = _grid(4)
        observed = np.full(total, 3, dtype=int)
        observed[5] = 10
        observed[6] = 10
        expected = np.full(total, 3.0)
        permutation = np.random.default_rng(4).permutation(total)
        for label, order in (("identity", np.arange(total)), ("permuted", permutation)):
            common = dict(
                coords=coords[order],
                adjacency=_permute_csr(adjacency, order),
                observed=observed[order],
                expected=expected[order],
                model="POISSON",
                scanmethod="CIRCULAR",
                stattype="ORIGINAL",
                clustertype="HOT",
                clustersize=4,
                K=5,
                simcount=0,
                n_jobs_centers=1,
                knn_mode="K_FIXED",
            )
            with pytest.warns(UserWarning, match="equidistant neighbours at the K=5") as record:
                first = flx.scan(**common)
            assert sum("equidistant" in str(w.message) for w in record) == 1, label
            with pytest.warns(UserWarning, match="equidistant neighbours"):
                second = flx.scan(**common)
            assert _areas(first) == _areas(second), label
            assert first["stats"] == pytest.approx(second["stats"], abs=0.0), label

    def test_tie_free_geometry_is_relabelling_invariant_and_silent(self) -> None:
        """Breaking the exact ties removes both the warning and the labelling dependence."""
        coords, adjacency, total = _grid(4)
        coords = coords + np.random.default_rng(11).uniform(-1e-3, 1e-3, coords.shape)
        observed = np.full(total, 3, dtype=int)
        observed[5] = 10
        observed[6] = 10
        expected = np.full(total, 3.0)
        results = []
        for order in (np.arange(total), np.random.default_rng(4).permutation(total)):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                res = flx.scan(
                    coords=coords[order],
                    adjacency=_permute_csr(adjacency, order),
                    observed=observed[order],
                    expected=expected[order],
                    model="POISSON",
                    scanmethod="CIRCULAR",
                    stattype="ORIGINAL",
                    clustertype="HOT",
                    clustersize=4,
                    K=5,
                    simcount=0,
                    n_jobs_centers=1,
                    knn_mode="K_FIXED",
                )
            # Map 1-based area ids of the permuted input back to the original labels.
            results.append((sorted(int(order[a - 1]) + 1 for a in res["area"]), res["stats"]))
        assert results[0][0] == results[1][0]
        assert results[0][1] == pytest.approx(results[1][1], abs=0.0)


class TestNUM3_large_N_and_clustersize:
    """NUM-3 (A15): flexscanx uses ``int``/``size_t`` indices (no 16-bit ``short``
    wrap at N>32767) and validates ``clustersize`` rather than overrunning a fixed
    64-slot buffer.
    """

    def test_large_n_grid_runs_without_index_overflow(self) -> None:
        n = 40_000  # > SHRT_MAX (32767): would wrap a 16-bit area index
        coords = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
        row_ptr = np.empty(n + 1, dtype=np.int64)
        col_idx = np.empty(2 * (n - 1), dtype=np.int64)
        row_ptr[0] = 0
        pos = 0
        for i in range(n):
            if i > 0:
                col_idx[pos] = i - 1
                pos += 1
            if i < n - 1:
                col_idx[pos] = i + 1
                pos += 1
            row_ptr[i + 1] = pos
        observed = np.full(n, 2, dtype=int)
        observed[20_000:20_003] = 15  # a planted hot blob in the interior
        expected = np.full(n, 2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # benign K_REFILL guardrail on large N
            res = flx.scan(
                coords=coords,
                adjacency=(row_ptr, col_idx[:pos]),
                observed=observed,
                expected=expected,
                model="POISSON",
                scanmethod="CIRCULAR",
                stattype="ORIGINAL",
                clustertype="HOT",
                clustersize=4,
                K=4,
                simcount=0,
                n_jobs_centers=1,
            )
        assert math.isfinite(res["stats"]) and res["stats"] > 0.0
        assert 20_001 in _areas(res)  # 1-based id of the planted centre

    def test_clustersize_is_validated_not_silently_truncated(self) -> None:
        coords, adjacency = _chain(4)
        common = dict(
            coords=coords,
            adjacency=adjacency,
            observed=[0, 1, 5, 5],
            expected=[1.0, 1.0, 4.0, 4.0],
            scanmethod="CIRCULAR",
            K=3,
            simcount=0,
            n_jobs_centers=1,
        )
        with pytest.raises(ValueError, match="clustersize"):
            flx.scan(clustersize=0, **common)
        # Oversized clustersize is announced (capped to K with a warning), never a
        # silent buffer overrun as in rflexscan's z[KLIM=64].
        with pytest.warns(UserWarning, match="exceeds K"):
            flx.scan(clustersize=99, **common)


class TestNUM5_reproducibility:
    """NUM-5 (A16): identical ``seed`` reproduces identical clusters and p-values
    across ``n_jobs_centers`` and ``both_parallel`` -- counter-based common random
    numbers make each replicate draw a pure function of (seed, rep_index).
    rflexscan offers no internal seed and an unstable neighbour sort.
    """

    @staticmethod
    def _signature(res: dict) -> tuple:
        def sub(d: dict) -> tuple:
            return tuple(_areas(d)), d["stats"], d["pval"]

        return sub(res["hot"]), sub(res["cold"])

    def test_same_seed_same_result_under_relabelling_on_tie_free_geometry(self) -> None:
        """D4 / FXA-187: with no exact distance ties the reported clusters, statistics
        and p-values are a function of the geography and the counts alone, so
        permuting the input rows (and remapping the area ids) changes nothing."""
        n = 10
        spacing = 1.0 + 0.37 * np.arange(n - 1)  # strictly distinct pairwise distances
        xs = np.concatenate([[0.0], np.cumsum(spacing)])
        coords = np.column_stack([xs, np.zeros(n)])
        _, adjacency = _chain(n)
        observed = np.array([0, 0, 3, 3, 3, 3, 3, 9, 9, 9])
        expected = np.full(n, 3.0)
        rng = np.random.default_rng(5)
        signatures = []
        for order in (np.arange(n), rng.permutation(n), rng.permutation(n)):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                res = flx.scan(
                    coords=coords[order],
                    adjacency=_permute_csr(adjacency, order),
                    observed=observed[order],
                    expected=expected[order],
                    model="POISSON",
                    scanmethod="FLEXIBLE",
                    stattype="RESTRICTED",
                    clustertype="BOTH",
                    ralpha=_RALPHA_DEFAULT,
                    clustersize=3,
                    K=5,
                    simcount=199,
                    seed=12345,
                )

            def back(d: dict) -> tuple:
                return (
                    tuple(sorted(int(order[a - 1]) + 1 for a in d["area"])),
                    d["stats"],
                    d["pval"],
                )

            signatures.append((back(res["hot"]), back(res["cold"])))
        assert signatures[0] == signatures[1] == signatures[2]

    def test_same_seed_same_result_across_parallelism(self) -> None:
        coords, adjacency = _chain(10)
        observed = [0, 0, 3, 3, 3, 3, 3, 9, 9, 9]
        expected = [3.0] * 10
        common = dict(
            coords=coords,
            adjacency=adjacency,
            observed=observed,
            expected=expected,
            model="POISSON",
            scanmethod="FLEXIBLE",
            stattype="RESTRICTED",
            clustertype="BOTH",
            ralpha=_RALPHA_DEFAULT,
            clustersize=3,
            K=5,
            simcount=199,
            seed=12345,
        )
        serial = flx.scan(n_jobs_centers=1, both_parallel=False, **common)
        threaded = flx.scan(n_jobs_centers=4, both_parallel=False, **common)
        both_par = flx.scan(n_jobs_centers=1, both_parallel=True, **common)
        assert self._signature(serial) == self._signature(threaded)
        assert self._signature(serial) == self._signature(both_par)


class TestCC3_binom_cdf_large_n:
    """CC-3 / NUM-1 (A17): the large-``n`` binomial middle-p must not collapse to
    0/1. This was flexscanx's *own* historical bug (the ``(1-p)^n`` recurrence-base
    underflow), since fixed in log space (``middle_p_binomial.hpp:70-91``).

    The scan-level regression already lives in
    ``tests/test_audit_v001c.py::test_binomial_restricted_gating_large_population``
    (AUD-3) and the identity checks in ``tests/test_core_bounds.py``; this pin adds
    the direct value check against the log-space reference.
    """

    # SciPy references (binom.cdf - 0.5*binom.pmf); the naive base (1-p)**n
    # underflows to exactly 0.0 for each of these, so a non-log-space CDF collapses.
    _LARGE_N_LOWER_MIDP = {
        (1100, 0.5, 540): 0.273348,
        (2000, 0.99, 1980): 0.485548,
        (2000, 0.5, 1000): 0.500000,
        (5000, 0.999, 4995): 0.471817,
    }

    def test_large_n_lower_midp_matches_reference_and_does_not_collapse(self) -> None:
        for (n, p, k), reference in self._LARGE_N_LOWER_MIDP.items():
            assert (1.0 - p) ** n == 0.0, "precondition: naive recurrence base underflows"
            lo = core.binomial_middle_p_lower(k, n, p)
            hi = core.binomial_middle_p_upper(k, n, p)
            assert 0.0 < lo < 1.0, f"n={n} p={p} k={k}: collapsed to {lo}"
            assert lo == pytest.approx(reference, abs=1e-6)
            assert lo + hi == pytest.approx(1.0, abs=1e-10)
