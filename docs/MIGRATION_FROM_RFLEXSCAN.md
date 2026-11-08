# Migrating from rflexscan — why flexscanx results can differ

<!-- BEGIN GENERATED: maintained_last_updated -->
**Last Updated:** 2026-11-08
<!-- END GENERATED: maintained_last_updated -->

This note explains, for a user moving from **rflexscan** (the R/FleXScan family)
to **flexscanx**, why the two engines can report different clusters and different
Monte-Carlo p-values for COLD, BOTH, RESTRICTED, BINOMIAL, and secondary-cluster
scans. It is the user-facing companion to the developer audit report and its
action register — the completed rflexscan-flaw audit + remediation, archived for
provenance under
[archive/reports/2026/rflexscan_flaw_audit_2026-06-15/](../archive/reports/2026/rflexscan_flaw_audit_2026-06-15/RFLEXSCAN_FLAWS_REPORT_2026-06-15.md)
([action register](../archive/reports/2026/rflexscan_flaw_audit_2026-06-15/ISSUES_AND_FIXES.md)).

## Parity vs correctness

rflexscan is flexscanx's *starting point*, not its yardstick. The **source of
truth is the method papers** — Tango & Takahashi's flexible scan statistic and
Otani et al.'s restricted/secondary-cluster definitions — not rflexscan's output.
rflexscan has several confirmed bugs (catalogued in the report); where flexscanx
disagrees with rflexscan because rflexscan is wrong, the divergence is a
**correction**, and parity with rflexscan would be a *regression away from the
papers*. So: treat numeric agreement with rflexscan as **evidence**, never as the
goal. The audit deliberately measured every divergence against the paper
equations (report Appendix A, "DEF-* paper equations"), and the differences below
are grouped by *who is wrong*.

Two kinds of difference follow. **(1)** rflexscan flaws that flexscanx already
gets right — flexscanx's behaviour did not change; results differ because
rflexscan is wrong. **(2)** flexscanx behaviour changes made in the 2026-06
remediation — p-values move relative to a *prior flexscanx release*, with an
opt-out where one applies.

---

## 1. rflexscan flaws that flexscanx corrects

No flexscanx behaviour change here. Results differ from rflexscan because
rflexscan deviates from the papers. Each item cites its flaw ID and the direction
of error; see report §2 ("rflexscan flaw catalogue") for the code-level evidence.

| Flaw | What rflexscan does wrong | Direction of error | What you see in flexscanx |
|---|---|---|---|
| **GATE-1 / GATE-2 / GATE-3** (RESTRICTED COLD) | COLD/BOTH centers are gated by the **HOT** tail (GATE-2), the COLD per-region middle-p is off-by-one (GATE-1), and at zero counts the COLD middle-p escapes `[0,1]` so every zero-count region auto-passes the cold gate (GATE-3). | rflexscan **never enumerates genuinely cold clusters** (GATE-2) yet **over-admits** sparse/zero-count cold nodes (GATE-1/3). | RESTRICTED COLD/BOTH detects pure-cold clusters rflexscan misses, and screens sparse cold regions at the paper-correct middle-p. |
| **LLR-5 / MC-2** (zero-case cold null) | A COLD/BOTH window with **zero in-window cases** evaluates to `NaN` and is silently dropped — from the observed MLC search **and** every Monte-Carlo replicate's null max. | rflexscan's COLD/BOTH **p-values are too small (anti-conservative)**, and the reported cold MLC is wrong when the coldest cluster is a zero-case region. | flexscanx evaluates zero-case windows to the finite paper LLR; the coldest cluster is selectable and the null max reflects it, giving the (larger, correct) p-value. |
| **ENUM-3** (binomial radius cap) | For `model="BINOMIAL"`, a finite `clusterradius` is **silently ignored** (the radius guard is missing from all four binomial scan functions). | rflexscan returns **larger clusters than requested** + biased p-values whenever `clusterradius` is finite under BINOMIAL. | flexscanx honours `clusterradius` model-agnostically, so BINOMIAL clusters respect the cap exactly like POISSON. |
| **MC-4** (binomial default null) | The default binomial null (`rantype="MULTINOMIAL"`) is a **clamped multinomial**, not an independent binomial; the proper binomial null is reachable only via the mislabeled `rantype="POISSON"`. | rflexscan computes binomial p-values against a **model-mismatched** reference distribution. | flexscanx's BINOMIAL model uses the independent-binomial null and **rejects** `MULTINOMIAL`/`POISSON` rantypes. For a like-for-like comparison, force rflexscan onto its hidden `rantype="POISSON"` binomial path. |
| **NUM-2** (coordinates) | The lat/lon path uses the spherical law of cosines `R·acos(dot)`, which loses precision for nearby points (can reorder neighbours), has a latent `acos(>1)=NaN`, and uses `R=6370 km`. rflexscan's own docs deprecate lat/lon "due to accuracy issues." | Geometry-dependent **neighbour mis-ordering** at fine spatial resolution on the lat/lon path. | flexscanx has **no great-circle path**: it requires **projected Cartesian** coordinates and warns (`ensure_projected`) on a geographic CRS. **Project your coordinates first** — as rflexscan also advises. |
| **NUM-5** (reproducibility) | An unstable `qsort` on distance ties makes the K-NN set platform-dependent, and there is no internal seed (`set.seed()` in R is the only control). | **Cross-platform non-reproducibility** under ties; silent non-reproducibility if the user forgets to seed. | flexscanx is reproducible **from `seed` alone**: identical results across thread counts and scheduler backends on the same build (counter-based common random numbers + an index tie-break on neighbours). Bit-for-bit agreement **across machines** holds for identical builds only — the null draws go through `std::poisson_distribution`, `std::binomial_distribution` and `std::discrete_distribution`, whose algorithms the C++ standard leaves to each standard library, and exact-tie handling depends on the compiler's `long double` width — so a scan replayed under a different standard library can draw a different null sample (same seed, same model, different replicate values). |

> rflexscan upstream courtesy patches for these are listed in the report; the
> vendored rflexscan source is read-only here, so flexscanx does not modify it.

---

## 2. flexscanx behaviour changes in this remediation

These move p-values **relative to the prior flexscanx release** (not just relative
to rflexscan). All three make flexscanx more paper-faithful; two are *more
conservative*, and one has an opt-out. See report §5 ("flexscanx flaws found along
the way").

### MC-3 — BOTH is now a proper two-sided test

Previously, `clustertype="BOTH"` reported the winning direction's **own one-sided**
p-value. flexscanx now ranks the BOTH primary (and its secondaries) against **one
combined two-sided null** = the per-replicate `max(hot_max, cold_max)`, following
DEF-BOTH (Tango 2008, p.81: the two-sided statistic deletes the direction
indicator).

- **Effect:** because `max(hot, cold) ≥` either one-sided max, the BOTH p-value is
  now **larger (more conservative)** than the old one-sided value — it pays the
  multiplicity cost of looking in both directions. This is the correct two-sided
  test, not a regression.
- **vs rflexscan:** flexscanx now matches rflexscan's *construction* of the BOTH
  null, but the *value* still differs because rflexscan's combined null is
  corrupted by MC-2 (it drops zero-case cold windows).
- **No opt-out** — the prior one-sided-for-BOTH behaviour was a defect. The
  per-direction one-sided p-values remain available under the result's `hot` and
  `cold` keys if you want them.

### MC-6 — secondary clusters now use the full-region null

Previously, every secondary cluster was ranked against a **conditional** null
recomputed over only the windows that avoid already-detected areas. flexscanx now
ranks each secondary against the **unconditional full-region null** already
computed for the primary, per DEF-STOP ("the same Monte-Carlo null distribution of
the maximum … as if it were the primary cluster").

- **Effect:** the conditional null's maxima are `≤` the full-region maxima, so the
  old secondary p-values were **anti-conservative**; the new full-region secondary
  p-values are **larger (more conservative)**. The primary/MLC is **unchanged**.
- **Opt-out:** pass `secondary_null="conditional"` to restore the prior behaviour;
  the new default is `secondary_null="full"`.
- **Note on BOTH:** under `clustertype="BOTH"`, the envelope primary is always a
  full-region *two-sided* rank by construction (MC-3), independent of this switch.

### GATE-4 — BOTH RESTRICTED screens at α₁/2 (shipped 2026-06-15)

For `clustertype="BOTH"` **and** `stattype="RESTRICTED"`, each of the two one-sided
sub-scans now screens each region at `ralpha/2` (Tango 2008, p.81: "replace α₁ with
α₁/2" — a Bonferroni split across the two tails). Previously each tail screened at
the full `ralpha`.

- **Effect:** BOTH/RESTRICTED admits **fewer windows** (the per-region screen is
  half as permissive). HOT-only and COLD-only scans are unchanged; `ORIGINAL` is a
  no-op (no per-region gate). The merged envelope still reports the configured
  `ralpha`; each sub-scan records the `ralpha/2` actually applied.
- This was the one place flexscanx itself was wrong (shared with rflexscan); see
  the `CHANGELOG.md` GATE-4 entry.

> **ENUM-2 (internal, no behaviour change).** The flexible-DFS enumeration comment
> was corrected — it had claimed each connected subset is generated exactly once,
> which is false for cyclic induced subgraphs. The duplicate visits are
> inference-neutral (the scan keeps the max LLR under a strict total-order
> tie-break), so the MLC and all p-values are unaffected. Mentioned only for
> completeness; nothing user-visible changed.

---

## Quick reference

| You are running… | Expect vs rflexscan | Expect vs prior flexscanx |
|---|---|---|
| RESTRICTED COLD / BOTH | flexscanx finds cold clusters rflexscan misses; sparse cold regions screened correctly (GATE-1/2/3) | BOTH/RESTRICTED admits fewer windows (GATE-4, α₁/2) |
| COLD / BOTH with zero-case regions | flexscanx p-values larger/correct; rflexscan's are too small (LLR-5/MC-2) | — |
| BOTH (any model) | matches rflexscan's null *construction*, not its MC-2-corrupted value | BOTH p larger/more conservative — now a true two-sided test (MC-3) |
| BINOMIAL with finite `clusterradius` | flexscanx respects the cap; rflexscan ignores it (ENUM-3) | — |
| BINOMIAL p-values | flexscanx uses the independent-binomial null; force rflexscan to `rantype="POISSON"` for parity (MC-4) | — |
| Multi-cluster scans (secondaries) | flexscanx matches DEF-STOP / rflexscan's full-region null | secondary p larger/more conservative; `secondary_null="conditional"` restores the old value (MC-6) |
| lat/lon inputs | project first — flexscanx has no great-circle path (NUM-2) | — |
| reproducibility | reproducible from `seed` alone across threads and scheduler backends on one build; across machines only for identical builds (standard-library distribution algorithms are implementation-defined) (NUM-5) | — |

For the full evidence, severity ratings, and the paper equations each behaviour is
checked against, see the archived audit report under
[archive/reports/2026/rflexscan_flaw_audit_2026-06-15/](../archive/reports/2026/rflexscan_flaw_audit_2026-06-15/RFLEXSCAN_FLAWS_REPORT_2026-06-15.md).
