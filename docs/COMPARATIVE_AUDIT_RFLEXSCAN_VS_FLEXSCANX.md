# flexscanx vs rflexscan

**Status:** Maintained comparison note  
<!-- BEGIN GENERATED: maintained_last_updated -->
**Last Updated:** 2026-11-08
<!-- END GENERATED: maintained_last_updated -->

This document is the current high-level comparison between the original `rflexscan`
R package and `flexscanx`. For the user-facing, item-by-item catalogue of *why
results differ* (directions of error and opt-outs), see
[`docs/MIGRATION_FROM_RFLEXSCAN.md`](MIGRATION_FROM_RFLEXSCAN.md).

It is intentionally narrower than the historical deep audit. It avoids hard-coded
file counts, deleted-file references, and stale benchmark numbers. For the full
completed audits and older comparative reports, see `archive/reports/` and
`archive/development_audits/`.

> **Read this first — rflexscan is the starting point, not the source of truth.**
> Per `ECOSYSTEM.md` ("Source of truth") and `DUPLICATES.md` (fact 4), flexscanx
> deliberately **corrects rflexscan/FleXScan's algorithmic and mathematical
> errors** and may expand beyond it. This comparison is a **divergence detector**,
> not a pass/fail oracle: a difference from rflexscan is **not** automatically a
> flexscanx defect. Each divergence is judged against the scan statistic's
> mathematical definition (Otani & Takahashi / Kulldorff method papers); where
> rflexscan is wrong, flexscanx's behaviour is the intended correction, and
> flexscanx is never adjusted to reproduce an rflexscan bug.

## Bottom Line

- `flexscanx` is a reimplementation, not a port or wrapper of `rflexscan`.
- **rflexscan is the starting point, not the source of truth.** Correctness is
  judged against the method papers / mathematical definitions, not against
  rflexscan's output; documented divergences where rflexscan is wrong are
  deliberate corrections.
- The mathematical target is the same flexible spatial scan family, but some
  user-visible behavior differs by design.
- `flexscanx` adds a modern Python API, a C++17 core, time-series workflows,
  sequence-analysis helpers, plotting/reporting utilities, and release-gated
  repository tooling.

## What `rflexscan` Is

`rflexscan` is the original R package implementation of the flexible spatial scan
statistic described by Otani and Takahashi.

In this repository, the original upstream source is kept under
`RFLEXSCAN_ORIGINAL/` as read-only reference material for validation, comparison,
and historical context.

## What `flexscanx` Is

`flexscanx` provides:

- a Python-facing API in `src/flexscanx/`
- a compiled C++17 core in `src/flexscanx_core/`
- additional workflow support for validation, tuning, plotting, reporting, and
  temporal/sequence analysis

It is designed as a first-release codebase with one canonical API surface and
without backward-compatibility shims for unreleased interfaces.

## Stable Differences

### 1. Implementation Model

- `rflexscan`: R package with R/Rcpp integration
- `flexscanx`: Python package with pybind11 and a C++17 core

This changes the integration surface, developer ergonomics, and extension model,
even when the underlying statistical goal is the same.

### 2. BOTH-Mode Semantics

`flexscanx` builds `clustertype="BOTH"` from separate HOT and COLD scans, then
ranks the combined primary against **one combined two-sided null** (per
Monte-Carlo replicate, `max(hot_max, cold_max)`; Tango 2008, p.81). The result
still exposes `hot`, `cold`, `primary`, and `primary_type`, and retains the
per-direction one-sided p-values under `hot`/`cold`. The BOTH p-value is therefore
a proper two-sided test — more conservative than a single one-sided tail. Do not
assume byte-for-byte p-value parity with `rflexscan` in mixed-direction cases.

### 3. COLD and RESTRICTED Handling

`flexscanx` uses the paper-correct lower-tail middle-p (Tango 2008) and screens
RESTRICTED COLD/BOTH centers and nodes by the scan direction. This corrects
several rflexscan defects — HOT-gated cold centers, an off-by-one cold middle-p, an
out-of-range cold middle-p at zero counts, and zero-case cold windows dropped from
the null. COLD/BOTH/RESTRICTED results therefore differ from rflexscan by design;
the per-flaw detail and directions of error are in
[`docs/MIGRATION_FROM_RFLEXSCAN.md`](MIGRATION_FROM_RFLEXSCAN.md).

### 4. Extended Workflow Surface

Compared with the original R package, `flexscanx` includes repository-native
support for:

- `scan_series()` time-slice workflows
- sequence-analysis utilities under `flexscanx.sequence`
- plotting helpers in `flexscanx.plotting`
- report export helpers in `flexscanx.report`
- validation and tuning helpers in `flexscanx.validation` and `flexscanx.tuning`

These are project-level capabilities, not strict parity features.

## Audit and Measured Parity

Two completed audits underpin this comparison:

- The **2026-06-11/12 end-to-end audit** ran identical synthetic inputs through
  both implementations across the full grid (POISSON/BINOMIAL × FLEXIBLE/CIRCULAR
  × ORIGINAL/RESTRICTED × HOT/COLD/BOTH): **20 of 24 cells return the identical
  primary cluster, relative LLR deltas ≤ 4e-13** (matched secondary clusters agree
  to the same precision). The 4 non-matching cells are all RESTRICTED+COLD — the
  intentional gating/middle-p correction above. HOT Monte-Carlo p-values agree
  within MC noise.
- The **2026-06-15 rflexscan-flaw audit** then resolved the one question the first
  audit left open: the systematic COLD/BOTH Monte-Carlo p-value divergence. It is
  **rflexscan's flaw** (zero-case cold windows are dropped from the null, so
  rflexscan's COLD p-values are anti-conservative); the BOTH gap was a flexscanx
  issue, now fixed to a combined two-sided null. flexscanx's COLD/BOTH p-values are
  the paper-correct values, not a defect to "fix" toward rflexscan.

Full evidence, severity ratings, and the parity matrix are in the archived
completed audit at
[`archive/reports/2026/rflexscan_flaw_audit_2026-06-15/`](../archive/reports/2026/rflexscan_flaw_audit_2026-06-15/RFLEXSCAN_FLAWS_REPORT_2026-06-15.md)
(the 2026-06-11/12 end-to-end report is archived alongside under
`archive/reports/2026/superseded/`). The measured parity holds under the
post-audit defaults (exact scan engine); the pre-audit default pruning heuristics
broke BINOMIAL HOT primary parity and are now opt-in.

## What To Use For Current Truth

For current behavior and contracts, prefer these sources in this order:

1. `API_REFERENCE.md`
2. `README.md`
3. executable vignettes in `vignettes/`
4. current tests and release-gated scripts

For current benchmark data, run:

- `scripts/bench.py`
- `scripts/perf_curves.py`

Do not rely on older audit documents for current performance numbers or file-layout
descriptions.

## Related Files

- `docs/MIGRATION_FROM_RFLEXSCAN.md`: user-facing catalogue of result differences
- `RFLEXSCAN_ORIGINAL/README.md`: upstream reference-material overview
- `docs/DEVELOPER_GUIDE.md`: current repository/developer notes
- `vignettes/notes/HOT_ORIGINAL_Parity_Otani_Takahashi_2021.qmd`: parity-oriented example
- `archive/reports/`: completed and timestamped historical reports
- `archive/development_audits/`: superseded development audits

## Historical Note

Earlier versions of this document were much larger and included detailed benchmark
claims, file inventories, and line-by-line comparative audit material. Those
details became stale as the codebase evolved. The active document is now kept short
and maintained; the historical material remains in the archive.
