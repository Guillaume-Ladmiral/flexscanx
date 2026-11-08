# Documentation Index

<!-- BEGIN GENERATED: maintained_last_updated -->
**Last Updated:** 2026-11-08
<!-- END GENERATED: maintained_last_updated -->

Canonical map of documentation for `PROJECT_FLEXSCANX`.

## Canonical Active Documentation

- `README.md`: Human-facing start page for installation, quick start, and practical usage.
- `API_REFERENCE.md`: Canonical public API contract (the single Python API reference).
- `CHANGELOG.md`: Release notes for published versions.
- `docs/README.md`: **Single hub** for all Python-implementation documentation — guides, the API reference, and every vignette in one place.
- `docs/ADVANCED_USAGE.md`: Practical runtime guidance beyond the first scan.
- `docs/MIGRATION_FROM_RFLEXSCAN.md`: Why flexscanx results can differ from rflexscan — per-flaw catalogue, directions of error, and opt-outs.
- `docs/COMPARATIVE_AUDIT_RFLEXSCAN_VS_FLEXSCANX.md`: Maintained high-level comparison with rflexscan.
- `vignettes/README.md`: Index of active executable workflows plus maintained method notes.
- `vignettes/workflows/*.qmd`: Runnable end-user workflows and examples.
- `vignettes/workflows/rflexscan_manual_examples.qmd`: Executable reproduction of every worked example in the original rflexscan 1.1.0 manual.
- `vignettes/notes/*.qmd`: Maintained validation, parity, and method notes.

## Project Family and Governance

- `AGENTS.md`: Environment policy and this project's role/responsibilities (read first).
- `ECOSYSTEM.md`: Points to the four-project family charter, whose one master copy is `ECOSYSTEM.md` in PROJECT_TRAMINEX (since 2026-10-26), and holds this project's source of truth.
- `DUPLICATES.md`: Function-by-function duplicate inventory and convergence decisions (shared; master copy in PROJECT_TRAMINEX).
- `MIGRATION_PLAN.md`: Cross-project convergence plan (shared; master copy in PROJECT_TRAMINEX).

## Reference-Only Material

- `DOCUMENTATION/README.md`: Academic reference map and citation guidance.
- `RFLEXSCAN_ORIGINAL/README.md`: Read-only upstream reference-material guide.
- `archive/generated/reference_extracts/README.md`: Manifest for generated text extracts derived from academic PDFs.
- `archive/generated/raw_dumps/README.md`: Manifest for uncurated repository dumps and raw inventories.
- `archive/internal_notes/README.md`: Manifest for internal notes and quarantined scratch material.

## Developer and Operations Notes

- `docs/DEVELOPER_GUIDE.md`: Repository-oriented engineering notes.
- `src/flexscanx_core/DEVELOPER.md`: C++ core developer notes.
- `src/flexscanx/sequence/NOTICE.txt`: Third-party attribution notice for the `sequence` sub-package.
- `scripts/README.md`: Supported build, validation, and release scripts.

## Completed Audits (Archived)

The 2026 audit work is complete and archived under `archive/reports/2026/`; the
maintained user-facing summary of rflexscan divergences is
`docs/MIGRATION_FROM_RFLEXSCAN.md`.

- `archive/reports/2026/rflexscan_flaw_audit_2026-06-15/`: the completed 2026-06-15
  rflexscan-flaw audit + remediation (integrated report, per-phase `findings/`,
  action register, completed plans, parity matrix). Current high-signal audit record.
- [The 2026-06-11/12 independent end-to-end audit](archive/reports/2026/superseded/AUDIT_REPORT_2026-06-11.md):
  all fixes merged; its open COLD/BOTH question was resolved by the flaw audit above.
- `archive/reports/2026/README.md`: full current-vs-superseded map for the year.

## Planning

- Local planning material contains the live engineering backlog (DRAFT; Phase 3
  done, the remaining phases are open proposals). It is intentionally excluded
  from release snapshots and package artifacts.

## Historical, Generated, and Internal Material

- `archive/reports/`: Timestamped audit/fix reports retained for provenance; year-folder READMEs mark current vs superseded reports.
- `archive/development_audits/`: Older development-cycle audits preserved only as historical context.
- `archive/generated/reference_extracts/README.md`: Entry point for generated academic-reference extracts.
- `archive/generated/raw_dumps/README.md`: Entry point for raw repository dumps and audit snapshots.
- `archive/internal_notes/README.md`: Entry point for curated internal notes and quarantined scratch material.

## Usage Policy

- Treat the files under "Canonical Active Documentation" as the current project documentation surface.
- Treat the files under "Reference-Only Material" as supporting context, not as user-facing contracts.
- Treat everything under `archive/` as non-canonical unless a README in that archive explicitly marks a file as current high-signal context.
