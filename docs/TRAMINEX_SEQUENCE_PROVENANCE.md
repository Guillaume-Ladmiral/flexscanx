# Traminex sequence-code provenance and license evidence

<!-- BEGIN GENERATED: maintained_last_updated -->
**Last Updated:** 2026-11-08
<!-- END GENERATED: maintained_last_updated -->

This is the maintained engineering evidence record for the packaged
`src/flexscanx/sequence/` code. It records repository facts and the decision
that remains for the maintainer; it is not a substitute for legal advice.

## Evidence reviewed

The review covered both repositories through these revisions:

- traminex `c2e0bf3acec8bfec4dff239f902e97fb663208b7`;
- Flexscanx `f24df71de2b66a0b02b334910e5fa2aa410c6b1b`.

Traminex currently declares `GPL-3.0-or-later` in its root `LICENSE`,
`pyproject.toml`, and README. Commit
`84c9f71e5a4527e838eba89eb15a1f5f31bf493e` added that root license on
2025-09-11. Searches of all refs and the complete license/metadata history found
no MIT text, dual-license statement, permission file, or other demonstrated
license grant for traminex.

**Re-checked 2026-09-07** against traminex
`6e76440b8ed7fc4a1c885c78546425a877bac275` (the revision the parity tests now
pin, and traminex's `master` on that date): `LICENSE` and `pyproject.toml`
still declare `GPL-3.0-or-later`; no commit on any ref has ever added or removed
the string "MIT License" in `LICENSE`, `pyproject.toml` or a README; the only
commits touching a licence-named file are `84c9f71e` and `b0ebaed7`
(2025-09-11). `src/flexscanx/sequence/NOTICE.txt` still carries the "Original
traminex MIT License" text. Nothing below changes; the decision remains the
maintainer's (evidence:
`audit/whole_codebase_audit_and_fix_20260907/evidence/47_d8_licensing_facts.out`).

Flexscanx commit `b647b78b006e696ab3dd87236e69dc3d534657e7`
simultaneously added its MIT root license and the packaged sequence modules:

- `src/flexscanx/sequence/__init__.py`;
- `src/flexscanx/sequence/_utils.py`;
- `src/flexscanx/sequence/analysis.py`;
- `src/flexscanx/sequence/distances.py`;
- `src/flexscanx/sequence/plotting.py`.

Those files identify portions as adapted from traminex; individual docstrings
also use terms such as “Adapted from” and “Ported from.” Commit
`3ee12d1c5d82590521b1f3ddd99dad653737fd42` later added
`src/flexscanx/sequence/NOTICE.txt`, including an alleged “Original traminex MIT
License.” That assertion is unsupported by the traminex repository evidence.
Flexscanx's root `LICENSE` and `pyproject.toml` nevertheless still declare the
distribution MIT.

## What the evidence does and does not establish

The contemporaneous file headers and function docstrings are direct
repository provenance claims that the sequence package was adapted or ported
from traminex. The history contains no separately reviewed clean-room record
that displaces those claims.

The later B1-B5 commits and tests establish behavioral relationships, not a
license grant:

- B1 proves Hamming/DHD numerical parity;
- B2 proves transition-rate/count matrix parity;
- B3 proves cross-sectional entropy parity;
- B4 proves frequency-table parity;
- B5 proves plot-data parity.

Those tests do not show that every implementation line was copied, nor can code
similarity establish copyright ownership or permission. Conversely, independent
or differently structured code that produces the same numbers does not erase
the repository's explicit adaptation statements. Flexscanx-only behavior
(`lag > 1`, longitudinal overall entropy, `extract_sequences`, and styling)
has no traminex parity target, but it is packaged in the same sequence
sub-package whose headers carry the adaptation claim.

No reviewed evidence identifies a copyright-holder-authorized MIT or
dual-license grant for traminex. README prose, generated notices, parity tests,
and commit authorship are not such a grant.

## Internally consistent remediation routes

The maintainer must explicitly authorize one route before package-wide license
metadata, copyright notices, or the provenance model changes.

1. **Make the affected distribution GPL-3.0-or-later-compatible.** Replace the
   unsupported MIT assertions, make the root license and package metadata
   consistent with the adapted GPL code, retain accurate attribution, include
   the required license files in sdist/wheels, and add artifact-level regression
   tests. This is the smallest route supported by the current provenance facts.
   Its release consequence is that Flexscanx distributions containing the
   affected sequence code are conveyed under GPL-compatible terms rather than
   as an MIT-only package; downstream redistribution obligations change and a
   new release is required.

2. **Replace the adapted implementation through a reviewed clean-room route.**
   Preserve the desired MIT package license only after an independently
   specified and implemented replacement has been reviewed for provenance,
   while retaining behavioral parity tests. The affected sequence modules,
   headers, NOTICE, provenance documentation, and built artifacts must all be
   re-reviewed. This is substantially more work, must be independently
   documented, and requires a new release.

3. **Provide authoritative permission or a historical license grant.** A
   copyright holder may supply evidence that authorizes MIT or dual licensing.
   Preserve that actual grant, update every notice and metadata field to match
   its precise scope, and test the built artifacts. A recollection, README
   statement, generated notice, or AI-authored text is not sufficient evidence.

Until the maintainer chooses a route, the unsupported “traminex MIT” assertion
remains an unresolved release blocker. The affected consistency surface is the
root license/metadata, all five sequence-module headers,
`src/flexscanx/sequence/NOTICE.txt`,
README and provenance documentation, repository contract tests, and the
license metadata/files embedded in sdists and wheels.
