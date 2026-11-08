from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sync_docs_metadata.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("sync_docs_metadata_test", SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Could not load module from {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sync_docs_metadata_check_passes_for_current_repo():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "[sync-docs-metadata] done" in proc.stdout


def test_sync_api_reference_updates_generated_contract_blocks(tmp_path):
    mod = _load_script_module()
    meta = mod.DocsMeta(
        version="0.0.1",
        status="In Development",
        year=2026,
        release_date="",
        last_updated="2026-04-04",
    )
    path = tmp_path / "API_REFERENCE.md"
    path.write_text(
        """# flexscanx API Reference

<!-- BEGIN GENERATED: api_reference_release_metadata -->
old release metadata
<!-- END GENERATED: api_reference_release_metadata -->

<!-- BEGIN GENERATED: scan_mc_rantype_policy -->
old rantype
<!-- END GENERATED: scan_mc_rantype_policy -->

<!-- BEGIN GENERATED: enumerating_all_clusters -->
old enum
<!-- END GENERATED: enumerating_all_clusters -->

<!-- BEGIN GENERATED: scan_series_returns -->
old returns
<!-- END GENERATED: scan_series_returns -->

<!-- BEGIN GENERATED: series_time_labels_note -->
old note
<!-- END GENERATED: series_time_labels_note -->

<!-- BEGIN GENERATED: series_result_contract -->
old contract
<!-- END GENERATED: series_result_contract -->

**flexscanx v0.0.0** | MIT License | 1999
""",
        encoding="utf-8",
    )

    updated, changed = mod.sync_api_reference(path, meta)

    assert changed is True
    assert "**Release**: Unreleased (date not set)" in updated
    assert "**Status**: In Development" in updated
    assert '`result["times"]` when provided' in updated
    assert "'times': List[Any]" in updated
    assert "**flexscanx v0.0.1** | MIT License | 2026" in updated
    assert all(line == line.rstrip() for line in updated.splitlines())


def test_sync_vignette_qmd_inserts_date_and_normalizes_footer_metadata(tmp_path):
    mod = _load_script_module()
    meta = mod.DocsMeta(
        version="0.0.1",
        status="In Development",
        year=2026,
        release_date="",
        last_updated="2026-04-04",
    )
    path = tmp_path / "workflow.qmd"
    path.write_text(
        """---
title: Demo Workflow
format:
  html: default
---

Body text.

**Tutorial Version**: 0.0.0
**Date**: 2026-01-01
""",
        encoding="utf-8",
    )

    updated, changed = mod.sync_vignette_qmd(path, meta)

    assert changed is True
    assert "date: 2026-04-04" in updated
    assert "**Tutorial Version**: 0.0.1" in updated
    assert "**Last Updated**: 2026-04-04" in updated
    assert "**Date**:" not in updated
