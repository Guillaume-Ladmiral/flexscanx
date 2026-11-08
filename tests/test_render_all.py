from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_all.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("render_all_test", SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Could not load module from {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_vignette(root: Path, relative_path: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\ntitle: test\n---\n", encoding="utf-8")


def test_build_render_commands_orders_workflows_then_notes_and_sets_execution_mode(tmp_path):
    mod = _load_module()
    _write_vignette(tmp_path, "vignettes/workflows/zeta.qmd")
    _write_vignette(tmp_path, "vignettes/workflows/alpha.qmd")
    _write_vignette(tmp_path, "vignettes/notes/zeta.qmd")
    _write_vignette(tmp_path, "vignettes/notes/alpha.qmd")

    commands = mod.build_render_commands(tmp_path, "/opt/quarto", include_heavy=False)

    assert commands == [
        ["/opt/quarto", "render", "vignettes/workflows/alpha.qmd", "--execute"],
        ["/opt/quarto", "render", "vignettes/workflows/zeta.qmd", "--execute"],
        ["/opt/quarto", "render", "vignettes/notes/alpha.qmd", "--no-execute"],
        ["/opt/quarto", "render", "vignettes/notes/zeta.qmd", "--no-execute"],
    ]


def test_build_render_commands_skips_heavy_vignette_unless_requested(tmp_path):
    mod = _load_module()
    _write_vignette(tmp_path, "vignettes/workflows/Manhattan_Breast_Cancer.qmd")
    _write_vignette(tmp_path, "vignettes/workflows/quick.qmd")

    default_commands = mod.build_render_commands(tmp_path, "quarto", include_heavy=False)
    heavy_commands = mod.build_render_commands(tmp_path, "quarto", include_heavy=True)

    assert [command[2] for command in default_commands] == ["vignettes/workflows/quick.qmd"]
    assert [command[2] for command in heavy_commands] == [
        "vignettes/workflows/Manhattan_Breast_Cancer.qmd",
        "vignettes/workflows/quick.qmd",
    ]


def test_build_render_commands_can_direct_all_render_output_outside_the_repository(tmp_path):
    mod = _load_module()
    _write_vignette(tmp_path, "vignettes/workflows/alpha.qmd")
    output_dir = tmp_path.parent / "render-output"

    commands = mod.build_render_commands(
        tmp_path,
        "quarto",
        include_heavy=False,
        output_dir=output_dir,
    )

    assert commands == [
        [
            "quarto",
            "render",
            "vignettes/workflows/alpha.qmd",
            "--output-dir",
            str(output_dir),
            "--execute",
        ]
    ]


def test_main_reports_missing_quarto_with_actionable_message(monkeypatch, capsys):
    mod = _load_module()
    monkeypatch.setattr(mod.shutil, "which", lambda _name: None)

    result = mod.main([])

    captured = capsys.readouterr()
    assert result != 0
    assert "Quarto executable not found" in captured.err
    assert "--quarto" in captured.err


def test_main_dry_run_prints_commands_without_invoking_quarto(tmp_path, monkeypatch, capsys):
    mod = _load_module()
    _write_vignette(tmp_path, "vignettes/workflows/alpha.qmd")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("dry-run must not invoke subprocess.run")

    monkeypatch.setattr(mod.subprocess, "run", fail_if_called)

    result = mod.main(
        [
            "--repo-root",
            str(tmp_path),
            "--quarto",
            "/opt/quarto",
            "--python",
            sys.executable,
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert f"QUARTO_PYTHON={sys.executable}" in captured.out
    assert "/opt/quarto render vignettes/workflows/alpha.qmd --execute" in captured.out


def test_main_stops_on_first_failed_render_and_propagates_exit_status(tmp_path, monkeypatch):
    mod = _load_module()
    _write_vignette(tmp_path, "vignettes/workflows/alpha.qmd")
    _write_vignette(tmp_path, "vignettes/workflows/zeta.qmd")
    calls: list[list[str]] = []

    class Result:
        returncode = 17

    def fail_first(command, **_kwargs):
        calls.append(command)
        return Result()

    monkeypatch.setattr(mod.subprocess, "run", fail_first)

    result = mod.main(
        ["--repo-root", str(tmp_path), "--quarto", "/opt/quarto", "--python", sys.executable]
    )

    assert result == 17
    assert calls == [["/opt/quarto", "render", "vignettes/workflows/alpha.qmd", "--execute"]]
