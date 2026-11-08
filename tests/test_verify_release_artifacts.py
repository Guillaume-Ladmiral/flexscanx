from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_release_artifacts.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("verify_release_artifacts_test", SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Could not load module from {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_read_project_metadata_derives_distribution_and_wheel_names(tmp_path):
    mod = _load_module()
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "Flex-Scan.X"\nversion = "1.2.3"\n',
        encoding="utf-8",
    )

    metadata = mod.read_project_metadata(tmp_path)

    assert metadata.name == "Flex-Scan.X"
    assert metadata.version == "1.2.3"
    assert metadata.sdist_stem == "flex_scan_x-1.2.3"
    assert metadata.wheel_stem == "flex_scan_x-1.2.3"


def test_select_artifacts_requires_exactly_one_expected_sdist_and_wheel(tmp_path):
    mod = _load_module()
    metadata = mod.ProjectMetadata("flexscanx", "0.0.2")
    sdist = tmp_path / "flexscanx-0.0.2.tar.gz"
    wheel = tmp_path / "flexscanx-0.0.2-cp312-cp312-linux_x86_64.whl"
    sdist.touch()
    wheel.touch()
    (tmp_path / "unrelated.txt").touch()

    assert mod.select_artifacts(tmp_path, metadata) == (sdist, wheel)

    (tmp_path / "flexscanx-0.0.2-py3-none-any.whl").touch()
    with pytest.raises(ValueError, match="exactly one expected wheel"):
        mod.select_artifacts(tmp_path, metadata)


@pytest.mark.parametrize(
    "unsafe_name",
    (
        "../outside",
        "prefix/../../outside",
        "/absolute/path",
        r"C:\absolute\path",
    ),
)
def test_validate_archive_member_name_rejects_unsafe_paths(unsafe_name):
    mod = _load_module()

    with pytest.raises(ValueError, match="unsafe archive member"):
        mod.validate_archive_member_name(unsafe_name)


def test_validate_wheel_members_requires_compiled_core_and_excludes_repository_trees():
    mod = _load_module()
    valid = {
        "flexscanx/__init__.py",
        "flexscanx/py.typed",
        "flexscanx/_core.cpython-312-x86_64-linux-gnu.so",
        "flexscanx-0.0.2.dist-info/METADATA",
    }

    mod.validate_wheel_members(valid)

    with pytest.raises(ValueError, match="exactly one compiled"):
        mod.validate_wheel_members(valid - {"flexscanx/_core.cpython-312-x86_64-linux-gnu.so"})
    with pytest.raises(ValueError, match="repository-only"):
        mod.validate_wheel_members(valid | {"audit/private-report.md"})


def test_validate_public_artifact_text_rejects_private_home_paths():
    mod = _load_module()

    mod.validate_public_artifact_text({"METADATA": "Name: flexscanx\n"})
    with pytest.raises(ValueError, match="private home path"):
        # Assembled at run time so this test file itself carries no home path.
        mod.validate_public_artifact_text({"METADATA": "built at " + "/home" + "/alice/project\n"})


def test_validate_sdist_members_requires_rebuild_sources_and_excludes_planning_material():
    mod = _load_module()
    prefix = "flexscanx-0.0.2/"
    valid = {
        f"{prefix}pyproject.toml",
        f"{prefix}CMakeLists.txt",
        f"{prefix}README.md",
        f"{prefix}LICENSE",
        f"{prefix}src/flexscanx/__init__.py",
        f"{prefix}src/flexscanx/py.typed",
        f"{prefix}src/flexscanx_core/bindings.cpp",
        f"{prefix}src/flexscanx_core/bindings_support.hpp",
        f"{prefix}src/flexscanx_core/version.hpp.in",
        f"{prefix}tests/test_smoke.py",
    }

    assert mod.validate_sdist_members(valid) == "flexscanx-0.0.2"

    with pytest.raises(ValueError, match="required rebuild content"):
        mod.validate_sdist_members(valid - {f"{prefix}CMakeLists.txt"})
    with pytest.raises(ValueError, match="repository-only"):
        mod.validate_sdist_members(valid | {f"{prefix}docs/superpowers/private-plan.md"})


def test_build_command_keeps_outputs_and_backend_state_under_temp_root(tmp_path):
    mod = _load_module()
    repo_root = tmp_path / "repo"
    output_dir = tmp_path / "artifacts"
    build_dir = tmp_path / "backend-build"

    command = mod.build_artifact_command(
        "/opt/python", repo_root, output_dir=output_dir, build_dir=build_dir, sdist=True
    )

    assert command == [
        "/opt/python",
        "-m",
        "build",
        "--wheel",
        "--sdist",
        "--outdir",
        str(output_dir),
        "--config-setting",
        f"build-dir={build_dir}",
        str(repo_root),
    ]


def test_isolated_install_commands_use_fresh_venv_and_built_wheel(tmp_path):
    mod = _load_module()
    env_dir = tmp_path / "wheel-env"
    wheel = tmp_path / "artifacts" / "flexscanx-0.0.2-cp312-linux_x86_64.whl"

    commands = mod.isolated_install_commands("/opt/python", env_dir, wheel=wheel, version="0.0.2")

    env_python = str(env_dir / "bin" / "python")
    assert commands[0] == ["/opt/python", "-m", "venv", str(env_dir)]
    assert commands[1] == [env_python, "-m", "pip", "install", str(wheel)]
    assert commands[2] == [env_python, "-m", "pip", "check"]
    assert commands[3][:3] == [env_python, "-c", mod.ISOLATED_SMOKE_CODE]
    assert commands[3][3] == "0.0.2"


def test_format_command_abbreviates_multiline_inline_python():
    mod = _load_module()

    displayed = mod.format_command(["/opt/python", "-c", "line_one\nline_two", "0.0.2"])

    assert "line_one" not in displayed
    assert "line_two" not in displayed
    assert "<inline Python validation>" in displayed
    assert displayed.endswith("0.0.2")


def test_resolve_python_preserves_virtualenv_interpreter_symlink(tmp_path):
    mod = _load_module()
    env_python = tmp_path / "cleanroom-env" / "bin" / "python"
    env_python.parent.mkdir(parents=True)
    env_python.symlink_to(Path(sys.executable))

    assert mod.resolve_python(str(env_python)) == str(env_python.absolute())


def test_managed_temp_root_cleans_up_on_failure_unless_kept(tmp_path, monkeypatch):
    mod = _load_module()
    removable = tmp_path / "remove-me"
    kept = tmp_path / "keep-me"
    roots = iter((removable, kept))

    def fake_mkdtemp(*_args, **_kwargs):
        path = next(roots)
        path.mkdir()
        return str(path)

    monkeypatch.setattr(mod.tempfile, "mkdtemp", fake_mkdtemp)

    with pytest.raises(RuntimeError):
        with mod.managed_temp_root(keep=False) as temp_root:
            (temp_root / "evidence.txt").write_text("test", encoding="utf-8")
            raise RuntimeError("expected failure")
    assert not removable.exists()

    with mod.managed_temp_root(keep=True) as temp_root:
        (temp_root / "evidence.txt").write_text("test", encoding="utf-8")
    assert (kept / "evidence.txt").read_text(encoding="utf-8") == "test"


# ---------------------------------------------------------------------------
# FXA-183 (2026-09-07): the gate scans every decodable sdist member and the policy
# names the repository-governance material that must not ship
# ---------------------------------------------------------------------------

GOVERNANCE_DOCS = (
    "AGENTS.md",
    "ECOSYSTEM.md",
    "DUPLICATES.md",
    "MIGRATION_PLAN.md",
    "CLAUDE.md",
    "docs/DEVELOPER_GUIDE.md",
    "docs/README.md",
    "tests/test_repository_guidance.py",
)


def test_release_gate_policy_names_the_same_repository_only_paths() -> None:
    mod = _load_module()
    for path in GOVERNANCE_DOCS:
        assert path in mod.SDIST_POLICY_EXCLUSIONS, path


def test_release_gate_scans_every_decodable_sdist_member_for_private_paths(tmp_path) -> None:
    import io
    import tarfile

    mod = _load_module()
    archive = tmp_path / "pkg-0.0.0.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, payload in (
            ("pkg-0.0.0/README.md", b"clean readme\n"),
            ("pkg-0.0.0/tests/helper.py", "x = 1\n".encode()),
            (
                "pkg-0.0.0/deep/nested/notes.txt",
                ("see " + "/home" + "/someone/private/notes\n").encode(),
            ),
            ("pkg-0.0.0/blob.bin", bytes([0xFF, 0xFE, 0x00, 0x80])),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    members = mod.read_sdist_text_members(archive)
    assert "pkg-0.0.0/deep/nested/notes.txt" in members
    assert "pkg-0.0.0/blob.bin" not in members  # undecodable members are skipped, not fatal
    with pytest.raises(ValueError, match="deep/nested/notes.txt"):
        mod.validate_public_artifact_text(members)
