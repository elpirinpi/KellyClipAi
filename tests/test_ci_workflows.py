from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_ci_and_windows_release_workflows_run_security_checks_and_pin_actions() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    release = (ROOT / ".github" / "workflows" / "windows-release.yml").read_text(encoding="utf-8")

    for workflow in (ci, release):
        assert re.search(r"uses:\s+[\w.-]+/[\w.-]+@[0-9a-f]{40}", workflow)
        assert "Set up verified FFmpeg for media tests" in workflow
        assert "60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba" in workflow
        assert workflow.index("Set up verified FFmpeg") < workflow.index("pytest")

    assert re.search(r"(?m)^on:\s*$[\s\S]*?^  push:", ci)
    assert re.search(r"(?m)^  pull_request:", ci)
    for command in (
        "pytest",
        "ruff check .",
        "ruff format --check .",
        "tsc --noEmit",
        "pip-audit",
        "npm audit",
    ):
        assert command in ci
    for lock in ("requirements.lock", "requirements-dev.lock", "requirements-desktop.lock"):
        assert f"pip-audit -r {lock}" in ci
    assert "pip-compile" in ci
    assert "--dry-run" in ci

    assert "wait-for-ci:" in release
    assert "needs: [wait-for-ci]" in release
    assert "check-runs" in release
    assert (
        "ref: ${{ github.event_name == 'workflow_dispatch' && inputs.release_tag || github.ref }}"
        in release
    )
    assert "KELLYCLIP_SIGN_PFX_BASE64" in release
    assert "pytest" in release
    assert release.index("pytest") < release.index("build_windows.ps1")
    attach_step = release.split("name: Attach installer", 1)[1]
    upload_script = attach_step.split("run: |", 1)[1]
    assert "RELEASE_TAG: ${{ inputs.release_tag }}" in release
    assert "${{ inputs.release_tag }}" not in upload_script


def test_ruff_config_exists_and_ci_lints_whole_tree() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.ruff]" in pyproject
    for rule in ('"E"', '"F"', '"I"', '"UP"', '"B"'):
        assert rule in pyproject
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "ruff check ." in ci
    assert "ruff format --check ." in ci
