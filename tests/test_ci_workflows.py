from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_ci_and_windows_release_workflows_run_security_checks_and_pin_actions() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    release = (ROOT / ".github" / "workflows" / "windows-release.yml").read_text(
        encoding="utf-8"
    )

    for workflow in (ci, release):
        assert re.search(r"uses:\s+[\w.-]+/[\w.-]+@[0-9a-f]{40}", workflow)

    assert re.search(r"(?m)^on:\s*$[\s\S]*?^  push:", ci)
    assert re.search(r"(?m)^  pull_request:", ci)
    for command in ("pytest", "ruff check backend", "tsc --noEmit", "pip-audit", "npm audit"):
        assert command in ci

    assert "pytest" in release
    assert release.index("pytest") < release.index("build_windows.ps1")
    attach_step = release.split("name: Attach installer", 1)[1]
    upload_script = attach_step.split("run: |", 1)[1]
    assert "RELEASE_TAG: ${{ inputs.release_tag }}" in release
    assert "${{ inputs.release_tag }}" not in upload_script
