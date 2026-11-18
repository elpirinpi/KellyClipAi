from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_python_test_dependencies_are_separate_and_lockfiles_are_hashed() -> None:
    runtime = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    development = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")

    assert not re.search(r"(?m)^pytest(?:\[.*\])?==", runtime)
    assert "-r requirements.txt" in development
    assert re.search(r"(?m)^pytest(?:\[.*\])?==", development)

    for lock_name in (
        "requirements.lock",
        "requirements-dev.lock",
        "requirements-desktop.lock",
    ):
        lock_path = ROOT / lock_name
        assert lock_path.is_file()
        lock = lock_path.read_text(encoding="utf-8")
        assert "--hash=sha256:" in lock
        assert re.search(r"(?m)^[a-zA-Z0-9_.-]+==[^\s\\]+", lock)

    assert "--require-hashes" in (ROOT / "build_windows.ps1").read_text(
        encoding="utf-8"
    )
    assert "requirements-dev.lock" in (
        ROOT / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")
    assert "requirements-dev.lock" in (
        ROOT / ".github" / "workflows" / "windows-release.yml"
    ).read_text(encoding="utf-8")
