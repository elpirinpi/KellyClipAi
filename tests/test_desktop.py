from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

import pytest

from desktop import _create_log_handler, _SingleInstance


def test_desktop_log_handler_rotates_old_log(tmp_path: Path) -> None:
    log_path = tmp_path / "desktop.log"
    handler = _create_log_handler(log_path)
    try:
        handler.emit(logging.LogRecord("desktop-test", logging.INFO, __file__, 1, "start", (), None))
        handler.emit(
            logging.LogRecord(
                "desktop-test",
                logging.INFO,
                __file__,
                2,
                "x" * (5 * 1024 * 1024 + 1),
                (),
                None,
            )
        )
        handler.flush()

        assert Path(f"{log_path}.1").read_text(encoding="utf-8").strip().endswith("start")
        assert handler.backupCount == 3
    finally:
        handler.close()


@pytest.mark.skipif(os.name != "nt", reason="La instancia única usa el mutex de Windows.")
def test_windows_named_mutex_allows_only_one_instance() -> None:
    mutex_name = f"Local\\kellyClipAi-test-{uuid.uuid4().hex}"
    first = _SingleInstance(mutex_name)
    second = _SingleInstance(mutex_name)
    third = _SingleInstance(mutex_name)

    try:
        assert first.acquire()
        assert not second.acquire()
    finally:
        second.close()
        first.close()

    try:
        assert third.acquire()
    finally:
        third.close()
