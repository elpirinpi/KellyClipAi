from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from backend import database


def test_initialize_tracks_schema_with_user_version(tmp_path: Path) -> None:
    database_path = tmp_path / "data" / "app.sqlite3"

    database.initialize(database_path)
    with database.connection(database_path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]

    assert version == 1


def test_initialize_rejects_a_database_from_a_newer_schema(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "data" / "app.sqlite3"
    database.initialize(database_path)
    with database.connection(database_path) as connection:
        connection.execute("PRAGMA user_version = 999")

    with pytest.raises(sqlite3.DatabaseError, match="newer schema version"):
        database.initialize(database_path)
