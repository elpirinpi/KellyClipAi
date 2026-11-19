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

    assert version == 3


def test_v2_migrates_only_the_retired_groq_model(tmp_path: Path) -> None:
    database_path = tmp_path / "data" / "app.sqlite3"
    database.initialize(database_path)
    database.save_llm_credentials(
        database_path,
        "groq-owner",
        "Groq",
        b"encrypted-groq-key",
        "llama-3.3-70b-versatile",
        "https://custom.groq.example/v1",
    )
    database.save_llm_credentials(
        database_path,
        "other-owner",
        "OpenAI",
        b"encrypted-other-key",
        "llama-3.3-70b-versatile",
        "",
    )

    with database.connection(database_path) as connection:
        connection.execute("PRAGMA user_version = 1")

    database.initialize(database_path)

    migrated = database.get_llm_credentials(database_path, "groq-owner")
    unchanged = database.get_llm_credentials(database_path, "other-owner")
    assert migrated is not None
    # v2 lo pasa a gpt-oss-120b y v3 al modelo predeterminado actual.
    assert migrated["model"] == "qwen/qwen3.8-27b"
    assert migrated["encrypted_api_key"] == b"encrypted-groq-key"
    assert migrated["base_url"] == "https://custom.groq.example/v1"
    assert unchanged is not None
    assert unchanged["provider"] == "OpenAI"
    assert unchanged["model"] == "llama-3.3-70b-versatile"
    assert unchanged["encrypted_api_key"] == b"encrypted-other-key"


def test_v3_migrates_groq_model_to_qwen(tmp_path: Path) -> None:
    database_path = tmp_path / "data" / "app.sqlite3"
    database.initialize(database_path)
    database.save_llm_credentials(
        database_path,
        "groq-owner",
        "Groq",
        b"encrypted-groq-key",
        "openai/gpt-oss-120b",
        "",
    )
    database.save_llm_credentials(
        database_path,
        "openai-owner",
        "OpenAI",
        b"encrypted-openai-key",
        "gpt-4o-mini",
        "",
    )

    with database.connection(database_path) as connection:
        connection.execute("PRAGMA user_version = 2")

    database.initialize(database_path)

    migrated = database.get_llm_credentials(database_path, "groq-owner")
    unchanged = database.get_llm_credentials(database_path, "openai-owner")
    assert migrated is not None
    assert migrated["model"] == "qwen/qwen3.8-27b"
    assert migrated["encrypted_api_key"] == b"encrypted-groq-key"
    assert unchanged is not None
    assert unchanged["model"] == "gpt-4o-mini"


def test_initialize_rejects_a_database_from_a_newer_schema(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "data" / "app.sqlite3"
    database.initialize(database_path)
    with database.connection(database_path) as connection:
        connection.execute("PRAGMA user_version = 999")

    with pytest.raises(sqlite3.DatabaseError, match="newer schema version"):
        database.initialize(database_path)
