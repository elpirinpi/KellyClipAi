from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from backend import database
from backend.process_queue import ProjectProcessingQueue


@pytest.mark.skipif(os.name != "nt", reason="DPAPI solo está disponible en Windows")
def test_oauth_tokens_are_encrypted_and_legacy_tokens_migrate(tmp_path: Path) -> None:
    database_path = tmp_path / "data" / "app.sqlite3"
    database.initialize(database_path)

    with database.connection(database_path) as connection:
        connection.execute(
            """
            INSERT INTO connected_accounts (
                owner_id, provider, account_id, display_name, access_token,
                refresh_token, expires_at, metadata_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "test-owner",
                "instagram",
                "account-id",
                "Kelly",
                "legacy-access-token",
                "legacy-refresh-token",
                None,
                json.dumps({"user_access_token": "legacy-user-token"}),
                database.utc_now(),
            ),
        )

    database.initialize(database_path)
    with database.connection(database_path) as connection:
        stored = connection.execute(
            """
            SELECT access_token, refresh_token, metadata_json
            FROM connected_accounts WHERE owner_id = ? AND provider = ?
            """,
            ("test-owner", "instagram"),
        ).fetchone()
    account = database.get_connected_account(database_path, "instagram", "test-owner")

    assert stored["access_token"] != "legacy-access-token"
    assert stored["refresh_token"] != "legacy-refresh-token"
    assert "legacy-user-token" not in stored["metadata_json"]
    assert account is not None
    assert account["access_token"] == "legacy-access-token"
    assert account["refresh_token"] == "legacy-refresh-token"
    assert account["metadata"]["user_access_token"] == "legacy-user-token"

    database.save_connected_account(
        database_path,
        "instagram",
        "new-account",
        "Kelly",
        "new-access-token",
        "new-refresh-token",
        None,
        {"user_access_token": "new-user-token"},
        "new-owner",
    )
    with database.connection(database_path) as connection:
        newly_saved = connection.execute(
            """
            SELECT access_token, refresh_token, metadata_json
            FROM connected_accounts WHERE owner_id = ? AND provider = ?
            """,
            ("new-owner", "instagram"),
        ).fetchone()
    assert "new-access-token" not in newly_saved["access_token"]
    assert "new-refresh-token" not in newly_saved["refresh_token"]
    assert "new-user-token" not in newly_saved["metadata_json"]
    saved_account = database.get_connected_account(
        database_path, "instagram", "new-owner"
    )
    assert saved_account is not None
    assert saved_account["access_token"] == "new-access-token"
    assert saved_account["refresh_token"] == "new-refresh-token"
    assert saved_account["metadata"]["user_access_token"] == "new-user-token"

    database.initialize(database_path)
    assert database.get_connected_account(database_path, "instagram", "test-owner") == account


@pytest.mark.parametrize(("max_concurrency", "expected_parallel"), [(1, 1), (2, 2)])
def test_project_queue_enforces_configured_concurrency(
    max_concurrency: int,
    expected_parallel: int,
) -> None:
    async def verify() -> None:
        active = 0
        max_active = 0
        lock = asyncio.Lock()
        parallel_started = asyncio.Event()
        release_workers = asyncio.Event()

        async def process_project(_project_id: str, _llm_config: object) -> None:
            nonlocal active, max_active
            async with lock:
                active += 1
                max_active = max(max_active, active)
                if active == expected_parallel:
                    parallel_started.set()
            await release_workers.wait()
            async with lock:
                active -= 1

        queue = ProjectProcessingQueue(max_concurrency, process_project)
        await queue.start()
        for project_id in ("project-1", "project-2"):
            await queue.submit(project_id, None)

        await asyncio.wait_for(parallel_started.wait(), timeout=1)
        assert max_active == expected_parallel
        release_workers.set()
        await queue.join()
        await queue.stop()

    asyncio.run(verify())
