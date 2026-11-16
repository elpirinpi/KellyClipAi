from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_lock = threading.RLock()
_JSON_COLUMNS = {"config", "metadata", "transcript", "clips"}
_PROJECT_COLUMNS = {
    "title",
    "status",
    "step",
    "progress",
    "error",
    "config",
    "metadata",
    "transcript",
    "clips",
    "updated_at",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connection(database_path: Path) -> Iterator[sqlite3.Connection]:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize(database_path: Path) -> None:
    with _lock, connection(database_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS staged_media (
                id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL CHECK (source_type IN ('upload', 'url')),
                status TEXT NOT NULL CHECK (status IN ('receiving', 'downloading', 'ready', 'failed', 'finalizing', 'deleting', 'completed')),
                filename TEXT NOT NULL,
                title TEXT,
                file_size INTEGER NOT NULL DEFAULT 0,
                duration REAL NOT NULL DEFAULT 0,
                width INTEGER NOT NULL DEFAULT 0,
                height INTEGER NOT NULL DEFAULT 0,
                kind TEXT NOT NULL DEFAULT 'video',
                bytes_received INTEGER NOT NULL DEFAULT 0,
                total_bytes INTEGER,
                speed REAL,
                eta REAL,
                error TEXT,
                config_json TEXT,
                project_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS staged_media_source_created
                ON staged_media(source_type, created_at DESC);
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                input_path TEXT NOT NULL,
                status TEXT NOT NULL,
                step TEXT NOT NULL,
                progress INTEGER,
                error TEXT,
                config_json TEXT NOT NULL,
                metadata_json TEXT,
                transcript_json TEXT,
                clips_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS project_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS project_events_project_id_id
                ON project_events(project_id, id);
            CREATE TABLE IF NOT EXISTS custom_templates (
                name TEXT PRIMARY KEY COLLATE NOCASE,
                base_template TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS clip_shares (
                token TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                clip_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(project_id, clip_id)
            );
            CREATE TABLE IF NOT EXISTS oauth_states (
                state TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS connected_accounts (
                provider TEXT PRIMARY KEY,
                account_id TEXT NOT NULL,
                display_name TEXT NOT NULL,
                access_token TEXT NOT NULL,
                refresh_token TEXT,
                expires_at TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS scheduled_jobs (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                clip_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                scheduled_at TEXT NOT NULL,
                visibility TEXT NOT NULL DEFAULT 'private',
                status TEXT NOT NULL CHECK (status IN ('scheduled', 'processing', 'published', 'exported', 'failed')),
                output_path TEXT,
                external_id TEXT,
                external_url TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(project_id, clip_id, platform, scheduled_at)
            );
            CREATE INDEX IF NOT EXISTS scheduled_jobs_due
                ON scheduled_jobs(status, scheduled_at);
            """
        )
        schedule_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(scheduled_jobs)")
        }
        if "visibility" not in schedule_columns:
            conn.execute(
                "ALTER TABLE scheduled_jobs ADD COLUMN visibility TEXT NOT NULL DEFAULT 'private'"
            )
        if "external_url" not in schedule_columns:
            conn.execute("ALTER TABLE scheduled_jobs ADD COLUMN external_url TEXT")
        conn.execute("PRAGMA foreign_keys=ON")


def fail_interrupted_staged_media(database_path: Path) -> None:
    with _lock, connection(database_path) as conn:
        conn.execute(
            """
            UPDATE staged_media
            SET status = 'failed',
                error = 'La aplicación se cerró antes de completar la importación. Inténtalo de nuevo.',
                updated_at = ?
            WHERE status IN ('receiving', 'downloading', 'finalizing', 'deleting')
            """,
            (utc_now(),),
        )


def create_staged_media(
    database_path: Path,
    staged_id: str,
    source_type: str,
    status: str,
    filename: str,
    *,
    file_size: int = 0,
    duration: float = 0,
    width: int = 0,
    height: int = 0,
    kind: str = "video",
    config: dict[str, Any] | None = None,
) -> None:
    now = utc_now()
    with _lock, connection(database_path) as conn:
        conn.execute(
            """
            INSERT INTO staged_media (
                id, source_type, status, filename, file_size, duration, width,
                height, kind, config_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                staged_id,
                source_type,
                status,
                filename,
                file_size,
                duration,
                width,
                height,
                kind,
                json.dumps(config, ensure_ascii=False) if config is not None else None,
                now,
                now,
            ),
        )


def get_staged_media(database_path: Path, staged_id: str) -> dict[str, Any] | None:
    with connection(database_path) as conn:
        row = conn.execute(
            "SELECT * FROM staged_media WHERE id = ?", (staged_id,)
        ).fetchone()
    return staged_media_from_row(row) if row else None


def latest_staged_media(
    database_path: Path, source_type: str
) -> dict[str, Any] | None:
    with connection(database_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM staged_media
            WHERE source_type = ? AND status IN ('receiving', 'downloading', 'ready', 'failed')
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """,
            (source_type,),
        ).fetchone()
    return staged_media_from_row(row) if row else None


def staged_media_from_row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["config"] = (
        json.loads(item.pop("config_json")) if item.get("config_json") is not None else None
    )
    return item


def update_staged_media(
    database_path: Path, staged_id: str, **values: Any
) -> None:
    allowed = {
        "status",
        "filename",
        "title",
        "file_size",
        "bytes_received",
        "total_bytes",
        "speed",
        "eta",
        "error",
        "project_id",
    }
    updates = {key: value for key, value in values.items() if key in allowed}
    if not updates:
        return
    assignments = [f"{key} = ?" for key in updates]
    parameters = list(updates.values())
    assignments.append("updated_at = ?")
    parameters.extend([utc_now(), staged_id])
    with _lock, connection(database_path) as conn:
        conn.execute(
            f"UPDATE staged_media SET {', '.join(assignments)} WHERE id = ?",
            parameters,
        )


def claim_staged_media(database_path: Path, staged_id: str) -> bool:
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            """
            UPDATE staged_media
            SET status = 'finalizing', error = NULL, updated_at = ?
            WHERE id = ? AND status = 'ready'
            """,
            (utc_now(), staged_id),
        )
    return cursor.rowcount == 1


def claim_staged_media_for_deletion(database_path: Path, staged_id: str) -> bool:
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            """
            UPDATE staged_media
            SET status = 'deleting', updated_at = ?
            WHERE id = ? AND status IN ('receiving', 'ready', 'failed')
            """,
            (utc_now(), staged_id),
        )
    return cursor.rowcount == 1


def delete_staged_media(database_path: Path, staged_id: str) -> bool:
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            "DELETE FROM staged_media WHERE id = ? AND status = 'deleting'",
            (staged_id,),
        )
    return cursor.rowcount == 1


def create_project(
    database_path: Path,
    project_id: str,
    title: str,
    input_path: Path,
    config: dict[str, Any],
) -> None:
    now = utc_now()
    with _lock, connection(database_path) as conn:
        conn.execute(
            """
            INSERT INTO projects (
                id, title, input_path, status, step, progress, config_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, 'ready', 'project', 0, ?, ?, ?)
            """,
            (project_id, title, str(input_path), json.dumps(config, ensure_ascii=False), now, now),
        )
        append_event(
            conn,
            project_id,
            "progress",
            {
                "status": "ready",
                "step": "project",
                "progress": 0,
                "message": "Proyecto creado.",
            },
        )


def get_project(database_path: Path, project_id: str) -> dict[str, Any] | None:
    with connection(database_path) as conn:
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    return project_from_row(row) if row else None


def list_projects(database_path: Path) -> list[dict[str, Any]]:
    with connection(database_path) as conn:
        rows = conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
    return [project_from_row(row) for row in rows]


def project_from_row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    for column in _JSON_COLUMNS:
        json_key = f"{column}_json"
        item[column] = json.loads(item.pop(json_key)) if item.get(json_key) is not None else None
    item.pop("input_path", None)
    return item


def update_project(database_path: Path, project_id: str, **values: Any) -> None:
    updates = {key: value for key, value in values.items() if key in _PROJECT_COLUMNS}
    if not updates:
        return
    assignments: list[str] = []
    parameters: list[Any] = []
    for key, value in updates.items():
        column = f"{key}_json" if key in _JSON_COLUMNS else key
        assignments.append(f"{column} = ?")
        parameters.append(json.dumps(value, ensure_ascii=False) if key in _JSON_COLUMNS else value)
    assignments.append("updated_at = ?")
    parameters.extend([utc_now(), project_id])
    with _lock, connection(database_path) as conn:
        conn.execute(
            f"UPDATE projects SET {', '.join(assignments)} WHERE id = ?",
            parameters,
        )


def queue_project(
    database_path: Path,
    project_id: str,
    config: dict[str, Any],
) -> bool:
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            """
            UPDATE projects
            SET config_json = ?, status = 'queued', step = 'upload', progress = 0,
                error = NULL, clips_json = '[]', updated_at = ?
            WHERE id = ? AND status NOT IN ('processing', 'queued')
            """,
            (json.dumps(config, ensure_ascii=False), utc_now(), project_id),
        )
        if cursor.rowcount != 1:
            return False
        conn.execute("DELETE FROM project_events WHERE project_id = ?", (project_id,))
        append_event(
            conn,
            project_id,
            "progress",
            {
                "status": "queued",
                "step": "upload",
                "progress": 0,
                "message": "Video recibido por el backend. Proyecto creado.",
            },
        )
        return True


def append_event(
    conn: sqlite3.Connection,
    project_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO project_events (project_id, event_type, payload_json, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (project_id, event_type, json.dumps(payload, ensure_ascii=False), utc_now()),
    )
    return int(cursor.lastrowid)


def publish_event(
    database_path: Path,
    project_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> int:
    with _lock, connection(database_path) as conn:
        return append_event(conn, project_id, event_type, payload)


def events_after(database_path: Path, project_id: str, after_id: int) -> list[dict[str, Any]]:
    with connection(database_path) as conn:
        rows = conn.execute(
            """
            SELECT id, event_type, payload_json
            FROM project_events
            WHERE project_id = ? AND id > ?
            ORDER BY id
            """,
            (project_id, after_id),
        ).fetchall()
    return [
        {"id": row["id"], "event": row["event_type"], "data": json.loads(row["payload_json"])}
        for row in rows
    ]


def list_custom_templates(database_path: Path) -> list[dict[str, str]]:
    with connection(database_path) as conn:
        rows = conn.execute(
            "SELECT name, base_template FROM custom_templates ORDER BY created_at, name"
        ).fetchall()
    return [{"name": row["name"], "base_template": row["base_template"]} for row in rows]


def save_custom_template(
    database_path: Path,
    name: str,
    base_template: str,
) -> dict[str, str]:
    with _lock, connection(database_path) as conn:
        conn.execute(
            """
            INSERT INTO custom_templates (name, base_template, created_at)
            VALUES (?, ?, ?)
            """,
            (name, base_template, utc_now()),
        )
    return {"name": name, "base_template": base_template}


def resolve_subtitle_template(database_path: Path, name: str) -> str:
    with connection(database_path) as conn:
        row = conn.execute(
            "SELECT base_template FROM custom_templates WHERE name = ?",
            (name,),
        ).fetchone()
    return str(row["base_template"]) if row else name


def get_clip_share(database_path: Path, token: str) -> dict[str, str] | None:
    with connection(database_path) as conn:
        row = conn.execute(
            "SELECT token, project_id, clip_id FROM clip_shares WHERE token = ?",
            (token,),
        ).fetchone()
    return dict(row) if row else None


def create_or_get_clip_share(
    database_path: Path,
    project_id: str,
    clip_id: str,
    token: str,
) -> str:
    with _lock, connection(database_path) as conn:
        row = conn.execute(
            "SELECT token FROM clip_shares WHERE project_id = ? AND clip_id = ?",
            (project_id, clip_id),
        ).fetchone()
        if row:
            return str(row["token"])
        conn.execute(
            """
            INSERT INTO clip_shares (token, project_id, clip_id, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (token, project_id, clip_id, utc_now()),
        )
    return token


def save_oauth_state(
    database_path: Path,
    state: str,
    provider: str,
    expires_at: str,
) -> None:
    with _lock, connection(database_path) as conn:
        conn.execute(
            "INSERT INTO oauth_states (state, provider, expires_at, created_at) VALUES (?, ?, ?, ?)",
            (state, provider, expires_at, utc_now()),
        )


def consume_oauth_state(database_path: Path, state: str, now: str) -> str | None:
    with _lock, connection(database_path) as conn:
        row = conn.execute(
            "SELECT provider, expires_at FROM oauth_states WHERE state = ?",
            (state,),
        ).fetchone()
        conn.execute("DELETE FROM oauth_states WHERE state = ?", (state,))
    if row and row["expires_at"] > now:
        return str(row["provider"])
    return None


def save_connected_account(
    database_path: Path,
    provider: str,
    account_id: str,
    display_name: str,
    access_token: str,
    refresh_token: str | None,
    expires_at: str | None,
    metadata: dict[str, Any],
) -> None:
    with _lock, connection(database_path) as conn:
        conn.execute(
            """
            INSERT INTO connected_accounts (
                provider, account_id, display_name, access_token, refresh_token,
                expires_at, metadata_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider) DO UPDATE SET
                account_id = excluded.account_id,
                display_name = excluded.display_name,
                access_token = excluded.access_token,
                refresh_token = excluded.refresh_token,
                expires_at = excluded.expires_at,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                provider,
                account_id,
                display_name,
                access_token,
                refresh_token,
                expires_at,
                json.dumps(metadata, ensure_ascii=False),
                utc_now(),
            ),
        )


def get_connected_account(
    database_path: Path,
    provider: str,
) -> dict[str, Any] | None:
    with connection(database_path) as conn:
        row = conn.execute(
            "SELECT * FROM connected_accounts WHERE provider = ?",
            (provider,),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["metadata"] = json.loads(result.pop("metadata_json"))
    return result


def disconnect_account(database_path: Path, provider: str) -> None:
    with _lock, connection(database_path) as conn:
        conn.execute("DELETE FROM connected_accounts WHERE provider = ?", (provider,))


def enqueue_schedule_jobs(
    database_path: Path,
    project_id: str,
    clip_ids: list[str],
    platform: str,
    scheduled_at: str,
    visibility: str,
) -> list[dict[str, Any]]:
    now = utc_now()
    with _lock, connection(database_path) as conn:
        for clip_id in clip_ids:
            conn.execute(
                """
                INSERT OR IGNORE INTO scheduled_jobs (
                    id, project_id, clip_id, platform, scheduled_at, status,
                    visibility, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'scheduled', ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    project_id,
                    clip_id,
                    platform,
                    scheduled_at,
                    visibility,
                    now,
                    now,
                ),
            )
        rows = conn.execute(
            """
            SELECT * FROM scheduled_jobs
            WHERE project_id = ? AND platform = ? AND scheduled_at = ?
            ORDER BY created_at, id
            """,
            (project_id, platform, scheduled_at),
        ).fetchall()
    return [schedule_job_from_row(row) for row in rows]


def schedule_job_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def list_schedule_jobs(
    database_path: Path,
    project_id: str | None = None,
) -> list[dict[str, Any]]:
    with connection(database_path) as conn:
        if project_id:
            rows = conn.execute(
                "SELECT * FROM scheduled_jobs WHERE project_id = ? ORDER BY scheduled_at, created_at",
                (project_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM scheduled_jobs ORDER BY scheduled_at, created_at"
            ).fetchall()
    return [schedule_job_from_row(row) for row in rows]


def get_schedule_job(database_path: Path, job_id: str) -> dict[str, Any] | None:
    with connection(database_path) as conn:
        row = conn.execute(
            "SELECT * FROM scheduled_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    return schedule_job_from_row(row) if row else None


def due_schedule_jobs(database_path: Path, now: str) -> list[dict[str, Any]]:
    with connection(database_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM scheduled_jobs
            WHERE status = 'scheduled' AND scheduled_at <= ?
            ORDER BY scheduled_at, created_at
            """,
            (now,),
        ).fetchall()
    return [schedule_job_from_row(row) for row in rows]


def fail_interrupted_schedule_jobs(database_path: Path) -> int:
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            """
            UPDATE scheduled_jobs
            SET status = 'failed',
                error = 'El backend se cerró durante la operación. Verifica la cuenta antes de reintentar.',
                updated_at = ?
            WHERE status = 'processing'
            """,
            (utc_now(),),
        )
    return cursor.rowcount


def claim_schedule_job(database_path: Path, job_id: str) -> bool:
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            """
            UPDATE scheduled_jobs
            SET status = 'processing', error = NULL, updated_at = ?
            WHERE id = ? AND status = 'scheduled'
            """,
            (utc_now(), job_id),
        )
    return cursor.rowcount == 1


def finish_schedule_job(
    database_path: Path,
    job_id: str,
    status: str,
    *,
    output_path: str | None = None,
    external_id: str | None = None,
    external_url: str | None = None,
    error: str | None = None,
) -> None:
    if status not in {"published", "exported", "failed"}:
        raise ValueError(f"Estado terminal no válido para una tarea: {status}")
    with _lock, connection(database_path) as conn:
        conn.execute(
            """
            UPDATE scheduled_jobs
            SET status = ?, output_path = ?, external_id = ?, external_url = ?,
                error = ?, updated_at = ?
            WHERE id = ? AND status = 'processing'
            """,
            (status, output_path, external_id, external_url, error, utc_now(), job_id),
        )
