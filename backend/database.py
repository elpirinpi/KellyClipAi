from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .credential_store import (
    decrypt_oauth_token,
    encrypt_oauth_token,
    is_oauth_token_encrypted,
)

logger = logging.getLogger(__name__)


class ProjectBusyError(RuntimeError):
    """El proyecto está encolado o procesándose y no se puede borrar."""


_lock = threading.RLock()
_LATEST_SCHEMA_VERSION = 3
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

# --- Backup automático de la BD (Fase 2, B-S8) -------------------------------
# Las copias quedan en <data_dir>/backups/kellyclipai-<timestamp>-<motivo>.sqlite3
# (data_dir es la carpeta que contiene kellyclipai.sqlite3, p. ej.
# %LOCALAPPDATA%\kellyClipAi\data\backups en Windows). Se conserva solo
# las _BACKUP_RETENTION_COUNT más recientes; las antiguas se borran al
# crear una nueva. Para restaurar: cerrar la app, copiar el archivo de
# respaldo sobre kellyclipai.sqlite3 (borrando antes kellyclipai.sqlite3-wal
# y -shm si existen) y volver a abrir. La copia se hace con la API
# sqlite3.Connection.backup, segura con la BD abierta y en modo WAL.
_BACKUP_DIR_NAME = "backups"
_BACKUP_FILE_PREFIX = "kellyclipai-"
_BACKUP_RETENTION_COUNT = 5
_BACKUP_PERIODIC_MAX_AGE_SECONDS = 24 * 3600  # 24 h

# Caché de transcripción: entradas más recientes que se conservan. Cada
# entrada guarda la transcripción completa con tiempos por palabra, así
# que el límite es deliberadamente bajo.
_TRANSCRIPT_CACHE_MAX_ENTRIES = 10


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@contextmanager
def connection(database_path: Path) -> Iterator[sqlite3.Connection]:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
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
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version > _LATEST_SCHEMA_VERSION:
            raise sqlite3.DatabaseError(
                f"Database has newer schema version {version}; "
                f"this application supports up to {_LATEST_SCHEMA_VERSION}."
            )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS staged_media (
                id TEXT PRIMARY KEY,
                owner_id TEXT,
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
                owner_id TEXT,
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
                owner_id TEXT NOT NULL,
                name TEXT NOT NULL COLLATE NOCASE,
                base_template TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (owner_id, name)
            );
            CREATE TABLE IF NOT EXISTS clip_shares (
                token TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                clip_id TEXT NOT NULL,
                expires_at TEXT,
                revoked_at TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(project_id, clip_id)
            );
            CREATE TABLE IF NOT EXISTS oauth_states (
                state TEXT PRIMARY KEY,
                owner_id TEXT,
                provider TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS connected_accounts (
                owner_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                account_id TEXT NOT NULL,
                display_name TEXT NOT NULL,
                access_token TEXT NOT NULL,
                refresh_token TEXT,
                expires_at TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL,
                PRIMARY KEY (owner_id, provider)
            );
            CREATE TABLE IF NOT EXISTS llm_credentials (
                owner_id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                encrypted_api_key BLOB NOT NULL,
                model TEXT NOT NULL,
                base_url TEXT NOT NULL DEFAULT '',
                -- 1 cuando el usuario eligió el modelo explícitamente en
                -- Ajustes: las migraciones no deben reescribirlo (B21).
                model_customized INTEGER NOT NULL DEFAULT 0,
                -- Modelo Whisper elegido en Ajustes; NULL = nunca guardado
                -- (se usa el de entorno). (B2)
                whisper_model TEXT,
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
            CREATE TABLE IF NOT EXISTS schedule_job_confirmations (
                job_id TEXT PRIMARY KEY REFERENCES scheduled_jobs(id) ON DELETE CASCADE,
                detected_at TEXT NOT NULL
            );
            -- Caché de transcripción por hash de archivo (Fase 2, B-R5):
            -- el mismo video en dos proyectos no se transcribe dos veces.
            -- La clave incluye modelo e idioma de Whisper: si cambian, la
            -- entrada no se reutiliza (regla de la Fase 1, A6).
            CREATE TABLE IF NOT EXISTS transcript_cache (
                file_hash TEXT NOT NULL,
                whisper_model TEXT NOT NULL,
                whisper_language TEXT NOT NULL,
                language TEXT NOT NULL DEFAULT '',
                transcript_json TEXT NOT NULL,
                duration REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                PRIMARY KEY (file_hash, whisper_model, whisper_language)
            );
            -- Cola de procesamiento persistente (Fase 2, B-S10): los
            -- trabajos sobreviven a un reinicio y se recuperan al
            -- arrancar. El índice parcial impide dos trabajos activos
            -- (pending/running) del mismo proyecto: deduplicación.
            -- La configuración LLM NO se persiste aquí (contiene la clave
            -- API en claro); al recuperar, el pipeline la resuelve desde
            -- las credenciales cifradas del dueño del proyecto.
            CREATE TABLE IF NOT EXISTS processing_jobs (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                retry_failed_only INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'done', 'failed', 'cancelled')),
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS processing_jobs_active_project
                ON processing_jobs(project_id) WHERE status IN ('pending', 'running');
            """
        )
        # Columnas añadidas en v1.0.11 sin subir la versión de esquema: las
        # bases creadas por versiones anteriores (ya en versión 3) no
        # volverían a pasar por ninguna migración, así que se añaden aquí de
        # forma idempotente, antes de ejecutar las migraciones versionadas
        # (v2/v3 consultan model_customized).
        llm_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(llm_credentials)")
        }
        if "model_customized" not in llm_columns:
            conn.execute(
                "ALTER TABLE llm_credentials "
                "ADD COLUMN model_customized INTEGER NOT NULL DEFAULT 0"
            )
        if "whisper_model" not in llm_columns:
            conn.execute(
                "ALTER TABLE llm_credentials ADD COLUMN whisper_model TEXT"
            )
        if version < _LATEST_SCHEMA_VERSION:
            # Copia de seguridad ANTES de migrar (B-S8): si una migración
            # corrompe algo, la BD previa está en <data_dir>/backups/.
            # Best-effort: un fallo del respaldo nunca debe impedir el
            # arranque. Se copia el estado confirmado en disco (la
            # transacción DDL de arriba aún no se ha confirmado), que es
            # exactamente el estado previo a la migración.
            try:
                backup_database(database_path, reason=f"pre-migration-v{version}")
            except Exception as error:
                logger.warning(
                    "No se pudo respaldar la base de datos antes de migrar: %s",
                    error,
                )
        _run_migrations(conn)
    # Copia periódica al arrancar si la última tiene más de 24 h (B-S8).
    # Fuera del bloque anterior para copiar el estado ya migrado.
    try:
        ensure_periodic_backup(database_path)
    except Exception as error:
        logger.warning("No se pudo hacer la copia periódica de la base de datos: %s", error)


def _run_migrations(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    migrations = {1: _migrate_schema_v1, 2: _migrate_schema_v2, 3: _migrate_schema_v3}
    for next_version in range(version + 1, _LATEST_SCHEMA_VERSION + 1):
        conn.execute("BEGIN IMMEDIATE")
        migrations[next_version](conn)
        conn.execute(f"PRAGMA user_version = {next_version}")
        conn.commit()


def _migrate_schema_v1(conn: sqlite3.Connection) -> None:
    schedule_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(scheduled_jobs)")
    }
    if "visibility" not in schedule_columns:
        conn.execute(
            "ALTER TABLE scheduled_jobs ADD COLUMN visibility TEXT NOT NULL DEFAULT 'private'"
        )
    if "external_url" not in schedule_columns:
        conn.execute("ALTER TABLE scheduled_jobs ADD COLUMN external_url TEXT")
    share_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(clip_shares)")
    }
    if "expires_at" not in share_columns:
        conn.execute("ALTER TABLE clip_shares ADD COLUMN expires_at TEXT")
    if "revoked_at" not in share_columns:
        conn.execute("ALTER TABLE clip_shares ADD COLUMN revoked_at TEXT")
    conn.execute(
        """
        UPDATE clip_shares
        SET expires_at = datetime(created_at, '+30 days')
        WHERE expires_at IS NULL
        """
    )
    for table in ("staged_media", "projects", "oauth_states"):
        columns = {
            row["name"] for row in conn.execute(f"PRAGMA table_info({table})")
        }
        if "owner_id" not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN owner_id TEXT")
    _migrate_owned_table(
        conn,
        "custom_templates",
        """
        CREATE TABLE custom_templates (
            owner_id TEXT NOT NULL,
            name TEXT NOT NULL COLLATE NOCASE,
            base_template TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (owner_id, name)
        )
        """,
        "INSERT INTO custom_templates (owner_id, name, base_template, created_at) "
        "SELECT 'legacy', name, base_template, created_at FROM custom_templates_legacy",
    )
    _migrate_owned_table(
        conn,
        "connected_accounts",
        """
        CREATE TABLE connected_accounts (
            owner_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            account_id TEXT NOT NULL,
            display_name TEXT NOT NULL,
            access_token TEXT NOT NULL,
            refresh_token TEXT,
            expires_at TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL,
            PRIMARY KEY (owner_id, provider)
        )
        """,
        "INSERT INTO connected_accounts (owner_id, provider, account_id, display_name, "
        "access_token, refresh_token, expires_at, metadata_json, updated_at) "
        "SELECT 'legacy', provider, account_id, display_name, access_token, refresh_token, "
        "expires_at, metadata_json, updated_at FROM connected_accounts_legacy",
    )
    _migrate_oauth_tokens(conn)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS projects_owner_created "
        "ON projects(owner_id, created_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS staged_media_owner_source_created "
        "ON staged_media(owner_id, source_type, created_at DESC)"
    )


def _migrate_schema_v2(conn: sqlite3.Connection) -> None:
    # Solo reescribe el modelo en filas que el usuario nunca personalizó
    # (model_customized = 0): quien eligió un modelo a propósito lo conserva.
    conn.execute(
        """
        UPDATE llm_credentials
        SET model = ?, updated_at = ?
        WHERE provider = ? AND model = ? AND model_customized = 0
        """,
        (
            "openai/gpt-oss-120b",
            utc_now(),
            "Groq",
            "llama-3.3-70b-versatile",
        ),
    )


def _migrate_schema_v3(conn: sqlite3.Connection) -> None:
    # Modelo predeterminado de Groq: qwen/qwen3.8-27b (plan gratuito:
    # 8K TPM / 200K TPD, los mismos límites que gpt-oss-120b; los planes
    # de pago tienen límites mayores). Actualiza tanto a
    # los usuarios que venían de gpt-oss-120b como a los que aún tenían
    # llama-3.3-70b-versatile (si saltaron la migración v2). Como en v2,
    # respeta las filas personalizadas por el usuario.
    conn.execute(
        """
        UPDATE llm_credentials
        SET model = ?, updated_at = ?
        WHERE provider = ? AND model IN (?, ?) AND model_customized = 0
        """,
        (
            "qwen/qwen3.8-27b",
            utc_now(),
            "Groq",
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
        ),
    )


def _migrate_owned_table(
    conn: sqlite3.Connection,
    table_name: str,
    create_sql: str,
    copy_sql: str,
) -> None:
    columns = {
        row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})")
    }
    if "owner_id" in columns:
        return
    legacy_name = f"{table_name}_legacy"
    conn.execute(f"ALTER TABLE {table_name} RENAME TO {legacy_name}")
    conn.execute(create_sql)
    conn.execute(copy_sql)
    conn.execute(f"DROP TABLE {legacy_name}")


def _migrate_oauth_tokens(conn: sqlite3.Connection) -> None:
    # Migración tolerante a fallos (A10): cada cuenta se migra de forma
    # aislada. Si DPAPI falla (otro usuario de Windows, perfil dañado, BD
    # abierta en otro SO) o el metadata_json está corrupto, esa cuenta se
    # marca como "necesita reconectar" (tokens limpiados + nota en
    # metadata) en vez de impedir el arranque de toda la aplicación.
    rows = conn.execute(
        """
        SELECT owner_id, provider, access_token, refresh_token, metadata_json
        FROM connected_accounts
        """
    ).fetchall()
    for row in rows:
        try:
            access_token = row["access_token"]
            refresh_token = row["refresh_token"]
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except ValueError:
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            user_access_token = metadata.get("user_access_token")
            if not is_oauth_token_encrypted(access_token):
                access_token = encrypt_oauth_token(access_token)
            if refresh_token and not is_oauth_token_encrypted(refresh_token):
                refresh_token = encrypt_oauth_token(refresh_token)
            if user_access_token and not is_oauth_token_encrypted(user_access_token):
                metadata["user_access_token"] = encrypt_oauth_token(user_access_token)
            conn.execute(
                """
                UPDATE connected_accounts
                SET access_token = ?, refresh_token = ?, metadata_json = ?
                WHERE owner_id = ? AND provider = ?
                """,
                (
                    access_token,
                    refresh_token,
                    json.dumps(metadata, ensure_ascii=False),
                    row["owner_id"],
                    row["provider"],
                ),
            )
        except Exception as error:
            logger.warning(
                "No se pudo migrar el token de %s (%s); la cuenta queda "
                "pendiente de reconectar: %s",
                row["provider"],
                row["owner_id"],
                error,
            )
            note = {
                "needs_reconnect": True,
                "migration_error": (
                    str(error).strip() or "No se pudo migrar el token guardado."
                )[:240],
            }
            try:
                conn.execute(
                    """
                    UPDATE connected_accounts
                    SET access_token = '', refresh_token = NULL, metadata_json = ?
                    WHERE owner_id = ? AND provider = ?
                    """,
                    (
                        json.dumps(note, ensure_ascii=False),
                        row["owner_id"],
                        row["provider"],
                    ),
                )
            except Exception:
                logger.exception(
                    "No se pudo marcar la cuenta %s (%s) como pendiente de reconectar",
                    row["provider"],
                    row["owner_id"],
                )


def _backup_files(backup_dir: Path) -> list[Path]:
    """Copias existentes, de la más reciente a la más antigua."""
    if not backup_dir.is_dir():
        return []
    backups = [
        path
        for path in backup_dir.glob(f"{_BACKUP_FILE_PREFIX}*.sqlite3")
        if path.is_file()
    ]
    backups.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return backups


def _prune_backups(backup_dir: Path) -> None:
    """Borra las copias más antiguas, conservando las N más recientes."""
    for stale in _backup_files(backup_dir)[_BACKUP_RETENTION_COUNT:]:
        try:
            stale.unlink()
        except OSError as error:
            logger.warning("No se pudo borrar la copia antigua %s: %s", stale, error)


def backup_database(database_path: Path, *, reason: str = "manual") -> Path | None:
    """Copia la base de datos con la API sqlite3 Connection.backup.

    Destino: ``<data_dir>/backups/kellyclipai-<AAAAMMDD-HHMMSS>-<motivo>.sqlite3``
    (ver las constantes _BACKUP_* al inicio del módulo). La copia es
    consistente aunque la app esté usando la BD (la API backup lee a
    través de WAL) y no interfiere con las escrituras: se toma el lock
    global solo para serializarla con otras operaciones de este módulo.
    Devuelve la ruta de la copia, o None si la BD no existe o está vacía
    (una BD nueva sin confirmar no merece respaldo).
    """
    try:
        if not database_path.is_file() or database_path.stat().st_size == 0:
            return None
    except OSError:
        return None
    backup_dir = database_path.parent / _BACKUP_DIR_NAME
    backup_dir.mkdir(parents=True, exist_ok=True)
    safe_reason = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in reason
    )[:40] or "manual"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = f"{_BACKUP_FILE_PREFIX}{timestamp}-{safe_reason}"
    target = backup_dir / f"{stem}.sqlite3"
    suffix = 2
    while target.exists():
        target = backup_dir / f"{stem}-{suffix}.sqlite3"
        suffix += 1
    with _lock:
        source = sqlite3.connect(database_path, timeout=30)
        try:
            destination = sqlite3.connect(target)
            try:
                source.backup(destination)
            finally:
                destination.close()
        finally:
            source.close()
    _prune_backups(backup_dir)
    logger.info("Copia de seguridad de la base de datos en %s", target)
    return target


def ensure_periodic_backup(database_path: Path) -> Path | None:
    """Copia periódica: solo si no hay copia o la última tiene más de 24 h.

    Se llama al arrancar (desde initialize). Devuelve la ruta de la copia
    creada, o None si no hacía falta ninguna.
    """
    backup_dir = database_path.parent / _BACKUP_DIR_NAME
    backups = _backup_files(backup_dir)
    if backups:
        newest_age = time.time() - backups[0].stat().st_mtime
        if newest_age < _BACKUP_PERIODIC_MAX_AGE_SECONDS:
            return None
    return backup_database(database_path, reason="periodic")


def claim_legacy_data(database_path: Path, owner_id: str) -> None:
    with _lock, connection(database_path) as conn:
        for table in (
            "projects",
            "staged_media",
            "custom_templates",
            "connected_accounts",
            "oauth_states",
        ):
            conn.execute(
                f"UPDATE {table} SET owner_id = ? WHERE owner_id IS NULL OR owner_id = 'legacy'",
                (owner_id,),
            )
        # llm_credentials también se reclama (B20): la clave guardada antes
        # del primer inicio de sesión no debe quedar huérfana. Como
        # owner_id es la clave primaria, solo se puede reclamar una fila
        # legacy y solo si el usuario aún no tiene credenciales propias;
        # las filas legacy restantes se descartan para que no las reclame
        # otro usuario después.
        legacy_rows = conn.execute(
            "SELECT rowid FROM llm_credentials "
            "WHERE owner_id IS NULL OR owner_id = 'legacy' ORDER BY rowid"
        ).fetchall()
        if not legacy_rows:
            return
        has_credentials = (
            conn.execute(
                "SELECT 1 FROM llm_credentials WHERE owner_id = ?",
                (owner_id,),
            ).fetchone()
            is not None
        )
        if has_credentials:
            conn.execute(
                "DELETE FROM llm_credentials "
                "WHERE owner_id IS NULL OR owner_id = 'legacy'"
            )
            return
        conn.execute(
            "UPDATE llm_credentials SET owner_id = ? WHERE rowid = ?",
            (owner_id, legacy_rows[0]["rowid"]),
        )
        if len(legacy_rows) > 1:
            conn.executemany(
                "DELETE FROM llm_credentials WHERE rowid = ?",
                [(row["rowid"],) for row in legacy_rows[1:]],
            )


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


def _has_processing_jobs_table(conn: sqlite3.Connection) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name = 'processing_jobs'"
        ).fetchone()
        is not None
    )


def fail_interrupted_projects(database_path: Path) -> int:
    error = (
        "El procesamiento se interrumpió al cerrar o reiniciar la aplicación. "
        "Puedes volver a intentarlo."
    )
    with _lock, connection(database_path) as conn:
        # Los proyectos con un trabajo persistido (pending/running) NO se
        # marcan como fallidos: la cola los recupera al arrancar (B-S10).
        # Solo se fallan los que no tienen trabajo que recuperar (p. ej.
        # encolados por una versión anterior a la cola persistente).
        recoverable_filter = (
            " AND NOT EXISTS ("
            " SELECT 1 FROM processing_jobs"
            " WHERE processing_jobs.project_id = projects.id"
            " AND processing_jobs.status IN ('pending', 'running')"
            ")"
            if _has_processing_jobs_table(conn)
            else ""
        )
        rows = conn.execute(
            "SELECT id, progress FROM projects WHERE status IN ('queued', 'processing')"
            + recoverable_filter
        ).fetchall()
        if not rows:
            return 0
        now = utc_now()
        conn.execute(
            """
            UPDATE projects
            SET status = 'failed', step = 'failed', error = ?, updated_at = ?
            WHERE status IN ('queued', 'processing')
            """
            + recoverable_filter,
            (error, now),
        )
        for row in rows:
            append_event(
                conn,
                row["id"],
                "failed",
                {
                    "status": "failed",
                    "step": "failed",
                    "progress": row["progress"],
                    "message": error,
                },
            )
    return len(rows)


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
    owner_id: str | None = None,
) -> None:
    now = utc_now()
    with _lock, connection(database_path) as conn:
        conn.execute(
            """
            INSERT INTO staged_media (
                id, owner_id, source_type, status, filename, file_size, duration, width,
                height, kind, config_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                staged_id,
                owner_id,
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


def get_staged_media(
    database_path: Path,
    staged_id: str,
    owner_id: str | None = None,
) -> dict[str, Any] | None:
    query = "SELECT * FROM staged_media WHERE id = ?"
    parameters: tuple[Any, ...] = (staged_id,)
    if owner_id is not None:
        query += " AND owner_id = ?"
        parameters += (owner_id,)
    with connection(database_path) as conn:
        row = conn.execute(query, parameters).fetchone()
    return staged_media_from_row(row) if row else None


def latest_staged_media(
    database_path: Path,
    source_type: str,
    owner_id: str | None = None,
) -> dict[str, Any] | None:
    query = """
        SELECT * FROM staged_media
        WHERE source_type = ? AND status IN ('receiving', 'downloading', 'ready', 'failed')
    """
    parameters: tuple[Any, ...] = (source_type,)
    if owner_id is not None:
        query += " AND owner_id = ?"
        parameters += (owner_id,)
    query += " ORDER BY created_at DESC, rowid DESC LIMIT 1"
    with connection(database_path) as conn:
        row = conn.execute(query, parameters).fetchone()
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
    # 'downloading' también se puede reclamar: cancelar una descarga por
    # enlace en curso es el caso principal del botón "Cancelar descarga".
    # La tarea de descarga detecta el estado 'deleting' (o la fila
    # borrada) y limpia los archivos parciales en vez de completarla.
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            """
            UPDATE staged_media
            SET status = 'deleting', updated_at = ?
            WHERE id = ? AND status IN ('receiving', 'downloading', 'ready', 'failed')
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
    owner_id: str | None = None,
) -> None:
    now = utc_now()
    with _lock, connection(database_path) as conn:
        conn.execute(
            """
            INSERT INTO projects (
                id, owner_id, title, input_path, status, step, progress, config_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'ready', 'project', 0, ?, ?, ?)
            """,
            (
                project_id,
                owner_id,
                title,
                str(input_path),
                json.dumps(config, ensure_ascii=False),
                now,
                now,
            ),
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


def get_project(
    database_path: Path,
    project_id: str,
    owner_id: str | None = None,
) -> dict[str, Any] | None:
    query = "SELECT * FROM projects WHERE id = ?"
    parameters: tuple[Any, ...] = (project_id,)
    if owner_id is not None:
        query += " AND owner_id = ?"
        parameters += (owner_id,)
    with connection(database_path) as conn:
        row = conn.execute(query, parameters).fetchone()
    return project_from_row(row) if row else None


def project_exists(
    database_path: Path,
    project_id: str,
    owner_id: str | None = None,
) -> bool:
    query = "SELECT 1 FROM projects WHERE id = ?"
    parameters: tuple[Any, ...] = (project_id,)
    if owner_id is not None:
        query += " AND owner_id = ?"
        parameters += (owner_id,)
    with connection(database_path) as conn:
        return conn.execute(query, parameters).fetchone() is not None


def get_project_status(database_path: Path, project_id: str) -> str | None:
    with connection(database_path) as conn:
        row = conn.execute(
            "SELECT status FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
    return str(row["status"]) if row else None


def get_project_outcome(
    database_path: Path, project_id: str
) -> tuple[str | None, str | None]:
    """Estado y error de un proyecto, sin parsear los JSON pesados.

    La usa la cola de procesamiento para decidir si un trabajo fallido
    merece reintentarse (B-S10): necesita el mensaje de error sin pagar
    el coste de leer clips_json/transcript_json.
    """
    with connection(database_path) as conn:
        row = conn.execute(
            "SELECT status, error FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
    if row is None:
        return None, None
    return str(row["status"]), row["error"]


def count_projects(
    database_path: Path,
    owner_id: str | None = None,
) -> int:
    """Número total de proyectos (para la paginación de GET /api/projects)."""
    query = "SELECT COUNT(*) FROM projects"
    parameters: tuple[Any, ...] = ()
    if owner_id is not None:
        query += " WHERE owner_id = ?"
        parameters = (owner_id,)
    with connection(database_path) as conn:
        row = conn.execute(query, parameters).fetchone()
    return int(row[0])


def list_projects(
    database_path: Path,
    owner_id: str | None = None,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Lista proyectos, sin la transcripción del proyecto (B-R10).

    `limit=None` (defecto) devuelve todos, como antes; los llamadores
    del listado HTTP pasan limit/offset para paginar. La transcripción
    por clip se excluye en la serialización HTTP, no aquí: el detalle
    de proyecto la necesita.
    """
    query = (
        "SELECT id, owner_id, title, input_path, status, step, progress, error, "
        "config_json, metadata_json, clips_json, created_at, updated_at FROM projects"
    )
    parameters: list[Any] = []
    if owner_id is not None:
        query += " WHERE owner_id = ?"
        parameters.append(owner_id)
    query += " ORDER BY created_at DESC"
    if limit is not None:
        if limit < 1:
            raise ValueError("El límite de paginación debe ser mayor que cero.")
        if offset < 0:
            raise ValueError("El desplazamiento de paginación no puede ser negativo.")
        query += " LIMIT ? OFFSET ?"
        parameters.extend([limit, offset])
    with connection(database_path) as conn:
        rows = conn.execute(query, parameters).fetchall()
    return [project_from_row(row, include_transcript=False) for row in rows]


def project_from_row(
    row: sqlite3.Row,
    *,
    include_transcript: bool = True,
) -> dict[str, Any]:
    item = dict(row)
    for column in _JSON_COLUMNS:
        json_key = f"{column}_json"
        if column == "transcript" and not include_transcript:
            item.pop(json_key, None)
            continue
        item[column] = json.loads(item.pop(json_key)) if item.get(json_key) is not None else None
    item.pop("input_path", None)
    return item


def _update_project_conn(
    conn: sqlite3.Connection, project_id: str, **values: Any
) -> None:
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
    conn.execute(
        f"UPDATE projects SET {', '.join(assignments)} WHERE id = ?",
        parameters,
    )


def update_project(database_path: Path, project_id: str, **values: Any) -> None:
    with _lock, connection(database_path) as conn:
        _update_project_conn(conn, project_id, **values)


def update_project_with_event(
    database_path: Path,
    project_id: str,
    event_type: str,
    payload: dict[str, Any],
    **values: Any,
) -> int:
    """Actualiza el proyecto y publica su evento en UNA transacción.

    BUG 2 (v1.3.2): el SSE de progreso cierra el flujo al ver un estado
    terminal; si el estado se confirmaba en una transacción y el evento
    en otra posterior, el flujo podía cerrarse entre ambas sin entregar
    el evento final y el frontend se quedaba en 98 %. Con una sola
    transacción, quien ve el estado terminal ve también su evento.
    """
    with _lock, connection(database_path) as conn:
        _update_project_conn(conn, project_id, **values)
        return append_event(conn, project_id, event_type, payload)


def delete_project(
    database_path: Path,
    project_id: str,
    owner_id: str | None = None,
) -> list[str] | None:
    with _lock, connection(database_path) as conn:
        query = "SELECT id, status FROM projects WHERE id = ?"
        parameters: tuple[Any, ...] = (project_id,)
        if owner_id is not None:
            query += " AND owner_id = ?"
            parameters += (owner_id,)
        row = conn.execute(query, parameters).fetchone()
        if row is None:
            return None
        # Guarda atómica (B5): el estado se comprueba dentro de la misma
        # transacción que el borrado, así que un proceso que encole el
        # proyecto entre la comprobación de la ruta y el borrado no puede
        # dejar un pipeline escribiendo sobre un proyecto ya borrado. Para
        # borrar un proyecto en curso hay que cancelarlo primero.
        if row["status"] in {"processing", "queued"}:
            raise ProjectBusyError(
                "No se puede eliminar un proyecto mientras se procesa."
            )
        staged_ids = [
            str(row["id"])
            for row in conn.execute(
                "SELECT id FROM staged_media WHERE project_id = ?",
                (project_id,),
            ).fetchall()
        ]
        conn.execute("DELETE FROM scheduled_jobs WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM clip_shares WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM staged_media WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    return staged_ids


def cleanup_expired_staged_media(
    database_path: Path,
    cutoff: str,
) -> list[str]:
    with _lock, connection(database_path) as conn:
        rows = conn.execute(
            "SELECT id FROM staged_media WHERE updated_at < ?",
            (cutoff,),
        ).fetchall()
        staged_ids = [str(row["id"]) for row in rows]
        if staged_ids:
            conn.executemany(
                "DELETE FROM staged_media WHERE id = ?",
                [(staged_id,) for staged_id in staged_ids],
            )
    return staged_ids


def queue_project(
    database_path: Path,
    project_id: str,
    config: dict[str, Any],
    *,
    preserve_clips: bool = False,
) -> bool:
    # A1: encolar NUNCA borra clips_json. Los clips anteriores se
    # conservan hasta que el reproceso renderiza al menos un clip nuevo
    # (el pipeline hace el swap); si el reproceso falla, los viejos
    # siguen intactos. El parámetro preserve_clips queda como no-op por
    # compatibilidad: preservar es ahora el único comportamiento.
    del preserve_clips
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            """
            UPDATE projects
            SET config_json = ?, status = 'queued', step = 'upload', progress = 0,
                error = NULL, updated_at = ?
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


def list_custom_templates(
    database_path: Path,
    owner_id: str = "legacy",
) -> list[dict[str, str]]:
    with connection(database_path) as conn:
        rows = conn.execute(
            "SELECT name, base_template FROM custom_templates "
            "WHERE owner_id = ? ORDER BY created_at, name",
            (owner_id,),
        ).fetchall()
    return [{"name": row["name"], "base_template": row["base_template"]} for row in rows]


def save_custom_template(
    database_path: Path,
    name: str,
    base_template: str,
    owner_id: str = "legacy",
) -> dict[str, str]:
    with _lock, connection(database_path) as conn:
        conn.execute(
            """
            INSERT INTO custom_templates (owner_id, name, base_template, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (owner_id, name, base_template, utc_now()),
        )
    return {"name": name, "base_template": base_template}


def resolve_subtitle_template(
    database_path: Path,
    name: str,
    owner_id: str = "legacy",
) -> str:
    with connection(database_path) as conn:
        row = conn.execute(
            "SELECT base_template FROM custom_templates WHERE owner_id = ? AND name = ?",
            (owner_id, name),
        ).fetchone()
    return str(row["base_template"]) if row else name


def get_clip_share(database_path: Path, token: str) -> dict[str, str] | None:
    with connection(database_path) as conn:
        row = conn.execute(
            """
            SELECT token, project_id, clip_id, expires_at
            FROM clip_shares
            WHERE token = ? AND revoked_at IS NULL
              AND expires_at IS NOT NULL AND julianday(expires_at) > julianday(?)
            """,
            (token, utc_now()),
        ).fetchone()
    return dict(row) if row else None


def create_or_get_clip_share(
    database_path: Path,
    project_id: str,
    clip_id: str,
    token: str,
) -> str:
    now = datetime.now(UTC)
    expires_at = (now + timedelta(days=30)).isoformat()
    with _lock, connection(database_path) as conn:
        row = conn.execute(
            """
            SELECT token, expires_at, revoked_at
            FROM clip_shares WHERE project_id = ? AND clip_id = ?
            """,
            (project_id, clip_id),
        ).fetchone()
        if (
            row
            and row["revoked_at"] is None
            and row["expires_at"] is not None
            and conn.execute(
                "SELECT julianday(?) > julianday(?)",
                (row["expires_at"], utc_now()),
            ).fetchone()[0]
        ):
            return str(row["token"])
        if row:
            conn.execute(
                """
                UPDATE clip_shares
                SET token = ?, expires_at = ?, revoked_at = NULL, created_at = ?
                WHERE project_id = ? AND clip_id = ?
                """,
                (token, expires_at, now.isoformat(), project_id, clip_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO clip_shares (
                    token, project_id, clip_id, expires_at, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (token, project_id, clip_id, expires_at, now.isoformat()),
            )
    return token


def revoke_clip_share(
    database_path: Path,
    project_id: str,
    clip_id: str,
) -> bool:
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            """
            UPDATE clip_shares
            SET revoked_at = ?
            WHERE project_id = ? AND clip_id = ?
              AND revoked_at IS NULL
              AND expires_at IS NOT NULL
              AND julianday(expires_at) > julianday(?)
            """,
            (utc_now(), project_id, clip_id, utc_now()),
        )
    return cursor.rowcount == 1
def save_oauth_state(
    database_path: Path,
    state: str,
    provider: str,
    expires_at: str,
    owner_id: str = "legacy",
) -> None:
    with _lock, connection(database_path) as conn:
        conn.execute(
            "INSERT INTO oauth_states (state, owner_id, provider, expires_at, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (state, owner_id, provider, expires_at, utc_now()),
        )


def consume_oauth_state_record(
    database_path: Path,
    state: str,
    now: str,
) -> dict[str, str] | None:
    with _lock, connection(database_path) as conn:
        row = conn.execute(
            "SELECT owner_id, provider, expires_at FROM oauth_states WHERE state = ?",
            (state,),
        ).fetchone()
        conn.execute("DELETE FROM oauth_states WHERE state = ?", (state,))
    if row and row["expires_at"] > now:
        return {"owner_id": str(row["owner_id"]), "provider": str(row["provider"])}
    return None


def consume_oauth_state(database_path: Path, state: str, now: str) -> str | None:
    record = consume_oauth_state_record(database_path, state, now)
    return record["provider"] if record else None


def save_connected_account(
    database_path: Path,
    provider: str,
    account_id: str,
    display_name: str,
    access_token: str,
    refresh_token: str | None,
    expires_at: str | None,
    metadata: dict[str, Any],
    owner_id: str = "legacy",
) -> None:
    with _lock, connection(database_path) as conn:
        conn.execute(
            """
            INSERT INTO connected_accounts (
                owner_id, provider, account_id, display_name, access_token, refresh_token,
                expires_at, metadata_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner_id, provider) DO UPDATE SET
                account_id = excluded.account_id,
                display_name = excluded.display_name,
                access_token = excluded.access_token,
                refresh_token = excluded.refresh_token,
                expires_at = excluded.expires_at,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                owner_id,
                provider,
                account_id,
                display_name,
                encrypt_oauth_token(access_token),
                encrypt_oauth_token(refresh_token) if refresh_token else None,
                expires_at,
                json.dumps(
                    {
                        **metadata,
                        **(
                            {
                                "user_access_token": encrypt_oauth_token(
                                    str(metadata["user_access_token"])
                                )
                            }
                            if metadata.get("user_access_token")
                            else {}
                        ),
                    },
                    ensure_ascii=False,
                ),
                utc_now(),
            ),
        )


def get_connected_account(
    database_path: Path,
    provider: str,
    owner_id: str = "legacy",
) -> dict[str, Any] | None:
    with connection(database_path) as conn:
        row = conn.execute(
            "SELECT * FROM connected_accounts WHERE owner_id = ? AND provider = ?",
            (owner_id, provider),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["metadata"] = json.loads(result.pop("metadata_json"))
    result["access_token"] = decrypt_oauth_token(result["access_token"])
    result["refresh_token"] = decrypt_oauth_token(result["refresh_token"])
    user_access_token = result["metadata"].get("user_access_token")
    if user_access_token:
        result["metadata"]["user_access_token"] = decrypt_oauth_token(
            user_access_token
        )
    return result


def disconnect_account(
    database_path: Path,
    provider: str,
    owner_id: str = "legacy",
) -> None:
    with _lock, connection(database_path) as conn:
        conn.execute(
            "DELETE FROM connected_accounts WHERE owner_id = ? AND provider = ?",
            (owner_id, provider),
        )


def save_llm_credentials(
    database_path: Path,
    owner_id: str,
    provider: str,
    encrypted_api_key: bytes,
    model: str,
    base_url: str,
    *,
    model_customized: bool = False,
    whisper_model: str | None = None,
) -> None:
    # model_customized es "pegajoso": una vez que el usuario personalizó
    # el modelo, un guardado posterior sin la marca no lo despersonaliza.
    # whisper_model None = sin cambio: conserva el valor guardado.
    with _lock, connection(database_path) as conn:
        conn.execute(
            """
            INSERT INTO llm_credentials (
                owner_id, provider, encrypted_api_key, model, base_url,
                model_customized, whisper_model, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner_id) DO UPDATE SET
                provider = excluded.provider,
                encrypted_api_key = excluded.encrypted_api_key,
                model = excluded.model,
                base_url = excluded.base_url,
                model_customized = MAX(
                    llm_credentials.model_customized, excluded.model_customized
                ),
                whisper_model = COALESCE(
                    excluded.whisper_model, llm_credentials.whisper_model
                ),
                updated_at = excluded.updated_at
            """,
            (
                owner_id,
                provider,
                encrypted_api_key,
                model,
                base_url,
                1 if model_customized else 0,
                whisper_model,
                utc_now(),
            ),
        )


def get_llm_credentials(
    database_path: Path,
    owner_id: str,
) -> dict[str, Any] | None:
    with connection(database_path) as conn:
        row = conn.execute(
            "SELECT owner_id, provider, encrypted_api_key, model, base_url, "
            "model_customized, whisper_model, updated_at "
            "FROM llm_credentials WHERE owner_id = ?",
            (owner_id,),
        ).fetchone()
    if row is None:
        return None
    credentials = dict(row)
    credentials["model_customized"] = bool(credentials.get("model_customized"))
    return credentials


def delete_llm_credentials(database_path: Path, owner_id: str) -> None:
    with _lock, connection(database_path) as conn:
        conn.execute("DELETE FROM llm_credentials WHERE owner_id = ?", (owner_id,))


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
    job = dict(row)
    job["confirmation_required"] = bool(job.get("confirmation_required", False))
    return job


def _schedule_job_select() -> str:
    return """
        scheduled_jobs.*,
        EXISTS(
            SELECT 1 FROM schedule_job_confirmations
            WHERE schedule_job_confirmations.job_id = scheduled_jobs.id
        ) AS confirmation_required
    """


def list_schedule_jobs(
    database_path: Path,
    project_id: str | None = None,
    owner_id: str | None = None,
) -> list[dict[str, Any]]:
    with connection(database_path) as conn:
        if project_id and owner_id:
            rows = conn.execute(
                f"""
                SELECT {_schedule_job_select()} FROM scheduled_jobs
                JOIN projects ON projects.id = scheduled_jobs.project_id
                WHERE scheduled_jobs.project_id = ? AND projects.owner_id = ?
                ORDER BY scheduled_jobs.scheduled_at, scheduled_jobs.created_at
                """,
                (project_id, owner_id),
            ).fetchall()
        elif project_id:
            rows = conn.execute(
                f"SELECT {_schedule_job_select()} FROM scheduled_jobs WHERE project_id = ? ORDER BY scheduled_at, created_at",
                (project_id,),
            ).fetchall()
        elif owner_id:
            rows = conn.execute(
                f"""
                SELECT {_schedule_job_select()} FROM scheduled_jobs
                JOIN projects ON projects.id = scheduled_jobs.project_id
                WHERE projects.owner_id = ?
                ORDER BY scheduled_jobs.scheduled_at, scheduled_jobs.created_at
                """,
                (owner_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {_schedule_job_select()} FROM scheduled_jobs ORDER BY scheduled_at, created_at"
            ).fetchall()
    return [schedule_job_from_row(row) for row in rows]


def get_schedule_job(
    database_path: Path,
    job_id: str,
    owner_id: str | None = None,
) -> dict[str, Any] | None:
    query = f"SELECT {_schedule_job_select()} FROM scheduled_jobs"
    parameters: tuple[Any, ...] = (job_id,)
    if owner_id is not None:
        query += " JOIN projects ON projects.id = scheduled_jobs.project_id"
    query += " WHERE scheduled_jobs.id = ?"
    if owner_id is not None:
        query += " AND projects.owner_id = ?"
        parameters += (owner_id,)
    with connection(database_path) as conn:
        row = conn.execute(query, parameters).fetchone()
    return schedule_job_from_row(row) if row else None


def due_schedule_jobs(database_path: Path, now: str) -> list[dict[str, Any]]:
    with connection(database_path) as conn:
        rows = conn.execute(
            """
            SELECT scheduled_jobs.*, projects.owner_id FROM scheduled_jobs
            JOIN projects ON projects.id = scheduled_jobs.project_id
            WHERE scheduled_jobs.status = 'scheduled' AND scheduled_jobs.scheduled_at <= ?
              AND NOT EXISTS (
                  SELECT 1 FROM schedule_job_confirmations
                  WHERE schedule_job_confirmations.job_id = scheduled_jobs.id
              )
            ORDER BY scheduled_jobs.scheduled_at, scheduled_jobs.created_at
            """,
            (now,),
        ).fetchall()
    return [schedule_job_from_row(row) for row in rows]


def hold_overdue_schedule_jobs(database_path: Path, now: str) -> int:
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO schedule_job_confirmations (job_id, detected_at)
            SELECT id, ?
            FROM scheduled_jobs
            WHERE status = 'scheduled' AND scheduled_at <= ?
            """,
            (now, now),
        )
    return cursor.rowcount


def resolve_overdue_schedule_job(
    database_path: Path,
    job_id: str,
    action: str,
) -> bool:
    if action not in {"publish_now", "discard"}:
        raise ValueError(f"Acción de publicación vencida no válida: {action}")
    now = utc_now()
    with _lock, connection(database_path) as conn:
        row = conn.execute(
            """
            SELECT scheduled_jobs.id FROM scheduled_jobs
            JOIN schedule_job_confirmations
              ON schedule_job_confirmations.job_id = scheduled_jobs.id
            WHERE scheduled_jobs.id = ? AND scheduled_jobs.status = 'scheduled'
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            return False
        if action == "publish_now":
            conn.execute(
                """
                UPDATE scheduled_jobs
                SET scheduled_at = ?, error = NULL, updated_at = ?
                WHERE id = ? AND status = 'scheduled'
                """,
                (now, now, job_id),
            )
        else:
            conn.execute(
                """
                UPDATE scheduled_jobs
                SET status = 'failed',
                    error = 'La publicación vencida se descartó al reabrir la aplicación.',
                    updated_at = ?
                WHERE id = ? AND status = 'scheduled'
                """,
                (now, job_id),
            )
        conn.execute(
            "DELETE FROM schedule_job_confirmations WHERE job_id = ?",
            (job_id,),
        )
    return True


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
              AND NOT EXISTS (
                  SELECT 1 FROM schedule_job_confirmations
                  WHERE schedule_job_confirmations.job_id = scheduled_jobs.id
              )
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


# ---------------------------------------------------------------------------
# Caché de transcripción por hash de archivo (Fase 2, B-R5)
# ---------------------------------------------------------------------------


def get_cached_transcript(
    database_path: Path,
    file_hash: str,
    whisper_model: str,
    whisper_language: str,
) -> dict[str, Any] | None:
    """Transcripción guardada para este archivo, modelo e idioma.

    None si no hay entrada: la clave incluye modelo e idioma, así que un
    cambio en cualquiera de ellos invalida la caché (regla de la Fase 1).
    """
    with connection(database_path) as conn:
        row = conn.execute(
            """
            SELECT language, transcript_json, duration FROM transcript_cache
            WHERE file_hash = ? AND whisper_model = ? AND whisper_language = ?
            """,
            (file_hash, whisper_model, whisper_language),
        ).fetchone()
    if row is None:
        return None
    return {
        "language": str(row["language"]),
        "transcript": json.loads(row["transcript_json"]),
        "duration": float(row["duration"]),
    }


def save_cached_transcript(
    database_path: Path,
    file_hash: str,
    whisper_model: str,
    whisper_language: str,
    language: str,
    transcript: list[dict[str, Any]],
    duration: float,
) -> None:
    """Guarda una transcripción en la caché y poda las entradas viejas."""
    now = utc_now()
    with _lock, connection(database_path) as conn:
        conn.execute(
            """
            INSERT INTO transcript_cache (
                file_hash, whisper_model, whisper_language, language,
                transcript_json, duration, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_hash, whisper_model, whisper_language) DO UPDATE SET
                language = excluded.language,
                transcript_json = excluded.transcript_json,
                duration = excluded.duration,
                created_at = excluded.created_at
            """,
            (
                file_hash,
                whisper_model,
                whisper_language,
                language,
                json.dumps(transcript, ensure_ascii=False),
                duration,
                now,
            ),
        )
        # Se conservan las N entradas más recientes; cada una guarda una
        # transcripción completa con tiempos por palabra.
        conn.execute(
            """
            DELETE FROM transcript_cache
            WHERE rowid NOT IN (
                SELECT rowid FROM transcript_cache
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
            )
            """,
            (_TRANSCRIPT_CACHE_MAX_ENTRIES,),
        )


# ---------------------------------------------------------------------------
# Cola de procesamiento persistente (Fase 2, B-S10)
# ---------------------------------------------------------------------------


def enqueue_processing_job(
    database_path: Path,
    project_id: str,
    retry_failed_only: bool = False,
) -> tuple[str, bool]:
    """Persiste un trabajo de procesamiento. Devuelve (id, creado).

    Si el proyecto ya tiene un trabajo activo (pending/running), no se
    crea otro: devuelve el existente con creado=False (deduplicación).
    """
    job_id = str(uuid.uuid4())
    now = utc_now()
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO processing_jobs (
                id, project_id, retry_failed_only, attempts, status,
                created_at, updated_at
            ) VALUES (?, ?, ?, 0, 'pending', ?, ?)
            """,
            (job_id, project_id, 1 if retry_failed_only else 0, now, now),
        )
        if cursor.rowcount == 1:
            return job_id, True
        row = conn.execute(
            """
            SELECT id FROM processing_jobs
            WHERE project_id = ? AND status IN ('pending', 'running')
            ORDER BY created_at LIMIT 1
            """,
            (project_id,),
        ).fetchone()
    if row is None:  # No debería pasar: el IGNORE implica que existe uno.
        return job_id, True
    return str(row["id"]), False


def get_processing_job(
    database_path: Path, job_id: str
) -> dict[str, Any] | None:
    with connection(database_path) as conn:
        row = conn.execute(
            "SELECT * FROM processing_jobs WHERE id = ?", (job_id,)
        ).fetchone()
    if row is None:
        return None
    job = dict(row)
    job["retry_failed_only"] = bool(job["retry_failed_only"])
    return job


def claim_processing_job(
    database_path: Path, job_id: str, attempts: int
) -> bool:
    """Marca un trabajo como en curso (pending/running -> running)."""
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            """
            UPDATE processing_jobs
            SET status = 'running', attempts = ?, error = NULL, updated_at = ?
            WHERE id = ? AND status IN ('pending', 'running')
            """,
            (attempts, utc_now(), job_id),
        )
    return cursor.rowcount == 1


def requeue_processing_job(database_path: Path, job_id: str) -> bool:
    """Devuelve un trabajo en curso a pendiente (para reintentarlo)."""
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            """
            UPDATE processing_jobs
            SET status = 'pending', updated_at = ?
            WHERE id = ? AND status = 'running'
            """,
            (utc_now(), job_id),
        )
    return cursor.rowcount == 1


def finish_processing_job(
    database_path: Path,
    job_id: str,
    status: str,
    *,
    error: str | None = None,
) -> None:
    if status not in {"done", "failed", "cancelled"}:
        raise ValueError(f"Estado terminal no válido para un trabajo: {status}")
    with _lock, connection(database_path) as conn:
        # Un estado terminal es final: un trabajo cancelado desde el
        # endpoint no se sobrescribe cuando el worker termina de drenar.
        conn.execute(
            """
            UPDATE processing_jobs
            SET status = ?, error = ?, updated_at = ?
            WHERE id = ? AND status NOT IN ('done', 'failed', 'cancelled')
            """,
            (status, error, utc_now(), job_id),
        )


def cancel_processing_jobs(database_path: Path, project_id: str) -> int:
    """Cancela los trabajos activos de un proyecto. Devuelve cuántos."""
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            """
            UPDATE processing_jobs
            SET status = 'cancelled', updated_at = ?
            WHERE project_id = ? AND status IN ('pending', 'running')
            """,
            (utc_now(), project_id),
        )
    return cursor.rowcount


def reset_interrupted_processing_jobs(database_path: Path) -> int:
    """Trabajos 'running' de un arranque anterior -> 'pending'.

    Se llama al recuperar la cola: un trabajo en curso cuando la app se
    cerró quedó a medias y debe reencolarse, no quedarse en 'running'
    para siempre.
    """
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            """
            UPDATE processing_jobs
            SET status = 'pending', updated_at = ?
            WHERE status = 'running'
            """,
            (utc_now(),),
        )
    return cursor.rowcount


def list_recoverable_processing_jobs(database_path: Path) -> list[dict[str, Any]]:
    """Trabajos pendientes de recuperar al arrancar, en orden de llegada."""
    with connection(database_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM processing_jobs
            WHERE status = 'pending'
            ORDER BY created_at, rowid
            """
        ).fetchall()
    jobs = []
    for row in rows:
        job = dict(row)
        job["retry_failed_only"] = bool(job["retry_failed_only"])
        jobs.append(job)
    return jobs
