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
            CREATE TABLE IF NOT EXISTS brand_kits (
                owner_id TEXT PRIMARY KEY,
                name TEXT NOT NULL DEFAULT '',
                primary_color TEXT NOT NULL DEFAULT '#ffffff',
                accent_color TEXT NOT NULL DEFAULT '#facc15',
                font_family TEXT,
                logo_filename TEXT,
                updated_at TEXT NOT NULL
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
            -- Cuentas adicionales por proveedor (Fase 3, ítem 10): la tabla
            -- connected_accounts solo admite una cuenta por proveedor (PK
            -- owner_id, provider) y no se migra su PK; las cuentas extra
            -- viven aquí con la misma forma. La cuenta de
            -- connected_accounts es la predeterminada de cada proveedor.
            CREATE TABLE IF NOT EXISTS connected_accounts_extra (
                owner_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                account_id TEXT NOT NULL,
                display_name TEXT NOT NULL,
                access_token TEXT NOT NULL,
                refresh_token TEXT,
                expires_at TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL,
                PRIMARY KEY (owner_id, provider, account_id)
            );
            -- Preferencias por usuario (Fase 3, ítem 11): la franja
            -- nocturna se guarda con owner_id '__device__' porque
            -- protege el PC compartido, no a un usuario concreto.
            CREATE TABLE IF NOT EXISTS user_preferences (
                owner_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (owner_id, key)
            );
            -- Biblioteca de B-roll (Fase 3, ítem 13): caché local de
            -- videos reutilizables, con favoritos.
            CREATE TABLE IF NOT EXISTS broll_items (
                id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                title TEXT NOT NULL,
                query TEXT NOT NULL DEFAULT '',
                provider TEXT NOT NULL DEFAULT 'local',
                external_id TEXT,
                creator TEXT NOT NULL DEFAULT '',
                attribution_url TEXT NOT NULL DEFAULT '',
                preview_url TEXT NOT NULL DEFAULT '',
                local_path TEXT,
                thumbnail_path TEXT,
                duration REAL NOT NULL DEFAULT 0,
                width INTEGER NOT NULL DEFAULT 0,
                height INTEGER NOT NULL DEFAULT 0,
                favorite INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS broll_items_owner_created
                ON broll_items(owner_id, created_at DESC);
            -- B-roll por frase/segmento (Fase 3, ítem 13): asigna un
            -- elemento de la biblioteca a un clip (start=end=0) o a un
            -- segmento concreto de ese clip.
            CREATE TABLE IF NOT EXISTS broll_segments (
                id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                broll_id TEXT NOT NULL REFERENCES broll_items(id) ON DELETE CASCADE,
                project_id TEXT NOT NULL,
                clip_id TEXT NOT NULL,
                start REAL NOT NULL DEFAULT 0,
                end REAL NOT NULL DEFAULT 0,
                label TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS broll_segments_clip
                ON broll_segments(project_id, clip_id);
            -- Biblioteca de momentos guardados (Fase 3, ítem 14):
            -- favoritos reutilizables entre proyectos (el título y la
            -- descripción son el tema; start/end solo aplican al proyecto
            -- de origen).
            CREATE TABLE IF NOT EXISTS saved_moments (
                id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                project_id TEXT,
                clip_id TEXT,
                start REAL,
                end REAL,
                excerpt TEXT NOT NULL DEFAULT '',
                tags_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS saved_moments_owner_created
                ON saved_moments(owner_id, created_at DESC);
            -- Carpetas observadas (Fase 3, ítem 11): videos nuevos en
            -- estas carpetas crean un proyecto y se encolan solos.
            CREATE TABLE IF NOT EXISTS watch_folders (
                id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                path TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                config_json TEXT NOT NULL DEFAULT '{}',
                poll_seconds INTEGER NOT NULL DEFAULT 300,
                last_scan_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(owner_id, path)
            );
            CREATE TABLE IF NOT EXISTS watch_folder_seen (
                folder_id TEXT NOT NULL REFERENCES watch_folders(id) ON DELETE CASCADE,
                file_key TEXT NOT NULL,
                project_id TEXT,
                seen_at TEXT NOT NULL,
                PRIMARY KEY (folder_id, file_key)
            );
            -- Elementos de lote (Fase 3, ítem 11): cada URL de un lote se
            -- descarga como staged media; el worker la convierte en
            -- proyecto cuando queda lista.
            CREATE TABLE IF NOT EXISTS batch_items (
                id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                staged_id TEXT,
                url TEXT,
                status TEXT NOT NULL CHECK (status IN ('downloading', 'pending', 'done', 'failed')),
                project_id TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS batch_items_owner_created
                ON batch_items(owner_id, created_at DESC);
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
        # Columnas de la Fase 3 (ítems 9–10) en scheduled_jobs, añadidas
        # de forma idempotente como las de arriba: attempts y
        # next_attempt_at sostienen el reintento por job con backoff;
        # account_id, la cuenta concreta de la publicación cruzada; y
        # group_id agrupa los jobs de una misma publicación cruzada.
        schedule_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(scheduled_jobs)")
        }
        for column, definition in (
            ("attempts", "INTEGER NOT NULL DEFAULT 0"),
            ("next_attempt_at", "TEXT"),
            ("account_id", "TEXT"),
            ("group_id", "TEXT"),
        ):
            if column not in schedule_columns:
                conn.execute(
                    f"ALTER TABLE scheduled_jobs ADD COLUMN {column} {definition}"
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
        # brand_kits también se reclama: owner_id es la clave primaria,
        # así que la fila legacy solo se reclama si el usuario aún no
        # tiene kit propio; si ya tiene, la legacy se descarta.
        has_kit = conn.execute(
            "SELECT 1 FROM brand_kits WHERE owner_id = ?", (owner_id,)
        ).fetchone()
        if has_kit is not None:
            conn.execute(
                "DELETE FROM brand_kits "
                "WHERE owner_id IS NULL OR owner_id = 'legacy'"
            )
        else:
            conn.execute(
                "UPDATE brand_kits SET owner_id = ? "
                "WHERE owner_id IS NULL OR owner_id = 'legacy'",
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


def get_brand_kit(
    database_path: Path,
    owner_id: str = "legacy",
) -> dict[str, Any] | None:
    """Plantilla de marca del usuario (Fase 3, ítem 5), o None."""
    with connection(database_path) as conn:
        row = conn.execute(
            """
            SELECT name, primary_color, accent_color, font_family, logo_filename
            FROM brand_kits WHERE owner_id = ?
            """,
            (owner_id,),
        ).fetchone()
    return dict(row) if row else None


def save_brand_kit(
    database_path: Path,
    kit: dict[str, Any],
    owner_id: str = "legacy",
) -> dict[str, Any]:
    """Crea o actualiza la plantilla de marca (una por usuario).

    ``logo_filename`` se conserva si el kit entrante no lo trae (el
    logo se sube por su propio endpoint).
    """
    with _lock, connection(database_path) as conn:
        existing = conn.execute(
            "SELECT logo_filename FROM brand_kits WHERE owner_id = ?",
            (owner_id,),
        ).fetchone()
        logo_filename = kit.get("logo_filename")
        if logo_filename is None and existing is not None:
            logo_filename = existing["logo_filename"]
        conn.execute(
            """
            INSERT INTO brand_kits (
                owner_id, name, primary_color, accent_color,
                font_family, logo_filename, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner_id) DO UPDATE SET
                name = excluded.name,
                primary_color = excluded.primary_color,
                accent_color = excluded.accent_color,
                font_family = excluded.font_family,
                logo_filename = excluded.logo_filename,
                updated_at = excluded.updated_at
            """,
            (
                owner_id,
                str(kit.get("name") or ""),
                str(kit.get("primary_color") or "#ffffff"),
                str(kit.get("accent_color") or "#facc15"),
                kit.get("font_family"),
                logo_filename,
                utc_now(),
            ),
        )
    saved = get_brand_kit(database_path, owner_id)
    assert saved is not None
    return saved


def set_brand_kit_logo(
    database_path: Path,
    logo_filename: str | None,
    owner_id: str = "legacy",
) -> dict[str, Any] | None:
    """Actualiza solo el logo de la plantilla de marca (o None)."""
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            "UPDATE brand_kits SET logo_filename = ?, updated_at = ? "
            "WHERE owner_id = ?",
            (logo_filename, utc_now(), owner_id),
        )
        if cursor.rowcount == 0:
            return None
    return get_brand_kit(database_path, owner_id)


def delete_brand_kit(database_path: Path, owner_id: str = "legacy") -> bool:
    """Borra la plantilla de marca del usuario."""
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            "DELETE FROM brand_kits WHERE owner_id = ?",
            (owner_id,),
        )
    return cursor.rowcount == 1


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
    *,
    account_id: str | None = None,
    group_id: str | None = None,
) -> list[dict[str, Any]]:
    now = utc_now()
    with _lock, connection(database_path) as conn:
        for clip_id in clip_ids:
            conn.execute(
                """
                INSERT OR IGNORE INTO scheduled_jobs (
                    id, project_id, clip_id, platform, scheduled_at, status,
                    visibility, account_id, group_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'scheduled', ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    project_id,
                    clip_id,
                    platform,
                    scheduled_at,
                    visibility,
                    account_id,
                    group_id,
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
    # Un job en backoff (next_attempt_at futuro) no está vencido aunque su
    # scheduled_at ya pasara: el reintento espera a su hora (Fase 3, ítem 9).
    with connection(database_path) as conn:
        rows = conn.execute(
            """
            SELECT scheduled_jobs.*, projects.owner_id FROM scheduled_jobs
            JOIN projects ON projects.id = scheduled_jobs.project_id
            WHERE scheduled_jobs.status = 'scheduled' AND scheduled_jobs.scheduled_at <= ?
              AND (scheduled_jobs.next_attempt_at IS NULL OR scheduled_jobs.next_attempt_at <= ?)
              AND NOT EXISTS (
                  SELECT 1 FROM schedule_job_confirmations
                  WHERE schedule_job_confirmations.job_id = scheduled_jobs.id
              )
            ORDER BY scheduled_jobs.scheduled_at, scheduled_jobs.created_at
            """,
            (now, now),
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
              AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
            """,
            (now, now, now),
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
    # Reclamar cuenta como un intento (Fase 3, ítem 9): attempts sube en
    # cada toma del job, y el backoff se calcula sobre ese contador.
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            """
            UPDATE scheduled_jobs
            SET status = 'processing', attempts = attempts + 1,
                error = NULL, updated_at = ?
            WHERE id = ? AND status = 'scheduled'
              AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
              AND NOT EXISTS (
                  SELECT 1 FROM schedule_job_confirmations
                  WHERE schedule_job_confirmations.job_id = scheduled_jobs.id
              )
            """,
            (utc_now(), job_id, utc_now()),
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


def list_processing_jobs(
    database_path: Path,
    owner_id: str | None = None,
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Vista global de la cola (Fase 3, ítem 11): trabajos con el título
    y el estado del proyecto, los activos primero."""
    query = """
        SELECT processing_jobs.*, projects.title AS project_title,
               projects.status AS project_status,
               projects.progress AS project_progress
        FROM processing_jobs
        JOIN projects ON projects.id = processing_jobs.project_id
    """
    parameters: tuple[Any, ...] = ()
    if owner_id is not None:
        query += " WHERE projects.owner_id = ?"
        parameters = (owner_id,)
    query += (
        " ORDER BY CASE processing_jobs.status"
        " WHEN 'running' THEN 0 WHEN 'pending' THEN 1 ELSE 2 END,"
        " processing_jobs.created_at DESC LIMIT ?"
    )
    parameters += (limit,)
    with connection(database_path) as conn:
        rows = conn.execute(query, parameters).fetchall()
    jobs = []
    for row in rows:
        job = dict(row)
        job["retry_failed_only"] = bool(job["retry_failed_only"])
        jobs.append(job)
    return jobs


# ---------------------------------------------------------------------------
# Cuentas conectadas: varias cuentas por plataforma (Fase 3, ítem 10)
# ---------------------------------------------------------------------------
# La tabla connected_accounts guarda la cuenta PREDETERMINADA de cada
# proveedor (PK owner_id, provider). Las demás viven en
# connected_accounts_extra con los mismos campos y el mismo cifrado de
# tokens. save_account_document enruta cada documento OAuth al lugar
# correcto: si la cuenta ya existe en algún lado se actualiza ahí; si es
# nueva y ya hay una predeterminada distinta, va a extras.


def _account_row_to_dict(row: sqlite3.Row, *, is_primary: bool) -> dict[str, Any]:
    result = dict(row)
    result["metadata"] = json.loads(result.pop("metadata_json"))
    result["access_token"] = decrypt_oauth_token(result["access_token"])
    result["refresh_token"] = decrypt_oauth_token(result["refresh_token"])
    user_access_token = result["metadata"].get("user_access_token")
    if user_access_token:
        result["metadata"]["user_access_token"] = decrypt_oauth_token(
            user_access_token
        )
    result["is_primary"] = is_primary
    return result


def _encrypted_account_values(
    account_id: str,
    display_name: str,
    access_token: str,
    refresh_token: str | None,
    expires_at: str | None,
    metadata: dict[str, Any],
) -> tuple[Any, ...]:
    return (
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
    )


def list_connected_accounts(
    database_path: Path,
    provider: str,
    owner_id: str = "legacy",
) -> list[dict[str, Any]]:
    """Todas las cuentas de un proveedor: la predeterminada primero."""
    accounts: list[dict[str, Any]] = []
    primary = get_connected_account(database_path, provider, owner_id)
    if primary is not None:
        primary["is_primary"] = True
        accounts.append(primary)
    with connection(database_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM connected_accounts_extra
            WHERE owner_id = ? AND provider = ?
            ORDER BY updated_at, account_id
            """,
            (owner_id, provider),
        ).fetchall()
    accounts.extend(
        _account_row_to_dict(row, is_primary=False) for row in rows
    )
    return accounts


def account_summary(account: dict[str, Any]) -> dict[str, Any]:
    """Resumen sin tokens para devolver al navegador."""
    return {
        "provider": account["provider"],
        "account_id": account["account_id"],
        "display_name": account["display_name"],
        "is_primary": bool(account.get("is_primary")),
        "expires_at": account.get("expires_at"),
    }


def get_account_by_id(
    database_path: Path,
    provider: str,
    account_id: str,
    owner_id: str = "legacy",
) -> dict[str, Any] | None:
    """Una cuenta concreta (predeterminada o extra), con tokens."""
    primary = get_connected_account(database_path, provider, owner_id)
    if primary is not None and str(primary["account_id"]) == account_id:
        primary["is_primary"] = True
        return primary
    with connection(database_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM connected_accounts_extra
            WHERE owner_id = ? AND provider = ? AND account_id = ?
            """,
            (owner_id, provider, account_id),
        ).fetchone()
    if row is None:
        return None
    return _account_row_to_dict(row, is_primary=False)


def save_account_document(
    database_path: Path,
    provider: str,
    account_id: str,
    display_name: str,
    access_token: str,
    refresh_token: str | None,
    expires_at: str | None,
    metadata: dict[str, Any],
    owner_id: str = "legacy",
) -> bool:
    """Guarda un documento OAuth y devuelve True si quedó como extra.

    Misma cuenta que la predeterminada → actualiza la predeterminada.
    Cuenta ya guardada como extra → actualiza esa extra. Cuenta nueva con
    predeterminada distinta ocupando el sitio → se guarda como extra, sin
    desplazar a la predeterminada (antes, conectar una segunda cuenta
    borraba la primera en silencio).
    """
    with _lock, connection(database_path) as conn:
        primary = conn.execute(
            "SELECT account_id FROM connected_accounts WHERE owner_id = ? AND provider = ?",
            (owner_id, provider),
        ).fetchone()
        extra = conn.execute(
            """
            SELECT account_id FROM connected_accounts_extra
            WHERE owner_id = ? AND provider = ? AND account_id = ?
            """,
            (owner_id, provider, account_id),
        ).fetchone()
        values = _encrypted_account_values(
            account_id,
            display_name,
            access_token,
            refresh_token,
            expires_at,
            metadata,
        )
        if extra is not None or (
            primary is not None and str(primary["account_id"]) != account_id
        ):
            conn.execute(
                """
                INSERT INTO connected_accounts_extra (
                    owner_id, provider, account_id, display_name,
                    access_token, refresh_token, expires_at, metadata_json,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(owner_id, provider, account_id) DO UPDATE SET
                    display_name = excluded.display_name,
                    access_token = excluded.access_token,
                    refresh_token = excluded.refresh_token,
                    expires_at = excluded.expires_at,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (owner_id, provider, *values),
            )
            return True
    save_connected_account(
        database_path,
        provider,
        account_id,
        display_name,
        access_token,
        refresh_token,
        expires_at,
        metadata,
        owner_id,
    )
    return False


def set_default_account(
    database_path: Path,
    provider: str,
    account_id: str,
    owner_id: str = "legacy",
) -> bool:
    """Hace predeterminada una cuenta extra (intercambia con la actual).

    Devuelve False si esa cuenta no existe o ya es la predeterminada.
    """
    with _lock, connection(database_path) as conn:
        extra_row = conn.execute(
            """
            SELECT * FROM connected_accounts_extra
            WHERE owner_id = ? AND provider = ? AND account_id = ?
            """,
            (owner_id, provider, account_id),
        ).fetchone()
        if extra_row is None:
            return False
        primary_row = conn.execute(
            "SELECT * FROM connected_accounts WHERE owner_id = ? AND provider = ?",
            (owner_id, provider),
        ).fetchone()
        conn.execute(
            "DELETE FROM connected_accounts_extra WHERE owner_id = ? AND provider = ? AND account_id = ?",
            (owner_id, provider, account_id),
        )
        if primary_row is not None:
            # La principal degradada conserva su updated_at original: si
            # luego se borra la nueva predeterminada, la promoción por
            # antigüedad devuelve a la anterior predeterminada, no a una
            # extra más reciente (menos sorpresa para el usuario).
            conn.execute(
                """
                INSERT INTO connected_accounts_extra (
                    owner_id, provider, account_id, display_name,
                    access_token, refresh_token, expires_at, metadata_json,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    owner_id,
                    provider,
                    primary_row["account_id"],
                    primary_row["display_name"],
                    primary_row["access_token"],
                    primary_row["refresh_token"],
                    primary_row["expires_at"],
                    primary_row["metadata_json"],
                    primary_row["updated_at"],
                ),
            )
        conn.execute(
            """
            INSERT INTO connected_accounts (
                owner_id, provider, account_id, display_name, access_token,
                refresh_token, expires_at, metadata_json, updated_at
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
                extra_row["account_id"],
                extra_row["display_name"],
                extra_row["access_token"],
                extra_row["refresh_token"],
                extra_row["expires_at"],
                extra_row["metadata_json"],
                utc_now(),
            ),
        )
    return True


def delete_account_by_id(
    database_path: Path,
    provider: str,
    account_id: str,
    owner_id: str = "legacy",
) -> bool:
    """Desconecta una cuenta concreta. Si era la predeterminada y quedan
    extras, la extra más antigua pasa a ser predeterminada. Devuelve
    False si esa cuenta no existe."""
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            """
            DELETE FROM connected_accounts_extra
            WHERE owner_id = ? AND provider = ? AND account_id = ?
            """,
            (owner_id, provider, account_id),
        )
        if cursor.rowcount == 1:
            return True
        primary_row = conn.execute(
            """
            SELECT account_id FROM connected_accounts
            WHERE owner_id = ? AND provider = ? AND account_id = ?
            """,
            (owner_id, provider, account_id),
        ).fetchone()
        if primary_row is None:
            return False
        conn.execute(
            "DELETE FROM connected_accounts WHERE owner_id = ? AND provider = ?",
            (owner_id, provider),
        )
        promoted = conn.execute(
            """
            SELECT * FROM connected_accounts_extra
            WHERE owner_id = ? AND provider = ?
            ORDER BY updated_at, account_id LIMIT 1
            """,
            (owner_id, provider),
        ).fetchone()
        if promoted is not None:
            conn.execute(
                "DELETE FROM connected_accounts_extra WHERE owner_id = ? AND provider = ? AND account_id = ?",
                (owner_id, provider, promoted["account_id"]),
            )
            conn.execute(
                """
                INSERT INTO connected_accounts (
                    owner_id, provider, account_id, display_name,
                    access_token, refresh_token, expires_at, metadata_json,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    owner_id,
                    provider,
                    promoted["account_id"],
                    promoted["display_name"],
                    promoted["access_token"],
                    promoted["refresh_token"],
                    promoted["expires_at"],
                    promoted["metadata_json"],
                    promoted["updated_at"],
                ),
            )
    return True


# ---------------------------------------------------------------------------
# Publicaciones programadas: reintento, reprogramación y grupos (Fase 3,
# ítems 9–10)
# ---------------------------------------------------------------------------


def requeue_schedule_job_for_retry(
    database_path: Path,
    job_id: str,
    next_attempt_at: str,
    error: str | None,
) -> bool:
    """Devuelve un job en curso a 'scheduled' para reintentarlo a la hora
    indicada (backoff). Solo desde 'processing'."""
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            """
            UPDATE scheduled_jobs
            SET status = 'scheduled', next_attempt_at = ?, error = ?,
                updated_at = ?
            WHERE id = ? AND status = 'processing'
            """,
            (next_attempt_at, error, utc_now(), job_id),
        )
    return cursor.rowcount == 1


def reschedule_schedule_job(
    database_path: Path,
    job_id: str,
    scheduled_at: str,
) -> bool:
    """Mueve un job pendiente a otra fecha/hora (calendario, drag & drop).
    Solo jobs 'scheduled'; cancela cualquier backoff pendiente."""
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            """
            UPDATE scheduled_jobs
            SET scheduled_at = ?, next_attempt_at = NULL, error = NULL,
                updated_at = ?
            WHERE id = ? AND status = 'scheduled'
            """,
            (scheduled_at, utc_now(), job_id),
        )
    return cursor.rowcount == 1


def cancel_schedule_job(database_path: Path, job_id: str) -> bool:
    """Cancela un job pendiente (queda 'failed' con el motivo)."""
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            """
            UPDATE scheduled_jobs
            SET status = 'failed',
                error = 'Publicación cancelada por el usuario.',
                updated_at = ?
            WHERE id = ? AND status = 'scheduled'
            """,
            (utc_now(), job_id),
        )
    return cursor.rowcount == 1


def retry_schedule_job(database_path: Path, job_id: str) -> bool:
    """Reintento manual de un job fallido: vuelve a 'scheduled' para
    publicarse cuanto antes, con el contador de intentos a cero."""
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            """
            UPDATE scheduled_jobs
            SET status = 'scheduled', attempts = 0, next_attempt_at = NULL,
                scheduled_at = ?, error = NULL, updated_at = ?
            WHERE id = ? AND status = 'failed'
            """,
            (utc_now(), utc_now(), job_id),
        )
    return cursor.rowcount == 1


def list_schedule_jobs_by_group(
    database_path: Path,
    group_id: str,
    owner_id: str | None = None,
) -> list[dict[str, Any]]:
    """Jobs de una publicación cruzada (mismo group_id)."""
    query = f"SELECT {_schedule_job_select()} FROM scheduled_jobs"
    parameters: tuple[Any, ...] = (group_id,)
    if owner_id is not None:
        query += " JOIN projects ON projects.id = scheduled_jobs.project_id"
    query += " WHERE scheduled_jobs.group_id = ?"
    if owner_id is not None:
        query += " AND projects.owner_id = ?"
        parameters += (owner_id,)
    query += " ORDER BY scheduled_jobs.platform, scheduled_jobs.created_at"
    with connection(database_path) as conn:
        rows = conn.execute(query, parameters).fetchall()
    return [schedule_job_from_row(row) for row in rows]


# ---------------------------------------------------------------------------
# Preferencias por usuario (Fase 3, ítem 11)
# ---------------------------------------------------------------------------


def get_preference(
    database_path: Path,
    owner_id: str,
    key: str,
    default: Any = None,
) -> Any:
    with connection(database_path) as conn:
        row = conn.execute(
            "SELECT value_json FROM user_preferences WHERE owner_id = ? AND key = ?",
            (owner_id, key),
        ).fetchone()
    if row is None:
        return default
    return json.loads(row["value_json"])


def get_preferences(database_path: Path, owner_id: str) -> dict[str, Any]:
    with connection(database_path) as conn:
        rows = conn.execute(
            "SELECT key, value_json FROM user_preferences WHERE owner_id = ?",
            (owner_id,),
        ).fetchall()
    return {str(row["key"]): json.loads(row["value_json"]) for row in rows}


def set_preference(
    database_path: Path,
    owner_id: str,
    key: str,
    value: Any,
) -> None:
    with _lock, connection(database_path) as conn:
        conn.execute(
            """
            INSERT INTO user_preferences (owner_id, key, value_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(owner_id, key) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = excluded.updated_at
            """,
            (owner_id, key, json.dumps(value, ensure_ascii=False), utc_now()),
        )


# ---------------------------------------------------------------------------
# Biblioteca de B-roll (Fase 3, ítem 13)
# ---------------------------------------------------------------------------


def _broll_item_from_row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["favorite"] = bool(item["favorite"])
    return item


def create_broll_item(
    database_path: Path,
    owner_id: str,
    *,
    title: str,
    query: str = "",
    provider: str = "local",
    external_id: str | None = None,
    creator: str = "",
    attribution_url: str = "",
    preview_url: str = "",
    local_path: str | None = None,
    thumbnail_path: str | None = None,
    duration: float = 0,
    width: int = 0,
    height: int = 0,
) -> dict[str, Any]:
    item_id = str(uuid.uuid4())
    now = utc_now()
    with _lock, connection(database_path) as conn:
        conn.execute(
            """
            INSERT INTO broll_items (
                id, owner_id, title, query, provider, external_id, creator,
                attribution_url, preview_url, local_path, thumbnail_path,
                duration, width, height, favorite, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                item_id,
                owner_id,
                title[:200],
                query[:200],
                provider,
                external_id,
                creator[:200],
                attribution_url,
                preview_url,
                local_path,
                thumbnail_path,
                duration,
                width,
                height,
                now,
                now,
            ),
        )
        row = conn.execute(
            "SELECT * FROM broll_items WHERE id = ?", (item_id,)
        ).fetchone()
    assert row is not None
    return _broll_item_from_row(row)


def list_broll_items(
    database_path: Path,
    owner_id: str,
    *,
    query: str | None = None,
    favorites_only: bool = False,
    limit: int = 200,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM broll_items WHERE owner_id = ?"
    parameters: list[Any] = [owner_id]
    if favorites_only:
        sql += " AND favorite = 1"
    if query and query.strip():
        sql += " AND (title LIKE ? OR query LIKE ? OR creator LIKE ?)"
        like = f"%{query.strip()}%"
        parameters.extend([like, like, like])
    sql += " ORDER BY favorite DESC, created_at DESC LIMIT ?"
    parameters.append(limit)
    with connection(database_path) as conn:
        rows = conn.execute(sql, parameters).fetchall()
    return [_broll_item_from_row(row) for row in rows]


def get_broll_item(
    database_path: Path,
    item_id: str,
    owner_id: str | None = None,
) -> dict[str, Any] | None:
    query = "SELECT * FROM broll_items WHERE id = ?"
    parameters: tuple[Any, ...] = (item_id,)
    if owner_id is not None:
        query += " AND owner_id = ?"
        parameters += (owner_id,)
    with connection(database_path) as conn:
        row = conn.execute(query, parameters).fetchone()
    return _broll_item_from_row(row) if row is not None else None


def update_broll_item(
    database_path: Path,
    item_id: str,
    owner_id: str,
    **values: Any,
) -> bool:
    allowed = {"title", "favorite", "local_path", "thumbnail_path"}
    updates = {key: value for key, value in values.items() if key in allowed}
    if not updates:
        return False
    if "title" in updates:
        updates["title"] = str(updates["title"])[:200]
    if "favorite" in updates:
        updates["favorite"] = 1 if updates["favorite"] else 0
    assignments = ", ".join(f"{key} = ?" for key in updates)
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            f"UPDATE broll_items SET {assignments}, updated_at = ? "
            "WHERE id = ? AND owner_id = ?",
            (*updates.values(), utc_now(), item_id, owner_id),
        )
    return cursor.rowcount == 1


def delete_broll_item(database_path: Path, item_id: str, owner_id: str) -> bool:
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            "DELETE FROM broll_items WHERE id = ? AND owner_id = ?",
            (item_id, owner_id),
        )
    return cursor.rowcount == 1


def assign_broll_segment(
    database_path: Path,
    owner_id: str,
    *,
    broll_id: str,
    project_id: str,
    clip_id: str,
    start: float = 0,
    end: float = 0,
    label: str = "",
) -> dict[str, Any]:
    segment_id = str(uuid.uuid4())
    now = utc_now()
    with _lock, connection(database_path) as conn:
        conn.execute(
            """
            INSERT INTO broll_segments (
                id, owner_id, broll_id, project_id, clip_id, start, end,
                label, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                segment_id,
                owner_id,
                broll_id,
                project_id,
                clip_id,
                start,
                end,
                label[:200],
                now,
            ),
        )
        row = conn.execute(
            "SELECT * FROM broll_segments WHERE id = ?", (segment_id,)
        ).fetchone()
    assert row is not None
    return dict(row)


def list_broll_segments(
    database_path: Path,
    project_id: str,
    clip_id: str | None = None,
    owner_id: str | None = None,
) -> list[dict[str, Any]]:
    query = """
        SELECT broll_segments.*, broll_items.title AS broll_title,
               broll_items.provider AS broll_provider,
               broll_items.local_path AS broll_local_path,
               broll_items.creator AS broll_creator,
               broll_items.attribution_url AS broll_attribution_url
        FROM broll_segments
        JOIN broll_items ON broll_items.id = broll_segments.broll_id
        WHERE broll_segments.project_id = ?
    """
    parameters: list[Any] = [project_id]
    if clip_id is not None:
        query += " AND broll_segments.clip_id = ?"
        parameters.append(clip_id)
    if owner_id is not None:
        query += " AND broll_segments.owner_id = ?"
        parameters.append(owner_id)
    query += " ORDER BY broll_segments.clip_id, broll_segments.start"
    with connection(database_path) as conn:
        rows = conn.execute(query, parameters).fetchall()
    return [dict(row) for row in rows]


def delete_broll_segment(
    database_path: Path,
    segment_id: str,
    owner_id: str,
) -> bool:
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            "DELETE FROM broll_segments WHERE id = ? AND owner_id = ?",
            (segment_id, owner_id),
        )
    return cursor.rowcount == 1


def broll_assignment_for_clip(
    database_path: Path,
    project_id: str,
    clip_id: str,
) -> dict[str, Any] | None:
    """Primera asignación de B-roll de un clip con archivo local.

    El pipeline la usa al renderizar: un B-roll asignado explícitamente
    gana a la descarga automática del proveedor.
    """
    segments = list_broll_segments(database_path, project_id, clip_id)
    for segment in segments:
        local_path = segment.get("broll_local_path")
        if local_path and Path(str(local_path)).is_file():
            return segment
    return None


# ---------------------------------------------------------------------------
# Biblioteca de momentos guardados (Fase 3, ítem 14)
# ---------------------------------------------------------------------------


def _saved_moment_from_row(row: sqlite3.Row) -> dict[str, Any]:
    moment = dict(row)
    moment["tags"] = json.loads(moment.pop("tags_json"))
    return moment


def create_saved_moment(
    database_path: Path,
    owner_id: str,
    *,
    title: str,
    description: str = "",
    project_id: str | None = None,
    clip_id: str | None = None,
    start: float | None = None,
    end: float | None = None,
    excerpt: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    moment_id = str(uuid.uuid4())
    now = utc_now()
    with _lock, connection(database_path) as conn:
        conn.execute(
            """
            INSERT INTO saved_moments (
                id, owner_id, title, description, project_id, clip_id,
                start, end, excerpt, tags_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                moment_id,
                owner_id,
                title[:200],
                description[:1000],
                project_id,
                clip_id,
                start,
                end,
                excerpt[:2000],
                json.dumps(
                    [str(tag)[:50] for tag in (tags or [])][:10],
                    ensure_ascii=False,
                ),
                now,
                now,
            ),
        )
        row = conn.execute(
            "SELECT * FROM saved_moments WHERE id = ?", (moment_id,)
        ).fetchone()
    assert row is not None
    return _saved_moment_from_row(row)


def list_saved_moments(
    database_path: Path,
    owner_id: str,
    *,
    query: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM saved_moments WHERE owner_id = ?"
    parameters: list[Any] = [owner_id]
    if query and query.strip():
        sql += " AND (title LIKE ? OR description LIKE ? OR tags_json LIKE ?)"
        like = f"%{query.strip()}%"
        parameters.extend([like, like, like])
    sql += " ORDER BY created_at DESC LIMIT ?"
    parameters.append(limit)
    with connection(database_path) as conn:
        rows = conn.execute(sql, parameters).fetchall()
    return [_saved_moment_from_row(row) for row in rows]


def get_saved_moment(
    database_path: Path,
    moment_id: str,
    owner_id: str | None = None,
) -> dict[str, Any] | None:
    query = "SELECT * FROM saved_moments WHERE id = ?"
    parameters: tuple[Any, ...] = (moment_id,)
    if owner_id is not None:
        query += " AND owner_id = ?"
        parameters += (owner_id,)
    with connection(database_path) as conn:
        row = conn.execute(query, parameters).fetchone()
    return _saved_moment_from_row(row) if row is not None else None


def update_saved_moment(
    database_path: Path,
    moment_id: str,
    owner_id: str,
    **values: Any,
) -> bool:
    allowed = {"title", "description", "excerpt", "tags"}
    updates: dict[str, Any] = {}
    for key, value in values.items():
        if key not in allowed:
            continue
        if key == "tags":
            updates["tags_json"] = json.dumps(
                [str(tag)[:50] for tag in (value or [])][:10],
                ensure_ascii=False,
            )
        elif key == "title":
            updates[key] = str(value)[:200]
        elif key == "description":
            updates[key] = str(value)[:1000]
        else:
            updates[key] = str(value)[:2000]
    if not updates:
        return False
    assignments = ", ".join(f"{key} = ?" for key in updates)
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            f"UPDATE saved_moments SET {assignments}, updated_at = ? "
            "WHERE id = ? AND owner_id = ?",
            (*updates.values(), utc_now(), moment_id, owner_id),
        )
    return cursor.rowcount == 1


def delete_saved_moment(
    database_path: Path,
    moment_id: str,
    owner_id: str,
) -> bool:
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            "DELETE FROM saved_moments WHERE id = ? AND owner_id = ?",
            (moment_id, owner_id),
        )
    return cursor.rowcount == 1


# ---------------------------------------------------------------------------
# Carpetas observadas y lotes (Fase 3, ítem 11)
# ---------------------------------------------------------------------------


def _watch_folder_from_row(row: sqlite3.Row) -> dict[str, Any]:
    folder = dict(row)
    folder["enabled"] = bool(folder["enabled"])
    folder["config"] = json.loads(folder.pop("config_json"))
    return folder


def create_watch_folder(
    database_path: Path,
    owner_id: str,
    path: str,
    *,
    enabled: bool = True,
    config: dict[str, Any] | None = None,
    poll_seconds: int = 300,
) -> dict[str, Any]:
    folder_id = str(uuid.uuid4())
    now = utc_now()
    with _lock, connection(database_path) as conn:
        conn.execute(
            """
            INSERT INTO watch_folders (
                id, owner_id, path, enabled, config_json, poll_seconds,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                folder_id,
                owner_id,
                path,
                1 if enabled else 0,
                json.dumps(config or {}, ensure_ascii=False),
                poll_seconds,
                now,
                now,
            ),
        )
        row = conn.execute(
            "SELECT * FROM watch_folders WHERE id = ?", (folder_id,)
        ).fetchone()
    assert row is not None
    return _watch_folder_from_row(row)


def list_watch_folders(
    database_path: Path,
    owner_id: str | None = None,
) -> list[dict[str, Any]]:
    query = "SELECT * FROM watch_folders"
    parameters: tuple[Any, ...] = ()
    if owner_id is not None:
        query += " WHERE owner_id = ?"
        parameters = (owner_id,)
    query += " ORDER BY created_at"
    with connection(database_path) as conn:
        rows = conn.execute(query, parameters).fetchall()
    return [_watch_folder_from_row(row) for row in rows]


def get_watch_folder(
    database_path: Path,
    folder_id: str,
    owner_id: str | None = None,
) -> dict[str, Any] | None:
    query = "SELECT * FROM watch_folders WHERE id = ?"
    parameters: tuple[Any, ...] = (folder_id,)
    if owner_id is not None:
        query += " AND owner_id = ?"
        parameters += (owner_id,)
    with connection(database_path) as conn:
        row = conn.execute(query, parameters).fetchone()
    return _watch_folder_from_row(row) if row is not None else None


def update_watch_folder(
    database_path: Path,
    folder_id: str,
    owner_id: str,
    **values: Any,
) -> bool:
    allowed = {"enabled", "config", "poll_seconds"}
    updates: dict[str, Any] = {}
    for key, value in values.items():
        if key not in allowed:
            continue
        if key == "config":
            updates["config_json"] = json.dumps(value or {}, ensure_ascii=False)
        elif key == "enabled":
            updates["enabled"] = 1 if value else 0
        else:
            updates["poll_seconds"] = int(value)
    if not updates:
        return False
    assignments = ", ".join(f"{key} = ?" for key in updates)
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            f"UPDATE watch_folders SET {assignments}, updated_at = ? "
            "WHERE id = ? AND owner_id = ?",
            (*updates.values(), utc_now(), folder_id, owner_id),
        )
    return cursor.rowcount == 1


def delete_watch_folder(
    database_path: Path,
    folder_id: str,
    owner_id: str,
) -> bool:
    with _lock, connection(database_path) as conn:
        conn.execute(
            "DELETE FROM watch_folder_seen WHERE folder_id = ?", (folder_id,)
        )
        cursor = conn.execute(
            "DELETE FROM watch_folders WHERE id = ? AND owner_id = ?",
            (folder_id, owner_id),
        )
    return cursor.rowcount == 1


def touch_watch_folder_scan(database_path: Path, folder_id: str) -> None:
    with _lock, connection(database_path) as conn:
        conn.execute(
            "UPDATE watch_folders SET last_scan_at = ?, updated_at = ? WHERE id = ?",
            (utc_now(), utc_now(), folder_id),
        )


def list_seen_file_keys(database_path: Path, folder_id: str) -> set[str]:
    with connection(database_path) as conn:
        rows = conn.execute(
            "SELECT file_key FROM watch_folder_seen WHERE folder_id = ?",
            (folder_id,),
        ).fetchall()
    return {str(row["file_key"]) for row in rows}


def list_seen_files(database_path: Path, folder_id: str) -> dict[str, str | None]:
    """Claves vistas con su proyecto (None = visto pero aún sin proyecto:
    candidato en espera de estabilizarse)."""
    with connection(database_path) as conn:
        rows = conn.execute(
            "SELECT file_key, project_id FROM watch_folder_seen WHERE folder_id = ?",
            (folder_id,),
        ).fetchall()
    return {
        str(row["file_key"]): (
            str(row["project_id"]) if row["project_id"] is not None else None
        )
        for row in rows
    }


def mark_file_seen(
    database_path: Path,
    folder_id: str,
    file_key: str,
    *,
    project_id: str | None = None,
) -> None:
    with _lock, connection(database_path) as conn:
        conn.execute(
            """
            INSERT INTO watch_folder_seen (folder_id, file_key, project_id, seen_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(folder_id, file_key) DO UPDATE SET
                project_id = COALESCE(excluded.project_id, watch_folder_seen.project_id),
                seen_at = excluded.seen_at
            """,
            (folder_id, file_key, project_id, utc_now()),
        )


def create_batch_item(
    database_path: Path,
    batch_id: str,
    owner_id: str,
    *,
    staged_id: str | None = None,
    url: str | None = None,
    status: str = "downloading",
    project_id: str | None = None,
) -> dict[str, Any]:
    item_id = str(uuid.uuid4())
    now = utc_now()
    with _lock, connection(database_path) as conn:
        conn.execute(
            """
            INSERT INTO batch_items (
                id, batch_id, owner_id, staged_id, url, status, project_id,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                batch_id,
                owner_id,
                staged_id,
                url,
                status,
                project_id,
                now,
                now,
            ),
        )
        row = conn.execute(
            "SELECT * FROM batch_items WHERE id = ?", (item_id,)
        ).fetchone()
    assert row is not None
    return dict(row)


def list_pending_batch_items(
    database_path: Path,
) -> list[dict[str, Any]]:
    with connection(database_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM batch_items
            WHERE status IN ('downloading', 'pending')
            ORDER BY created_at
            """
        ).fetchall()
    return [dict(row) for row in rows]


def list_batch_items(
    database_path: Path,
    owner_id: str,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    with connection(database_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM batch_items WHERE owner_id = ?
            ORDER BY created_at DESC LIMIT ?
            """,
            (owner_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def update_batch_item(
    database_path: Path,
    item_id: str,
    *,
    status: str | None = None,
    project_id: str | None = None,
    error: str | None = None,
) -> bool:
    updates: dict[str, Any] = {}
    if status is not None:
        updates["status"] = status
    if project_id is not None:
        updates["project_id"] = project_id
    if error is not None:
        updates["error"] = error[:1000]
    if not updates:
        return False
    assignments = ", ".join(f"{key} = ?" for key in updates)
    with _lock, connection(database_path) as conn:
        cursor = conn.execute(
            f"UPDATE batch_items SET {assignments}, updated_at = ? WHERE id = ?",
            (*updates.values(), utc_now(), item_id),
        )
    return cursor.rowcount == 1
