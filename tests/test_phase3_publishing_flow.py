"""Tests de la Fase 3 (ítems 9–15): publicación y flujo.

Cubre: calendario + mejor hora + reintento por job con backoff (9),
publicación cruzada multi-cuenta (10), cola por lotes + carpeta observada
+ procesar de noche (11), generador de títulos/descripciones/hashtags
(12), biblioteca de B-roll (13), momentos guardados (14) y duplicar
proyecto (15).
"""

from __future__ import annotations

import asyncio
import base64
import json
import sqlite3
import uuid
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import backend.captions as captions_module
import backend.main as main_module
import backend.scheduler as scheduler_module
import backend.watch_folder as watch_folder_module
from backend import database, publish_guide
from backend.integrations import IntegrationError
from backend.main import create_app as create_application
from backend.process_queue import ProjectProcessingQueue
from backend.schemas import ProjectConfig
from backend.settings import AppSettings


def create_app(settings: AppSettings):
    return create_application(settings, auth_required=False)


@pytest.fixture(autouse=True)
def _fake_token_encryption(monkeypatch: pytest.MonkeyPatch) -> None:
    """DPAPI solo existe en Windows: en Linux los tests usan un cifrado
    reversible de pruebas para poder guardar cuentas OAuth. Solo afecta a
    este archivo; el cifrado real no se toca."""
    # El no_proxy de esta VM contiene IPv6 entre corchetes que la versión
    # de httpx instalada no sabe interpretar: construir un httpx.Client()
    # con trust_env (el valor por defecto, que usa todo el backend) lanza
    # InvalidURL. Sin estas variables, el cliente se construye bien.
    for variable in (
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
    ):
        monkeypatch.delenv(variable, raising=False)

    def fake_encrypt(token: str | None) -> str | None:
        if token is None:
            return None
        return "test-enc:" + base64.b64encode(token.encode()).decode()

    def fake_decrypt(value: str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str) and value.startswith("test-enc:"):
            return base64.b64decode(value[len("test-enc:"):]).decode()
        return value

    monkeypatch.setattr(database, "encrypt_oauth_token", fake_encrypt)
    monkeypatch.setattr(database, "decrypt_oauth_token", fake_decrypt)


def _settings(data_dir: Path, **overrides: Any) -> AppSettings:
    values: dict[str, Any] = {
        "data_dir": data_dir,
        "cors_origins": (),
        "llm_provider": "Groq",
        "llm_api_key": "test-key",
        "llm_model": "test-model",
        "llm_base_url": "",
        "whisper_model": "small",
        "youtube_client_id": "yt-id",
        "youtube_client_secret": "yt-secret",
        "tiktok_client_key": "tt-key",
        "tiktok_client_secret": "tt-secret",
        "meta_app_id": "meta-id",
        "meta_app_secret": "meta-secret",
    }
    values.update(overrides)
    return AppSettings(**values)


def _create_project(
    settings: AppSettings,
    project_id: str,
    *,
    status: str = "ready",
    config: dict[str, Any] | None = None,
) -> Path:
    project_dir = settings.data_dir / "projects" / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "source").write_bytes(b"video-bytes")
    database.create_project(
        settings.database_path,
        project_id,
        f"Proyecto {project_id}",
        project_dir / "source",
        config if config is not None else ProjectConfig().model_dump(mode="json"),
        "test-user",
    )
    if status != "ready":
        database.update_project(settings.database_path, project_id, status=status)
    return project_dir


def _create_completed_project_with_clip(
    settings: AppSettings,
    project_id: str,
    clip_id: str = "clip-1",
) -> Path:
    project_dir = _create_project(settings, project_id, status="completed")
    clips_dir = project_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clips_dir / "clip.mp4"
    clip_path.write_bytes(b"clip-bytes")
    database.update_project(
        settings.database_path,
        project_id,
        clips=[
            {
                "id": clip_id,
                "title": "Clip de prueba",
                "description": "Descripción de prueba",
                "viral_reason": "Un gancho fuerte",
                "keywords": ["prueba"],
                "emojis": ["🔥"],
                "hashtags": {"youtube": ["prueba"], "tiktok": [], "instagram": []},
                "filename": "clip.mp4",
                "status": "ready",
                "start": 0.0,
                "end": 10.0,
                "duration": 10.0,
            }
        ],
    )
    return clip_path


def _save_account(
    settings: AppSettings,
    provider: str,
    account_id: str,
    display_name: str = "Cuenta",
) -> None:
    database.save_connected_account(
        settings.database_path,
        provider,
        account_id,
        display_name,
        f"token-{account_id}",
        f"refresh-{account_id}",
        None,
        {},
        "test-user",
    )


def _idle_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """El worker del programador y el de carpetas en reposo: los tests de
    endpoints no deben publicar de verdad ni sondear carpetas."""

    async def idle_schedule_worker(_settings: AppSettings) -> None:
        await asyncio.sleep(3600)

    async def idle_watch_worker(_settings: AppSettings, _submit: Any) -> None:
        await asyncio.sleep(3600)

    monkeypatch.setattr(main_module, "schedule_worker", idle_schedule_worker)
    monkeypatch.setattr(
        watch_folder_module, "watch_folder_worker", idle_watch_worker
    )


def _future_iso(**kwargs: Any) -> str:
    return (datetime.now(UTC) + timedelta(**kwargs)).isoformat()


def _clear_next_attempt(settings: AppSettings, job_id: str) -> None:
    with database.connection(settings.database_path) as conn:
        conn.execute(
            "UPDATE scheduled_jobs SET next_attempt_at = NULL WHERE id = ?",
            (job_id,),
        )


# ---------------------------------------------------------------------------
# Ítem 9: calendario + mejor hora + reintento por job
# ---------------------------------------------------------------------------


def test_best_times_guide_is_labelled_as_guidance_not_audience_data(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/publish/best-times?platform=youtube")
    assert response.status_code == 200
    payload = response.json()
    assert payload["platform"] == "youtube"
    assert payload["weekday_hours"]
    assert payload["weekend_hours"]
    assert "audiencia" in payload["note"]
    assert len(payload["next_slots"]["youtube"]) == 3
    assert all(
        datetime.fromisoformat(slot) > datetime.now()  # noqa: DTZ005 - hora local
        for slot in payload["next_slots"]["youtube"]
    )


def test_best_times_unknown_platform_is_404(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/publish/best-times?platform=twitch")
    assert response.status_code == 404


def test_next_best_slots_skips_past_hours_today() -> None:
    now = datetime(2026, 11, 23, 20, 30)  # Lunes 20:30.
    slots = publish_guide.next_best_slots("youtube", count=3, now=now)
    assert slots[0] == "2026-11-23T21:00"
    assert slots[1].startswith("2026-11-24T19:00")


def test_group_jobs_by_date_groups_local_dates() -> None:
    grouped = publish_guide.group_jobs_by_date(
        [
            {"id": "a", "scheduled_at": "2026-11-23T19:00:00+00:00"},
            {"id": "b", "scheduled_at": "2026-11-23T21:00:00+00:00"},
            {"id": "c", "scheduled_at": "2026-11-24T10:00:00+00:00"},
        ]
    )
    assert [job["id"] for job in grouped["2026-11-23"]] == ["a", "b"]
    assert [job["id"] for job in grouped["2026-11-24"]] == ["c"]


def test_transient_publish_error_retries_and_terminal_after_max_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    _create_completed_project_with_clip(settings, project_id)
    _save_account(settings, "youtube", "yt-account")

    def failing_publish(*_args: Any, **_kwargs: Any) -> dict[str, str]:
        raise IntegrationError("connection timeout al subir a YouTube")

    monkeypatch.setattr(scheduler_module, "publish_clip", failing_publish)
    jobs = database.enqueue_schedule_jobs(
        settings.database_path,
        project_id,
        ["clip-1"],
        "youtube",
        _future_iso(days=-1),
        "private",
    )
    job_id = jobs[0]["id"]
    for expected_attempt in (1, 2):
        assert database.claim_schedule_job(settings.database_path, job_id)
        job = database.get_schedule_job(settings.database_path, job_id)
        assert job is not None
        scheduler_module.process_schedule_job(settings, job)
        job = database.get_schedule_job(settings.database_path, job_id)
        assert job is not None
        assert job["status"] == "scheduled"
        assert job["attempts"] == expected_attempt
        assert job["next_attempt_at"] is not None
        assert datetime.fromisoformat(job["next_attempt_at"]) > datetime.now(UTC)
        assert f"intento {expected_attempt + 1} de 3" in (job["error"] or "")
        _clear_next_attempt(settings, job_id)
    assert database.claim_schedule_job(settings.database_path, job_id)
    job = database.get_schedule_job(settings.database_path, job_id)
    assert job is not None
    scheduler_module.process_schedule_job(settings, job)
    job = database.get_schedule_job(settings.database_path, job_id)
    assert job is not None
    assert job["status"] == "failed"
    assert job["attempts"] == 3


def test_permanent_publish_error_fails_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    _create_completed_project_with_clip(settings, project_id)
    _save_account(settings, "youtube", "yt-account")

    def failing_publish(*_args: Any, **_kwargs: Any) -> dict[str, str]:
        raise IntegrationError("La sesión de youtube venció.")

    monkeypatch.setattr(scheduler_module, "publish_clip", failing_publish)
    jobs = database.enqueue_schedule_jobs(
        settings.database_path,
        project_id,
        ["clip-1"],
        "youtube",
        _future_iso(days=-1),
        "private",
    )
    assert database.claim_schedule_job(settings.database_path, jobs[0]["id"])
    job = database.get_schedule_job(settings.database_path, jobs[0]["id"])
    assert job is not None
    scheduler_module.process_schedule_job(settings, job)
    job = database.get_schedule_job(settings.database_path, jobs[0]["id"])
    assert job is not None
    assert job["status"] == "failed"
    assert job["attempts"] == 1
    assert job["next_attempt_at"] is None


def test_due_jobs_excludes_backoff_jobs(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    _create_project(settings, project_id)
    jobs = database.enqueue_schedule_jobs(
        settings.database_path,
        project_id,
        ["clip-1"],
        "youtube",
        _future_iso(days=-1),
        "private",
    )
    job_id = jobs[0]["id"]
    assert database.claim_schedule_job(settings.database_path, job_id)
    assert database.requeue_schedule_job_for_retry(
        settings.database_path,
        job_id,
        _future_iso(minutes=30),
        "transitorio",
    )
    assert database.due_schedule_jobs(settings.database_path, database.utc_now()) == []
    _clear_next_attempt(settings, job_id)
    assert [
        job["id"] for job in database.due_schedule_jobs(
            settings.database_path, database.utc_now()
        )
    ] == [job_id]


def test_reschedule_retry_and_cancel_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _idle_workers(monkeypatch)
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        project_id = str(uuid.uuid4())
        _create_completed_project_with_clip(settings, project_id)
        jobs = database.enqueue_schedule_jobs(
            settings.database_path,
            project_id,
            ["clip-1"],
            "youtube",
            _future_iso(days=1),
            "private",
        )
        job_id = jobs[0]["id"]
        moved = client.post(
            f"/api/scheduled-jobs/{job_id}/reschedule",
            json={"scheduled_at": _future_iso(days=2)},
        )
        assert moved.status_code == 200
        assert moved.json()["scheduled_at"].startswith(
            _future_iso(days=2)[:10]
        )
        past = client.post(
            f"/api/scheduled-jobs/{job_id}/reschedule",
            json={"scheduled_at": _future_iso(days=-1)},
        )
        assert past.status_code == 422
        retry_too_soon = client.post(f"/api/scheduled-jobs/{job_id}/retry")
        assert retry_too_soon.status_code == 409
        cancelled = client.delete(f"/api/scheduled-jobs/{job_id}")
        assert cancelled.status_code == 204
        job = database.get_schedule_job(settings.database_path, job_id)
        assert job is not None and job["status"] == "failed"
        cancelled_again = client.delete(f"/api/scheduled-jobs/{job_id}")
        assert cancelled_again.status_code == 409
        retried = client.post(f"/api/scheduled-jobs/{job_id}/retry")
        assert retried.status_code == 200
        assert retried.json()["status"] == "scheduled"
        assert retried.json()["attempts"] == 0


# ---------------------------------------------------------------------------
# Ítem 10: publicación cruzada multi-cuenta
# ---------------------------------------------------------------------------


def test_second_account_goes_to_extras_without_displacing_primary(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    _save_account(settings, "youtube", "yt-main", "Canal principal")
    was_extra = database.save_account_document(
        settings.database_path,
        "youtube",
        "yt-second",
        "Canal secundario",
        "token-yt-second",
        "refresh-yt-second",
        None,
        {},
        "test-user",
    )
    assert was_extra is True
    accounts = database.list_connected_accounts(
        settings.database_path, "youtube", "test-user"
    )
    assert [account["account_id"] for account in accounts] == [
        "yt-main",
        "yt-second",
    ]
    assert accounts[0]["is_primary"] is True
    assert accounts[1]["is_primary"] is False
    assert accounts[1]["access_token"] == "token-yt-second"
    # Guardar de nuevo la principal la actualiza en su sitio, no la duplica.
    was_extra = database.save_account_document(
        settings.database_path,
        "youtube",
        "yt-main",
        "Canal principal",
        "token-yt-main-v2",
        "refresh-yt-main-v2",
        None,
        {},
        "test-user",
    )
    assert was_extra is False
    accounts = database.list_connected_accounts(
        settings.database_path, "youtube", "test-user"
    )
    assert len(accounts) == 2
    assert accounts[0]["access_token"] == "token-yt-main-v2"


def test_set_default_swaps_and_delete_promotes_oldest_extra(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    _save_account(settings, "youtube", "yt-main")
    database.save_account_document(
        settings.database_path, "youtube", "yt-b", "B",
        "token-yt-b", None, None, {}, "test-user",
    )
    database.save_account_document(
        settings.database_path, "youtube", "yt-c", "C",
        "token-yt-c", None, None, {}, "test-user",
    )
    assert database.set_default_account(
        settings.database_path, "youtube", "yt-c", "test-user"
    )
    accounts = database.list_connected_accounts(
        settings.database_path, "youtube", "test-user"
    )
    assert accounts[0]["account_id"] == "yt-c"
    assert database.delete_account_by_id(
        settings.database_path, "youtube", "yt-c", "test-user"
    )
    accounts = database.list_connected_accounts(
        settings.database_path, "youtube", "test-user"
    )
    # La extra más antigua (yt-main, que era la principal anterior) promociona.
    assert accounts[0]["account_id"] == "yt-main"
    assert accounts[0]["is_primary"] is True
    assert database.delete_account_by_id(
        settings.database_path, "youtube", "yt-b", "test-user"
    )
    assert len(
        database.list_connected_accounts(
            settings.database_path, "youtube", "test-user"
        )
    ) == 1
    assert not database.delete_account_by_id(
        settings.database_path, "youtube", "yt-nope", "test-user"
    )


def test_publish_clip_uses_the_selected_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import backend.integrations as integrations_module

    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    _save_account(settings, "youtube", "yt-main")
    database.save_account_document(
        settings.database_path, "youtube", "yt-second", "Segundo",
        "token-yt-second", None, None, {}, "test-user",
    )
    seen: dict[str, str] = {}

    def fake_youtube(token: str, *_args: Any, **_kwargs: Any) -> dict[str, str]:
        seen["token"] = token
        return {"external_id": "vid-1", "url": "https://example.invalid/vid-1"}

    monkeypatch.setattr(integrations_module, "_publish_youtube", fake_youtube)
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"clip-bytes")
    integrations_module.publish_clip(
        settings, "youtube", {"title": "T"}, clip_path, "private",
        "test-user", account_id="yt-second",
    )
    assert seen["token"] == "token-yt-second"
    integrations_module.publish_clip(
        settings, "youtube", {"title": "T"}, clip_path, "private", "test-user"
    )
    assert seen["token"] == "token-yt-main"


def test_cross_publish_creates_one_job_per_target_with_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _idle_workers(monkeypatch)
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        project_id = str(uuid.uuid4())
        _create_completed_project_with_clip(settings, project_id)
        _save_account(settings, "youtube", "yt-main")
        _save_account(settings, "tiktok", "tt-main")
        response = client.post(
            f"/api/projects/{project_id}/cross-publish",
            json={
                "clip_ids": ["clip-1"],
                "targets": [
                    {"platform": "youtube", "visibility": "private"},
                    {
                        "platform": "tiktok",
                        "visibility": "private",
                        "scheduled_at": _future_iso(days=1),
                    },
                ],
            },
        )
        assert response.status_code == 201, response.text
        payload = response.json()
        assert len(payload["jobs"]) == 2
        assert {job["platform"] for job in payload["jobs"]} == {
            "youtube",
            "tiktok",
        }
        assert {job["group_id"] for job in payload["jobs"]} == {
            payload["group_id"]
        }
        grouped = client.get(
            f"/api/schedule-groups/{payload['group_id']}"
        )
        assert grouped.status_code == 200
        assert len(grouped.json()) == 2


def test_cross_publish_two_accounts_same_platform_staggers_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _idle_workers(monkeypatch)
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        project_id = str(uuid.uuid4())
        _create_completed_project_with_clip(settings, project_id)
        _save_account(settings, "youtube", "yt-main")
        database.save_account_document(
            settings.database_path, "youtube", "yt-second", "Segundo",
            "token-yt-second", None, None, {}, "test-user",
        )
        when = _future_iso(days=1)
        response = client.post(
            f"/api/projects/{project_id}/cross-publish",
            json={
                "clip_ids": ["clip-1"],
                "targets": [
                    {"platform": "youtube", "account_id": "yt-main",
                     "visibility": "private", "scheduled_at": when},
                    {"platform": "youtube", "account_id": "yt-second",
                     "visibility": "private", "scheduled_at": when},
                ],
            },
        )
        assert response.status_code == 201, response.text
        jobs = response.json()["jobs"]
        assert len(jobs) == 2
        assert {job["account_id"] for job in jobs} == {"yt-main", "yt-second"}
        assert jobs[0]["scheduled_at"] != jobs[1]["scheduled_at"]


def test_cross_publish_rejects_duplicate_target_and_unknown_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _idle_workers(monkeypatch)
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        project_id = str(uuid.uuid4())
        _create_completed_project_with_clip(settings, project_id)
        _save_account(settings, "youtube", "yt-main")
        duplicate = client.post(
            f"/api/projects/{project_id}/cross-publish",
            json={
                "clip_ids": ["clip-1"],
                "targets": [
                    {"platform": "youtube", "visibility": "private"},
                    {"platform": "youtube", "visibility": "private"},
                ],
            },
        )
        assert duplicate.status_code == 422
        unknown = client.post(
            f"/api/projects/{project_id}/cross-publish",
            json={
                "clip_ids": ["clip-1"],
                "targets": [
                    {"platform": "youtube", "account_id": "yt-nope",
                     "visibility": "private"},
                ],
            },
        )
        assert unknown.status_code == 409


def test_accounts_endpoint_lists_summaries_without_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _idle_workers(monkeypatch)
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        _save_account(settings, "youtube", "yt-main", "Principal")
        database.save_account_document(
            settings.database_path, "youtube", "yt-second", "Segundo",
            "token-yt-second", None, None, {}, "test-user",
        )
        response = client.get("/api/integrations/accounts")
        assert response.status_code == 200
        assert "token-yt-second" not in response.text
        assert "access_token" not in response.text
        accounts = response.json()["youtube"]
        assert [item["account_id"] for item in accounts] == [
            "yt-main",
            "yt-second",
        ]
        made_default = client.post(
            "/api/integrations/youtube/accounts/yt-second/set-default"
        )
        assert made_default.status_code == 200
        assert made_default.json()["youtube"][0]["account_id"] == "yt-second"
        deleted = client.delete("/api/integrations/youtube/accounts/yt-main")
        assert deleted.status_code == 204


# ---------------------------------------------------------------------------
# Ítem 11: cola por lotes + carpeta observada + procesar de noche
# ---------------------------------------------------------------------------


def test_night_window_handles_overnight_ranges() -> None:
    assert publish_guide.within_window(time(23, 0), time(22, 0), time(6, 0))
    assert publish_guide.within_window(time(3, 0), time(22, 0), time(6, 0))
    assert not publish_guide.within_window(time(12, 0), time(22, 0), time(6, 0))
    assert not publish_guide.within_window(time(6, 0), time(22, 0), time(6, 0))
    assert publish_guide.within_window(time(12, 0), time(12, 0), time(12, 0))
    with pytest.raises(ValueError, match="HH:MM"):
        publish_guide.parse_hhmm("25:00")


def test_night_window_preference_round_trip(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        initial = client.get("/api/preferences/night-window")
        assert initial.status_code == 200
        assert initial.json()["enabled"] is False
        updated = client.put(
            "/api/preferences/night-window",
            json={"enabled": True, "start": "23:00", "end": "05:00"},
        )
        assert updated.status_code == 200
        assert updated.json() == {
            "enabled": True,
            "start": "23:00",
            "end": "05:00",
        }
        invalid = client.put(
            "/api/preferences/night-window",
            json={"enabled": True, "start": "23:60", "end": "05:00"},
        )
        assert invalid.status_code == 422


def test_processing_gate_holds_jobs_outside_night_window(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    _create_project(settings, project_id)

    async def run() -> None:
        gate_state = {"allowed": False}
        processed = asyncio.Event()

        async def processor(_pid: str, _llm: Any, _retry: bool) -> None:
            processed.set()

        queue = ProjectProcessingQueue(
            1,
            processor,
            gate=lambda: gate_state["allowed"],
        )
        queue._gate_wait_seconds = 0.05  # noqa: SLF001 - ajuste del test
        await queue.start()
        await queue.submit(project_id, None)
        await asyncio.sleep(0.2)
        assert not processed.is_set()
        gate_state["allowed"] = True
        await asyncio.wait_for(processed.wait(), timeout=5)
        await queue.join()
        await queue.stop()

    asyncio.run(run())


def test_watch_folder_detects_only_new_videos(tmp_path: Path) -> None:
    folder = tmp_path / "watched"
    folder.mkdir()
    (folder / "video.mp4").write_bytes(b"a" * 10)
    (folder / "notas.txt").write_text("no es video")
    (folder / ".oculto.mp4").write_bytes(b"a" * 10)
    (folder / "otro.MOV").write_bytes(b"b" * 5)
    found = watch_folder_module.detect_new_videos(folder, set())
    assert [path.name for path in found] == ["otro.MOV", "video.mp4"]
    seen = {watch_folder_module.file_key(folder / "video.mp4")}
    found = watch_folder_module.detect_new_videos(folder, seen)
    assert [path.name for path in found] == ["otro.MOV"]


def test_watch_folder_scan_needs_two_stable_scans(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    folder_path = tmp_path / "watched"
    folder_path.mkdir()
    (folder_path / "video.mp4").write_bytes(b"a" * 10)
    folder = database.create_watch_folder(
        settings.database_path,
        "test-user",
        str(folder_path),
        config=ProjectConfig().model_dump(mode="json"),
    )

    async def run() -> None:
        submitted: list[str] = []

        async def submit(project_id: str) -> None:
            submitted.append(project_id)

        created = await watch_folder_module.scan_folder_once(
            settings, folder, submit
        )
        assert created == 0
        assert submitted == []
        created = await watch_folder_module.scan_folder_once(
            settings, folder, submit
        )
        assert created == 1
        assert len(submitted) == 1
        project = database.get_project(settings.database_path, submitted[0])
        assert project is not None
        assert project["title"] == "video"
        # Tercer sondeo: nada nuevo.
        created = await watch_folder_module.scan_folder_once(
            settings, folder, submit
        )
        assert created == 0
        # Un archivo que crece entre sondeos vuelve a ser candidato.
        (folder_path / "video.mp4").write_bytes(b"a" * 20)
        created = await watch_folder_module.scan_folder_once(
            settings, folder, submit
        )
        assert created == 0
        created = await watch_folder_module.scan_folder_once(
            settings, folder, submit
        )
        assert created == 1

    asyncio.run(run())


def test_batch_sweep_converts_ready_staged_into_projects(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    staged_id = str(uuid.uuid4())
    config = ProjectConfig().model_dump(mode="json")
    database.create_staged_media(
        settings.database_path,
        staged_id,
        "url",
        "ready",
        "video.mp4",
        config=config,
        owner_id="test-user",
    )
    staging_path = settings.data_dir / "staging" / f"{staged_id}.source"
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path.write_bytes(b"video-bytes")
    batch_id = str(uuid.uuid4())
    item = database.create_batch_item(
        settings.database_path,
        batch_id,
        "test-user",
        staged_id=staged_id,
        url="https://example.invalid/video",
        status="downloading",
    )

    async def run() -> None:
        submitted: list[str] = []

        async def submit(project_id: str) -> None:
            submitted.append(project_id)

        created = await watch_folder_module.sweep_batch_items(settings, submit)
        assert created == 1
        assert len(submitted) == 1
        project = database.get_project(settings.database_path, submitted[0])
        assert project is not None
        items = database.list_batch_items(settings.database_path, "test-user")
        assert items[0]["id"] == item["id"]
        assert items[0]["status"] == "done"
        assert items[0]["project_id"] == submitted[0]

    asyncio.run(run())


def test_batch_endpoint_enqueues_staged_and_rejects_bad_urls(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    app = create_app(settings)
    staged_id = str(uuid.uuid4())
    config = ProjectConfig().model_dump(mode="json")
    database.create_staged_media(
        settings.database_path,
        staged_id,
        "upload",
        "ready",
        "video.mp4",
        config=config,
        owner_id="test-user",
    )
    staging_path = settings.data_dir / "staging" / f"{staged_id}.source"
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path.write_bytes(b"video-bytes")

    async def run() -> None:
        processed: list[str] = []

        async def processor(pid: str, _llm: Any, _retry: bool) -> None:
            processed.append(pid)
            database.update_project(
                settings.database_path, pid, status="completed"
            )

        queue = ProjectProcessingQueue(
            1, processor, database_path=settings.database_path
        )
        await queue.start()
        app.state.processing_queue = queue

        async def submit_background(project_id: str) -> None:
            project = database.get_project(settings.database_path, project_id)
            assert project is not None
            assert database.queue_project(
                settings.database_path,
                project_id,
                project["config"],
            )
            await queue.submit(project_id, None)

        app.state.submit_background_project = submit_background
        client = TestClient(app)
        response = await asyncio.to_thread(
            client.post,
            "/api/queue/batch",
            json={
                "staged_ids": [staged_id, "desaparecido"],
                "urls": ["nota-url"],
            },
        )
        assert response.status_code == 201, response.text
        items = response.json()["items"]
        assert items[0]["status"] == "done"
        assert items[0]["project_id"]
        assert items[1]["status"] == "failed"
        assert items[2]["status"] == "failed"
        await queue.join()
        assert processed == [items[0]["project_id"]]
        await queue.stop()

    asyncio.run(run())


def test_global_queue_lists_jobs_with_project_titles(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        project_id = str(uuid.uuid4())
        _create_project(settings, project_id)
        database.enqueue_processing_job(settings.database_path, project_id)
        response = client.get("/api/queue")
        assert response.status_code == 200
        jobs = response.json()
        assert len(jobs) == 1
        assert jobs[0]["project_id"] == project_id
        assert jobs[0]["project_title"] == f"Proyecto {project_id}"
        assert jobs[0]["status"] == "pending"


def test_watch_folder_crud_endpoints(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    watched = tmp_path / "watched"
    watched.mkdir()
    with TestClient(create_app(settings)) as client:
        missing = client.post(
            "/api/watch-folders",
            json={"path": str(tmp_path / "no-existe")},
        )
        assert missing.status_code == 422
        created = client.post(
            "/api/watch-folders",
            json={"path": str(watched), "poll_seconds": 120},
        )
        assert created.status_code == 201, created.text
        folder_id = created.json()["id"]
        assert created.json()["enabled"] is True
        duplicate = client.post("/api/watch-folders", json={"path": str(watched)})
        assert duplicate.status_code == 409
        listed = client.get("/api/watch-folders")
        assert len(listed.json()) == 1
        updated = client.put(
            f"/api/watch-folders/{folder_id}",
            json={"enabled": False},
        )
        assert updated.status_code == 200
        assert updated.json()["enabled"] is False
        scanned = client.post(f"/api/watch-folders/{folder_id}/scan")
        assert scanned.status_code == 200
        assert scanned.json()["created"] == 0
        deleted = client.delete(f"/api/watch-folders/{folder_id}")
        assert deleted.status_code == 204
        assert client.get("/api/watch-folders").json() == []


# ---------------------------------------------------------------------------
# Ítem 12: generador de títulos/descripciones/hashtags
# ---------------------------------------------------------------------------


def test_caption_variants_normalize_and_pad_from_clip_data() -> None:
    clips = [
        {
            "id": "clip-1",
            "title": "Título original",
            "description": "Descripción original",
            "viral_reason": "Gancho",
            "keywords": ["precio"],
            "emojis": ["🔥"],
            "hashtags": {"youtube": ["precios"], "tiktok": [], "instagram": []},
        }
    ]
    parsed = {
        "clips": [
            {
                "id": "clip-1",
                "titles": ["Solo una variante", "con #hashtag raro!"],
                "description": "",
                "hashtags": {"youtube": ["Precios", "precios", "mal tag!"]},
            }
        ]
    }
    result = captions_module.normalize_caption_response(clips, parsed)
    variant = result["clip-1"]
    assert variant["titles"][0] == "Solo una variante"
    assert "Título original" in variant["titles"]
    assert len(variant["titles"]) == 3
    assert variant["description"] == "Descripción original"
    assert variant["hashtags"]["youtube"] == ["Precios"]
    assert variant["hashtags"]["tiktok"] == []


def test_generate_captions_endpoint_stores_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _idle_workers(monkeypatch)
    settings = _settings(tmp_path / "data")

    def fake_completion(_llm: Any, _system: str, _user: str) -> str:
        return json.dumps(
            {
                "clips": [
                    {
                        "id": "clip-1",
                        "titles": ["A", "B", "C"],
                        "description": "Nueva descripción",
                        "hashtags": {
                            "youtube": ["uno"],
                            "tiktok": ["dos"],
                            "instagram": ["tres"],
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr(
        captions_module, "request_text_completion", fake_completion
    )
    with TestClient(create_app(settings)) as client:
        project_id = str(uuid.uuid4())
        _create_completed_project_with_clip(settings, project_id)
        response = client.post(
            f"/api/projects/{project_id}/generate-captions",
            json={},
        )
        assert response.status_code == 200, response.text
        variant = response.json()["variants"]["clip-1"]
        assert variant["titles"] == ["A", "B", "C"]
        assert variant["description"] == "Nueva descripción"
        project = database.get_project(settings.database_path, project_id)
        assert project is not None
        stored = (project.get("metadata") or {}).get("caption_variants") or {}
        assert stored["clip-1"]["titles"] == ["A", "B", "C"]
        applied = client.put(
            f"/api/projects/{project_id}/clips/clip-1/caption",
            json={
                "title": "B",
                "description": "Nueva descripción",
                "hashtags": {"youtube": ["uno"], "tiktok": [], "instagram": []},
            },
        )
        assert applied.status_code == 200
        assert applied.json()["title"] == "B"
        assert applied.json()["description"] == "Nueva descripción"


def test_generate_captions_without_llm_key_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _idle_workers(monkeypatch)
    settings = _settings(tmp_path / "data", llm_api_key="")
    with TestClient(create_app(settings)) as client:
        project_id = str(uuid.uuid4())
        _create_completed_project_with_clip(settings, project_id)
        response = client.post(
            f"/api/projects/{project_id}/generate-captions",
            json={},
        )
        assert response.status_code == 409
        assert "Ajustes" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Ítem 13: biblioteca de B-roll
# ---------------------------------------------------------------------------


def test_broll_search_needs_provider_key(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/broll/search",
            json={"provider": "pexels", "query": "ciudad"},
        )
        assert response.status_code == 422
        assert "PEXELS_API_KEY" in response.json()["detail"]


def test_broll_search_returns_provider_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import backend.integrations as integrations_module

    settings = _settings(tmp_path / "data", pexels_api_key="pexels-key")

    def fake_request_json(
        _client: Any, _method: str, _url: str, _purpose: str, **_kwargs: Any
    ) -> dict[str, Any]:
        return {
            "videos": [
                {
                    "id": 123,
                    "duration": 12,
                    "url": "https://www.pexels.com/video/123/",
                    "image": "https://images.pexels.com/123.jpg",
                    "user": {"name": "Autora"},
                    "video_files": [
                        {
                            "file_type": "video/mp4",
                            "width": 1280,
                            "height": 720,
                            "link": "https://videos.pexels.com/123.mp4",
                        }
                    ],
                }
            ]
        }

    monkeypatch.setattr(integrations_module, "_request_json", fake_request_json)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/broll/search",
            json={"provider": "pexels", "query": "ciudad"},
        )
        assert response.status_code == 200, response.text
        results = response.json()["results"]
        assert len(results) == 1
        assert results[0]["external_id"] == "123"
        assert results[0]["creator"] == "Autora"
        assert results[0]["download_url"] == "https://videos.pexels.com/123.mp4"


def test_broll_import_rejects_foreign_download_urls(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/broll/import",
            json={
                "provider": "pexels",
                "title": "Video",
                "download_url": "https://evil.example/video.mp4",
            },
        )
        assert response.status_code == 422


def test_broll_import_downloads_into_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path / "data")

    class _FakeStream:
        is_error = False
        status_code = 200

        def __init__(self, data: bytes) -> None:
            self._data = data

        def __enter__(self) -> _FakeStream:
            return self

        def __exit__(self, *_args: Any) -> bool:
            return False

        def iter_bytes(self, _size: int) -> Any:
            yield self._data

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> _FakeClient:
            return self

        def __exit__(self, *_args: Any) -> bool:
            return False

        def stream(self, *args: Any, **kwargs: Any) -> _FakeStream:
            return _FakeStream(b"fake-video-bytes")

    with TestClient(create_app(settings)) as client:
        assert client.get("/api/health").status_code == 200
        with monkeypatch.context() as patch:
            patch.setattr(httpx, "Client", _FakeClient)
            response = client.post(
                "/api/broll/import",
                json={
                    "provider": "pexels",
                    "external_id": "123",
                    "title": "Ciudad de noche",
                    "query": "ciudad",
                    "creator": "Autora",
                    "attribution_url": "https://www.pexels.com/video/123/",
                    "download_url": "https://videos.pexels.com/123.mp4",
                    "duration": 12,
                    "width": 1280,
                    "height": 720,
                },
            )
        assert response.status_code == 201, response.text
        item = response.json()
        assert item["title"] == "Ciudad de noche"
        assert item["favorite"] is False
        assert Path(item["local_path"]).is_file()
        listed = client.get("/api/broll?q=ciudad")
        assert len(listed.json()) == 1
        favorited = client.put(
            f"/api/broll/{item['id']}", json={"favorite": True}
        )
        assert favorited.status_code == 200
        assert favorited.json()["favorite"] is True
        only_favorites = client.get("/api/broll?favorites=true")
        assert len(only_favorites.json()) == 1
        streamed = client.get(f"/api/broll/{item['id']}/file")
        assert streamed.status_code == 200
        assert streamed.content == b"fake-video-bytes"
        deleted = client.delete(f"/api/broll/{item['id']}")
        assert deleted.status_code == 204
        assert not Path(item["local_path"]).exists()


def test_broll_segment_assignment_and_clip_lookup(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        project_id = str(uuid.uuid4())
        _create_completed_project_with_clip(settings, project_id)
        broll_dir = settings.data_dir / "broll"
        broll_dir.mkdir(parents=True, exist_ok=True)
        local_path = broll_dir / "local.mp4"
        local_path.write_bytes(b"broll-bytes")
        item = database.create_broll_item(
            settings.database_path,
            "test-user",
            title="Local",
            local_path=str(local_path),
        )
        assigned = client.post(
            "/api/broll/assign",
            json={
                "broll_id": item["id"],
                "project_id": project_id,
                "clip_id": "clip-1",
                "start": 2.0,
                "end": 5.0,
                "label": "cuando hablo de precios",
            },
        )
        assert assigned.status_code == 201, assigned.text
        segment_id = assigned.json()["id"]
        segments = client.get(
            f"/api/projects/{project_id}/broll?clip_id=clip-1"
        )
        assert segments.status_code == 200
        assert len(segments.json()) == 1
        assert segments.json()[0]["label"] == "cuando hablo de precios"
        found = database.broll_assignment_for_clip(
            settings.database_path, project_id, "clip-1"
        )
        assert found is not None
        assert found["broll_id"] == item["id"]
        removed = client.delete(f"/api/broll/segments/{segment_id}")
        assert removed.status_code == 204
        assert (
            database.broll_assignment_for_clip(
                settings.database_path, project_id, "clip-1"
            )
            is None
        )


# ---------------------------------------------------------------------------
# Ítem 14: biblioteca de momentos guardados
# ---------------------------------------------------------------------------


def test_saved_moments_crud_and_apply_to_project(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        source_id = str(uuid.uuid4())
        _create_completed_project_with_clip(settings, source_id)
        target_id = str(uuid.uuid4())
        _create_project(settings, target_id)
        created = client.post(
            "/api/saved-moments",
            json={
                "title": "cuando hablo de precios",
                "description": "Momentos en los que menciono precios",
                "project_id": source_id,
                "clip_id": "clip-1",
                "start": 2.0,
                "end": 8.0,
                "excerpt": "y esto cuesta...",
                "tags": ["precios"],
            },
        )
        assert created.status_code == 201, created.text
        moment_id = created.json()["id"]
        assert created.json()["tags"] == ["precios"]
        found = client.get("/api/saved-moments?q=precios")
        assert len(found.json()) == 1
        applied = client.post(
            f"/api/saved-moments/{moment_id}/apply",
            json={"project_id": target_id},
        )
        assert applied.status_code == 200, applied.text
        target = database.get_project(settings.database_path, target_id)
        assert target is not None
        assert "cuando hablo de precios" in str(target["config"].get("moment"))
        renamed = client.put(
            f"/api/saved-moments/{moment_id}",
            json={"title": "precios"},
        )
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "precios"
        deleted = client.delete(f"/api/saved-moments/{moment_id}")
        assert deleted.status_code == 204
        assert client.get("/api/saved-moments").json() == []


# ---------------------------------------------------------------------------
# Ítem 15: duplicar proyecto / reprocesar con otra configuración
# ---------------------------------------------------------------------------


def test_duplicate_project_clones_source_and_config(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        project_id = str(uuid.uuid4())
        _create_completed_project_with_clip(settings, project_id)
        transcript = [{"start": 0.0, "end": 5.0, "text": "hola"}]
        database.update_project(
            settings.database_path,
            project_id,
            transcript=transcript,
        )
        response = client.post(
            f"/api/projects/{project_id}/duplicate",
            json={"ratio": "16:9", "duration": "30–60s"},
        )
        assert response.status_code == 201, response.text
        duplicated = response.json()
        assert duplicated["id"] != project_id
        assert duplicated["title"].endswith("(copia)")
        assert duplicated["config"]["ratio"] == "16:9"
        assert duplicated["config"]["duration"] == "30–60s"
        assert duplicated["clips"] == []
        assert duplicated["status"] == "ready"
        stored = database.get_project(settings.database_path, duplicated["id"])
        assert stored is not None
        assert stored.get("transcript") == transcript
        new_source = (
            settings.data_dir / "projects" / duplicated["id"] / "source"
        )
        assert new_source.is_file()
        assert new_source.read_bytes() == b"video-bytes"
        # Borrar el original no rompe la copia: la fuente está copiada.
        client.delete(f"/api/projects/{project_id}")
        assert new_source.is_file()


def test_duplicate_project_rejects_bad_ratio_and_missing_source(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        project_id = str(uuid.uuid4())
        project_dir = _create_completed_project_with_clip(settings, project_id)
        bad_ratio = client.post(
            f"/api/projects/{project_id}/duplicate",
            json={"ratio": "4:3"},
        )
        assert bad_ratio.status_code == 422
        (settings.data_dir / "projects" / project_id / "source").unlink()
        missing = client.post(
            f"/api/projects/{project_id}/duplicate",
            json={},
        )
        assert missing.status_code == 404
        assert project_dir.is_file()  # Solo se borró la fuente, no el clip.


def test_duplicate_and_process_enqueues_with_fake_queue(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    app = create_app(settings)
    project_id = str(uuid.uuid4())
    _create_completed_project_with_clip(settings, project_id)

    async def run() -> None:
        processed: list[str] = []

        async def processor(pid: str, _llm: Any, _retry: bool) -> None:
            processed.append(pid)
            database.update_project(
                settings.database_path, pid, status="completed"
            )

        queue = ProjectProcessingQueue(
            1, processor, database_path=settings.database_path
        )
        await queue.start()
        app.state.processing_queue = queue

        async def submit_background(duplicate_id: str) -> None:
            project = database.get_project(settings.database_path, duplicate_id)
            assert project is not None
            assert database.queue_project(
                settings.database_path,
                duplicate_id,
                project["config"],
            )
            await queue.submit(duplicate_id, None)

        app.state.submit_background_project = submit_background
        client = TestClient(app)
        response = await asyncio.to_thread(
            client.post,
            f"/api/projects/{project_id}/duplicate",
            json={"process": True},
        )
        assert response.status_code == 201, response.text
        new_id = response.json()["id"]
        await queue.join()
        assert processed == [new_id]
        await queue.stop()

    asyncio.run(run())


def test_migration_keeps_old_schedule_jobs_working(tmp_path: Path) -> None:
    """Una BD creada antes de la Fase 3 (sin las columnas nuevas) se
    actualiza sola al abrirla: los jobs viejos siguen ahí y se pueden
    reclamar."""
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    _create_project(settings, project_id)
    jobs = database.enqueue_schedule_jobs(
        settings.database_path,
        project_id,
        ["clip-1"],
        "youtube",
        _future_iso(days=-1),
        "private",
    )
    # Simular una BD vieja: quitar las columnas añadidas en la Fase 3.
    with sqlite3.connect(settings.database_path) as conn:
        conn.execute("ALTER TABLE scheduled_jobs DROP COLUMN attempts")
        conn.execute("ALTER TABLE scheduled_jobs DROP COLUMN next_attempt_at")
        conn.execute("ALTER TABLE scheduled_jobs DROP COLUMN account_id")
        conn.execute("ALTER TABLE scheduled_jobs DROP COLUMN group_id")
        conn.commit()
    database.initialize(settings.database_path)
    job = database.get_schedule_job(settings.database_path, jobs[0]["id"])
    assert job is not None
    assert job["attempts"] == 0
    assert job["next_attempt_at"] is None
    assert database.claim_schedule_job(settings.database_path, jobs[0]["id"])
