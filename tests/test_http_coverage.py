from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest

from backend import database, integrations, llm, pipeline
from backend.llm import ResolvedLLM
from backend.schemas import ProjectConfig, TranscriptSegment, Word
from backend.settings import AppSettings


def _settings(data_dir: Path, **overrides: Any) -> AppSettings:
    settings = AppSettings(
        data_dir=data_dir,
        cors_origins=(),
        llm_provider="Groq",
        llm_api_key="",
        llm_model="",
        llm_base_url="",
        whisper_model="small",
        youtube_client_id="youtube-client",
        youtube_client_secret="youtube-secret",
        tiktok_client_key="tiktok-client",
        tiktok_client_secret="tiktok-secret",
        meta_app_id="meta-client",
        meta_app_secret="meta-secret",
        zoom_client_id="zoom-client",
        zoom_client_secret="zoom-secret",
        pexels_api_key="pexels-key",
        pixabay_api_key="pixabay-key",
    )
    return replace(settings, **overrides)


def _install_transport(monkeypatch, module, handler) -> None:
    client_type = httpx.Client
    monkeypatch.setattr(
        module.httpx,
        "Client",
        lambda *args, **kwargs: client_type(
            *args,
            transport=httpx.MockTransport(handler),
            **kwargs,
        ),
    )


@pytest.mark.parametrize(
    ("provider", "expected_account_id"),
    [
        ("youtube", "youtube-channel"),
        ("tiktok", "tiktok-user"),
        ("instagram", "instagram-user"),
        ("zoom", "zoom-user"),
    ],
)
def test_oauth_exchange_and_identity_use_provider_http_flows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider: str,
    expected_account_id: str,
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if provider == "youtube" and path.endswith("/token"):
            return httpx.Response(
                200,
                json={"access_token": "yt-access", "refresh_token": "yt-refresh", "expires_in": 3600},
            )
        if provider == "youtube":
            return httpx.Response(
                200,
                json={"items": [{"id": expected_account_id, "snippet": {"title": "Canal"}}]},
            )
        if provider == "tiktok" and path.endswith("/oauth/token/"):
            return httpx.Response(
                200,
                json={"data": {"access_token": "tt-access", "refresh_token": "tt-refresh"}},
            )
        if provider == "tiktok":
            return httpx.Response(
                200,
                json={"data": {"user": {"open_id": expected_account_id, "display_name": "Cuenta"}}},
            )
        if provider == "instagram" and path.endswith("/oauth/access_token"):
            if request.url.params.get("grant_type") == "fb_exchange_token":
                return httpx.Response(200, json={"access_token": "ig-user-long"})
            return httpx.Response(200, json={"access_token": "ig-user-short"})
        if provider == "instagram":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "facebook-page",
                            "name": "Página",
                            "access_token": "ig-page",
                            "instagram_business_account": {
                                "id": expected_account_id,
                                "username": "usuario",
                            },
                        }
                    ]
                },
            )
        if path.endswith("/oauth/token"):
            return httpx.Response(
                200,
                json={"access_token": "zoom-access", "refresh_token": "zoom-refresh"},
            )
        return httpx.Response(
            200,
            json={"id": expected_account_id, "email": "zoom@example.test"},
        )

    _install_transport(monkeypatch, integrations, handle)
    account = integrations.complete_oauth(
        _settings(tmp_path / "data"),
        provider,
        "authorization-code",
    )

    assert account["provider"] == provider
    assert account["account_id"] == expected_account_id
    assert account["access_token"]
    assert len(requests) >= 2
    if provider == "instagram":
        assert account["metadata"]["user_access_token"] == "ig-user-long"


@pytest.mark.parametrize(
    ("provider", "expected_access_token"),
    [
        ("youtube", "youtube-new"),
        ("tiktok", "tiktok-new"),
        ("instagram", "instagram-new"),
        ("zoom", "zoom-new"),
    ],
)
def test_expired_provider_accounts_refresh_and_persist_tokens(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider: str,
    expected_access_token: str,
) -> None:
    saved: list[tuple[Any, ...]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if provider == "youtube":
            return httpx.Response(200, json={"access_token": expected_access_token})
        if provider == "tiktok":
            return httpx.Response(
                200,
                json={"data": {"access_token": expected_access_token, "refresh_token": "next-refresh"}},
            )
        if provider == "zoom":
            return httpx.Response(
                200,
                json={"access_token": expected_access_token, "refresh_token": "next-refresh"},
            )
        if request.url.path.endswith("/oauth/access_token"):
            return httpx.Response(200, json={"access_token": "instagram-user-new"})
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "access_token": expected_access_token,
                        "instagram_business_account": {"id": "account-id"},
                    }
                ]
            },
        )

    monkeypatch.setattr(
        integrations.database,
        "save_connected_account",
        lambda *args: saved.append(args),
    )
    _install_transport(monkeypatch, integrations, handle)
    account = {
        "account_id": "account-id",
        "display_name": "Cuenta",
        "access_token": "old-access",
        "refresh_token": None if provider == "instagram" else "old-refresh",
        "metadata": {"user_access_token": "old-user"} if provider == "instagram" else {},
    }

    refreshed = integrations._refresh_account(
        _settings(tmp_path / "data"),
        provider,
        account,
        "owner",
    )

    assert refreshed["access_token"] == expected_access_token
    assert saved[0][1:5] == (
        provider,
        "account-id",
        "Cuenta",
        expected_access_token,
    )
    if provider == "instagram":
        assert refreshed["metadata"]["user_access_token"] == "instagram-user-new"


def test_tiktok_publish_uploads_file_and_confirms_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(b"real-mp4")
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/creator_info/query/"):
            return httpx.Response(
                200,
                json={"data": {"privacy_level_options": ["SELF_ONLY"]}},
            )
        if request.url.path.endswith("/video/init/"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "upload_url": "https://open-upload.tiktokapis.com/upload/1",
                        "publish_id": "publish-1",
                    }
                },
            )
        if request.method == "PUT":
            assert request.content == b"real-mp4"
            assert request.headers["Content-Range"] == "bytes 0-7/8"
            return httpx.Response(200)
        return httpx.Response(
            200,
            json={"data": {"status": "PUBLISH_COMPLETE"}},
        )

    monkeypatch.setattr(
        integrations,
        "_access_token",
        lambda *_args: ("access-token", {"account_id": "account"}),
    )
    monkeypatch.setattr(integrations.time, "sleep", lambda _seconds: None)
    _install_transport(monkeypatch, integrations, handle)

    result = integrations.publish_clip(
        _settings(tmp_path / "data"),
        "tiktok",
        {
            "title": "Consejo",
            "description": "Un consejo práctico",
            "hashtags": {"tiktok": ["Aprender"]},
        },
        file_path,
        "private",
    )

    assert result == {"external_id": "publish-1", "url": ""}
    init_payload = json.loads(requests[1].content)
    assert init_payload["post_info"]["title"] == "Un consejo práctico\n\n#Aprender"
    assert len(requests) == 4


def test_instagram_publish_creates_uploads_and_publishes_reel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(b"real-mp4")
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/media_publish"):
            return httpx.Response(200, json={"id": "instagram-media"})
        if request.url.host == "rupload.facebook.com":
            assert request.headers["Authorization"] == "OAuth access-token"
            return httpx.Response(200, json={"success": True})
        if request.method == "GET":
            return httpx.Response(200, json={"status_code": "FINISHED"})
        return httpx.Response(200, json={"id": "container-1"})

    monkeypatch.setattr(
        integrations,
        "_access_token",
        lambda *_args: ("access-token", {"account_id": "instagram-account"}),
    )
    monkeypatch.setattr(integrations.time, "sleep", lambda _seconds: None)
    _install_transport(monkeypatch, integrations, handle)

    result = integrations.publish_clip(
        _settings(tmp_path / "data"),
        "instagram",
        {
            "title": "Consejo",
            "description": "Una idea útil",
            "hashtags": {"instagram": ["Ideas"]},
        },
        file_path,
        "public",
    )

    assert result == {"external_id": "instagram-media", "url": ""}
    assert requests[0].method == "POST"
    caption = parse_qs(requests[0].content.decode())["caption"][0]
    assert caption == "Una idea útil\n\n#Ideas"
    assert [request.url.path.rsplit("/", 1)[-1] for request in requests] == [
        "media",
        "container-1",
        "container-1",
        "media_publish",
    ]


def test_zoom_recordings_are_filtered_and_selected_file_downloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == "cdn.zoom.us":
            return httpx.Response(200, content=b"zoom-mp4")
        if "/meetings/" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "recording_files": [
                        {
                            "id": "file-1",
                            "file_type": "MP4",
                            "status": "completed",
                            "download_url": "https://cdn.zoom.us/file.mp4",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "meetings": [
                    {
                        "uuid": "meeting-1",
                        "topic": "Reunión",
                        "recording_files": [
                            {
                                "id": "file-1",
                                "file_type": "MP4",
                                "status": "completed",
                                "file_size": 8,
                            },
                            {
                                "id": "audio-1",
                                "file_type": "M4A",
                                "status": "completed",
                            },
                        ],
                    }
                ]
            },
        )

    monkeypatch.setattr(
        integrations,
        "_access_token",
        lambda *_args: ("zoom-token", {}),
    )
    _install_transport(monkeypatch, integrations, handle)
    settings = _settings(tmp_path / "data")

    recordings = integrations.list_zoom_recordings(settings, "owner")
    target_path = tmp_path / "recordings" / "zoom.mp4"
    integrations.download_zoom_recording(
        settings,
        "meeting-1",
        "file-1",
        target_path,
        "owner",
    )

    assert recordings == [
        {
            "meeting_id": "meeting-1",
            "topic": "Reunión",
            "start_time": None,
            "files": [
                {
                    "file_id": "file-1",
                    "file_type": "MP4",
                    "recording_type": "",
                    "file_size": 8,
                }
            ],
        }
    ]
    assert target_path.read_bytes() == b"zoom-mp4"


@pytest.mark.parametrize(
    ("provider", "search_host", "download_url"),
    [
        (
            "Pexels",
            "api.pexels.com",
            "https://videos.pexels.com/video-files/1/clip.mp4",
        ),
        (
            "Pixabay",
            "pixabay.com",
            "https://cdn.pixabay.com/video/1/clip.mp4",
        ),
    ],
)
def test_broll_search_and_download_use_mocked_provider_http(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider: str,
    search_host: str,
    download_url: str,
) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host != search_host:
            assert str(request.url) == download_url
            return httpx.Response(200, content=b"broll-mp4")
        if provider == "Pexels":
            return httpx.Response(
                200,
                json={
                    "videos": [
                        {
                            "url": "https://www.pexels.com/video/1",
                            "user": {"name": "Creador"},
                            "video_files": [
                                {
                                    "file_type": "video/mp4",
                                    "width": 640,
                                    "link": download_url,
                                }
                            ],
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "hits": [
                    {
                        "pageURL": "https://pixabay.com/videos/1",
                        "user": "Creador",
                        "videos": {"medium": {"url": download_url}},
                    }
                ]
            },
        )

    settings = _settings(
        tmp_path / "data",
        broll_provider=provider,
    )
    _install_transport(monkeypatch, integrations, handle)
    target = tmp_path / "broll" / "clip.mp4"

    attribution = integrations.download_broll(settings, "  ciencia   aplicada ", target)

    assert attribution == {
        "provider": provider,
        "creator": "Creador",
        "attribution_url": (
            "https://www.pexels.com/video/1"
            if provider == "Pexels"
            else "https://pixabay.com/videos/1"
        ),
    }
    assert target.read_bytes() == b"broll-mp4"


def test_anthropic_clip_proposal_uses_messages_api_with_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "content": [
                    {
                        "type": "text",
                        "text": '```json\n[{"inicio": 1, "fin": 4}]\n```',
                    }
                ]
            },
        )

    _install_transport(monkeypatch, llm, handle)
    result = llm.propose_clips(
        [],
        ProjectConfig(),
        10,
        ResolvedLLM(
            "Anthropic",
            "anthropic-key",
            "claude-test",
            "https://anthropic.example",
        ),
    )

    assert result == [{"inicio": 1, "fin": 4}]
    assert requests[0].url.path == "/v1/messages"
    assert requests[0].headers["x-api-key"] == "anthropic-key"
    assert json.loads(requests[0].content)["model"] == "claude-test"


def test_groq_gpt_oss_json_request_hides_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"clips": []}'}}]},
        )

    _install_transport(monkeypatch, llm, handle)
    proposals = llm.propose_clips(
        [],
        ProjectConfig(),
        60,
        ResolvedLLM(
            "Groq",
            "test-key",
            "openai/gpt-oss-120b",
            "https://api.groq.com/openai/v1",
        ),
    )

    assert proposals == []
    request_payload = json.loads(requests[0].content)
    assert request_payload["response_format"] == {"type": "json_object"}
    assert request_payload["reasoning_format"] == "hidden"
    assert 'objeto JSON con una única clave "clips"' in request_payload["messages"][0]["content"]
    assert "o con un array JSON" not in request_payload["messages"][1]["content"]


@pytest.mark.parametrize(
    ("clips", "expected_clip_count"),
    [
        (
            [
                {
                    "inicio": 0,
                    "fin": 3,
                    "titulo": "Idea completa",
                    "puntuacion": 8,
                    "razon_viral": "Una conclusión útil.",
                    "descripcion": "Una idea práctica.",
                    "hashtags": {"youtube": ["Ideas"]},
                    "palabras_clave": ["idea"],
                    "emojis": [],
                }
            ],
            1,
        ),
        ([], 0),
    ],
)
def test_process_project_runs_llm_http_and_persists_real_pipeline_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clips: list[dict[str, Any]],
    expected_clip_count: int,
) -> None:
    settings = _settings(
        tmp_path / "data",
        llm_provider="OpenAI",
        llm_api_key="test-key",
        llm_model="test-model",
        llm_base_url="https://llm.example/v1",
    )
    database.initialize(settings.database_path)
    project_id = "http-pipeline"
    project_dir = settings.data_dir / "projects" / project_id
    input_path = project_dir / "source"
    input_path.parent.mkdir(parents=True)
    input_path.write_bytes(b"video")
    config = ProjectConfig()
    database.create_project(
        settings.database_path,
        project_id,
        "Pipeline",
        input_path,
        config.model_dump(mode="json"),
        "test-owner",
    )
    database.update_project(
        settings.database_path,
        project_id,
        metadata={"duration": 12},
        transcript=[
            TranscriptSegment(
                start=0,
                end=3,
                text="Una idea útil.",
                words=[
                    Word(start=1, end=2, text="idea"),
                    Word(start=2, end=3, text="útil."),
                ],
            ).model_dump(mode="json")
        ],
    )
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"clips": clips}, ensure_ascii=False)
                        }
                    }
                ]
            },
        )

    _install_transport(monkeypatch, llm, handle)
    monkeypatch.setattr(pipeline.shutil, "which", lambda _name: "ffmpeg")
    render_calls: list[list[dict[str, Any]]] = []

    def render(_settings, _project_id, _input, _config, proposals, *_args):
        render_calls.append(proposals)
        return [
            {
                "id": record["id"],
                "title": record["proposal"]["titulo"],
                "start": record["proposal"]["inicio"],
                "status": "ready",
            }
            for record in proposals
        ]

    monkeypatch.setattr(pipeline, "_render_projects", render)

    pipeline.process_project(project_id, None, settings)

    stored = database.get_project(settings.database_path, project_id)
    assert stored is not None
    assert stored["status"] == "completed"
    assert len(stored["clips"]) == expected_clip_count
    assert len(requests) == 1
    assert requests[0].headers["Authorization"] == "Bearer test-key"
    assert len(render_calls) == (1 if clips else 0)
    if clips:
        assert stored["metadata"]["clip_proposals"][0]["proposal"]["titulo"] == "Idea completa"
