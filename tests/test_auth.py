from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import database
from backend.auth import AuthenticatedUser, AuthenticationError
from backend.credential_store import (
    CredentialStoreError,
    decrypt_api_key,
    encrypt_api_key,
)
from backend.main import create_app
from backend.schemas import ProjectConfig
from backend.settings import AppSettings


USER_A = "10000000-0000-4000-8000-000000000001"
USER_B = "20000000-0000-4000-8000-000000000002"


class TokenVerifier:
    async def verify(self, token: str) -> AuthenticatedUser:
        users = {"token-a": USER_A, "token-b": USER_B}
        if token not in users:
            raise AuthenticationError("Sesión no válida.")
        return AuthenticatedUser(users[token], None)


def test_protected_api_requires_a_valid_supabase_session(tmp_path: Path) -> None:
    settings = AppSettings(
        data_dir=tmp_path / "data",
        cors_origins=(),
        llm_provider="Groq",
        llm_api_key="",
        llm_model="",
        llm_base_url="",
        whisper_model="small",
    )
    with TestClient(create_app(settings, auth_verifier=TokenVerifier())) as client:
        assert client.get("/api/settings").status_code == 401
        assert client.get(
            "/api/settings", headers={"Authorization": "Bearer invalid"}
        ).status_code == 401
        assert client.get("/api/auth/config").status_code == 200


def test_users_cannot_read_each_others_projects_and_legacy_data_is_claimed_once(
    tmp_path: Path,
) -> None:
    settings = AppSettings(
        data_dir=tmp_path / "data",
        cors_origins=(),
        llm_provider="Groq",
        llm_api_key="",
        llm_model="",
        llm_base_url="",
        whisper_model="small",
    )
    database.initialize(settings.database_path)
    config = ProjectConfig().model_dump(mode="json")
    database.create_project(
        settings.database_path,
        "legacy-project",
        "Proyecto anterior",
        tmp_path / "legacy.mp4",
        config,
    )
    database.create_project(
        settings.database_path,
        "owned-by-a",
        "Proyecto privado",
        tmp_path / "private.mp4",
        config,
        owner_id=USER_A,
    )

    with TestClient(create_app(settings, auth_verifier=TokenVerifier())) as client:
        owned = client.get(
            "/api/projects/owned-by-a",
            headers={"Authorization": "Bearer token-a"},
        )
        denied = client.get(
            "/api/projects/owned-by-a",
            headers={"Authorization": "Bearer token-b"},
        )
        legacy_owner = database.get_project(
            settings.database_path, "legacy-project", USER_A
        )
        other_user_legacy = client.get(
            "/api/projects/legacy-project",
            headers={"Authorization": "Bearer token-b"},
        )

    assert owned.status_code == 200
    assert denied.status_code == 404
    assert legacy_owner is not None
    assert legacy_owner["owner_id"] == USER_A
    assert other_user_legacy.status_code == 404


@pytest.mark.skipif(os.name != "nt", reason="DPAPI solo está disponible en Windows")
def test_llm_credentials_are_encrypted_and_scoped_to_the_signed_in_user(
    tmp_path: Path,
) -> None:
    settings = AppSettings(
        data_dir=tmp_path / "data",
        cors_origins=(),
        llm_provider="Groq",
        llm_api_key="",
        llm_model="",
        llm_base_url="",
        whisper_model="small",
    )
    with TestClient(create_app(settings, auth_verifier=TokenVerifier())) as client:
        saved = client.put(
            "/api/settings/llm-key",
            headers={"Authorization": "Bearer token-a"},
            json={
                "provider": "Groq",
                "api_key": "test-key-only",
                "model": "test-model",
                "base_url": "",
            },
        )
        user_a_status = client.get(
            "/api/settings", headers={"Authorization": "Bearer token-a"}
        )
        user_b_status = client.get(
            "/api/settings", headers={"Authorization": "Bearer token-b"}
        )
        credentials = database.get_llm_credentials(settings.database_path, USER_A)

    assert saved.status_code == 200
    assert saved.json() == {"saved": True, "has_api_key": True}
    assert user_a_status.json()["has_saved_llm_key"] is True
    assert user_b_status.json()["has_saved_llm_key"] is False
    assert "test-key-only" not in user_a_status.text
    assert credentials is not None
    assert credentials["encrypted_api_key"] != b"test-key-only"


@pytest.mark.skipif(os.name != "nt", reason="DPAPI solo está disponible en Windows")
def test_api_key_round_trips_through_windows_dpapi() -> None:
    ciphertext = encrypt_api_key("test-key-only")

    assert ciphertext != b"test-key-only"
    assert decrypt_api_key(ciphertext) == "test-key-only"


@pytest.mark.skipif(os.name == "nt", reason="Este caso solo aplica fuera de Windows")
def test_non_windows_credential_store_rejects_encryption() -> None:
    with pytest.raises(CredentialStoreError, match="requiere Windows"):
        encrypt_api_key("test-key-only")
