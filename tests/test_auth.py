from __future__ import annotations

import asyncio
import base64
import os
import time
from pathlib import Path
from uuid import uuid4

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient

from backend import database
from backend.auth import (
    AuthenticatedUser,
    AuthenticationError,
    SupabaseAuthVerifier,
)
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


def _b64url_uint(value: int) -> str:
    encoded = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(encoded).rstrip(b"=").decode("ascii")


def _signed_supabase_token(
    private_key: ec.EllipticCurvePrivateKey,
    *,
    user_id: str = USER_A,
    token_id: str | None = None,
) -> tuple[str, dict[str, str]]:
    public_numbers = private_key.public_key().public_numbers()
    jwk = {
        "kty": "EC",
        "crv": "P-256",
        "x": _b64url_uint(public_numbers.x),
        "y": _b64url_uint(public_numbers.y),
        "use": "sig",
        "alg": "ES256",
        "kid": "test-key",
    }
    token = jwt.encode(
        {
            "iss": "https://example.supabase.co/auth/v1",
            "aud": "authenticated",
            "role": "authenticated",
            "sub": user_id,
            "email": "kelly@example.test",
            "iat": int(time.time()),
            "exp": int(time.time()) + 600,
            "jti": token_id or str(uuid4()),
        },
        private_key,
        algorithm="ES256",
        headers={"kid": "test-key"},
    )
    return token, jwk


def test_supabase_jwt_verification_uses_cached_jwks_without_user_endpoint() -> None:
    private_key = ec.generate_private_key(ec.SECP256R1())
    token, jwk = _signed_supabase_token(private_key)
    requests: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        assert request.url.path == "/auth/v1/.well-known/jwks.json"
        return httpx.Response(200, json={"keys": [jwk]})

    verifier = SupabaseAuthVerifier(
        "https://example.supabase.co",
        "unused-publishable-key",
        transport=httpx.MockTransport(handle),
    )

    async def verify_twice() -> tuple[AuthenticatedUser, AuthenticatedUser]:
        try:
            first = await verifier.verify(token)
            second = await verifier.verify(token)
            unverified_payload = jwt.decode(token, options={"verify_signature": False})
            unknown_kid_token = jwt.encode(
                unverified_payload,
                private_key,
                algorithm="ES256",
                headers={"kid": "unknown-key"},
            )
            with pytest.raises(AuthenticationError):
                await verifier.verify(unknown_kid_token)
            return first, second
        finally:
            await verifier.close()

    first, second = asyncio.run(verify_twice())

    assert first == AuthenticatedUser(USER_A, "kelly@example.test")
    assert second == first
    assert requests == ["https://example.supabase.co/auth/v1/.well-known/jwks.json"]


def test_supabase_jwt_rejects_bad_signatures_and_expired_tokens() -> None:
    signing_key = ec.generate_private_key(ec.SECP256R1())
    wrong_key = ec.generate_private_key(ec.SECP256R1())
    token, jwk = _signed_supabase_token(signing_key)
    expired_token = jwt.encode(
        {
            "iss": "https://example.supabase.co/auth/v1",
            "aud": "authenticated",
            "role": "authenticated",
            "sub": USER_A,
            "iat": int(time.time()) - 1200,
            "exp": int(time.time()) - 600,
        },
        signing_key,
        algorithm="ES256",
        headers={"kid": "test-key"},
    )

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"keys": [jwk]})

    verifier = SupabaseAuthVerifier(
        "https://example.supabase.co",
        "unused-publishable-key",
        transport=httpx.MockTransport(handle),
    )

    async def verify_invalid_tokens() -> tuple[Exception, Exception]:
        try:
            with pytest.raises(AuthenticationError) as bad_signature:
                await verifier.verify(
                    jwt.encode(
                        jwt.decode(token, options={"verify_signature": False}),
                        wrong_key,
                        algorithm="ES256",
                        headers={"kid": "test-key"},
                    )
                )
            with pytest.raises(AuthenticationError) as expired:
                await verifier.verify(expired_token)
            return bad_signature.value, expired.value
        finally:
            await verifier.close()

    signature_error, expiration_error = asyncio.run(verify_invalid_tokens())

    assert "válida" in str(signature_error).casefold()
    assert "expiró" in str(expiration_error).casefold()


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
