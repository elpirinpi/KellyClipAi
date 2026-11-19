from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import httpx
import jwt

_JWKS_CACHE_SECONDS = 300
_UNKNOWN_KID_REFRESH_SECONDS = 30
_ALLOWED_ALGORITHMS = {"ES256", "RS256", "EdDSA"}


class AuthenticationError(RuntimeError):
    pass


class AuthenticationUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class AuthenticatedUser:
    id: str
    email: str | None


class SupabaseAuthVerifier:
    def __init__(
        self,
        project_url: str,
        publishable_key: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        auth_url = f"{project_url.rstrip('/')}/auth/v1"
        self._issuer = auth_url
        self._jwks_url = f"{auth_url}/.well-known/jwks.json"
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(10, connect=5),
            transport=transport,
        )
        self._jwks: dict[str, dict[str, Any]] = {}
        self._jwks_expires_at = 0.0
        self._jwks_refreshed_at = 0.0
        self._jwks_lock = asyncio.Lock()

    async def _refresh_jwks(self, *, force: bool = False) -> None:
        async with self._jwks_lock:
            now = time.monotonic()
            if not force and now < self._jwks_expires_at:
                return
            if force and now - self._jwks_refreshed_at < _UNKNOWN_KID_REFRESH_SECONDS:
                return
            try:
                response = await self._client.get(self._jwks_url)
                response.raise_for_status()
                payload = response.json()
            except (httpx.RequestError, httpx.HTTPStatusError) as error:
                raise AuthenticationUnavailable(
                    "No se pudieron consultar las claves públicas de autenticación. Comprueba tu conexión."
                ) from error
            except ValueError as error:
                raise AuthenticationUnavailable(
                    "El servidor de autenticación devolvió claves públicas no válidas."
                ) from error
            if not isinstance(payload, dict) or not isinstance(payload.get("keys"), list):
                raise AuthenticationUnavailable(
                    "El servidor de autenticación no devolvió un conjunto de claves válido."
                )
            keys = {
                jwk["kid"]: jwk
                for jwk in payload["keys"]
                if isinstance(jwk, dict)
                and isinstance(jwk.get("kid"), str)
                and jwk.get("use", "sig") == "sig"
            }
            if not keys:
                raise AuthenticationUnavailable(
                    "No hay claves públicas disponibles para verificar la sesión."
                )
            self._jwks = keys
            self._jwks_refreshed_at = time.monotonic()
            self._jwks_expires_at = self._jwks_refreshed_at + _JWKS_CACHE_SECONDS

    async def _signing_key(self, kid: str, algorithm: str) -> Any:
        await self._refresh_jwks()
        jwk = self._jwks.get(kid)
        if jwk is None:
            await self._refresh_jwks(force=True)
            jwk = self._jwks.get(kid)
        if jwk is None:
            raise AuthenticationError("La sesión expiró o no es válida. Inicia sesión nuevamente.")
        if algorithm not in _ALLOWED_ALGORITHMS or jwk.get("alg", algorithm) != algorithm:
            raise AuthenticationError("El método de firma de la sesión no es válido.")
        try:
            return jwt.PyJWK.from_dict(jwk, algorithm=algorithm).key
        except (jwt.PyJWTError, ValueError, TypeError) as error:
            raise AuthenticationUnavailable(
                "La clave pública de autenticación tiene un formato no válido."
            ) from error

    async def verify(self, access_token: str) -> AuthenticatedUser:
        try:
            header = jwt.get_unverified_header(access_token)
        except jwt.InvalidTokenError as error:
            raise AuthenticationError("La sesión expiró o no es válida. Inicia sesión nuevamente.") from error
        kid = header.get("kid")
        algorithm = header.get("alg")
        if not isinstance(kid, str) or not isinstance(algorithm, str):
            raise AuthenticationError("La sesión no incluye una firma verificable.")
        signing_key = await self._signing_key(kid, algorithm)
        try:
            payload = jwt.decode(
                access_token,
                signing_key,
                algorithms=[algorithm],
                audience="authenticated",
                issuer=self._issuer,
                leeway=30,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
            user_id = str(UUID(payload["sub"]))
            if payload.get("role") != "authenticated":
                raise AuthenticationError("La sesión no corresponde a un usuario autenticado.")
        except jwt.ExpiredSignatureError as error:
            raise AuthenticationError("La sesión expiró. Inicia sesión nuevamente.") from error
        except AuthenticationError:
            raise
        except (jwt.InvalidTokenError, KeyError, TypeError, ValueError) as error:
            raise AuthenticationError("La sesión expiró o no es válida. Inicia sesión nuevamente.") from error
        email = payload.get("email")
        return AuthenticatedUser(user_id, email if isinstance(email, str) else None)

    async def close(self) -> None:
        await self._client.aclose()
