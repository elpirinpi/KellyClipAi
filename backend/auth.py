from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import httpx


class AuthenticationError(RuntimeError):
    pass


class AuthenticationUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class AuthenticatedUser:
    id: str
    email: str | None


class SupabaseAuthVerifier:
    def __init__(self, project_url: str, publishable_key: str) -> None:
        self._user_url = f"{project_url.rstrip('/')}/auth/v1/user"
        self._publishable_key = publishable_key
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(10, connect=5))

    async def verify(self, access_token: str) -> AuthenticatedUser:
        try:
            response = await self._client.get(
                self._user_url,
                headers={
                    "apikey": self._publishable_key,
                    "Authorization": f"Bearer {access_token}",
                },
            )
        except httpx.RequestError as error:
            raise AuthenticationUnavailable(
                "No se pudo verificar la sesión con Supabase. Comprueba tu conexión."
            ) from error
        if response.status_code in {401, 403}:
            raise AuthenticationError("La sesión expiró. Inicia sesión nuevamente.")
        if response.is_error:
            raise AuthenticationUnavailable(
                f"Supabase no pudo verificar la sesión (HTTP {response.status_code})."
            )
        try:
            payload = response.json()
            user_id = str(UUID(payload["id"]))
            email = payload.get("email")
        except (KeyError, TypeError, ValueError) as error:
            raise AuthenticationError("Supabase devolvió una identidad no válida.") from error
        return AuthenticatedUser(user_id, email if isinstance(email, str) else None)

    async def close(self) -> None:
        await self._client.aclose()
