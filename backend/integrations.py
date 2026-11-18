from __future__ import annotations

import math
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlparse

import httpx

from . import database
from .settings import AppSettings

_GRAPH_VERSION = "v27.0"
_OAUTH_PROVIDERS = {"youtube", "tiktok", "instagram", "zoom"}
_PROVIDER_ENV = {
    "youtube": ("YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET"),
    "tiktok": ("TIKTOK_CLIENT_KEY", "TIKTOK_CLIENT_SECRET"),
    "instagram": ("META_APP_ID", "META_APP_SECRET"),
    "zoom": ("ZOOM_CLIENT_ID", "ZOOM_CLIENT_SECRET"),
    "pexels": ("PEXELS_API_KEY",),
    "pixabay": ("PIXABAY_API_KEY",),
}
_PROVIDER_SETTINGS = {
    "youtube": ("youtube_client_id", "youtube_client_secret"),
    "tiktok": ("tiktok_client_key", "tiktok_client_secret"),
    "instagram": ("meta_app_id", "meta_app_secret"),
    "zoom": ("zoom_client_id", "zoom_client_secret"),
    "pexels": ("pexels_api_key",),
    "pixabay": ("pixabay_api_key",),
}


class IntegrationError(RuntimeError):
    pass


def _publication_caption(
    clip: dict[str, Any],
    platform: str,
    *,
    add_shorts: bool = False,
    max_length: int,
) -> str:
    description = str(clip.get("description") or clip.get("title") or "").strip()
    platform_hashtags = clip.get("hashtags")
    values = (
        platform_hashtags.get(platform, [])
        if isinstance(platform_hashtags, dict)
        else []
    )
    hashtags: list[str] = []
    seen: set[str] = set()
    if isinstance(values, list):
        for value in values:
            tag = str(value).strip().lstrip("#")
            if not tag or not all(character.isalnum() or character == "_" for character in tag):
                continue
            normalized = tag.casefold()
            if normalized not in seen:
                hashtags.append(f"#{tag}")
                seen.add(normalized)
    if add_shorts and "shorts" not in seen:
        hashtags.append("#Shorts")
    parts = [part for part in (description, " ".join(hashtags)) if part]
    return "\n\n".join(parts)[:max_length]


def _missing_credentials(provider: str, settings: AppSettings) -> list[str]:
    attributes = _PROVIDER_SETTINGS.get(provider)
    if attributes is None:
        raise IntegrationError(f"Integración no admitida: {provider}.")
    return [
        env_name
        for env_name, attribute in zip(_PROVIDER_ENV[provider], attributes, strict=True)
        if not str(getattr(settings, attribute)).strip()
    ]


def provider_missing_credentials(provider: str, settings: AppSettings) -> list[str]:
    return _missing_credentials(provider, settings)


def integration_status(
    settings: AppSettings,
    database_path: Path,
    owner_id: str = "legacy",
) -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for provider in _PROVIDER_ENV:
        missing = _missing_credentials(provider, settings)
        account = database.get_connected_account(database_path, provider, owner_id)
        status[provider] = {
            "configured": not missing,
            "missing": missing,
            "connected": bool(account) and not missing,
            "display_name": account["display_name"] if account else None,
        }
    return status


def _redirect_uri(settings: AppSettings, provider: str) -> str:
    return f"{settings.oauth_redirect_base_url}/api/integrations/{provider}/callback"


def authorization_url(settings: AppSettings, provider: str, state: str) -> str:
    missing = _missing_credentials(provider, settings)
    if missing:
        raise IntegrationError(
            f"Faltan credenciales en .env: {', '.join(missing)}."
        )
    if provider not in _OAUTH_PROVIDERS:
        raise IntegrationError(f"{provider} no utiliza OAuth.")
    redirect_uri = _redirect_uri(settings, provider)
    values: dict[str, str]
    if provider == "youtube":
        # channels.list?mine=true requires youtube.readonly; verify OAuth with a real account.
        values = {
            "client_id": settings.youtube_client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": (
                "openid email https://www.googleapis.com/auth/youtube.upload "
                "https://www.googleapis.com/auth/youtube.readonly"
            ),
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        }
        return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(values)}"
    if provider == "tiktok":
        values = {
            "client_key": settings.tiktok_client_key,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "user.info.basic,video.publish",
            "state": state,
        }
        return f"https://www.tiktok.com/v2/auth/authorize/?{urlencode(values)}"
    if provider == "instagram":
        values = {
            "client_id": settings.meta_app_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "instagram_basic,instagram_content_publish,pages_show_list,pages_read_engagement",
            "state": state,
        }
        return f"https://www.facebook.com/{_GRAPH_VERSION}/dialog/oauth?{urlencode(values)}"
    values = {
        "client_id": settings.zoom_client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "state": state,
    }
    return f"https://zoom.us/oauth/authorize?{urlencode(values)}"


def _response_json(response: httpx.Response, purpose: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as error:
        raise IntegrationError(f"{purpose}: el proveedor devolvió una respuesta no válida.") from error
    if not isinstance(payload, dict):
        raise IntegrationError(f"{purpose}: el proveedor devolvió una respuesta no válida.")
    if response.is_error:
        detail = payload.get("error_description") or payload.get("message")
        if isinstance(payload.get("error"), dict):
            detail = payload["error"].get("message") or detail
        if isinstance(payload.get("data"), dict) and payload["data"].get("error"):
            detail = payload["data"]["error"].get("message") or detail
        message = str(detail)[:240] if detail else f"HTTP {response.status_code}"
        raise IntegrationError(f"{purpose}: {message}")
    return payload


def _request_json(
    client: httpx.Client,
    method: str,
    url: str,
    purpose: str,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        response = client.request(method, url, **kwargs)
    except httpx.RequestError as error:
        raise IntegrationError(f"{purpose}: no se pudo conectar con el proveedor.") from error
    return _response_json(response, purpose)


def _expires_at(expires_in: object) -> str | None:
    try:
        seconds = int(expires_in)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()


def complete_oauth(
    settings: AppSettings,
    provider: str,
    code: str,
) -> dict[str, Any]:
    if provider not in _OAUTH_PROVIDERS:
        raise IntegrationError(f"Proveedor OAuth no admitido: {provider}.")
    missing = _missing_credentials(provider, settings)
    if missing:
        raise IntegrationError(f"Faltan credenciales en .env: {', '.join(missing)}.")
    redirect_uri = _redirect_uri(settings, provider)
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        if provider == "youtube":
            token_payload = _request_json(
                client,
                "POST",
                "https://oauth2.googleapis.com/token",
                "Google OAuth",
                data={
                    "client_id": settings.youtube_client_id,
                    "client_secret": settings.youtube_client_secret,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": redirect_uri,
                },
            )
            token = _required_token(token_payload, "access_token", "Google")
            identity_payload = _request_json(
                client,
                "GET",
                "https://www.googleapis.com/youtube/v3/channels",
                "YouTube",
                headers={"Authorization": f"Bearer {token}"},
                params={"part": "snippet", "mine": "true"},
            )
            channels = identity_payload.get("items") or []
            if not channels:
                raise IntegrationError("La cuenta de Google no tiene un canal de YouTube disponible.")
            channel = channels[0]
            snippet = channel.get("snippet") or {}
            return _account_document(
                provider,
                str(channel["id"]),
                str(snippet.get("title") or "Canal de YouTube"),
                token_payload,
                token,
                str(token_payload.get("refresh_token") or "") or None,
                {"channel_title": snippet.get("title")},
            )

        if provider == "tiktok":
            token_payload = _request_json(
                client,
                "POST",
                "https://open.tiktokapis.com/v2/oauth/token/",
                "TikTok OAuth",
                data={
                    "client_key": settings.tiktok_client_key,
                    "client_secret": settings.tiktok_client_secret,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": redirect_uri,
                },
            )
            token_data = _provider_data(token_payload, "TikTok OAuth")
            token = _required_token(token_data, "access_token", "TikTok")
            identity_payload = _request_json(
                client,
                "GET",
                "https://open.tiktokapis.com/v2/user/info/",
                "TikTok",
                headers={"Authorization": f"Bearer {token}"},
                params={"fields": "open_id,display_name"},
            )
            identity_data = _provider_data(identity_payload, "TikTok")
            user = identity_data.get("user")
            if not isinstance(user, dict):
                raise IntegrationError("TikTok no devolvió los datos de la cuenta autorizada.")
            return _account_document(
                provider,
                _required_token(user, "open_id", "TikTok"),
                str(user.get("display_name") or "Cuenta de TikTok"),
                token_data,
                token,
                str(token_data.get("refresh_token") or "") or None,
                {},
            )

        if provider == "instagram":
            short_token = _request_json(
                client,
                "GET",
                f"https://graph.facebook.com/{_GRAPH_VERSION}/oauth/access_token",
                "Meta OAuth",
                params={
                    "client_id": settings.meta_app_id,
                    "client_secret": settings.meta_app_secret,
                    "redirect_uri": redirect_uri,
                    "code": code,
                },
            )
            user_token = _required_token(short_token, "access_token", "Meta")
            long_token_payload = _request_json(
                client,
                "GET",
                f"https://graph.facebook.com/{_GRAPH_VERSION}/oauth/access_token",
                "Meta OAuth",
                params={
                    "grant_type": "fb_exchange_token",
                    "client_id": settings.meta_app_id,
                    "client_secret": settings.meta_app_secret,
                    "fb_exchange_token": user_token,
                },
            )
            user_token = _required_token(long_token_payload, "access_token", "Meta")
            pages_payload = _request_json(
                client,
                "GET",
                f"https://graph.facebook.com/{_GRAPH_VERSION}/me/accounts",
                "Instagram",
                headers={"Authorization": f"Bearer {user_token}"},
                params={
                    "fields": "name,access_token,instagram_business_account{id,username}",
                    "limit": 100,
                },
            )
            pages = pages_payload.get("data") or []
            linked = next(
                (
                    page
                    for page in pages
                    if isinstance(page.get("instagram_business_account"), dict)
                ),
                None,
            )
            if linked is None:
                raise IntegrationError(
                    "No se encontró una cuenta profesional de Instagram conectada a una página de Facebook."
                )
            instagram = linked["instagram_business_account"]
            page_token = str(linked.get("access_token") or user_token)
            metadata = {
                "page_id": linked.get("id"),
                "user_access_token": user_token,
                "page_name": linked.get("name"),
            }
            return _account_document(
                provider,
                _required_token(instagram, "id", "Instagram"),
                str(instagram.get("username") or linked.get("name") or "Instagram"),
                long_token_payload,
                page_token,
                None,
                metadata,
            )

        token_payload = _request_json(
            client,
            "POST",
            "https://zoom.us/oauth/token",
            "Zoom OAuth",
            auth=(settings.zoom_client_id, settings.zoom_client_secret),
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
            },
        )
        token = _required_token(token_payload, "access_token", "Zoom")
        identity_payload = _request_json(
            client,
            "GET",
            "https://api.zoom.us/v2/users/me",
            "Zoom",
            headers={"Authorization": f"Bearer {token}"},
        )
        return _account_document(
            provider,
            _required_token(identity_payload, "id", "Zoom"),
            str(identity_payload.get("email") or identity_payload.get("display_name") or "Cuenta de Zoom"),
            token_payload,
            token,
            str(token_payload.get("refresh_token") or "") or None,
            {},
        )


def _provider_data(payload: dict[str, Any], provider: str) -> dict[str, Any]:
    outer_error = payload.get("error")
    if isinstance(outer_error, dict):
        if outer_error.get("code") not in (None, "ok", ""):
            raise IntegrationError(
                f"{provider}: {outer_error.get('message') or outer_error.get('code')}"
            )
    elif outer_error and outer_error not in ("ok", ""):
        raise IntegrationError(f"{provider}: {outer_error}")
    data = payload.get("data")
    if not isinstance(data, dict):
        return payload
    error = data.get("error")
    if isinstance(error, dict) and error.get("code") not in (None, "ok"):
        raise IntegrationError(f"{provider}: {error.get('message') or error.get('code')}")
    return data


def _required_token(payload: dict[str, Any], field: str, provider: str) -> str:
    value = payload.get(field)
    if not isinstance(value, (str, int)) or not str(value):
        raise IntegrationError(f"{provider} no devolvió el dato requerido ({field}).")
    return str(value)


def _account_document(
    provider: str,
    account_id: str,
    display_name: str,
    token_payload: dict[str, Any],
    access_token: str,
    refresh_token: str | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "provider": provider,
        "account_id": account_id,
        "display_name": display_name,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": _expires_at(token_payload.get("expires_in")),
        "metadata": metadata,
    }


def _refresh_account(
    settings: AppSettings,
    provider: str,
    account: dict[str, Any],
    owner_id: str = "legacy",
) -> dict[str, Any]:
    refresh_token = account.get("refresh_token")
    if provider == "instagram" and account.get("metadata", {}).get("user_access_token"):
        refresh_token = account["metadata"]["user_access_token"]
    if not refresh_token:
        raise IntegrationError(f"La sesión de {provider} venció. Desconecta y vuelve a conectar la cuenta.")
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        if provider == "youtube":
            payload = _request_json(
                client,
                "POST",
                "https://oauth2.googleapis.com/token",
                "Google OAuth",
                data={
                    "client_id": settings.youtube_client_id,
                    "client_secret": settings.youtube_client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            token = _required_token(payload, "access_token", "Google")
            refreshed = {
                **account,
                "access_token": token,
                "expires_at": _expires_at(payload.get("expires_in")),
            }
        elif provider == "tiktok":
            payload = _request_json(
                client,
                "POST",
                "https://open.tiktokapis.com/v2/oauth/token/",
                "TikTok OAuth",
                data={
                    "client_key": settings.tiktok_client_key,
                    "client_secret": settings.tiktok_client_secret,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
            )
            data = _provider_data(payload, "TikTok OAuth")
            refreshed = {
                **account,
                "access_token": _required_token(data, "access_token", "TikTok"),
                "refresh_token": data.get("refresh_token") or refresh_token,
                "expires_at": _expires_at(data.get("expires_in")),
            }
        elif provider == "zoom":
            payload = _request_json(
                client,
                "POST",
                "https://zoom.us/oauth/token",
                "Zoom OAuth",
                auth=(settings.zoom_client_id, settings.zoom_client_secret),
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
            )
            refreshed = {
                **account,
                "access_token": _required_token(payload, "access_token", "Zoom"),
                "refresh_token": payload.get("refresh_token") or refresh_token,
                "expires_at": _expires_at(payload.get("expires_in")),
            }
        elif provider == "instagram":
            payload = _request_json(
                client,
                "GET",
                f"https://graph.facebook.com/{_GRAPH_VERSION}/oauth/access_token",
                "Meta OAuth",
                params={
                    "grant_type": "fb_exchange_token",
                    "client_id": settings.meta_app_id,
                    "client_secret": settings.meta_app_secret,
                    "fb_exchange_token": refresh_token,
                },
            )
            user_token = _required_token(payload, "access_token", "Meta")
            pages = _request_json(
                client,
                "GET",
                f"https://graph.facebook.com/{_GRAPH_VERSION}/me/accounts",
                "Instagram",
                headers={"Authorization": f"Bearer {user_token}"},
                params={
                    "fields": "name,access_token,instagram_business_account{id,username}",
                    "limit": 100,
                },
            ).get("data") or []
            page = next(
                (
                    item
                    for item in pages
                    if item.get("instagram_business_account", {}).get("id")
                    == account["account_id"]
                ),
                None,
            )
            if not page:
                raise IntegrationError("No se pudo renovar el acceso a la cuenta profesional de Instagram.")
            refreshed = {
                **account,
                "access_token": str(page.get("access_token") or user_token),
                "expires_at": _expires_at(payload.get("expires_in")),
                "metadata": {
                    **account["metadata"],
                    "user_access_token": user_token,
                },
            }
        else:
            raise IntegrationError(f"No se puede renovar el acceso de {provider}.")
    database.save_connected_account(
        settings.database_path,
        provider,
        refreshed["account_id"],
        refreshed["display_name"],
        refreshed["access_token"],
        refreshed.get("refresh_token"),
        refreshed.get("expires_at"),
        refreshed.get("metadata") or {},
        owner_id,
    )
    return refreshed


def _access_token(
    settings: AppSettings,
    provider: str,
    owner_id: str = "legacy",
) -> tuple[str, dict[str, Any]]:
    account = database.get_connected_account(
        settings.database_path,
        provider,
        owner_id,
    )
    if account is None:
        raise IntegrationError(f"No hay una cuenta de {provider} conectada.")
    expires_at = account.get("expires_at")
    if expires_at:
        try:
            expiry = datetime.fromisoformat(expires_at)
        except ValueError as error:
            raise IntegrationError(f"La fecha de sesión de {provider} no es válida; vuelve a conectar.") from error
        if expiry <= datetime.now(UTC) + timedelta(minutes=2):
            account = _refresh_account(settings, provider, account, owner_id)
    return account["access_token"], account


def publish_clip(
    settings: AppSettings,
    platform: str,
    clip: dict[str, Any],
    file_path: Path,
    visibility: str,
    owner_id: str = "legacy",
) -> dict[str, str]:
    if platform not in {"youtube", "tiktok", "instagram"}:
        raise IntegrationError(f"Plataforma de publicación no admitida: {platform}.")
    if not file_path.is_file():
        raise IntegrationError("El MP4 del clip no está disponible para publicar.")
    token, account = _access_token(settings, platform, owner_id)
    if platform == "youtube":
        return _publish_youtube(token, clip, file_path, visibility)
    if platform == "tiktok":
        return _publish_tiktok(token, clip, file_path, visibility)
    if visibility != "public":
        raise IntegrationError("Instagram publica en la cuenta profesional; elige visibilidad pública para continuar.")
    return _publish_instagram(token, account, clip, file_path)


def _publish_youtube(
    token: str,
    clip: dict[str, Any],
    file_path: Path,
    visibility: str,
) -> dict[str, str]:
    if visibility not in {"private", "public", "unlisted"}:
        raise IntegrationError("La visibilidad de YouTube debe ser privada, pública o no listada.")
    with httpx.Client(timeout=httpx.Timeout(30, read=600), follow_redirects=False) as client:
        try:
            start = client.post(
                "https://www.googleapis.com/upload/youtube/v3/videos",
                params={"uploadType": "resumable", "part": "snippet,status"},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json; charset=UTF-8",
                    "X-Upload-Content-Type": "video/mp4",
                    "X-Upload-Content-Length": str(file_path.stat().st_size),
                },
                json={
                    "snippet": {
                        "title": str(clip["title"])[:100],
                        "description": _publication_caption(
                            clip, "youtube", add_shorts=True, max_length=4000
                        ),
                        "tags": [str(item)[:40] for item in clip.get("keywords") or []][:15],
                        "categoryId": "22",
                    },
                    "status": {
                        "privacyStatus": visibility,
                        "selfDeclaredMadeForKids": False,
                    },
                },
            )
        except httpx.RequestError as error:
            raise IntegrationError("YouTube no pudo iniciar la carga del clip.") from error
        if start.status_code >= 400:
            _response_json(start, "YouTube")
        upload_url = start.headers.get("Location")
        parsed = urlparse(upload_url or "")
        if parsed.scheme != "https" or parsed.hostname not in {"www.googleapis.com", "youtube.googleapis.com"}:
            raise IntegrationError("YouTube devolvió una URL de carga no válida.")
        try:
            with file_path.open("rb") as media:
                uploaded = client.put(
                    upload_url,
                    content=media,
                    headers={
                        "Content-Type": "video/mp4",
                        "Content-Length": str(file_path.stat().st_size),
                    },
                )
        except httpx.RequestError as error:
            raise IntegrationError("Se interrumpió la carga del MP4 a YouTube.") from error
        payload = _response_json(uploaded, "YouTube")
    video_id = _required_token(payload, "id", "YouTube")
    return {"external_id": video_id, "url": f"https://www.youtube.com/watch?v={video_id}"}


def _publish_tiktok(
    token: str,
    clip: dict[str, Any],
    file_path: Path,
    visibility: str,
) -> dict[str, str]:
    privacy_level = "PUBLIC_TO_EVERYONE" if visibility == "public" else "SELF_ONLY"
    file_size = file_path.stat().st_size
    chunk_size = 10 * 1024 * 1024
    chunks = math.ceil(file_size / chunk_size)
    with httpx.Client(timeout=httpx.Timeout(30, read=300), follow_redirects=False) as client:
        creator = _request_json(
            client,
            "POST",
            "https://open.tiktokapis.com/v2/post/publish/creator_info/query/",
            "TikTok",
            headers={"Authorization": f"Bearer {token}"},
            json={},
        )
        creator_data = _provider_data(creator, "TikTok")
        privacy_options = (creator_data.get("privacy_level_options") or [])
        if privacy_level not in privacy_options:
            raise IntegrationError(
                "La cuenta de TikTok no permite la visibilidad seleccionada. No se inició la publicación."
            )
        init = _request_json(
            client,
            "POST",
            "https://open.tiktokapis.com/v2/post/publish/video/init/",
            "TikTok",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=UTF-8",
            },
            json={
                "post_info": {
                    "title": _publication_caption(
                        clip, "tiktok", max_length=2200
                    ),
                    "privacy_level": privacy_level,
                    "disable_duet": False,
                    "disable_comment": False,
                    "disable_stitch": False,
                },
                "source_info": {
                    "source": "FILE_UPLOAD",
                    "video_size": file_size,
                    "chunk_size": min(chunk_size, file_size),
                    "total_chunk_count": chunks,
                },
            },
        )
        init_data = _provider_data(init, "TikTok")
        upload_url = _required_token(init_data, "upload_url", "TikTok")
        parsed = urlparse(upload_url)
        if parsed.scheme != "https" or parsed.hostname != "open-upload.tiktokapis.com":
            raise IntegrationError("TikTok devolvió una URL de carga no válida.")
        publish_id = _required_token(init_data, "publish_id", "TikTok")
        try:
            with file_path.open("rb") as media:
                offset = 0
                while offset < file_size:
                    chunk = media.read(chunk_size)
                    end = offset + len(chunk) - 1
                    response = client.put(
                        upload_url,
                        content=chunk,
                        headers={
                            "Content-Type": "video/mp4",
                            "Content-Length": str(len(chunk)),
                            "Content-Range": f"bytes {offset}-{end}/{file_size}",
                        },
                    )
                    if response.status_code >= 400:
                        _response_json(response, "TikTok")
                    offset = end + 1
        except httpx.RequestError as error:
            raise IntegrationError("Se interrumpió la carga del MP4 a TikTok.") from error

        for _ in range(40):
            status = _request_json(
                client,
                "POST",
                "https://open.tiktokapis.com/v2/post/publish/status/fetch/",
                "TikTok",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json; charset=UTF-8",
                },
                json={"publish_id": publish_id},
            )
            status_data = _provider_data(status, "TikTok")
            state = status_data.get("status")
            if state == "PUBLISH_COMPLETE":
                return {"external_id": publish_id, "url": ""}
            if state in {"FAILED", "PUBLISH_FAILED"}:
                raise IntegrationError(
                    f"TikTok no publicó el clip: {status_data.get('fail_reason') or state}."
                )
            time.sleep(3)
    raise IntegrationError(
        "TikTok aceptó el archivo, pero no confirmó la publicación a tiempo. Revisa la cuenta antes de reintentar."
    )


def _publish_instagram(
    token: str,
    account: dict[str, Any],
    clip: dict[str, Any],
    file_path: Path,
) -> dict[str, str]:
    account_id = account["account_id"]
    graph_url = f"https://graph.facebook.com/{_GRAPH_VERSION}"
    with httpx.Client(timeout=httpx.Timeout(30, read=300), follow_redirects=False) as client:
        container = _request_json(
            client,
            "POST",
            f"{graph_url}/{account_id}/media",
            "Instagram",
            headers={"Authorization": f"Bearer {token}"},
            data={
                "media_type": "REELS",
                "upload_type": "resumable",
                "caption": _publication_caption(
                    clip, "instagram", max_length=2200
                ),
            },
        )
        container_id = _required_token(container, "id", "Instagram")
        upload_url = f"https://rupload.facebook.com/ig-api-upload/{_GRAPH_VERSION}/{container_id}"
        try:
            with file_path.open("rb") as media:
                uploaded = client.post(
                    upload_url,
                    content=media,
                    headers={
                        "Authorization": f"OAuth {token}",
                        "offset": "0",
                        "file_size": str(file_path.stat().st_size),
                        "Content-Type": "application/octet-stream",
                    },
                )
        except httpx.RequestError as error:
            raise IntegrationError("Se interrumpió la carga del MP4 a Instagram.") from error
        upload_result = _response_json(uploaded, "Instagram")
        if upload_result.get("success") is not True:
            raise IntegrationError("Instagram no confirmó la carga del reel.")
        state = ""
        for _ in range(36):
            status = _request_json(
                client,
                "GET",
                f"{graph_url}/{container_id}",
                "Instagram",
                headers={"Authorization": f"Bearer {token}"},
                params={"fields": "status_code,status"},
            )
            state = str(status.get("status_code") or "")
            if state == "FINISHED":
                break
            if state == "ERROR":
                raise IntegrationError(
                    f"Instagram no pudo procesar el reel: {status.get('status') or 'error desconocido'}."
                )
            time.sleep(5)
        if state != "FINISHED":
            raise IntegrationError(
                "Instagram aceptó el reel, pero no confirmó que esté listo para publicar."
            )
        published = _request_json(
            client,
            "POST",
            f"{graph_url}/{account_id}/media_publish",
            "Instagram",
            headers={"Authorization": f"Bearer {token}"},
            data={"creation_id": container_id},
        )
    media_id = _required_token(published, "id", "Instagram")
    return {"external_id": media_id, "url": ""}


def list_zoom_recordings(
    settings: AppSettings,
    owner_id: str = "legacy",
) -> list[dict[str, Any]]:
    token, _ = _access_token(settings, "zoom", owner_id)
    today = datetime.now(UTC).date()
    start = today - timedelta(days=30)
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        payload = _request_json(
            client,
            "GET",
            "https://api.zoom.us/v2/users/me/recordings",
            "Zoom",
            headers={"Authorization": f"Bearer {token}"},
            params={"from": start.isoformat(), "to": today.isoformat(), "page_size": 100},
        )
    recordings: list[dict[str, Any]] = []
    for meeting in payload.get("meetings") or []:
        files = [
            {
                "file_id": str(item.get("id") or ""),
                "file_type": str(item.get("file_type") or ""),
                "recording_type": str(item.get("recording_type") or ""),
                "file_size": item.get("file_size"),
            }
            for item in meeting.get("recording_files") or []
            if item.get("file_type") == "MP4"
            and item.get("status") == "completed"
            and item.get("id")
        ]
        if files:
            recordings.append(
                {
                    "meeting_id": str(meeting.get("uuid") or meeting.get("id") or ""),
                    "topic": str(meeting.get("topic") or "Grabación de Zoom"),
                    "start_time": meeting.get("start_time"),
                    "files": files,
                }
            )
    return recordings


def download_zoom_recording(
    settings: AppSettings,
    meeting_id: str,
    file_id: str,
    target_path: Path,
    owner_id: str = "legacy",
) -> None:
    if len(meeting_id) > 256 or len(file_id) > 100:
        raise IntegrationError("El identificador de grabación de Zoom no es válido.")
    token, _ = _access_token(settings, "zoom", owner_id)
    recordings = list_zoom_recordings(settings, owner_id)
    selected = next(
        (
            (meeting, file)
            for meeting in recordings
            if meeting["meeting_id"] == meeting_id
            for file in meeting["files"]
            if file["file_id"] == file_id
        ),
        None,
    )
    if selected is None:
        raise IntegrationError("La grabación ya no está disponible en la cuenta de Zoom.")

    encoded_meeting_id = quote(quote(meeting_id, safe=""), safe="")
    with httpx.Client(timeout=httpx.Timeout(30, read=600), follow_redirects=True) as client:
        payload = _request_json(
            client,
            "GET",
            f"https://api.zoom.us/v2/meetings/{encoded_meeting_id}/recordings",
            "Zoom",
            headers={"Authorization": f"Bearer {token}"},
        )
        file_info = next(
            (
                item
                for item in payload.get("recording_files") or []
                if str(item.get("id") or "") == file_id
                and item.get("file_type") == "MP4"
                and item.get("status") == "completed"
            ),
            None,
        )
        if file_info is None:
            raise IntegrationError("Zoom no devolvió el MP4 seleccionado.")
        url = str(file_info.get("download_url") or "")
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname or not (
            parsed.hostname == "zoom.us"
            or parsed.hostname.endswith(".zoom.us")
            or parsed.hostname.endswith(".zoomcdn.net")
        ):
            raise IntegrationError("Zoom devolvió una URL de descarga no válida.")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        try:
            with client.stream(
                "GET",
                url,
                headers={"Authorization": f"Bearer {token}"},
            ) as response:
                if response.is_error:
                    raise IntegrationError(f"Zoom no pudo descargar la grabación (HTTP {response.status_code}).")
                with target_path.open("wb") as target:
                    for chunk in response.iter_bytes(1024 * 1024):
                        written += len(chunk)
                        if written > 2 * 1024 * 1024 * 1024:
                            raise IntegrationError("La grabación supera el límite local de 2 GB.")
                        target.write(chunk)
        except httpx.RequestError as error:
            target_path.unlink(missing_ok=True)
            raise IntegrationError("No se pudo descargar la grabación desde Zoom.") from error
        except Exception:
            target_path.unlink(missing_ok=True)
            raise
    if written == 0:
        target_path.unlink(missing_ok=True)
        raise IntegrationError("Zoom devolvió un archivo vacío.")


def download_broll(
    settings: AppSettings,
    query: str,
    target_path: Path,
) -> dict[str, str]:
    provider = settings.broll_provider.strip().lower()
    if provider not in {"pexels", "pixabay"}:
        raise IntegrationError("BROLL_PROVIDER debe ser Pexels o Pixabay.")
    missing = _missing_credentials(provider, settings)
    if missing:
        raise IntegrationError(f"Faltan credenciales en .env: {', '.join(missing)}.")
    search_term = " ".join(query.split())[:100]
    if not search_term:
        raise IntegrationError("El clip no tiene palabras clave para buscar B-roll.")
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        if provider == "pexels":
            payload = _request_json(
                client,
                "GET",
                "https://api.pexels.com/videos/search",
                "Pexels",
                headers={"Authorization": settings.pexels_api_key.strip()},
                params={"query": search_term, "per_page": 1},
            )
            videos = payload.get("videos") or []
            if not videos:
                raise IntegrationError(f"Pexels no encontró B-roll para “{search_term}”.")
            video = videos[0]
            files = [
                file
                for file in video.get("video_files") or []
                if file.get("file_type") == "video/mp4" and file.get("link")
            ]
            if not files:
                raise IntegrationError("Pexels no devolvió un MP4 descargable para ese resultado.")
            selected = min(files, key=lambda item: int(item.get("width") or 10**9))
            download_url = str(selected["link"])
            attribution = str(video.get("url") or "Pexels")
            creator = str((video.get("user") or {}).get("name") or "Pexels")
            allowed_host = lambda host: host == "pexels.com" or host.endswith(".pexels.com")
        else:
            payload = _request_json(
                client,
                "GET",
                "https://pixabay.com/api/videos/",
                "Pixabay",
                params={
                    "key": settings.pixabay_api_key.strip(),
                    "q": search_term,
                    "per_page": 1,
                    "safesearch": "true",
                },
            )
            videos = payload.get("hits") or []
            if not videos:
                raise IntegrationError(f"Pixabay no encontró B-roll para “{search_term}”.")
            video = videos[0]
            variants = video.get("videos") or {}
            selected = variants.get("medium") or variants.get("small") or variants.get("tiny")
            if not isinstance(selected, dict) or not selected.get("url"):
                raise IntegrationError("Pixabay no devolvió un MP4 descargable para ese resultado.")
            download_url = str(selected["url"])
            attribution = str(video.get("pageURL") or "Pixabay")
            creator = str(video.get("user") or "Pixabay")
            allowed_host = lambda host: host == "pixabay.com" or host.endswith(".pixabay.com")

    parsed = urlparse(download_url)
    if parsed.scheme != "https" or not parsed.hostname or not allowed_host(parsed.hostname):
        raise IntegrationError(f"{provider.title()} devolvió una URL de video no permitida.")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with (
            httpx.Client(
                timeout=httpx.Timeout(30, read=300), follow_redirects=True
            ) as client,
            client.stream("GET", download_url) as response,
        ):
            if response.is_error:
                raise IntegrationError(f"{provider.title()} no pudo descargar el B-roll (HTTP {response.status_code}).")
            with target_path.open("wb") as target:
                for chunk in response.iter_bytes(1024 * 1024):
                    written += len(chunk)
                    if written > 250 * 1024 * 1024:
                        raise IntegrationError("El B-roll supera el límite local de 250 MB.")
                    target.write(chunk)
    except httpx.RequestError as error:
        target_path.unlink(missing_ok=True)
        raise IntegrationError(f"No se pudo descargar el B-roll desde {provider.title()}.") from error
    except Exception:
        target_path.unlink(missing_ok=True)
        raise
    if written == 0:
        target_path.unlink(missing_ok=True)
        raise IntegrationError(f"{provider.title()} devolvió un archivo vacío.")
    return {"provider": provider.title(), "creator": creator, "attribution_url": attribution}
