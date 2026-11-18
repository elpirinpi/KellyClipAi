from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

_RELEASE_URL = "https://api.github.com/repos/elpirinpi/KellyClipAi/releases/latest"
_MAX_INSTALLER_SIZE = 1024 * 1024 * 1024
_VERSION_PATTERN = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")
logger = logging.getLogger(__name__)


class UpdateError(RuntimeError):
    pass


def _version_tuple(value: str) -> tuple[int, int, int]:
    match = _VERSION_PATTERN.fullmatch(value)
    if match is None:
        raise UpdateError("GitHub devolvió una versión no válida.")
    return tuple(int(part) for part in match.groups())


def _request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "kellyClipAi-updater",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise UpdateError(
            f"GitHub respondió con un error al consultar actualizaciones (HTTP {error.code})."
        ) from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise UpdateError(
            "No se pudo consultar GitHub. Comprueba tu conexión e inténtalo más tarde."
        ) from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise UpdateError("GitHub devolvió una respuesta de actualización no válida.") from error
    if not isinstance(payload, dict):
        raise UpdateError("GitHub devolvió una respuesta de actualización no válida.")
    return payload


class UpdateManager:
    def __init__(
        self,
        current_version: str,
        data_dir: Path,
        application_path: Path | None,
    ) -> None:
        self.current_version = current_version
        self.update_dir = data_dir / "updates"
        self.application_path = application_path
        self.enabled = os.name == "nt" and application_path is not None
        self._lock = threading.Lock()
        self._latest: dict[str, Any] | None = None
        self._status: dict[str, Any] = {
            "enabled": self.enabled,
            "current_version": current_version,
            "latest_version": None,
            "update_available": False,
            "state": "idle" if self.enabled else "unavailable",
            "progress": 0,
            "error": None,
        }
        self._thread: threading.Thread | None = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def start_check(self) -> dict[str, Any]:
        if not self.enabled:
            return self.status()
        with self._lock:
            if self._status["state"] in {
                "checking",
                "available",
                "downloading",
                "downloaded",
                "installer_opened",
            }:
                return dict(self._status)
            self._status.update(state="checking", error=None)
            self._thread = threading.Thread(
                target=self._check_latest_release,
                name="kellyclipai-update-check",
                daemon=True,
            )
            self._thread.start()
            return dict(self._status)

    def _check_latest_release(self) -> None:
        try:
            release = _request_json(_RELEASE_URL)
            if release.get("draft") or release.get("prerelease"):
                raise UpdateError("El release más reciente no es una versión estable.")
            version = str(release.get("tag_name", "")).removeprefix("v")
            latest_version = _version_tuple(version)
            current_version = _version_tuple(self.current_version)
            asset_name = f"KellyClipAi-Setup-{version}.exe"
            assets = release.get("assets")
            if not isinstance(assets, list):
                raise UpdateError("El release no incluye el instalador de Windows.")
            asset = next(
                (
                    item
                    for item in assets
                    if isinstance(item, dict) and item.get("name") == asset_name
                ),
                None,
            )
            is_newer = latest_version > current_version
            if is_newer and asset is None:
                raise UpdateError(
                    "Hay una versión nueva, pero todavía no está disponible su instalador."
                )
            digest = str(asset.get("digest", "")) if asset else ""
            if is_newer and not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest):
                raise UpdateError("El instalador no tiene una suma SHA-256 verificable.")
            download_url = str(asset.get("browser_download_url", "")) if asset else ""
            parsed_url = urllib.parse.urlparse(download_url)
            if is_newer and (
                parsed_url.scheme != "https"
                or parsed_url.hostname != "github.com"
                or not parsed_url.path.startswith(
                    f"/elpirinpi/KellyClipAi/releases/download/v{version}/"
                )
            ):
                raise UpdateError("La dirección del instalador no es confiable.")
            latest = {
                "version": version,
                "download_url": download_url,
                "sha256": digest.removeprefix("sha256:").lower(),
                "size": int(asset.get("size", 0)) if asset else 0,
                "asset_name": asset_name,
            }
            with self._lock:
                self._latest = latest if is_newer else None
                self._status.update(
                    latest_version=version,
                    update_available=is_newer,
                    state="available" if is_newer else "current",
                    progress=0,
                    error=None,
                )
        except UpdateError as error:
            with self._lock:
                self._status.update(state="error", error=str(error))
        except Exception:
            logger.exception("Falló la comprobación de actualizaciones")
            with self._lock:
                self._status.update(
                    state="error",
                    error="No se pudo comprobar si hay una actualización.",
                )

    def start_download(self) -> dict[str, Any]:
        if not self.enabled:
            raise UpdateError("Las actualizaciones están disponibles solo en la app instalada.")
        with self._lock:
            if self._status["state"] in {"checking", "downloading"}:
                return dict(self._status)
            if not self._latest:
                raise UpdateError("No hay una actualización disponible para descargar.")
            latest = dict(self._latest)
            self._status.update(state="downloading", progress=0, error=None)
            self._thread = threading.Thread(
                target=self._download_installer,
                args=(latest,),
                name="kellyclipai-update-download",
                daemon=True,
            )
            self._thread.start()
            return dict(self._status)

    def _download_installer(self, release: dict[str, Any]) -> None:
        self.update_dir.mkdir(parents=True, exist_ok=True)
        target = self.update_dir / release["asset_name"]
        temporary = target.with_suffix(".exe.part")
        request = urllib.request.Request(
            release["download_url"],
            headers={"Accept": "application/octet-stream", "User-Agent": "kellyClipAi-updater"},
        )
        digest = hashlib.sha256()
        downloaded = 0
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                content_length = int(response.headers.get("Content-Length", "0"))
                expected_size = release["size"] or content_length
                if expected_size <= 0 or expected_size > _MAX_INSTALLER_SIZE:
                    raise UpdateError("El tamaño del instalador no es válido.")
                if content_length and content_length != expected_size:
                    raise UpdateError("El tamaño del instalador no coincide con el release.")
                with temporary.open("wb") as installer:
                    while chunk := response.read(1024 * 1024):
                        downloaded += len(chunk)
                        if downloaded > _MAX_INSTALLER_SIZE or downloaded > expected_size:
                            raise UpdateError("La descarga excede el tamaño esperado.")
                        digest.update(chunk)
                        installer.write(chunk)
                        with self._lock:
                            self._status["progress"] = min(
                                99, round(downloaded * 100 / expected_size)
                            )
            if downloaded != expected_size:
                raise UpdateError("La descarga del instalador quedó incompleta.")
            if digest.hexdigest() != release["sha256"]:
                raise UpdateError("La verificación SHA-256 del instalador falló.")
            os.replace(temporary, target)
            with self._lock:
                self._status.update(
                    state="downloaded",
                    progress=100,
                    error=None,
                )
        except UpdateError as error:
            temporary.unlink(missing_ok=True)
            with self._lock:
                self._status.update(state="error", error=str(error), progress=0)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            temporary.unlink(missing_ok=True)
            with self._lock:
                self._status.update(
                    state="error",
                    error=f"No se pudo descargar el instalador: {error}",
                    progress=0,
                )
        except Exception:
            temporary.unlink(missing_ok=True)
            logger.exception("Falló la descarga de la actualización")
            with self._lock:
                self._status.update(
                    state="error",
                    error="Falló la descarga de la actualización.",
                    progress=0,
                )

    def open_installer(self) -> None:
        if not self.enabled or self._latest is None:
            raise UpdateError("No hay una actualización lista para instalar.")
        installer = self.update_dir / self._latest["asset_name"]
        if self.status()["state"] not in {"downloaded", "installer_opened"} or not installer.is_file():
            raise UpdateError("Descarga primero el instalador de la actualización.")
        try:
            os.startfile(str(installer))  # type: ignore[attr-defined]
        except OSError as error:
            raise UpdateError("No se pudo abrir el instalador descargado.") from error
        with self._lock:
            self._status.update(state="installer_opened")
