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
_INSTALLER_NAME_PATTERN = re.compile(r"^KellyClipAi-Setup-.*\.exe(\.part)?$")
logger = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_quietly(path: Path) -> None:
    """Borra un archivo parcial sin fallar si ya no existe o está bloqueado."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("No se pudo borrar el archivo parcial %s", path.name)


def _parse_content_range(value: str | None) -> tuple[int, int, int | None] | None:
    """Interpreta una cabecera Content-Range ("bytes inicio-fin/total",
    con "*" si el total es desconocido). Devuelve None si no se puede
    interpretar."""
    if not value:
        return None
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", value.strip())
    if match is None:
        return None
    total = match.group(3)
    return int(match.group(1)), int(match.group(2)), None if total == "*" else int(total)


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
            # Si quedó un .part de un intento anterior, el progreso inicial
            # refleja lo ya descargado en vez de volver a 0.
            part = (self.update_dir / latest["asset_name"]).with_suffix(".exe.part")
            expected = int(latest.get("size") or 0)
            initial_progress = 0
            if expected > 0 and part.is_file():
                existing = part.stat().st_size
                if 0 < existing < expected:
                    initial_progress = min(99, round(existing * 100 / expected))
            self._status.update(state="downloading", progress=initial_progress, error=None)
            self._thread = threading.Thread(
                target=self._download_installer,
                args=(latest,),
                name="kellyclipai-update-download",
                daemon=True,
            )
            self._thread.start()
            return dict(self._status)

    def _kept_progress(self, temporary: Path, expected_size: int) -> int:
        """Progreso que conserva un .part parcial tras un fallo (0 si no
        hay nada que reanudar). Así la UI muestra lo ya descargado en vez
        de volver a 0 cuando la descarga se interrumpe."""
        if expected_size <= 0 or not temporary.is_file():
            return 0
        return min(99, round(temporary.stat().st_size * 100 / expected_size))

    def _download_installer(self, release: dict[str, Any]) -> None:
        self.update_dir.mkdir(parents=True, exist_ok=True)
        target = self.update_dir / release["asset_name"]
        temporary = target.with_suffix(".exe.part")
        expected_size = int(release.get("size") or 0)
        if expected_size < 0 or expected_size > _MAX_INSTALLER_SIZE:
            with self._lock:
                self._status.update(
                    state="error",
                    error="El tamaño del instalador no es válido.",
                    progress=0,
                )
            return
        # Reanudación: si quedó un .part de un intento anterior, continuar
        # desde su tamaño en vez de descargar desde cero.
        existing = temporary.stat().st_size if temporary.is_file() else 0
        resume_from = 0
        if expected_size > 0 and 0 < existing < expected_size:
            resume_from = existing
        elif expected_size > 0 and existing >= expected_size:
            # El .part ya tiene el tamaño completo: si su hash coincide,
            # el instalador está listo sin descargar nada; si no, los
            # bytes están corruptos y hay que empezar de cero.
            try:
                complete = _sha256_file(temporary) == release["sha256"]
            except OSError:
                complete = False
            if complete:
                try:
                    os.replace(temporary, target)
                except OSError:
                    logger.warning(
                        "No se pudo promover el instalador parcial de %s; "
                        "se descarga de nuevo.",
                        release["asset_name"],
                    )
                else:
                    self._cleanup_old_installers(keep=target)
                    with self._lock:
                        self._status.update(
                            state="downloaded",
                            progress=100,
                            error=None,
                        )
                    return
            else:
                logger.warning(
                    "El instalador parcial de %s no coincide con el hash "
                    "esperado; se descarga de nuevo.",
                    release["asset_name"],
                )
                _remove_quietly(temporary)
        headers = {
            "Accept": "application/octet-stream",
            "User-Agent": "kellyClipAi-updater",
        }
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"
        request = urllib.request.Request(release["download_url"], headers=headers)
        downloaded = resume_from
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                status = getattr(response, "status", 200)
                content_length = int(response.headers.get("Content-Length", "0"))
                if resume_from and status == 206:
                    content_range = _parse_content_range(
                        response.headers.get("Content-Range")
                    )
                    if content_range is None or content_range[0] != resume_from:
                        raise UpdateError(
                            "El servidor no reanudó la descarga en el punto esperado."
                        )
                    if content_range[2] is not None and content_range[2] != expected_size:
                        raise UpdateError("El tamaño del instalador no coincide con el release.")
                    mode = "ab"
                elif resume_from:
                    # El servidor ignoró la cabecera Range: empezar desde
                    # cero en vez de mezclar bytes.
                    logger.warning(
                        "El servidor no admite reanudación (HTTP %s); "
                        "descargando el instalador desde cero.",
                        status,
                    )
                    resume_from = 0
                    downloaded = 0
                    mode = "wb"
                else:
                    mode = "wb"
                if expected_size <= 0:
                    expected_size = content_length
                if expected_size <= 0 or expected_size > _MAX_INSTALLER_SIZE:
                    raise UpdateError("El tamaño del instalador no es válido.")
                if content_length and content_length != expected_size - downloaded:
                    raise UpdateError("El tamaño del instalador no coincide con el release.")
                with temporary.open(mode) as installer:
                    while chunk := response.read(1024 * 1024):
                        downloaded += len(chunk)
                        if downloaded > _MAX_INSTALLER_SIZE or downloaded > expected_size:
                            # Datos corruptos: el .part no se puede
                            # reanudar con fiabilidad, así que se borra.
                            _remove_quietly(temporary)
                            raise UpdateError("La descarga excede el tamaño esperado.")
                        installer.write(chunk)
                        with self._lock:
                            self._status["progress"] = min(
                                99, round(downloaded * 100 / expected_size)
                            )
            if downloaded != expected_size:
                raise UpdateError("La descarga del instalador quedó incompleta.")
            # Verificación final con la misma función que usa open_installer
            # antes de ejecutar el instalador.
            if _sha256_file(temporary) != release["sha256"]:
                _remove_quietly(temporary)
                raise UpdateError("La verificación SHA-256 del instalador falló.")
            os.replace(temporary, target)
            self._cleanup_old_installers(keep=target)
            with self._lock:
                self._status.update(
                    state="downloaded",
                    progress=100,
                    error=None,
                )
        except UpdateError as error:
            # El .part parcial se conserva para reanudar en el próximo
            # intento; solo se borra en fallos de integridad (hash o
            # tamaño), que ya lo eliminaron antes de lanzar el error.
            with self._lock:
                self._status.update(
                    state="error",
                    error=str(error),
                    progress=self._kept_progress(temporary, expected_size),
                )
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            with self._lock:
                self._status.update(
                    state="error",
                    error=f"No se pudo descargar el instalador: {error}",
                    progress=self._kept_progress(temporary, expected_size),
                )
        except Exception:
            logger.exception("Falló la descarga de la actualización")
            with self._lock:
                self._status.update(
                    state="error",
                    error="Falló la descarga de la actualización.",
                    progress=self._kept_progress(temporary, expected_size),
                )

    def _cleanup_old_installers(self, keep: Path) -> None:
        """Borra instaladores de versiones anteriores del directorio de
        updates, conservando solo el descargado actual (`keep`).

        Solo se tocan archivos con el nombre de nuestros instaladores
        (KellyClipAi-Setup-*.exe y sus .part); cualquier otro archivo del
        directorio se deja intacto. La limpieza es best-effort: un archivo
        bloqueado (p. ej. en uso) se registra y se sigue con los demás.
        """
        try:
            entries = list(self.update_dir.iterdir())
        except OSError:
            return
        for entry in entries:
            if entry.resolve() == keep.resolve():
                continue
            if not _INSTALLER_NAME_PATTERN.fullmatch(entry.name):
                continue
            try:
                entry.unlink()
            except OSError:
                logger.warning(
                    "No se pudo borrar el instalador antiguo %s", entry.name
                )

    def open_installer(self) -> None:
        if not self.enabled or self._latest is None:
            raise UpdateError("No hay una actualización lista para instalar.")
        installer = self.update_dir / self._latest["asset_name"]
        if self.status()["state"] not in {"downloaded", "installer_opened"} or not installer.is_file():
            raise UpdateError("Descarga primero el instalador de la actualización.")
        # Re-verificar el SHA-256 inmediatamente antes de ejecutar: el
        # archivo pudo modificarse o corromperse después de la descarga.
        expected_sha256 = str(self._latest.get("sha256", "")).lower()
        try:
            actual_sha256 = _sha256_file(installer)
        except OSError as error:
            raise UpdateError(
                "No se pudo leer el instalador descargado para verificarlo."
            ) from error
        if not expected_sha256 or actual_sha256 != expected_sha256:
            raise UpdateError(
                "La verificación SHA-256 del instalador falló antes de "
                "ejecutarlo. Descarga de nuevo la actualización."
            )
        try:
            os.startfile(str(installer))  # type: ignore[attr-defined]
        except OSError as error:
            raise UpdateError("No se pudo abrir el instalador descargado.") from error
        with self._lock:
            self._status.update(state="installer_opened")
