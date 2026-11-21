from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from . import database
from .credential_store import CredentialStoreError, decrypt_api_key
from .integrations import download_broll
from .llm import (
    LLM_MAX_OUTPUT_TOKENS,
    LLMConfigurationError,
    LLMResponseError,
    ResolvedLLM,
    propose_clips,
    resolve_llm,
)
from .media import (
    MediaProcessingError,
    create_thumbnail,
    probe_media,
    render_clip,
    transcribe,
)
from .process_queue import is_project_cancelled
from .schemas import (
    MAX_VIDEO_DURATION_SECONDS,
    ClipProposal,
    LLMConfig,
    ProjectConfig,
    TranscriptSegment,
    Word,
)
from .settings import AppSettings, get_settings


class ProjectNotFoundError(RuntimeError):
    pass


class ProjectCancelled(RuntimeError):
    """El usuario canceló el procesamiento.

    Lleva los clips renderizados hasta el momento para conservarlos: si al
    menos uno quedó listo, el pipeline los publica; si no, se conservan
    los clips anteriores del proyecto.
    """

    def __init__(self, clips: list[dict[str, Any]] | None = None) -> None:
        super().__init__("Procesamiento cancelado por el usuario.")
        self.clips = clips or []


logger = logging.getLogger(__name__)

# Ventanas de 5 minutos: cada llamada al LLM lleva una transcripción del
# tamaño de un video de 5 minutos, que Groq procesa sin error. Las ventanas
# de 15 minutos generaban peticiones de ~5.000 tokens que superaban los
# límites de tokens/minuto del plan gratuito de Groq con videos largos.
_LLM_WINDOW_SECONDS = 5 * 60
_LLM_WINDOW_OVERLAP_SECONDS = 1 * 60

# --- Pacing TPM (Fase 2) ----------------------------------------------------
# El plan gratuito de Groq permite 8K TPM por modelo (qwen/qwen3.8-27b).
# Cada ventana de 5 min consume ~1.500-2.000 tokens de entrada + hasta
# 2.000 de salida (max_tokens, incluido el razonamiento de Qwen3 antes
# del JSON): dos ventanas seguidas sin pausa ya superan el límite, la
# segunda recibe 429 y los reintentos rápidos caen dentro del mismo
# minuto. El pacer estima los tokens de cada llamada y duerme lo
# necesario para mantenerse bajo el presupuesto en la ventana móvil de
# 60 s. El presupuesto por defecto (7.000) deja margen bajo 8.000. El
# pacing previene la mayoría de 429; el backoff paciente de llm.py cubre
# el resto. Puro Python, sin suposiciones de hardware (cualquier PC).
_LLM_TPM_BUDGET_TOKENS = 7000
_LLM_TPM_WINDOW_SECONDS = 60.0
# Caracteres de instrucciones fijas del prompt (system + reglas), ~500 tokens.
_LLM_PROMPT_OVERHEAD_CHARS = 2000
# ~4 caracteres por token (regla de oro para texto en español/inglés).
_LLM_CHARS_PER_TOKEN = 4


def _estimate_request_tokens(transcript: list[TranscriptSegment]) -> int:
    """Tokens estimados de una llamada de propuestas: entrada + salida.

    Entrada: el texto de la transcripción + las instrucciones fijas del
    prompt, a ~4 caracteres por token. Salida: el máximo (max_tokens),
    porque el razonamiento de Qwen3 antes del JSON también consume
    tokens de salida. La estimación es conservadora a propósito.
    """
    text_chars = sum(len(segment.text) for segment in transcript)
    return (
        (text_chars + _LLM_PROMPT_OVERHEAD_CHARS) // _LLM_CHARS_PER_TOKEN
        + LLM_MAX_OUTPUT_TOKENS
    )


class _TokenPacer:
    """Regula las llamadas LLM para no superar el presupuesto TPM.

    Lleva la cuenta de los tokens estimados en la ventana móvil de 60 s;
    antes de cada llamada, duerme lo necesario para que la petición quepa
    bajo el presupuesto. Una instancia por proyecto (sin estado global).
    `sleep` y `monotonic` son inyectables para los tests. `on_wait` se
    invoca antes de dormir para avisar al usuario (el pipeline lo conecta
    a los eventos de progreso SSE).
    """

    def __init__(
        self,
        *,
        budget_tokens: int = _LLM_TPM_BUDGET_TOKENS,
        window_seconds: float = _LLM_TPM_WINDOW_SECONDS,
        sleep: Callable[[float], None] | None = None,
        monotonic: Callable[[], float] | None = None,
        on_wait: Callable[[float], None] | None = None,
    ) -> None:
        self._budget = budget_tokens
        self._window = window_seconds
        self._sleep = sleep or time.sleep
        self._monotonic = monotonic or time.monotonic
        self._on_wait = on_wait
        self._usage: list[tuple[float, int]] = []

    def wait_before_call(self, estimated_tokens: int) -> float:
        """Duerme lo necesario antes de una llamada; devuelve lo esperado."""
        now = self._monotonic()
        self._prune(now)
        used = sum(tokens for _, tokens in self._usage)
        if used + estimated_tokens <= self._budget:
            self._usage.append((now, estimated_tokens))
            return 0.0
        # Quitar las entradas más antiguas hasta que la petición quepa;
        # hay que esperar a que caduque la última quitada. Si ni vaciando
        # la ventana cabe (una sola petición mayor que el presupuesto),
        # se espera a que se vacíe del todo y se procede: una petición no
        # se puede dividir y el backoff de 429 cubre ese caso.
        freed = 0
        wait_until = now
        for timestamp, tokens in self._usage:
            freed += tokens
            wait_until = timestamp + self._window
            if used - freed + estimated_tokens <= self._budget:
                break
        wait = max(0.0, wait_until - now)
        if wait > 0:
            if self._on_wait is not None:
                self._on_wait(wait)
            self._sleep(wait)
            now = self._monotonic()
            self._prune(now)
        self._usage.append((now, estimated_tokens))
        return wait

    def _prune(self, now: float) -> None:
        self._usage = [
            (timestamp, tokens)
            for timestamp, tokens in self._usage
            if now - timestamp < self._window
        ]

# Espacio libre mínimo para empezar a procesar: un proyecto puede producir
# ~10 clips renderizados, B-roll y la descarga del modelo Whisper. Si hay
# menos, se falla rápido con un mensaje claro en vez de dejar el disco
# lleno a mitad del render. En sistemas de archivos más pequeños que el
# propio umbral (tmpfs, RAM disks, contenedores con /tmp pequeño), exigir
# 1 GiB haría imposible procesar nada aunque hubiera espacio de sobra para
# el proyecto, así que el mínimo baja al 10 % del total (ver
# _check_free_disk_space). En cualquier disco real (>= 10 GB) el mínimo es
# exactamente 1 GiB.
_MIN_FREE_DISK_BYTES = 1 * 1024 * 1024 * 1024  # 1 GiB
_MIN_FREE_DISK_FRACTION = 0.10  # 10 % del total en sistemas pequeños

# Throttle de progreso (Fase 2, B-R3): cada _publish son dos escrituras a
# SQLite y los callbacks llegan decenas de veces por segundo (ffmpeg, el
# análisis facial a 4 Hz, la transcripción por segmento). Se publica solo
# si cambia el % entero o han pasado 500 ms desde la última publicación
# del proyecto. Los estados terminales (completed/failed/cancelled)
# siempre se publican y limpian el estado, para que un reproceso del
# mismo proyecto no herede el throttle del anterior.
_PROGRESS_THROTTLE_SECONDS = 0.5
_last_progress_publish: dict[str, tuple[int | None, str, float]] = {}
_last_progress_publish_lock = threading.Lock()


def _reset_progress_throttle() -> None:
    """Vacía el estado del throttle de progreso (tests)."""
    with _last_progress_publish_lock:
        _last_progress_publish.clear()


def _is_cancelled(settings: AppSettings, project_id: str) -> bool:
    """Cancelación cooperativa: flag en memoria o estado persistido."""
    if is_project_cancelled(project_id):
        return True
    return (
        database.get_project_status(settings.database_path, project_id)
        == "cancelled"
    )


def _check_not_cancelled(settings: AppSettings, project_id: str) -> None:
    if _is_cancelled(settings, project_id):
        raise ProjectCancelled()


def _check_video_duration(duration_seconds: float) -> None:
    if duration_seconds > MAX_VIDEO_DURATION_SECONDS:
        limit_hours = MAX_VIDEO_DURATION_SECONDS // 3600
        raise MediaProcessingError(
            f"El video dura {duration_seconds / 3600:.1f} horas y supera el "
            f"máximo de {limit_hours} horas por proyecto."
        )


def _format_bytes(size: int) -> str:
    if size >= 1024**3:
        return f"{size / 1024**3:.0f} GB"
    return f"{max(1, size // (1024**2))} MB"


def _check_free_disk_space(path: Path) -> None:
    try:
        usage = shutil.disk_usage(path)
    except OSError as error:
        # No se pudo comprobar (ruta inexistente, permiso): no bloquear el
        # procesamiento por una comprobación de diagnóstico.
        logger.warning("No se pudo comprobar el espacio libre en disco: %s", error)
        return
    required = min(
        _MIN_FREE_DISK_BYTES, int(usage.total * _MIN_FREE_DISK_FRACTION)
    )
    if usage.free < required:
        raise MediaProcessingError(
            f"Queda menos de {_format_bytes(required)} de espacio libre en "
            "disco. Libera espacio antes de procesar el video."
        )


def _transcript_cache_matches(
    metadata: dict[str, Any],
    project_config: ProjectConfig,
    settings: AppSettings,
) -> bool:
    """La transcripción guardada solo se reutiliza si se hizo con el mismo
    modelo e idioma de Whisper que la configuración actual (A6). Las
    transcripciones guardadas antes de v1.0.11 no registran estos datos y
    se reutilizan tal cual: no se puede saber con qué se hicieron y
    forzar una retranscripción en cada reproceso sería un coste
    inesperado para el usuario."""
    cached_model = metadata.get("whisper_model")
    if cached_model is not None and cached_model != (
        project_config.whisper_model or settings.whisper_model
    ):
        return False
    cached_language = metadata.get("whisper_language")
    if (
        cached_language is not None
        and cached_language != project_config.whisper_language
    ):
        return False
    return True


def _swap_staged_clips(project_dir: Path, staging_dir: Path) -> None:
    """Publica los clips renderizados en staging de forma atómica (A1).

    Solo se llama cuando al menos un clip nuevo quedó listo. El directorio
    viejo se aparta a clips.old/, el nuevo ocupa su lugar y luego se borra
    el respaldo. Si el swap falla a medias, se restauran los clips viejos.
    Un clips.old/ huérfano de un swap anterior que terminó bien se borra;
    si clips/ no existe pero el respaldo sí (swap interrumpido), el
    respaldo se restaura antes de continuar.
    """
    output_dir = project_dir / "clips"
    backup_dir = project_dir / "clips.old"
    if not staging_dir.exists():
        return
    if output_dir.exists() and backup_dir.exists():
        shutil.rmtree(backup_dir)
    elif not output_dir.exists() and backup_dir.exists():
        os.replace(backup_dir, output_dir)
    try:
        if output_dir.exists():
            os.replace(output_dir, backup_dir)
        os.replace(staging_dir, output_dir)
    except OSError:
        if backup_dir.exists() and not output_dir.exists():
            os.replace(backup_dir, output_dir)
        raise
    if backup_dir.exists():
        try:
            shutil.rmtree(backup_dir)
        except OSError as error:
            # El swap ya terminó bien; el respaldo huérfano se limpia en
            # el próximo reproceso. No merece fallar el proyecto.
            logger.warning(
                "No se pudo borrar el respaldo de clips %s: %s", backup_dir, error
            )


def _publish(
    settings: AppSettings,
    project_id: str,
    status: str,
    step: str,
    progress: int | None,
    message: str,
) -> None:
    terminal = status in {"completed", "failed", "cancelled"}
    if terminal:
        with _last_progress_publish_lock:
            _last_progress_publish.pop(project_id, None)
    elif progress is not None:
        now = time.monotonic()
        with _last_progress_publish_lock:
            last = _last_progress_publish.get(project_id)
            if (
                last is not None
                and last[0] == progress
                and last[1] == step
                and now - last[2] < _PROGRESS_THROTTLE_SECONDS
            ):
                return
            _last_progress_publish[project_id] = (progress, step, now)
    payload = {
        "status": status,
        "step": step,
        "progress": progress,
        "message": message,
    }
    database.update_project(
        settings.database_path,
        project_id,
        status=status,
        step=step,
        progress=progress,
        error=message if status == "failed" else None,
    )
    database.publish_event(
        settings.database_path,
        project_id,
        "failed" if status == "failed" else "progress",
        payload,
    )


def _file_sha256(path: Path) -> str:
    """SHA-256 del archivo, por bloques de 1 MiB (B-R5)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_llm_for_project(
    stored: dict[str, Any],
    request_llm: LLMConfig | None,
    settings: AppSettings,
) -> ResolvedLLM:
    """Resuelve el LLM del proyecto, con respaldo a las credenciales guardadas.

    Con configuración explícita (el flujo normal desde /process) se usa
    tal cual. Sin ella — el caso de la cola persistente al recuperar un
    trabajo tras un reinicio, donde la clave API no se guardó en la cola
    por seguridad — se leen las credenciales cifradas del dueño del
    proyecto; si tampoco las hay, manda el entorno (.env) como antes.
    """
    if request_llm is not None:
        return resolve_llm(request_llm, settings)
    owner_id = stored.get("owner_id")
    if owner_id:
        try:
            credentials = database.get_llm_credentials(
                settings.database_path, str(owner_id)
            )
        except Exception as error:
            logger.warning(
                "No se pudieron leer las credenciales LLM guardadas: %s", error
            )
            credentials = None
        if credentials is not None:
            try:
                return resolve_llm(
                    LLMConfig(
                        provider=credentials["provider"],
                        api_key=decrypt_api_key(
                            bytes(credentials["encrypted_api_key"])
                        ),
                        model=credentials["model"],
                        base_url=credentials["base_url"],
                    ),
                    settings,
                )
            except (CredentialStoreError, ValidationError, LLMConfigurationError) as error:
                logger.warning(
                    "Las credenciales LLM guardadas no son válidas (%s); "
                    "se usa la configuración de entorno.",
                    error,
                )
    return resolve_llm(None, settings)


def _duration_bounds(choice: str) -> tuple[float, float]:
    return {
        "Cualquiera": (3.0, float("inf")),
        "<30s": (3.0, 29.9),
        "30–60s": (30.0, 60.0),
        "60–90s": (60.0, 90.0),
        "90s–3min": (90.0, 180.0),
    }[choice]


def _validate_proposals(
    proposals: list[object],
    duration: float,
    choice: str,
    *,
    source: str = "del LLM",
) -> tuple[list[ClipProposal], list[str]]:
    minimum, maximum = _duration_bounds(choice)
    if choice != "Cualquiera" and duration + 0.05 < minimum:
        raise MediaProcessingError(
            f"El video dura {duration:.1f} segundos y no alcanza la duración mínima de {choice}."
        )
    valid: list[ClipProposal] = []
    errors: list[str] = []
    for index, candidate in enumerate(proposals[:10], start=1):
        label = (
            str(candidate.get("titulo") or f"#{index}")
            if isinstance(candidate, dict)
            else f"#{index}"
        )
        try:
            proposal = ClipProposal.model_validate(candidate)
        except (ValidationError, TypeError) as error:
            reason = f"clip {label}: estructura inválida ({str(error).splitlines()[0][:240]})"
            logger.warning("Se descarta la propuesta %s: %s", source, reason)
            errors.append(reason)
            continue

        if proposal.fin > duration + 0.05:
            reason = f"clip {label}: el fin supera la duración del video"
            logger.warning("Se descarta la propuesta %s: %s", source, reason)
            errors.append(reason)
            continue
        if proposal.fin > duration:
            proposal = proposal.model_copy(update={"fin": duration})

        clip_duration = proposal.fin - proposal.inicio
        if (
            clip_duration < minimum - 0.05
            or clip_duration > min(maximum, duration) + 0.05
        ):
            reason = f"clip {label}: duración fuera del rango solicitado ({choice})"
            logger.warning("Se descarta la propuesta %s: %s", source, reason)
            errors.append(reason)
            continue
        valid.append(proposal)

    previous_end = -1.0
    non_overlapping: list[ClipProposal] = []
    for proposal in sorted(valid, key=lambda clip: clip.inicio):
        if proposal.inicio < previous_end:
            reason = f"clip {proposal.titulo}: se solapa con otro clip"
            logger.warning("Se descarta la propuesta %s: %s", source, reason)
            errors.append(reason)
            continue
        previous_end = proposal.fin
        non_overlapping.append(proposal)
    return non_overlapping, errors


def _validate_and_complete_proposals(
    candidates: list[object],
    transcript: list[TranscriptSegment],
    duration: float,
    choice: str,
    *,
    source: str = "del LLM",
) -> tuple[list[ClipProposal], list[str]]:
    proposals, errors = _validate_proposals(
        candidates, duration, choice, source=source
    )
    if not proposals:
        return [], errors
    completed = _complete_clip_ends(proposals, transcript, duration, choice)
    completed, completion_errors = _validate_proposals(
        completed, duration, choice, source=source
    )
    return completed, errors + completion_errors


def _propose_valid_clips(
    transcript: list[TranscriptSegment],
    project: ProjectConfig,
    duration: float,
    llm: ResolvedLLM | None,
    should_cancel: Callable[[], bool] | None = None,
    *,
    on_rate_limit: Callable[[float, int], None] | None = None,
    token_pacer: _TokenPacer | None = None,
    input_path: Path | None = None,
    ffmpeg_path: str | None = None,
    on_event_download_progress: Callable[[float], None] | None = None,
    on_profile: Callable[[str], None] | None = None,
    on_semantic_download_progress: Callable[[float], None] | None = None,
    on_diarization_download_progress: Callable[[float], None] | None = None,
    channels: _ChannelsHandle | None = None,
    fast: bool = False,
) -> list[ClipProposal]:
    """Punto de entrada de propuestas: despacha según project.moment_engine.

    - "signals" / "signals+ai": motor local de señales (video completo,
      sin ventanas ni LLM para elegir; el pulido con IA de "signals+ai"
      lo hace process_project después de la selección).
    - "llm": camino anterior por ventanas de 5 min (llm obligatorio).
    ``channels`` (extracción iniciada en segundo plano) y ``fast``
    (modo rápido Tier 0) solo aplican al camino de señales.
    """
    if project.moment_engine in {"signals", "signals+ai"}:
        return _propose_valid_signals(
            transcript,
            project,
            duration,
            input_path,
            ffmpeg_path,
            should_cancel,
            on_event_download_progress=on_event_download_progress,
            on_profile=on_profile,
            on_semantic_download_progress=on_semantic_download_progress,
            on_diarization_download_progress=on_diarization_download_progress,
            channels=channels,
            fast=fast,
        )
    if llm is None:
        raise LLMConfigurationError(
            "El modo de motor 'llm' necesita un proveedor de IA configurado."
        )
    return _propose_valid_clips_llm(
        transcript,
        project,
        duration,
        llm,
        should_cancel,
        on_rate_limit=on_rate_limit,
        token_pacer=token_pacer,
    )


# --- extracción de canales de señales (v1.3.1, velocidad) ----------------------
# Los 5 canales (YAMNet, diarización, visual, MiniLM, audio) son
# independientes entre sí: cada uno decodifica con su propio ffmpeg y
# corre su propia inferencia. En serie sumaban ~156 s en un video de
# 320 s en esta VM; en paralelo el tiempo lo marca el canal más lento
# en equipos con núcleos de sobra (en esta VM, con 1,5 CPU de cuota,
# el paralelismo es neutro: medido 159 s frente a 156 s en serie).
# Además, la extracción puede solaparse con la transcripción (que no
# necesita la transcripción para nada): ver _start_signal_channels.

_OVERLAP_MIN_CPUS = 8  # Núcleos lógicos mínimos para solapar canales con Whisper.
_CHANNELS_ABANDON_TIMEOUT_SECONDS = 10.0  # Tope al abandonar una extracción.


@dataclass
class _SignalChannels:
    """Resultados de la extracción de canales (None = canal ausente)."""

    event_timeline: Any = None
    speaker_timeline: Any = None
    visual: Any = None  # VisualTimelines de backend.visual_signals
    semantic_ready: bool = False
    audio_timeline: Any = None


def _extract_signal_channels(
    input_path: Path | None,
    duration: float,
    ffmpeg_path: str | None,
    should_cancel: Callable[[], bool] | None = None,
    *,
    on_event_download_progress: Callable[[float], None] | None = None,
    on_diarization_download_progress: Callable[[float], None] | None = None,
    on_semantic_download_progress: Callable[[float], None] | None = None,
) -> _SignalChannels:
    """Extrae los 5 canales de señales en paralelo (hilos).

    Cada canal mantiene su semántica de fallo de siempre: solo la
    cancelación del usuario se propaga (ProjectCancelled); cualquier
    otro fallo (sin runtime, sin modelo, sin red, sin audio, ffmpeg
    roto) deja el canal apagado y el motor sigue con el resto. La
    descarga de modelos (solo la primera vez) avisa por los callbacks
    de progreso como antes; los callbacks escriben en SQLite (WAL,
    timeout 30 s) y son seguros desde hilos.
    """
    def _events() -> Any:
        try:
            from . import audio_events
        except ImportError:
            return None
        try:
            return audio_events.get_event_timeline(
                input_path,
                duration,
                ffmpeg_path,
                on_progress=on_event_download_progress,
                should_cancel=should_cancel,
            )
        except audio_events.AudioEventsCancelled as error:
            raise ProjectCancelled() from error

    def _speakers() -> Any:
        try:
            from . import diarization
        except ImportError:
            return None
        try:
            return diarization.get_speaker_timeline(
                input_path,
                duration,
                ffmpeg_path,
                on_progress=on_diarization_download_progress,
                should_cancel=should_cancel,
            )
        except diarization.DiarizationCancelled as error:
            raise ProjectCancelled() from error

    def _visual() -> Any:
        if input_path is None or not ffmpeg_path:
            return None
        try:
            from . import visual_signals

            return visual_signals.extract_visual_timelines(
                input_path, ffmpeg_path, duration, should_cancel=should_cancel
            )
        except Exception as error:  # noqa: BLE001 - lo visual jamás rompe Tier 0
            logger.warning(
                "La extracción de señales visuales falló (%s); "
                "el motor sigue sin el canal visual.",
                error,
            )
            return None

    def _semantic() -> bool:
        try:
            from . import semantic
        except ImportError:
            return False
        try:
            semantic.preload_minilm_model(
                on_semantic_download_progress, should_cancel=should_cancel
            )
            return True
        except semantic.SemanticCancelled as error:
            raise ProjectCancelled() from error
        except Exception as error:  # noqa: BLE001 - sin MiniLM, solo TF-IDF
            logger.warning(
                "La capa semántica MiniLM no está disponible (%s); "
                "la deduplicación sigue solo con TF-IDF.",
                error,
            )
            return False

    def _audio() -> Any:
        if input_path is None or not ffmpeg_path:
            return None
        try:
            from . import moments
        except ImportError:
            return None
        return moments.extract_audio_timeline(input_path, ffmpeg_path, duration)

    jobs = {
        "eventos": _events,
        "hablantes": _speakers,
        "visual": _visual,
        "semántico": _semantic,
        "audio": _audio,
    }
    results: dict[str, Any] = {}
    cancelled: ProjectCancelled | None = None
    with ThreadPoolExecutor(
        max_workers=len(jobs), thread_name_prefix="kca-channel"
    ) as pool:
        futures = {name: pool.submit(job) for name, job in jobs.items()}
        for name, future in futures.items():
            try:
                results[name] = future.result()
            except ProjectCancelled as error:
                # Se sigue esperando al resto (abortan pronto por
                # should_cancel) y se propaga al final, sin dejar
                # hilos sueltos.
                if cancelled is None:
                    cancelled = error
                results[name] = None
            except Exception as error:  # Defensa: un canal jamás rompe el proyecto.
                logger.warning(
                    "El canal %s falló de forma inesperada (%s); se sigue sin él.",
                    name,
                    error,
                )
                results[name] = None
    if cancelled is not None:
        raise cancelled
    return _SignalChannels(
        event_timeline=results.get("eventos"),
        speaker_timeline=results.get("hablantes"),
        visual=results.get("visual"),
        semantic_ready=bool(results.get("semántico")),
        audio_timeline=results.get("audio"),
    )


class _ChannelsHandle:
    """Extracción de canales iniciada en segundo plano (solape con la transcripción).

    Envuelve el Future de ``_extract_signal_channels``: ``results()``
    espera y devuelve los canales (o propaga ProjectCancelled);
    ``abandon()`` marca la extracción como innecesaria (p. ej. la
    transcripción falló) y espera a que los trabajadores aborten, con
    tope. Ambas son seguras de llamar en cualquier orden y repetidas.
    """

    def __init__(
        self,
        future: Any,
        abandoned: threading.Event,
        executor: ThreadPoolExecutor,
    ) -> None:
        self._future = future
        self._abandoned = abandoned
        self._executor = executor

    def results(self) -> _SignalChannels:
        """Espera a la extracción y devuelve los canales."""
        try:
            return self._future.result()
        finally:
            self._executor.shutdown(wait=True)

    def abandon(self) -> None:
        """Marca la extracción como innecesaria y espera a que aborte."""
        self._abandoned.set()
        try:
            self._future.result(timeout=_CHANNELS_ABANDON_TIMEOUT_SECONDS)
        except Exception:  # noqa: BLE001 - al abandonar, el resultado ya no importa
            logger.debug(
                "La extracción de canales en segundo plano no terminó "
                "al abandonarla; el hilo termina solo."
            )
        finally:
            self._executor.shutdown(wait=False, cancel_futures=True)


def _should_overlap_channels() -> bool:
    """True si la máquina tiene núcleos de sobra para solapar.

    Solapar la transcripción (Whisper: 2 workers + hilos de
    CTranslate2) con los 5 canales (ffmpeg + ONNX) solo compensa con
    núcleos de sobra: en esta VM (4 lógicos, 1,5 de cuota) el
    paralelismo de canales fue neutro (159 s frente a 156 s en serie).
    En el PC de Kelly (Xeon, 28 hilos) el solape ahorra ~100 s por
    proyecto. Con menos de 8 núcleos, el camino en serie de siempre.
    """
    return (os.cpu_count() or 1) >= _OVERLAP_MIN_CPUS


def _signal_models_cached() -> bool:
    """True si YAMNet, diarización y MiniLM pueden correr sin descargar nada.

    Puerta del solape con la transcripción: solo se solapa cuando no
    hay descargas pendientes (una descarga en segundo plano no avisaría
    al usuario con el progreso habitual y, en un video sin voz,
    descargaría ~70 MB de modelos para nada). Nunca descarga ni lanza.
    """
    try:
        from . import audio_events, diarization, semantic
    except ImportError:
        return False
    try:
        return (
            audio_events.is_tier1_available()
            and diarization.is_diarization_available()
            and semantic.is_semantic_available()
        )
    except Exception:  # noqa: BLE001 - sin certeza de caché, camino en serie
        return False


def _start_signal_channels(
    input_path: Path,
    duration: float,
    ffmpeg_path: str,
    should_cancel: Callable[[], bool] | None = None,
) -> _ChannelsHandle:
    """Inicia la extracción de canales en un hilo (solape con la transcripción).

    Los canales no necesitan la transcripción: solo el video y su
    duración. Solo se llama cuando los modelos ya están en caché (ver
    ``_signal_models_cached``), así que no hay descargas ni progreso
    que avisar. Si la transcripción falla o se cancela, el llamador
    abandona el handle y los trabajadores abortan en su siguiente
    chequeo de cancelación.
    """
    abandoned = threading.Event()

    def _combined_cancel() -> bool:
        if abandoned.is_set():
            return True
        return should_cancel() if should_cancel is not None else False

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kca-channels")
    future = executor.submit(
        _extract_signal_channels,
        input_path,
        duration,
        ffmpeg_path,
        _combined_cancel,
    )
    return _ChannelsHandle(future, abandoned, executor)


def _propose_valid_signals(
    transcript: list[TranscriptSegment],
    project: ProjectConfig,
    duration: float,
    input_path: Path | None,
    ffmpeg_path: str | None,
    should_cancel: Callable[[], bool] | None = None,
    *,
    on_event_download_progress: Callable[[float], None] | None = None,
    on_profile: Callable[[str], None] | None = None,
    on_semantic_download_progress: Callable[[float], None] | None = None,
    on_diarization_download_progress: Callable[[float], None] | None = None,
    channels: _ChannelsHandle | None = None,
    fast: bool = False,
) -> list[ClipProposal]:
    """Propuestas del motor de señales (Fases A–D, v1.2.0).

    Importación perezosa de backend.moments: numpy solo se necesita aquí
    (en producción siempre está, vía mediapipe/opencv/faster-whisper); si
    falta, el proyecto falla con un mensaje claro en vez de romper la
    importación de todo el pipeline. Cero clips es un resultado válido
    del motor de señales: se devuelve [] sin error.

    Tier 1 (Fase B) es opcional y jamás rompe un proyecto: si el módulo
    de eventos no se puede importar, o si get_event_timeline devuelve
    None (sin runtime, sin modelo, sin red, sin audio), el motor sigue
    con Tier 0. Solo la cancelación del usuario se propaga. Lo mismo
    vale para las señales visuales (Fase C, 0 MB, ffmpeg + YuNet), la
    diarización (Fase C, ~33 MB bajo demanda) y la capa semántica
    MiniLM (Fase D, ~23 MB bajo demanda): cualquier fallo deja el canal
    apagado y el motor sigue con el resto.

    Los 5 canales se extraen en paralelo (``_extract_signal_channels``)
    o se reutilizan si ``channels`` trae una extracción iniciada en
    segundo plano durante la transcripción. ``fast`` activa el modo
    rápido (Tier 0: solo audio + texto, sin canales opcionales).
    """
    if should_cancel is not None and should_cancel():
        raise ProjectCancelled()
    try:
        from . import moments
    except ImportError as error:
        raise MediaProcessingError(
            "El motor de señales locales necesita numpy, que no está "
            "instalado. Reinstala la aplicación."
        ) from error
    if fast:
        if channels is not None:
            # El modo rápido no usa los canales: se abandona la
            # extracción en segundo plano en vez de esperarla.
            channels.abandon()
        return _propose_valid_signals_fast(
            transcript,
            project,
            duration,
            input_path,
            ffmpeg_path,
            should_cancel,
            on_profile,
        )
    if channels is not None:
        # Extracción iniciada en segundo plano durante la transcripción.
        extracted = channels.results()
    elif transcript:
        # Sin frases no hay candidatos: no merece descargar modelos
        # (~15 MB YAMNet, ~33 MB diarización, ~23 MB MiniLM) ni correr
        # inferencias para un resultado vacío.
        extracted = _extract_signal_channels(
            input_path,
            duration,
            ffmpeg_path,
            should_cancel,
            on_event_download_progress=on_event_download_progress,
            on_diarization_download_progress=on_diarization_download_progress,
            on_semantic_download_progress=on_semantic_download_progress,
        )
    else:
        extracted = _SignalChannels()
    visual = extracted.visual
    try:
        from . import semantic as semantic_module
    except ImportError:
        semantic_module = None
    try:
        proposals = moments.detect_moments(
            transcript,
            project,
            duration,
            input_path=input_path,
            ffmpeg_path=ffmpeg_path,
            audio_timeline=extracted.audio_timeline,
            event_timeline=extracted.event_timeline,
            speaker_timeline=extracted.speaker_timeline,
            visual_timeline=visual.combined if visual is not None else None,
            face_timeline=visual.face if visual is not None else None,
            scene_cut_timeline=visual.scene_cuts if visual is not None else None,
            motion_timeline=visual.motion if visual is not None else None,
            visual_stats=visual.stats if visual is not None else None,
            should_cancel=should_cancel,
            on_profile=on_profile,
            enable_semantic=extracted.semantic_ready,
        )
    except Exception as error:  # noqa: BLE001 - solo se traduce la cancelación semántica
        if semantic_module is not None and isinstance(
            error, semantic_module.SemanticCancelled
        ):
            raise ProjectCancelled() from error
        raise
    if not proposals:
        return []
    valid, _errors = _validate_and_complete_proposals(
        proposals, transcript, duration, project.duration, source="de señales"
    )
    return valid


def _propose_valid_signals_fast(
    transcript: list[TranscriptSegment],
    project: ProjectConfig,
    duration: float,
    input_path: Path | None,
    ffmpeg_path: str | None,
    should_cancel: Callable[[], bool] | None = None,
    on_profile: Callable[[str], None] | None = None,
) -> list[ClipProposal]:
    """Propuestas Tier 0 (modo rápido): solo audio + texto.

    Sin YAMNet, diarización, visual ni semántica: la transcripción +
    el timeline de audio (RMS 1 Hz) + el scoring por señales de texto,
    ritmo y gancho. Es el mismo motor con los canales opcionales
    apagados (el motor completo es el predeterminado); en un video de
    320 s ahorra ~170 s de canales en esta VM. ``input_path=None`` en
    ``detect_moments`` evita que extraiga lo visual por su cuenta.
    """
    from . import moments

    audio_timeline = None
    if input_path is not None and ffmpeg_path:
        audio_timeline = moments.extract_audio_timeline(input_path, ffmpeg_path, duration)
    proposals = moments.detect_moments(
        transcript,
        project,
        duration,
        input_path=None,
        audio_timeline=audio_timeline,
        should_cancel=should_cancel,
        on_profile=on_profile,
        enable_semantic=False,
    )
    if not proposals:
        return []
    valid, _errors = _validate_and_complete_proposals(
        proposals, transcript, duration, project.duration, source="de señales"
    )
    return valid


def _propose_valid_clips_llm(
    transcript: list[TranscriptSegment],
    project: ProjectConfig,
    duration: float,
    llm: ResolvedLLM,
    should_cancel: Callable[[], bool] | None = None,
    *,
    on_rate_limit: Callable[[float, int], None] | None = None,
    token_pacer: _TokenPacer | None = None,
) -> list[ClipProposal]:
    """Propuestas válidas para todo el video, por ventanas de 5 min (LLM).

    token_pacer regula el ritmo de las llamadas LLM (presupuesto TPM);
    si es None no hay pacing (process_project siempre pasa uno).
    """
    if should_cancel is not None and should_cancel():
        raise ProjectCancelled()
    if duration <= _LLM_WINDOW_SECONDS:
        return _propose_valid_window(
            transcript,
            project,
            duration,
            llm,
            on_rate_limit=on_rate_limit,
            token_pacer=token_pacer,
        )

    stride = _LLM_WINDOW_SECONDS - _LLM_WINDOW_OVERLAP_SECONDS
    proposals: list[ClipProposal] = []
    errors: list[str] = []
    window_start = 0.0
    window_index = 0
    while window_start < duration:
        if should_cancel is not None and should_cancel():
            raise ProjectCancelled()
        window_end = min(duration, window_start + _LLM_WINDOW_SECONDS)
        window_transcript = _slice_transcript(transcript, window_start, window_end)
        window_duration = window_end - window_start
        try:
            window_proposals = _propose_valid_window(
                window_transcript,
                project,
                window_duration,
                llm,
                on_rate_limit=on_rate_limit,
                token_pacer=token_pacer,
            )
        except (
            MediaProcessingError,
            LLMResponseError,
            # Errores de parseo/JSON del LLM: LLMResponseError es el tipo
            # canónico desde v1.0.11; ValueError/TypeError cubren las
            # respuestas sin JSON válido que aún se propagan como errores
            # de parseo crudos. Un fallo en una ventana se registra y se
            # sigue con las restantes en vez de abortar el proyecto (A3).
            ValueError,
            TypeError,
        ) as error:
            errors.append(f"ventana {window_index + 1}: {error}")
            logger.warning("Se omite una ventana de transcripción: %s", errors[-1])
            window_proposals = []

        owned_duration = (
            window_duration
            if window_end >= duration
            else min(float(stride), window_duration)
        )
        proposals.extend(
            proposal.model_copy(
                update={
                    "inicio": proposal.inicio + window_start,
                    "fin": proposal.fin + window_start,
                }
            )
            for proposal in window_proposals
            if proposal.inicio < owned_duration
        )
        if window_end >= duration:
            break
        window_start += stride
        window_index += 1

    selected: list[ClipProposal] = []
    for proposal in sorted(
        proposals, key=lambda clip: (-clip.puntuacion, clip.inicio)
    ):
        if any(
            proposal.inicio < existing.fin and existing.inicio < proposal.fin
            for existing in selected
        ):
            logger.warning(
                "Se descarta el clip %s por solaparse con otra ventana.",
                proposal.titulo,
            )
            continue
        selected.append(proposal)
        if len(selected) == 10:
            break
    if not selected and errors:
        raise MediaProcessingError(
            "No se encontraron propuestas válidas en las ventanas analizadas. "
            + "; ".join(errors[:3])
        )
    return sorted(selected, key=lambda clip: clip.inicio)


def _slice_transcript(
    transcript: list[TranscriptSegment],
    start: float,
    end: float,
) -> list[TranscriptSegment]:
    sliced: list[TranscriptSegment] = []
    for segment in transcript:
        segment_start = max(start, segment.start)
        segment_end = min(end, segment.end)
        if segment_end <= segment_start:
            continue
        words = [
            Word(
                start=max(start, word.start) - start,
                end=min(end, word.end) - start,
                text=word.text,
            )
            for word in segment.words
            if word.start < end and word.end > start
        ]
        sliced.append(
            TranscriptSegment(
                start=segment_start - start,
                end=segment_end - start,
                text=segment.text,
                words=words,
            )
        )
    return sliced


def _propose_valid_window(
    transcript: list[TranscriptSegment],
    project: ProjectConfig,
    duration: float,
    llm: ResolvedLLM,
    *,
    on_rate_limit: Callable[[float, int], None] | None = None,
    token_pacer: _TokenPacer | None = None,
) -> list[ClipProposal]:
    # Cada llamada al LLM (la inicial y el reintento por validación) pasa
    # por el pacer: son peticiones independientes que consumen TPM.
    if token_pacer is not None:
        token_pacer.wait_before_call(_estimate_request_tokens(transcript))
    candidates = propose_clips(
        transcript, project, duration, llm, on_rate_limit=on_rate_limit
    )
    proposals, errors = _validate_and_complete_proposals(
        candidates, transcript, duration, project.duration
    )
    if errors:
        if token_pacer is not None:
            token_pacer.wait_before_call(_estimate_request_tokens(transcript))
        retry_candidates = propose_clips(
            transcript,
            project,
            duration,
            llm,
            validation_feedback=errors,
            on_rate_limit=on_rate_limit,
        )
        retried, retry_errors = _validate_and_complete_proposals(
            retry_candidates, transcript, duration, project.duration
        )
        proposals, merge_errors = _validate_proposals(
            [*proposals, *retried], duration, project.duration
        )
        errors.extend(retry_errors)
        errors.extend(merge_errors)
    if not proposals and errors:
        raise MediaProcessingError(
            "El modelo no propuso clips válidos tras un reintento. "
            + "; ".join(errors[:5])
        )
    return proposals


def _ends_sentence(text: str) -> bool:
    return text.rstrip("\"'”’)]}").endswith(
        (".", "!", "?", "…", "。", "！", "？")
    )


def _complete_clip_ends(
    proposals: list[ClipProposal],
    transcript: list[TranscriptSegment],
    duration: float,
    choice: str,
) -> list[ClipProposal]:
    _, maximum = _duration_bounds(choice)
    words = sorted(
        (word for segment in transcript for word in segment.words),
        key=lambda word: (word.start, word.end),
    )
    ordered = sorted(proposals, key=lambda clip: clip.inicio)
    completed: list[ClipProposal] = []

    for index, proposal in enumerate(ordered):
        end_limit = min(duration, proposal.inicio + maximum)
        if index + 1 < len(ordered):
            end_limit = min(end_limit, ordered[index + 1].inicio)
        if end_limit <= proposal.fin:
            completed.append(proposal)
            continue

        relevant = [
            word
            for word in words
            if proposal.inicio < word.end and word.start < end_limit
        ]
        words_before_end = [word for word in relevant if word.start < proposal.fin]
        # El clip se da por bueno solo si el fin NO corta ninguna palabra
        # (los timestamps de Whisper pueden solaparse: el fin puede caer
        # dentro de una palabra aunque la última palabra "termine en
        # frase") y la última palabra cierra una oración (v1.2.1).
        cuts_word = any(word.start < proposal.fin < word.end for word in relevant)
        if not cuts_word and words_before_end and _ends_sentence(words_before_end[-1].text):
            completed.append(proposal)
            continue

        boundary: float | None = None
        for word_index, word in enumerate(relevant):
            if word.end <= proposal.fin or word.end > min(end_limit, proposal.fin + 10):
                continue
            has_sentence_end = _ends_sentence(word.text)
            has_pause = (
                word_index + 1 < len(relevant)
                and relevant[word_index + 1].start - word.end >= 0.8
            )
            if has_sentence_end or has_pause:
                boundary = word.end
                break

        if boundary is None:
            next_word = next(
                (
                    word
                    for word in relevant
                    if proposal.fin < word.end <= min(end_limit, proposal.fin + 2)
                ),
                None,
            )
            if next_word is not None:
                boundary = next_word.end

        if boundary is None:
            completed.append(proposal)
            continue

        # La cola de +0.25 s no debe entrar en la palabra siguiente
        # (v1.2.1: en el video de Kelly, 33.36 + 0.25 = 33.61 caía dentro
        # de "truquito"): se acota al inicio de cualquier palabra que el
        # tramo de cola alcanzaría.
        tail_end = boundary + 0.25
        for word in relevant:
            if word.start < tail_end and word.end > boundary:
                tail_end = min(tail_end, word.start)
        completed_end = min(end_limit, tail_end)
        if completed_end <= proposal.fin:
            completed.append(proposal)
            continue
        completed.append(proposal.model_copy(update={"fin": completed_end}))

    return completed


def _progress_for_llm(
    settings: AppSettings,
    project_id: str,
    provider: str,
) -> None:
    _publish(
        settings,
        project_id,
        "processing",
        "finding",
        None,
        f"Analizando la transcripción con {provider}…",
    )


def _progress_for_signals(
    settings: AppSettings,
    project_id: str,
) -> None:
    _publish(
        settings,
        project_id,
        "processing",
        "finding",
        None,
        "Analizando momentos con señales locales (sin IA)…",
    )


def _render_projects(
    settings: AppSettings,
    project_id: str,
    input_path: Path,
    project_config: ProjectConfig,
    proposal_records: list[dict[str, Any] | ClipProposal],
    transcript: list[TranscriptSegment],
    ffmpeg_path: str,
    owner_id: str,
    existing_clips: list[dict[str, Any]] | None = None,
    output_dir: Path | None = None,
    persist_clips: bool = True,
    should_cancel: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:
    # Los parámetros nuevos van al final y se pasan por posición desde
    # process_project: algunos tests sustituyen esta función por un mock
    # que acepta *args pero no kwargs.
    #
    # clips_json NO se escribe por clip (Fase 2, B-R3): reescribir el
    # JSON completo tras cada clip es O(n²) en escrituras a SQLite. Los
    # clips se acumulan en `rendered` y process_project escribe una sola
    # vez al final (o al cancelar, con lo acumulado hasta ese momento).
    # El parámetro persist_clips queda como no-op por compatibilidad de
    # firma: persistir es ahora siempre al final, nunca por clip.
    del persist_clips
    project_dir = input_path.parent
    if output_dir is None:
        output_dir = project_dir / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    proposal_records = [
        {
            "id": str(uuid.uuid4()),
            "proposal": record.model_dump(mode="json"),
        }
        if isinstance(record, ClipProposal)
        else record
        for record in proposal_records
    ]
    replacing_ids = {str(record["id"]) for record in proposal_records}
    original_by_id = {
        str(clip.get("id")): clip for clip in existing_clips or []
    }
    rendered = [
        clip
        for clip in existing_clips or []
        if str(clip.get("id")) not in replacing_ids
    ]
    count = len(proposal_records)
    if not count:
        return rendered

    for index, record in enumerate(proposal_records):
        if should_cancel is not None and should_cancel():
            # Se conservan los clips ya renderizados en esta tanda Y las
            # entradas originales de los que aún no se re-renderizaron
            # (en el reintento de fallidos, los fallidos pendientes no
            # deben desaparecer del listado al cancelar).
            cancelled_clips = list(rendered)
            for pending in proposal_records[index:]:
                original = original_by_id.get(str(pending["id"]))
                if original is not None:
                    cancelled_clips.append(original)
            cancelled_clips.sort(key=lambda item: float(item.get("start") or 0))
            raise ProjectCancelled(cancelled_clips)
        clip_id = str(record["id"])
        proposal = ClipProposal.model_validate(record["proposal"])
        output_path = output_dir / f"{clip_id}.mp4"
        subtitle_path = output_dir / f"{clip_id}.ass"
        clip = {
            "id": clip_id,
            "title": proposal.titulo,
            "start": proposal.inicio,
            "end": proposal.fin,
            "duration": proposal.fin - proposal.inicio,
            "score": proposal.puntuacion,
            "viral_reason": proposal.razon_viral,
            "description": proposal.descripcion,
            "hashtags": proposal.hashtags,
            "keywords": proposal.palabras_clave,
            "emojis": proposal.emojis if project_config.options.emojis else [],
            "resolution": 720,
            "filename": output_path.name,
            "transcript": [],
            "width": 0,
            "height": 0,
        }
        if proposal.signal_breakdown is not None:
            # Desglose "por qué este clip" del motor de señales (la UI lo
            # muestra; las propuestas del LLM no lo traen).
            clip["signal_breakdown"] = proposal.signal_breakdown
        clip_start_progress = 72 + round(index / count * 26)
        _publish(
            settings,
            project_id,
            "processing",
            "editing",
            clip_start_progress,
            f"Preparando clip {index + 1} de {count}…",
        )

        def on_encode_progress(percent: float, clip_index: int = index) -> None:
            overall = 72 + round(((clip_index + percent / 100) / count) * 26)
            _publish(
                settings,
                project_id,
                "processing",
                "editing",
                min(overall, 98),
                f"Editando clip {clip_index + 1} de {count}…",
            )

        def on_analysis_progress(
            percent: float,
            clip_index: int = index,
            progress: int = clip_start_progress,
        ) -> None:
            _publish(
                settings,
                project_id,
                "processing",
                "editing",
                progress,
                f"Analizando reencuadre del clip {clip_index + 1} de {count}… "
                f"{percent:.0f}%",
            )

        broll_path: Path | None = None
        try:
            render_config = project_config.model_copy(
                update={
                    "subtitle_template": database.resolve_subtitle_template(
                        settings.database_path,
                        project_config.subtitle_template,
                        owner_id,
                    )
                }
            )
            broll_metadata: dict[str, str] | None = None
            if project_config.options.broll:
                broll_path = project_dir / "broll" / f"{clip_id}.mp4"
                broll_metadata = download_broll(
                    settings,
                    " ".join(proposal.palabras_clave) or proposal.titulo,
                    broll_path,
                )
            clip_duration, clip_transcript = render_clip(
                input_path,
                output_path,
                subtitle_path,
                proposal,
                render_config,
                transcript,
                ffmpeg_path,
                on_encode_progress,
                broll_path=broll_path,
                on_analysis_progress=on_analysis_progress,
            )
            metadata = probe_media(output_path, shutil.which("ffprobe") or "ffprobe")
            clip.update(
                {
                    "duration": clip_duration,
                    "width": metadata["width"],
                    "height": metadata["height"],
                    "audio_enhanced": False,
                    "transcript": clip_transcript,
                    "status": "ready",
                    **(
                        {
                            "broll_provider": broll_metadata["provider"],
                            "broll_creator": broll_metadata["creator"],
                            "broll_attribution_url": broll_metadata["attribution_url"],
                        }
                        if broll_metadata
                        else {}
                    ),
                }
            )
        except Exception as error:
            logger.exception("No se pudo renderizar el clip %s del proyecto %s", clip_id, project_id)
            clip.update(
                {
                    "status": "failed",
                    "error": str(error).strip() or "No se pudo renderizar este clip.",
                }
            )
            for partial_path in (output_path, subtitle_path, broll_path):
                if partial_path is None:
                    continue
                try:
                    partial_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "No se pudo eliminar el archivo parcial %s", partial_path
                    )
        rendered = [item for item in rendered if str(item.get("id")) != clip_id]
        rendered.append(clip)
        rendered.sort(key=lambda item: float(item.get("start") or 0))
    return rendered


def process_project(
    project_id: str,
    request_llm: LLMConfig | None,
    settings: AppSettings | None = None,
    *,
    retry_failed_only: bool = False,
) -> None:
    settings = settings or get_settings()
    project_dir = settings.data_dir / "projects" / project_id
    input_path = project_dir / "source"
    # Extracción de canales iniciada en segundo plano durante la
    # transcripción (solo con núcleos de sobra y modelos en caché):
    # se abandona en el finally si la transcripción falla o se cancela.
    channels_handle: _ChannelsHandle | None = None

    def should_cancel() -> bool:
        return _is_cancelled(settings, project_id)

    try:
        stored = database.get_project(settings.database_path, project_id)
        if stored is None or not input_path.is_file():
            raise ProjectNotFoundError("No se encontró el video guardado para este proyecto.")
        _check_not_cancelled(settings, project_id)
        _check_free_disk_space(project_dir)
        project_config = ProjectConfig.model_validate(stored["config"])
        # El motor de señales (por defecto) no necesita LLM: el proveedor
        # solo se resuelve en modo "llm". En "signals+ai" se resuelve más
        # abajo, best-effort, solo para pulir títulos.
        llm = (
            None
            if retry_failed_only or project_config.moment_engine != "llm"
            else _resolve_llm_for_project(stored, request_llm, settings)
        )
        ffmpeg_path = shutil.which("ffmpeg")
        ffprobe_path = shutil.which("ffprobe")
        if not ffmpeg_path or not ffprobe_path:
            raise MediaProcessingError(
                "FFmpeg y ffprobe son necesarios. Instálalos y comprueba que estén en PATH."
            )

        stored_metadata = stored.get("metadata")
        stored_transcript = stored.get("transcript")
        transcript_cache_valid = (
            isinstance(stored_metadata, dict)
            and stored_metadata.get("duration")
            and isinstance(stored_transcript, list)
            and stored_transcript
            and _transcript_cache_matches(stored_metadata, project_config, settings)
        )
        if transcript_cache_valid:
            metadata = stored_metadata
            _check_video_duration(float(metadata["duration"]))
            transcript = [
                TranscriptSegment.model_validate(segment) for segment in stored_transcript
            ]
            if not any(segment.words for segment in transcript):
                raise MediaProcessingError(
                    "La transcripción guardada no contiene tiempos por palabra. Reintenta el análisis."
                )
            _publish(
                settings,
                project_id,
                "processing",
                "finding",
                100,
                "Reutilizando la transcripción real guardada…",
            )
        else:
            if (
                isinstance(stored_metadata, dict)
                and isinstance(stored_transcript, list)
                and stored_transcript
            ):
                logger.info(
                    "La transcripción guardada del proyecto %s se descarta porque "
                    "cambió el modelo o el idioma de Whisper; se vuelve a transcribir.",
                    project_id,
                )
            _publish(
                settings,
                project_id,
                "processing",
                "process_video",
                0,
                "Leyendo metadatos y creando miniatura con ffprobe y FFmpeg…",
            )
            metadata = probe_media(input_path, ffprobe_path)
            _check_video_duration(float(metadata["duration"]))
            project_dir.mkdir(parents=True, exist_ok=True)
            create_thumbnail(
                input_path,
                project_dir / "thumbnail.jpg",
                float(metadata["duration"]),
                ffmpeg_path,
            )
            database.update_project(settings.database_path, project_id, metadata=metadata)
            # Solape con la transcripción (v1.3.1): los canales de
            # señales no necesitan la transcripción, así que con
            # núcleos de sobra y modelos ya en caché empiezan ahora,
            # en segundo plano, mientras Whisper transcribe (~107 s
            # en un video de 320 s en esta VM; en el PC de Kelly el
            # solape ahorra ~100 s por proyecto). Sin núcleos de sobra
            # o con modelos por descargar, el camino en serie de
            # siempre (los canales arrancan tras la transcripción).
            if (
                project_config.moment_engine in {"signals", "signals+ai"}
                and not project_config.fast_detection
                and _should_overlap_channels()
                and _signal_models_cached()
            ):
                channels_handle = _start_signal_channels(
                    input_path,
                    float(metadata["duration"]),
                    ffmpeg_path,
                    should_cancel,
                )
            _publish(
                settings,
                project_id,
                "processing",
                "finding",
                0,
                "Transcribiendo audio con faster-whisper en CPU…",
            )

            def on_transcription_progress(percent: float) -> None:
                progress = min(100, max(0, round(percent)))
                _publish(
                    settings,
                    project_id,
                    "processing",
                    "finding",
                    progress,
                    f"Transcribiendo el audio… {percent:.0f}%",
                )

            def on_whisper_download_progress(percent: float) -> None:
                progress = min(100, max(0, round(percent)))
                _publish(
                    settings,
                    project_id,
                    "processing",
                    "finding",
                    progress,
                    f"Descargando el modelo Whisper… {percent:.0f}%",
                )

            # Caché por hash de archivo (B-R5): si este mismo video ya se
            # transcribió en otro proyecto con el mismo modelo e idioma de
            # Whisper, se reutiliza sin volver a transcribir. La clave
            # incluye modelo e idioma, así que un cambio en cualquiera de
            # ellos invalida la entrada (regla de la Fase 1, A6).
            cached_transcript: dict[str, Any] | None = None
            file_hash: str | None = None
            try:
                file_hash = _file_sha256(input_path)
                cached_transcript = database.get_cached_transcript(
                    settings.database_path,
                    file_hash,
                    project_config.whisper_model or settings.whisper_model,
                    project_config.whisper_language,
                )
            except Exception as error:
                logger.warning(
                    "No se pudo consultar la caché de transcripción por hash: %s",
                    error,
                )
            if cached_transcript is not None:
                transcript = [
                    TranscriptSegment.model_validate(segment)
                    for segment in cached_transcript["transcript"]
                ]
                if not any(segment.words for segment in transcript):
                    raise MediaProcessingError(
                        "La transcripción guardada no contiene tiempos por palabra. Reintenta el análisis."
                    )
                language = str(cached_transcript["language"])
                _publish(
                    settings,
                    project_id,
                    "processing",
                    "finding",
                    100,
                    "Reutilizando la transcripción de otro proyecto con el mismo video…",
                )
            else:
                transcript, language = transcribe(
                    input_path,
                    project_config.whisper_model or settings.whisper_model,
                    on_transcription_progress,
                    language=project_config.whisper_language,
                    on_model_download_progress=on_whisper_download_progress,
                )
            _check_not_cancelled(settings, project_id)
            transcript_data = [segment.model_dump() for segment in transcript]
            metadata["language"] = language
            # Se registra con qué modelo e idioma se transcribió para
            # invalidar la caché si el usuario los cambia (A6).
            metadata["whisper_model"] = (
                project_config.whisper_model or settings.whisper_model
            )
            metadata["whisper_language"] = project_config.whisper_language
            database.update_project(
                settings.database_path,
                project_id,
                metadata=metadata,
                transcript=transcript_data,
            )
            if cached_transcript is None and file_hash is not None:
                # Caché por hash de archivo (B-R5): el mismo video en otro
                # proyecto reutiliza esta transcripción sin volver a llamar
                # a Whisper. Best-effort: si el guardado falla, el
                # proyecto sigue adelante con su propia transcripción.
                try:
                    database.save_cached_transcript(
                        settings.database_path,
                        file_hash,
                        project_config.whisper_model or settings.whisper_model,
                        project_config.whisper_language,
                        language,
                        transcript_data,
                        float(metadata["duration"]),
                    )
                except Exception as error:
                    logger.warning(
                        "No se pudo guardar la transcripción en la caché por hash: %s",
                        error,
                    )
        previous_clips = stored.get("clips") or []
        existing_clips = previous_clips
        if retry_failed_only:
            failed_ids = {
                str(clip["id"])
                for clip in existing_clips
                if clip.get("status") == "failed"
            }
            stored_proposals = (
                metadata.get("clip_proposals")
                if isinstance(metadata, dict)
                else None
            )
            proposal_records = [
                proposal
                for proposal in stored_proposals or []
                if isinstance(proposal, dict)
                and str(proposal.get("id")) in failed_ids
                and isinstance(proposal.get("proposal"), dict)
            ]
            if not proposal_records:
                raise MediaProcessingError(
                    "No hay propuestas guardadas para reintentar los clips fallidos."
                )
            _publish(
                settings,
                project_id,
                "processing",
                "editing",
                72,
                f"Reintentando {len(proposal_records)} clip(s) fallido(s) sin volver a llamar a la IA…",
            )
        else:
            # Perfil de contenido (Fase C): lo rellena el callback
            # on_profile del camino de señales; vacío en modo "llm".
            content_profile_holder: dict[str, str] = {}
            if project_config.moment_engine == "llm":
                if llm is None:
                    raise LLMConfigurationError("No se pudo configurar el proveedor de IA.")
                _progress_for_llm(settings, project_id, llm.provider)

                def on_rate_limit(wait_seconds: float, _attempt: int) -> None:
                    # Aviso visible mientras se espera por un 429: el usuario
                    # ve que la app está esperando y no colgada (Fase 2).
                    _publish(
                        settings,
                        project_id,
                        "processing",
                        "finding",
                        None,
                        f"Límite de {llm.provider} alcanzado, esperando "
                        f"{wait_seconds:.0f}s antes de reintentar…",
                    )

                def on_pacing_wait(wait_seconds: float) -> None:
                    # Aviso visible mientras el pacer TPM espera entre
                    # ventanas: sin él, la pausa de hasta un minuto parecería
                    # un cuelgue (Fase 2).
                    _publish(
                        settings,
                        project_id,
                        "processing",
                        "finding",
                        None,
                        f"Esperando {wait_seconds:.0f}s para respetar el límite de {llm.provider}…",
                    )

                token_pacer = _TokenPacer(on_wait=on_pacing_wait)
                proposals = _propose_valid_clips(
                    transcript,
                    project_config,
                    float(metadata["duration"]),
                    llm,
                    should_cancel,
                    on_rate_limit=on_rate_limit,
                    token_pacer=token_pacer,
                )
                engine_used = "llm"
            else:
                _progress_for_signals(settings, project_id)

                def on_event_download_progress(percent: float) -> None:
                    # Aviso visible mientras se descarga el modelo YAMNet
                    # (~15 MB, solo la primera vez): sin él, la descarga
                    # parecería un cuelgue (Fase B, Tier 1).
                    _publish(
                        settings,
                        project_id,
                        "processing",
                        "finding",
                        None,
                        f"Descargando el modelo de eventos de audio ({percent:.0f}%)…",
                    )

                def on_semantic_download_progress(percent: float) -> None:
                    # Aviso visible mientras se descarga el modelo MiniLM
                    # (~23 MB, solo la primera vez): deduplicación por
                    # paráfrasis (Fase D).
                    _publish(
                        settings,
                        project_id,
                        "processing",
                        "finding",
                        None,
                        f"Descargando el modelo semántico ({percent:.0f}%)…",
                    )

                def on_diarization_download_progress(percent: float) -> None:
                    # Aviso visible mientras se descargan los modelos de
                    # diarización (~33 MB, solo la primera vez): turnos
                    # de hablante (Fase C).
                    _publish(
                        settings,
                        project_id,
                        "processing",
                        "finding",
                        None,
                        f"Descargando el modelo de diarización ({percent:.0f}%)…",
                    )

                def on_content_profile(profile: str) -> None:
                    content_profile_holder["profile"] = profile

                proposals = _propose_valid_clips(
                    transcript,
                    project_config,
                    float(metadata["duration"]),
                    None,
                    should_cancel,
                    input_path=input_path,
                    ffmpeg_path=ffmpeg_path,
                    on_event_download_progress=on_event_download_progress,
                    on_profile=on_content_profile,
                    on_semantic_download_progress=on_semantic_download_progress,
                    on_diarization_download_progress=on_diarization_download_progress,
                    channels=channels_handle,
                    fast=project_config.fast_detection,
                )
                engine_used = "signals"
                if project_config.moment_engine == "signals+ai" and proposals:
                    # Pulido best-effort con UNA llamada corta al LLM: sin
                    # clave configurada, o si la llamada falla, se conservan
                    # los títulos por plantilla sin error.
                    polish_llm: ResolvedLLM | None = None
                    try:
                        polish_llm = _resolve_llm_for_project(
                            stored, request_llm, settings
                        )
                    except (
                        LLMConfigurationError,
                        CredentialStoreError,
                        ValidationError,
                    ) as error:
                        logger.info(
                            "Sin proveedor de IA para el pulido de títulos; "
                            "se conservan los títulos por plantilla: %s",
                            error,
                        )
                    if polish_llm is not None:
                        def on_polish_rate_limit(
                            wait_seconds: float, _attempt: int
                        ) -> None:
                            assert polish_llm is not None
                            _publish(
                                settings,
                                project_id,
                                "processing",
                                "finding",
                                None,
                                f"Límite de {polish_llm.provider} alcanzado, "
                                f"esperando {wait_seconds:.0f}s antes de "
                                "reintentar…",
                            )

                        from . import moments

                        polished = moments.polish_titles(
                            proposals,
                            transcript,
                            polish_llm,
                            on_rate_limit=on_polish_rate_limit,
                        )
                        if polished is not None:
                            proposals = polished
                            engine_used = "signals+ai"
            proposal_records = [
                {
                    "id": str(uuid.uuid4()),
                    "proposal": proposal.model_dump(mode="json"),
                }
                for proposal in proposals
            ]
            metadata = dict(metadata or {})
            metadata["clip_proposals"] = proposal_records
            # Motor efectivo registrado por proyecto (la UI lo muestra en
            # resultados). En "signals+ai" sin clave o con pulido fallido,
            # el motor efectivo es "signals".
            metadata["moment_engine"] = engine_used
            if engine_used in {"signals", "signals+ai"}:
                metadata["signal_score_threshold"] = (
                    project_config.signal_score_threshold
                )
                if content_profile_holder.get("profile"):
                    metadata["content_profile"] = content_profile_holder["profile"]
            database.update_project(
                settings.database_path, project_id, metadata=metadata
            )
            if not proposals:
                # No se toca clips_json: los clips anteriores se conservan.
                empty_message = (
                    "El análisis terminó. No se encontraron momentos que "
                    "cumplan los criterios solicitados."
                )
                if engine_used == "signals":
                    empty_message += (
                        " Puedes bajar el umbral de puntuación en la "
                        "configuración del proyecto y reprocesar."
                    )
                _publish(
                    settings,
                    project_id,
                    "completed",
                    "finalize",
                    100,
                    empty_message,
                )
                return

        if retry_failed_only:
            # El reintento renderiza directo en clips/: solo repone los
            # fallidos, sin tocar los que ya estaban listos. clips_json
            # se escribe una sola vez al final (B-R3); si se cancela a
            # mitad, se escriben los acumulados (Fase 1: la cancelación
            # publica los clips ya renderizados) y se relanza para que el
            # handler general publique el estado cancelled.
            try:
                clips = _render_projects(
                    settings,
                    project_id,
                    input_path,
                    project_config,
                    proposal_records,
                    transcript,
                    ffmpeg_path,
                    str(stored["owner_id"]),
                    existing_clips,
                    None,
                    True,
                    should_cancel,
                )
            except ProjectCancelled as cancelled:
                database.update_project(
                    settings.database_path, project_id, clips=cancelled.clips
                )
                raise
            database.update_project(settings.database_path, project_id, clips=clips)
        else:
            # Reproceso completo: se renderiza en clips.new/ y solo se
            # publica con un swap atómico si al menos un clip nuevo quedó
            # listo. Si el reproceso falla o se cancela sin clips listos,
            # los clips viejos y su clips_json quedan intactos (A1).
            staging_dir = project_dir / "clips.new"
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            staging_dir.mkdir(parents=True, exist_ok=True)
            try:
                clips = _render_projects(
                    settings,
                    project_id,
                    input_path,
                    project_config,
                    proposal_records,
                    transcript,
                    ffmpeg_path,
                    str(stored["owner_id"]),
                    [],
                    staging_dir,
                    False,
                    should_cancel,
                )
            except ProjectCancelled as cancelled:
                clips = cancelled.clips
                ready_cancelled = [
                    clip
                    for clip in clips
                    if clip.get("status", "ready") != "failed"
                ]
                if ready_cancelled:
                    try:
                        _swap_staged_clips(project_dir, staging_dir)
                        database.update_project(
                            settings.database_path, project_id, clips=clips
                        )
                    except OSError:
                        logger.exception(
                            "No se pudieron publicar los clips renderizados antes "
                            "de cancelar el proyecto %s; se conservan los anteriores.",
                            project_id,
                        )
                        shutil.rmtree(staging_dir, ignore_errors=True)
                else:
                    shutil.rmtree(staging_dir, ignore_errors=True)
                _publish(
                    settings,
                    project_id,
                    "cancelled",
                    "cancelled",
                    None,
                    "Procesamiento cancelado. Se conservaron los clips ya renderizados.",
                )
                return
            except Exception:
                # Un fallo inesperado del render no debe dejar un staging
                # huérfano ni tocar los clips viejos.
                shutil.rmtree(staging_dir, ignore_errors=True)
                raise
            ready_clips = [
                clip for clip in clips if clip.get("status", "ready") != "failed"
            ]
            if ready_clips:
                try:
                    _swap_staged_clips(project_dir, staging_dir)
                except OSError as error:
                    # Los clips se renderizaron pero no se pudieron
                    # publicar; los viejos quedaron restaurados por el swap.
                    shutil.rmtree(staging_dir, ignore_errors=True)
                    raise MediaProcessingError(
                        "Los clips se renderizaron, pero no se pudieron publicar "
                        "en la carpeta del proyecto. Los clips anteriores se "
                        "conservaron. Inténtalo de nuevo."
                    ) from error
                database.update_project(settings.database_path, project_id, clips=clips)
            else:
                shutil.rmtree(staging_dir, ignore_errors=True)
                if not previous_clips:
                    # Primer procesamiento sin clips previos: se guardan
                    # los fallidos para que el reintento pueda recuperarlos
                    # sin volver a llamar a la IA.
                    database.update_project(
                        settings.database_path, project_id, clips=clips
                    )
                # Con clips previos, clips_json queda intacto: los viejos
                # siguen publicados.
        scheduled_count = 0
        schedule_warning: str | None = None
        scheduled_clips = [
            clip for clip in clips if clip.get("status", "ready") != "failed"
        ]
        if project_config.schedule is not None and scheduled_clips:
            # La programación está desacoplada del estado del proyecto
            # (A7): si falla, el proyecto completa con una advertencia en
            # vez de marcarse como fallido con los clips ya listos.
            try:
                jobs = database.enqueue_schedule_jobs(
                    settings.database_path,
                    project_id,
                    [str(clip["id"]) for clip in scheduled_clips],
                    project_config.schedule.platform,
                    project_config.schedule.scheduled_at.isoformat(),
                    project_config.schedule.visibility,
                )
                scheduled_count = len(jobs)
            except Exception as error:
                logger.exception(
                    "No se pudieron programar las publicaciones del proyecto %s",
                    project_id,
                )
                schedule_warning = (
                    "No se pudieron programar las publicaciones "
                    f"({str(error).strip() or 'error desconocido'}). Los clips "
                    "están listos; programa las publicaciones de nuevo desde la app."
                )
        ready_count = len(scheduled_clips)
        failed_count = len(clips) - ready_count
        if not ready_count and failed_count:
            if previous_clips and not retry_failed_only:
                failure_message = (
                    f"No se pudo renderizar ninguno de los {failed_count} clips nuevos. "
                    "Se conservaron los clips anteriores."
                )
            else:
                failure_message = (
                    f"No se pudo renderizar ninguno de los {failed_count} clips. "
                    "Puedes reintentarlos."
                )
            _publish(
                settings,
                project_id,
                "failed",
                "failed",
                None,
                failure_message,
            )
            return
        completion_message = f"Procesamiento terminado: {ready_count} clips reales listos."
        if failed_count:
            completion_message += (
                f" {failed_count} clip(s) fallaron y pueden reintentarse."
            )
        if scheduled_count:
            completion_message += f" {scheduled_count} publicaciones quedaron programadas."
        if schedule_warning is not None:
            completion_message += f" Advertencia: {schedule_warning}"
            metadata = dict(metadata or {})
            metadata["schedule_warning"] = schedule_warning
            database.update_project(
                settings.database_path, project_id, metadata=metadata
            )
        _publish(
            settings,
            project_id,
            "completed",
            "finalize",
            100,
            completion_message,
        )
    except ProjectCancelled:
        logger.info("Procesamiento cancelado del proyecto %s", project_id)
        _publish(
            settings,
            project_id,
            "cancelled",
            "cancelled",
            None,
            "Procesamiento cancelado por el usuario.",
        )
    except Exception as error:
        logger.exception("Fallo inesperado al procesar el proyecto %s", project_id)
        message = str(error).strip() or "Error inesperado durante el procesamiento."
        _publish(settings, project_id, "failed", "failed", None, message)
    finally:
        # Si la extracción de canales en segundo plano no se consumió
        # (la transcripción falló o se canceló antes de las propuestas),
        # se abandona: los trabajadores abortan en su siguiente chequeo
        # de cancelación. Si ya se consumió, es un no-op.
        if channels_handle is not None:
            channels_handle.abandon()
