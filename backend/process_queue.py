from __future__ import annotations

import asyncio
import logging
import re
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path

from . import database
from .media import terminate_active_renders
from .schemas import LLMConfig

logger = logging.getLogger(__name__)
ProjectProcessor = Callable[[str, LLMConfig | None, bool], Awaitable[None]]

# Proyectos cuya cancelación pidió el usuario. El pipeline corre en hilos
# (asyncio.to_thread) y el endpoint de cancelación en el bucle de eventos,
# así que el conjunto vive a nivel de módulo y se protege con un lock. Es
# solo una vía rápida: la fuente de verdad es el estado persistido
# ("cancelled") en la base de datos, que el pipeline también consulta.
_cancelled_projects: set[str] = set()
_cancelled_lock = threading.Lock()


def request_project_cancellation(project_id: str) -> None:
    """Marca un proyecto como cancelado para que el pipeline se detenga."""
    with _cancelled_lock:
        _cancelled_projects.add(project_id)


def is_project_cancelled(project_id: str) -> bool:
    """Indica si el usuario pidió cancelar el proyecto."""
    with _cancelled_lock:
        return project_id in _cancelled_projects


def clear_project_cancellation(project_id: str) -> None:
    """Olvida una cancelación (al reprocesar o al terminar el trabajo)."""
    with _cancelled_lock:
        _cancelled_projects.discard(project_id)


# --- Cola persistente y reintentos (Fase 2, B-S10) ---------------------------
# Intentos máximos por trabajo (1 intento inicial + 2 reintentos). Solo se
# reintentan los fallos que huelen a transitorios: un error de red, un
# timeout del LLM o un bloqueo de SQLite pueden resolverse solos; un video
# corrupto o una clave inválida no. La detección es heurística sobre el
# mensaje de error del proyecto (ver _TRANSIENT_ERROR_PATTERN); ante la
# duda, no se reintenta.
_MAX_JOB_ATTEMPTS = 3
_RETRY_DELAYS_SECONDS = (5.0, 15.0, 45.0)
_TRANSIENT_ERROR_PATTERN = re.compile(
    r"timeout|timed out|connection|conexi[óo]n|temporarily|temporalmente"
    r"|429|502|503|504|database is locked|locked|network|red\b"
    r"|unavailable|no se pudo conectar|int[ée]ntalo (de nuevo|más tarde)",
    re.IGNORECASE,
)

# Apagado elegante (Fase 2, B-P14): tiempo máximo que stop() espera al
# trabajo en curso antes de matar los ffmpeg hijos y cancelar los workers.
# Los trabajos pendientes no se esperan: quedan persistidos en BD y se
# recuperan al arrancar.
_DRAIN_TIMEOUT_SECONDS = 30.0
_POST_KILL_GRACE_SECONDS = 5.0


def _is_transient_error(error: str | None) -> bool:
    return bool(error) and _TRANSIENT_ERROR_PATTERN.search(error) is not None


class ProjectProcessingQueue:
    """Cola de procesamiento con persistencia en BD, reintentos y drenado.

    Sin database_path funciona solo en memoria, como antes (tests y usos
    sin BD). Con database_path, cada trabajo se persiste al encolar y se
    recupera al arrancar (start): los trabajos encolados sobreviven a un
    reinicio de la app. La configuración LLM NO se persiste (contiene la
    clave API en claro): al recuperar, el pipeline la resuelve desde las
    credenciales cifradas del dueño del proyecto (pipeline.py).
    """

    def __init__(
        self,
        max_concurrency: int,
        processor: ProjectProcessor,
        *,
        database_path: Path | None = None,
        drain_timeout: float = _DRAIN_TIMEOUT_SECONDS,
    ) -> None:
        if max_concurrency not in {1, 2}:
            raise ValueError("La concurrencia de procesamiento debe ser 1 o 2.")
        self._max_concurrency = max_concurrency
        self._processor = processor
        self._database_path = database_path
        self._drain_timeout = drain_timeout
        # (job_id persistido o None, project_id, llm_config, retry_failed_only, intentos)
        self._queue: asyncio.Queue[
            tuple[str | None, str, LLMConfig | None, bool, int]
        ] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._stopping = False
        self._in_flight = 0
        self._in_flight_condition = asyncio.Condition()
        self._retry_tasks: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        if self._workers:
            raise RuntimeError("La cola de procesamiento ya está iniciada.")
        self._stopping = False
        self._workers = [
            asyncio.create_task(self._worker())
            for _ in range(self._max_concurrency)
        ]
        await self._recover_persisted_jobs()

    async def _recover_persisted_jobs(self) -> None:
        """Reencola los trabajos que quedaron pendientes al cerrar la app."""
        if self._database_path is None:
            return
        try:
            database.reset_interrupted_processing_jobs(self._database_path)
            jobs = database.list_recoverable_processing_jobs(self._database_path)
        except Exception:
            logger.exception(
                "No se pudieron recuperar los trabajos en cola; "
                "se fallarán al arrancar los proyectos afectados."
            )
            return
        for job in jobs:
            job_id = str(job["id"])
            project_id = str(job["project_id"])
            try:
                status, _ = database.get_project_outcome(
                    self._database_path, project_id
                )
            except Exception:
                logger.exception(
                    "No se pudo leer el estado del proyecto %s al recuperar "
                    "su trabajo en cola; se omite.",
                    project_id,
                )
                continue
            if status is None or status in {"completed", "failed", "cancelled"}:
                # El proyecto ya llegó a un estado terminal (o se borró)
                # aunque el trabajo quedara pendiente: no hay nada que
                # recuperar.
                database.finish_processing_job(
                    self._database_path,
                    job_id,
                    "done" if status == "completed" else "cancelled",
                )
                continue
            if int(job["attempts"]) >= _MAX_JOB_ATTEMPTS:
                message = (
                    "El procesamiento falló varias veces seguidas y se "
                    "abandonó. Revisa el video y la configuración e "
                    "inténtalo de nuevo."
                )
                database.finish_processing_job(
                    self._database_path, job_id, "failed", error=message
                )
                database.update_project(
                    self._database_path,
                    project_id,
                    status="failed",
                    step="failed",
                    error=message,
                )
                database.publish_event(
                    self._database_path,
                    project_id,
                    "failed",
                    {
                        "status": "failed",
                        "step": "failed",
                        "progress": None,
                        "message": message,
                    },
                )
                continue
            logger.info(
                "Se recupera el trabajo en cola del proyecto %s (intento %d).",
                project_id,
                int(job["attempts"]) + 1,
            )
            await self._queue.put(
                (
                    job_id,
                    project_id,
                    None,
                    bool(job["retry_failed_only"]),
                    int(job["attempts"]),
                )
            )

    async def submit(
        self,
        project_id: str,
        llm_config: LLMConfig | None,
        *,
        retry_failed_only: bool = False,
    ) -> None:
        if not self._workers or self._stopping:
            raise RuntimeError("La cola de procesamiento no está disponible.")
        # Un reproceso es un trabajo nuevo: cualquier cancelación anterior
        # del mismo proyecto queda olvidada.
        clear_project_cancellation(project_id)
        job_id: str | None = None
        if self._database_path is not None:
            try:
                job_id, created = database.enqueue_processing_job(
                    self._database_path, project_id, retry_failed_only
                )
            except Exception:
                logger.exception(
                    "No se pudo persistir el trabajo del proyecto %s; "
                    "se encola solo en memoria.",
                    project_id,
                )
                job_id, created = None, True
            if not created:
                raise RuntimeError(
                    "Este proyecto ya tiene un trabajo en la cola de procesamiento."
                )
        await self._queue.put((job_id, project_id, llm_config, retry_failed_only, 0))

    def discard_queued(self, project_id: str) -> int:
        """Saca de la cola los trabajos pendientes de un proyecto.

        Solo debe llamarse desde el bucle de eventos (el endpoint de
        cancelación es async): manipula la deque interna de asyncio.Queue
        sin awaits, así que es atómica respecto a los workers. Si el
        trabajo ya lo tomó un worker, no hay nada que sacar y la
        cancelación cooperativa del pipeline se encarga de detenerlo.
        También cancela las filas persistidas, para que no se recuperen
        al arrancar.
        """
        pending = self._queue._queue  # noqa: SLF001 - deque interna de asyncio.Queue
        remaining = [item for item in pending if item[1] != project_id]
        removed = len(pending) - len(remaining)
        if removed:
            pending.clear()
            pending.extend(remaining)
            # Los trabajos sacados nunca llamarán a task_done(); ajustar el
            # contador para que join() no espere por ellos para siempre.
            self._queue._unfinished_tasks -= removed  # noqa: SLF001
        if self._database_path is not None:
            try:
                database.cancel_processing_jobs(self._database_path, project_id)
            except Exception:
                logger.exception(
                    "No se pudieron cancelar los trabajos persistidos del "
                    "proyecto %s.",
                    project_id,
                )
        return removed

    def request_cancel(self, project_id: str) -> int:
        """Marca un proyecto como cancelado y saca sus trabajos pendientes."""
        request_project_cancellation(project_id)
        return self.discard_queued(project_id)

    async def join(self) -> None:
        await self._queue.join()

    async def stop(self) -> None:
        """Apagado elegante (B-P14): drena el trabajo en curso con timeout.

        Espera solo al trabajo en curso (no a los pendientes: quedan
        persistidos y se recuperan al arrancar). Si el drenado agota el
        timeout, mata los ffmpeg hijos — cancelar la tarea asyncio no
        detiene el hilo del render —, da un respiro breve y cancela los
        workers. El trabajo interrumpido queda 'running' en BD y se
        reencola en el próximo arranque.
        """
        if not self._workers:
            return
        self._stopping = True
        for task in list(self._retry_tasks):
            task.cancel()
        self._retry_tasks.clear()
        drained = await self._wait_for_drain(self._drain_timeout)
        if not drained:
            logger.warning(
                "El trabajo en curso no terminó en %.0f s durante el apagado; "
                "se terminan los renders ffmpeg en curso.",
                self._drain_timeout,
            )
            terminate_active_renders()
            await self._wait_for_drain(_POST_KILL_GRACE_SECONDS)
        # Sacar los pendientes de la cola en memoria (persisten en BD).
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._queue.task_done()
        workers = self._workers
        self._workers = []
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    async def _wait_for_drain(self, timeout: float) -> bool:
        try:
            async with self._in_flight_condition:
                await asyncio.wait_for(
                    self._in_flight_condition.wait_for(
                        lambda: self._in_flight == 0
                    ),
                    timeout=timeout,
                )
            return True
        except TimeoutError:
            return False

    async def _worker(self) -> None:
        while True:
            job_id, project_id, llm_config, retry_failed_only, attempts = (
                await self._queue.get()
            )
            try:
                async with self._in_flight_condition:
                    self._in_flight += 1
                try:
                    if is_project_cancelled(project_id):
                        logger.info(
                            "Se omite el trabajo cancelado del proyecto %s", project_id
                        )
                        self._finish_persisted_job(job_id, "cancelled")
                    else:
                        self._claim_persisted_job(job_id, attempts + 1)
                        await self._processor(
                            project_id, llm_config, retry_failed_only
                        )
                        await self._finish_job_after_run(
                            job_id,
                            project_id,
                            llm_config,
                            retry_failed_only,
                            attempts + 1,
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    logger.exception(
                        "Falló el trabajo en cola del proyecto %s", project_id
                    )
                    await self._maybe_retry(
                        job_id,
                        project_id,
                        llm_config,
                        retry_failed_only,
                        attempts + 1,
                        error,
                    )
            finally:
                async with self._in_flight_condition:
                    self._in_flight -= 1
                    if self._in_flight == 0:
                        self._in_flight_condition.notify_all()
                self._queue.task_done()
                clear_project_cancellation(project_id)

    def _claim_persisted_job(self, job_id: str | None, attempts: int) -> None:
        if job_id is None or self._database_path is None:
            return
        try:
            database.claim_processing_job(self._database_path, job_id, attempts)
        except Exception:
            logger.exception(
                "No se pudo marcar el trabajo %s como en curso.", job_id
            )

    def _finish_persisted_job(
        self,
        job_id: str | None,
        status: str,
        error: str | None = None,
    ) -> None:
        if job_id is None or self._database_path is None:
            return
        try:
            database.finish_processing_job(
                self._database_path, job_id, status, error=error
            )
        except Exception:
            logger.exception(
                "No se pudo marcar el trabajo %s como %s.", job_id, status
            )

    async def _finish_job_after_run(
        self,
        job_id: str | None,
        project_id: str,
        llm_config: LLMConfig | None,
        retry_failed_only: bool,
        attempts: int,
    ) -> None:
        """Cierra el trabajo según el estado final del proyecto.

        El procesador (process_project) captura sus propios errores y
        marca el proyecto failed/completed/cancelled, así que el estado
        del proyecto — no una excepción — dice si el trabajo terminó bien.
        Un failed por causa transitoria y con intentos restantes se
        reencola con backoff.
        """
        if job_id is None or self._database_path is None:
            return
        try:
            status, error = database.get_project_outcome(
                self._database_path, project_id
            )
        except Exception:
            logger.exception(
                "No se pudo leer el resultado del proyecto %s.", project_id
            )
            self._finish_persisted_job(job_id, "done")
            return
        if status == "failed" and _is_transient_error(error):
            if attempts < _MAX_JOB_ATTEMPTS:
                await self._schedule_retry(
                    job_id, project_id, llm_config, retry_failed_only, attempts
                )
                return
            logger.warning(
                "El proyecto %s falló %d veces por causas transitorias; "
                "se abandona: %s",
                project_id,
                attempts,
                error,
            )
            self._finish_persisted_job(job_id, "failed", error=error)
            return
        if status in {"completed", "cancelled"}:
            self._finish_persisted_job(job_id, "done")
        elif status == "failed":
            self._finish_persisted_job(job_id, "failed", error=error)
        elif status is None:
            self._finish_persisted_job(job_id, "cancelled")
        else:
            # El procesador terminó sin dejar el proyecto en estado
            # terminal (no debería pasar): no se reencola en bucle.
            logger.warning(
                "El trabajo del proyecto %s terminó con el proyecto en "
                "estado %r; se marca como fallido.",
                project_id,
                status,
            )
            self._finish_persisted_job(job_id, "failed", error=error)

    async def _maybe_retry(
        self,
        job_id: str | None,
        project_id: str,
        llm_config: LLMConfig | None,
        retry_failed_only: bool,
        attempts: int,
        error: Exception,
    ) -> None:
        """Reintento cuando el propio procesador lanzó (fallo de la cola).

        Solo se reintentan errores transitorios; un error de programa se
        marca como fallido sin reintentar. Sin persistencia no hay
        reintentos: el contador de intentos vive en la fila de BD.
        """
        if job_id is None or self._database_path is None:
            return
        if not _is_transient_error(str(error)):
            self._finish_persisted_job(job_id, "failed", error=str(error))
            return
        if attempts >= _MAX_JOB_ATTEMPTS:
            logger.warning(
                "El trabajo del proyecto %s falló %d veces; se abandona.",
                project_id,
                attempts,
            )
            self._finish_persisted_job(job_id, "failed", error=str(error))
            return
        await self._schedule_retry(
            job_id, project_id, llm_config, retry_failed_only, attempts
        )

    async def _schedule_retry(
        self,
        job_id: str,
        project_id: str,
        llm_config: LLMConfig | None,
        retry_failed_only: bool,
        attempts: int,
    ) -> None:
        """Reencola un trabajo tras un backoff (sin bloquear el worker)."""
        if self._stopping:
            # La fila queda 'running' y se recupera al arrancar.
            return
        delay = _RETRY_DELAYS_SECONDS[min(attempts - 1, len(_RETRY_DELAYS_SECONDS) - 1)]
        logger.info(
            "Se reintentará el proyecto %s en %.0f s (intento %d de %d).",
            project_id,
            delay,
            attempts + 1,
            _MAX_JOB_ATTEMPTS,
        )
        try:
            database.requeue_processing_job(self._database_path, job_id)  # type: ignore[arg-type]
        except Exception:
            logger.exception(
                "No se pudo devolver el trabajo %s a pendiente.", job_id
            )
            return

        async def _delayed_requeue() -> None:
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return
            if self._stopping or not self._workers:
                return
            await self._queue.put(
                (job_id, project_id, llm_config, retry_failed_only, attempts)
            )

        task = asyncio.create_task(_delayed_requeue())
        self._retry_tasks.add(task)
        task.add_done_callback(self._retry_tasks.discard)
