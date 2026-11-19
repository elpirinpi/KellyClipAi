from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable

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


class ProjectProcessingQueue:
    def __init__(
        self,
        max_concurrency: int,
        processor: ProjectProcessor,
    ) -> None:
        if max_concurrency not in {1, 2}:
            raise ValueError("La concurrencia de procesamiento debe ser 1 o 2.")
        self._max_concurrency = max_concurrency
        self._processor = processor
        self._queue: asyncio.Queue[tuple[str, LLMConfig | None, bool]] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._stopping = False

    async def start(self) -> None:
        if self._workers:
            raise RuntimeError("La cola de procesamiento ya está iniciada.")
        self._stopping = False
        self._workers = [
            asyncio.create_task(self._worker())
            for _ in range(self._max_concurrency)
        ]

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
        await self._queue.put((project_id, llm_config, retry_failed_only))

    def discard_queued(self, project_id: str) -> int:
        """Saca de la cola los trabajos pendientes de un proyecto.

        Solo debe llamarse desde el bucle de eventos (el endpoint de
        cancelación es async): manipula la deque interna de asyncio.Queue
        sin awaits, así que es atómica respecto a los workers. Si el
        trabajo ya lo tomó un worker, no hay nada que sacar y la
        cancelación cooperativa del pipeline se encarga de detenerlo.
        """
        pending = self._queue._queue  # noqa: SLF001 - deque interna de asyncio.Queue
        remaining = [item for item in pending if item[0] != project_id]
        removed = len(pending) - len(remaining)
        if removed:
            pending.clear()
            pending.extend(remaining)
            # Los trabajos sacados nunca llamarán a task_done(); ajustar el
            # contador para que join() no espere por ellos para siempre.
            self._queue._unfinished_tasks -= removed  # noqa: SLF001
        return removed

    def request_cancel(self, project_id: str) -> int:
        """Marca un proyecto como cancelado y saca sus trabajos pendientes."""
        request_project_cancellation(project_id)
        return self.discard_queued(project_id)

    async def join(self) -> None:
        await self._queue.join()

    async def stop(self) -> None:
        if not self._workers:
            return
        self._stopping = True
        workers = self._workers
        self._workers = []
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    async def _worker(self) -> None:
        while True:
            project_id, llm_config, retry_failed_only = await self._queue.get()
            try:
                if is_project_cancelled(project_id):
                    logger.info(
                        "Se omite el trabajo cancelado del proyecto %s", project_id
                    )
                else:
                    await self._processor(project_id, llm_config, retry_failed_only)
            except Exception:
                logger.exception(
                    "Falló el trabajo en cola del proyecto %s", project_id
                )
            finally:
                self._queue.task_done()
                clear_project_cancellation(project_id)
