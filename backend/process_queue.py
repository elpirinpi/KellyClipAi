from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from .schemas import LLMConfig

logger = logging.getLogger(__name__)
ProjectProcessor = Callable[[str, LLMConfig | None, bool], Awaitable[None]]


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
        await self._queue.put((project_id, llm_config, retry_failed_only))

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
                await self._processor(project_id, llm_config, retry_failed_only)
            except Exception:
                logger.exception(
                    "Falló el trabajo en cola del proyecto %s", project_id
                )
            finally:
                self._queue.task_done()
