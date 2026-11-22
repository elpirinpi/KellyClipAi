from __future__ import annotations

import asyncio
import logging
import os
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import database
from .media import MediaProcessingError, ProgressCallback, probe_media, render_clip
from .schemas import (
    ClipEditRequest,
    ClipProposal,
    ProjectConfig,
    SubtitleStyle,
    TranscriptSegment,
)
from .settings import AppSettings

logger = logging.getLogger(__name__)


class EditorRenderCancelled(RuntimeError):
    """El usuario canceló un render del editor.

    Se lanza desde el callback de progreso (que render_clip invoca dentro
    de su bucle de lectura de ffmpeg): al propagarse, el finally de
    _render_clip mata el ffmpeg hijo, así que la cancelación no deja
    procesos huérfanos. No hereda de ValueError/OSError para que ningún
    except de media.py la convierta en un error de render.
    """


def find_clip(project: dict[str, Any], clip_id: str) -> dict[str, Any] | None:
    return next(
        (clip for clip in project.get("clips") or [] if clip.get("id") == clip_id),
        None,
    )


def editor_phrases(clip: dict[str, Any]) -> list[dict[str, Any]]:
    clip_start = float(clip["start"])
    words = clip.get("transcript") or []
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0

    for word in words:
        text = str(word["text"]).strip()
        if not text:
            continue
        gap = (
            float(word["preview_start"]) - float(current[-1]["preview_end"])
            if current
            else 0.0
        )
        current_text = " ".join(str(item["text"]).strip() for item in current)
        if current and (
            len(current) >= 8
            or current_chars + len(text) > 48
            or float(word["preview_end"]) - float(current[0]["preview_start"]) > 3.5
            or gap > 1.2
            or current_text.endswith((".", "!", "?", "…"))
        ):
            groups.append(current)
            current, current_chars = [], 0
        current.append(word)
        current_chars += len(text) + 1
    if current:
        groups.append(current)

    phrases: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        start = max(0.0, float(group[0]["start"]) - clip_start)
        end = max(start, float(group[-1]["end"]) - clip_start)
        if end <= start:
            continue
        phrases.append(
            {
                "id": f"phrase-{index}-{round(start * 1000)}",
                "start": round(start, 3),
                "end": round(end, 3),
                "preview_start": round(float(group[0]["preview_start"]), 3),
                "text": " ".join(str(word["text"]).strip() for word in group),
                "removed": False,
            }
        )
    return phrases


def editor_state(project: dict[str, Any], clip: dict[str, Any]) -> dict[str, Any]:
    saved = clip.get("edit")
    subtitle_style = (
        saved.get("subtitle_style")
        if isinstance(saved, dict)
        else project["config"].get("subtitle_style")
    )
    serialized_style = SubtitleStyle.model_validate(
        subtitle_style if isinstance(subtitle_style, dict) else {}
    ).model_dump(mode="json")
    if isinstance(saved, dict) and isinstance(saved.get("phrases"), list):
        return {
            "ratio": saved.get("ratio", project["config"]["ratio"]),
            "subtitle_template": saved.get(
                "subtitle_template", project["config"]["subtitle_template"]
            ),
            "subtitle_style": serialized_style,
            "phrases": saved["phrases"],
        }
    return {
        "ratio": project["config"]["ratio"],
        "subtitle_template": project["config"]["subtitle_template"],
        "subtitle_style": serialized_style,
        "phrases": editor_phrases(clip),
    }


def _edit_document(
    clip: dict[str, Any],
    edit: ClipEditRequest,
) -> tuple[dict[str, Any], list[tuple[float, float]], list[tuple[float, float, str]]]:
    saved = clip.get("edit")
    base_phrases = (
        saved.get("base_phrases")
        if isinstance(saved, dict) and isinstance(saved.get("base_phrases"), list)
        else editor_phrases(clip)
    )
    base_by_id = {phrase["id"]: phrase for phrase in base_phrases}
    edited = edit.model_dump()
    edited_ids = {phrase["id"] for phrase in edited["phrases"]}
    if edited_ids != set(base_by_id):
        raise ValueError("El editor debe conservar todas las frases originales.")

    removed_ranges: list[tuple[float, float]] = []
    subtitle_overrides: list[tuple[float, float, str]] = []
    for phrase in edited["phrases"]:
        if phrase["end"] > float(clip["end"]) - float(clip["start"]) + 0.05:
            raise ValueError("Una frase excede la duración original del clip.")
        if phrase["removed"]:
            removed_ranges.append((phrase["start"], phrase["end"]))
        elif phrase["text"] != base_by_id[phrase["id"]]["text"]:
            subtitle_overrides.append(
                (phrase["start"], phrase["end"], phrase["text"])
            )
    document = {**edited, "base_phrases": base_phrases}
    return document, removed_ranges, subtitle_overrides


def _save_edit(
    settings: AppSettings,
    project: dict[str, Any],
    clip: dict[str, Any],
    document: dict[str, Any],
) -> None:
    clips = list(project.get("clips") or [])
    for index, current in enumerate(clips):
        if current.get("id") == clip["id"]:
            clips[index] = {**current, "edit": document}
            break
    else:
        raise MediaProcessingError("No se encontró el clip que se está editando.")
    database.update_project(settings.database_path, project["id"], clips=clips)


def save_editor_draft(
    settings: AppSettings,
    project: dict[str, Any],
    clip: dict[str, Any],
    edit: ClipEditRequest,
) -> dict[str, Any]:
    document, _, _ = _edit_document(clip, edit)
    _save_edit(settings, project, clip, document)
    return {
        "ratio": document["ratio"],
        "subtitle_template": document["subtitle_template"],
        "subtitle_style": document["subtitle_style"],
        "phrases": document["phrases"],
    }


def _brand_kit_for_project(
    settings: AppSettings,
    project: dict[str, Any],
    project_config: ProjectConfig,
) -> dict[str, Any] | None:
    """Kit de marca resuelto para un render, o None si no aplica."""
    if not project_config.options.brand:
        return None
    owner_id = str(project.get("owner_id") or "legacy")
    try:
        kit = database.get_brand_kit(settings.database_path, owner_id)
    except Exception as error:  # noqa: BLE001 - sin marca, render normal
        logger.warning("No se pudo cargar la plantilla de marca (%s).", error)
        return None
    if kit is None:
        return None
    from .brand import brand_logo_path

    brand_kit = dict(kit)
    logo_path = brand_logo_path(
        settings.data_dir, owner_id, kit.get("logo_filename")
    )
    if logo_path is not None:
        brand_kit["logo_path"] = str(logo_path)
    return brand_kit


def render_editor_version(
    settings: AppSettings,
    project: dict[str, Any],
    clip: dict[str, Any],
    edit: ClipEditRequest,
    preview: bool,
    output_resolution: int | None = None,
    enhance_audio: bool | None = None,
    on_progress: ProgressCallback | None = None,
    on_analysis_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    document, removed_ranges, subtitle_overrides = _edit_document(clip, edit)
    _save_edit(settings, project, clip, document)

    resolution = output_resolution or int(clip.get("resolution", 720))
    audio_is_enhanced = (
        bool(clip.get("audio_enhanced", False))
        if enhance_audio is None
        else enhance_audio
    )
    project_dir = settings.data_dir / "projects" / project["id"]
    source_path = project_dir / "source"
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    if not source_path.is_file():
        raise MediaProcessingError("No se encontró el video original para volver a renderizar.")
    if not ffmpeg_path or not ffprobe_path:
        raise MediaProcessingError("FFmpeg y ffprobe son necesarios para exportar el clip.")

    try:
        proposal = ClipProposal(
            inicio=clip["start"],
            fin=clip["end"],
            titulo=clip["title"],
            puntuacion=clip["score"],
            razon_viral=clip["viral_reason"],
            palabras_clave=clip.get("keywords") or [],
            emojis=clip.get("emojis") or [],
        )
        project_config = ProjectConfig.model_validate(project["config"]).model_copy(
            update={
                "ratio": edit.ratio,
                "subtitle_template": database.resolve_subtitle_template(
                    settings.database_path,
                    edit.subtitle_template,
                    str(project["owner_id"]),
                ),
                "subtitle_style": edit.subtitle_style,
            }
        )
        transcript = [
            TranscriptSegment.model_validate(segment)
            for segment in project.get("transcript") or []
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise MediaProcessingError("Los datos guardados del clip no permiten editarlo.") from error

    if preview:
        render_dir = project_dir / "edits" / clip["id"]
        output_path = render_dir / "preview.mp4"
        subtitle_path = render_dir / "preview.ass"
    else:
        filename = Path(str(clip.get("filename", ""))).name
        if not filename or not filename.lower().endswith(".mp4"):
            raise MediaProcessingError("El archivo guardado del clip no es válido.")
        render_dir = project_dir / "clips"
        output_path = render_dir / f"{clip['id']}.editing.mp4"
        subtitle_path = render_dir / f"{clip['id']}.editing.ass"
    render_dir.mkdir(parents=True, exist_ok=True)

    try:
        duration, rendered_transcript = render_clip(
            source_path,
            output_path,
            subtitle_path,
            proposal,
            project_config,
            transcript,
            ffmpeg_path,
            on_progress or (lambda _: None),
            removed_ranges=removed_ranges,
            subtitle_overrides=subtitle_overrides,
            output_resolution=resolution,
            enhance_audio=audio_is_enhanced,
            on_analysis_progress=on_analysis_progress,
            speaker_segments=(project.get("metadata") or {}).get("speaker_segments"),
            brand_kit=_brand_kit_for_project(settings, project, project_config),
        )
        metadata = probe_media(output_path, ffprobe_path)
        if preview:
            return {
                "preview_url": (
                    f"/api/projects/{project['id']}/clips/{clip['id']}/edit/preview"
                    f"?v={uuid.uuid4()}"
                ),
                "duration": duration,
                "width": metadata["width"],
                "height": metadata["height"],
                "resolution": resolution,
                "audio_enhanced": audio_is_enhanced,
            }

        final_path = render_dir / filename
        final_subtitle = final_path.with_suffix(".ass")
        # Versiones (Fase 3, ítem 6): el archivo vigente se archiva
        # como vN antes de sustituirlo, para poder volver atrás.
        from .clip_versions import archive_version

        archive_version(project_dir, render_dir, clip["id"], filename)
        os.replace(output_path, final_path)
        os.replace(subtitle_path, final_subtitle)
        clips = list(project.get("clips") or [])
        for index, current in enumerate(clips):
            if current.get("id") == clip["id"]:
                clips[index] = {
                    **current,
                    "duration": duration,
                    "width": metadata["width"],
                    "height": metadata["height"],
                    "transcript": rendered_transcript,
                    "ratio": edit.ratio,
                    "subtitle_template": edit.subtitle_template,
                    "resolution": resolution,
                    "audio_enhanced": audio_is_enhanced,
                    "edit": document,
                }
                break
        else:
            raise MediaProcessingError("No se encontró el clip que se está exportando.")
        database.update_project(settings.database_path, project["id"], clips=clips)
        return {
            "clip_id": clip["id"],
            "duration": duration,
            "width": metadata["width"],
            "height": metadata["height"],
            "resolution": resolution,
            "audio_enhanced": audio_is_enhanced,
        }
    finally:
        if not preview:
            output_path.unlink(missing_ok=True)
            subtitle_path.unlink(missing_ok=True)


def editor_preview_path(
    settings: AppSettings,
    project_id: str,
    clip_id: str,
) -> Path:
    preview = settings.data_dir / "projects" / project_id / "edits" / clip_id / "preview.mp4"
    if not preview.is_file():
        raise FileNotFoundError("La vista previa todavía no está disponible.")
    return preview


def validate_clip_edit(
    clip: dict[str, Any],
    edit: ClipEditRequest,
) -> None:
    """Valida una edición sin renderizar (para fallar rápido al encolar).

    Reutiliza _edit_document: comprueba que se conserven todas las frases
    originales, que ninguna exceda la duración del clip y que los rangos
    sean coherentes. Lanza ValueError con el motivo.
    """
    _edit_document(clip, edit)


# ---------------------------------------------------------------------------
# Renders del editor como trabajos asíncronos (Fase 2, B-P8)
# ---------------------------------------------------------------------------

# Trabajos que se conservan en memoria (los terminales más antiguos se
# descartan al encolar uno nuevo). Los trabajos son solo en memoria: un
# render del editor dura segundos o pocos minutos y no merece persistencia
# en BD como la cola de procesamiento; al reiniciar, el usuario vuelve a
# lanzar el render desde el editor.
_MAX_EDITOR_JOBS = 50

EditorRenderKind = str  # "preview" | "export" | "quality" | "improve"


@dataclass
class EditorRenderJob:
    """Un render del editor encolado: estado, progreso y resultado."""

    id: str
    project_id: str
    clip_id: str
    kind: EditorRenderKind
    status: str = "queued"  # queued | running | completed | failed | cancelled
    progress: float = 0.0
    message: str = "Render en cola…"
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str = field(default_factory=database.utc_now)
    updated_at: str = field(default_factory=database.utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.id,
            "project_id": self.project_id,
            "clip_id": self.clip_id,
            "kind": self.kind,
            "status": self.status,
            "progress": round(self.progress, 1),
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class EditorRenderJobManager:
    """Cola en memoria de renders del editor, con progreso y cancelación.

    Sigue el patrón de ProjectProcessingQueue: el render corre en un hilo
    (asyncio.to_thread) para no bloquear la petición HTTP; el progreso se
    publica en el trabajo y se expone por SSE; la cancelación es
    cooperativa a través del callback de progreso, que lanza
    EditorRenderCancelled y mata el ffmpeg hijo (ver la clase).
    """

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._jobs: dict[str, EditorRenderJob] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.Lock()

    def submit(
        self,
        project: dict[str, Any],
        clip: dict[str, Any],
        edit: ClipEditRequest,
        kind: EditorRenderKind,
        *,
        output_resolution: int | None = None,
        enhance_audio: bool | None = None,
    ) -> EditorRenderJob:
        """Encola un render. Valida la edición antes (ValueError si es mala).

        Debe llamarse desde el bucle de eventos (los endpoints son async):
        crea la tarea asyncio que ejecutará el render.
        """
        validate_clip_edit(clip, edit)
        job = EditorRenderJob(
            id=str(uuid.uuid4()),
            project_id=str(project["id"]),
            clip_id=str(clip["id"]),
            kind=kind,
        )
        with self._lock:
            self._jobs[job.id] = job
            self._prune_locked()
        task = asyncio.create_task(
            self._run_job(job, project, clip, edit, output_resolution, enhance_audio)
        )
        with self._lock:
            self._tasks[job.id] = task
        task.add_done_callback(lambda _: self._forget_task(job.id))
        return job

    def get(self, job_id: str) -> EditorRenderJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """Pide cancelar un trabajo. False si no existe o ya terminó.

        Cooperativa, a propósito: NO se cancela la tarea asyncio, porque
        eso dejaría el hilo del render huérfano (el render seguiría y
        sobrescribiría el clip). En su lugar se marca el trabajo y el
        callback de progreso lanzará EditorRenderCancelled en el
        siguiente tick, lo que mata el ffmpeg hijo y aborta el render
        antes de escribir nada. Si el trabajo aún no empezó, _run_job lo
        detecta al arrancar y termina como cancelado sin renderizar.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in {"completed", "failed", "cancelled"}:
                return False
            self._cancelled.add(job_id)
        return True

    def _forget_task(self, job_id: str) -> None:
        with self._lock:
            self._tasks.pop(job_id, None)

    def _prune_locked(self) -> None:
        """Descarta los trabajos terminales más antiguos (con el lock tomado)."""
        if len(self._jobs) <= _MAX_EDITOR_JOBS:
            return
        terminal = [
            job
            for job in self._jobs.values()
            if job.status in {"completed", "failed", "cancelled"}
        ]
        terminal.sort(key=lambda job: job.updated_at)
        for job in terminal[: len(self._jobs) - _MAX_EDITOR_JOBS]:
            self._jobs.pop(job.id, None)
            self._cancelled.discard(job.id)

    def _is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancelled

    def _update(self, job: EditorRenderJob, **values: Any) -> None:
        with self._lock:
            for key, value in values.items():
                setattr(job, key, value)
            job.updated_at = database.utc_now()

    async def _run_job(
        self,
        job: EditorRenderJob,
        project: dict[str, Any],
        clip: dict[str, Any],
        edit: ClipEditRequest,
        output_resolution: int | None,
        enhance_audio: bool | None,
    ) -> None:
        def check_cancelled() -> None:
            if self._is_cancelled(job.id):
                raise EditorRenderCancelled()

        def on_progress(percent: float) -> None:
            check_cancelled()
            self._update(
                job,
                progress=min(100.0, max(0.0, float(percent))),
                message=f"Renderizando… {percent:.0f}%",
            )

        def on_analysis_progress(percent: float) -> None:
            check_cancelled()
            self._update(
                job,
                message=f"Analizando reencuadre… {percent:.0f}%",
            )

        try:
            if self._is_cancelled(job.id):
                raise EditorRenderCancelled()
            self._update(job, status="running", message="Renderizando…")
            result = await asyncio.to_thread(
                render_editor_version,
                self._settings,
                project,
                clip,
                edit,
                job.kind == "preview",
                output_resolution=output_resolution,
                enhance_audio=enhance_audio,
                on_progress=on_progress,
                on_analysis_progress=on_analysis_progress,
            )
            self._update(
                job,
                status="completed",
                progress=100.0,
                message="Render terminado.",
                result=result,
            )
        except EditorRenderCancelled:
            self._update(
                job,
                status="cancelled",
                message="Render cancelado.",
                error="Render cancelado por el usuario.",
            )
        except asyncio.CancelledError:
            self._update(
                job,
                status="cancelled",
                message="Render cancelado.",
                error="Render cancelado por el usuario.",
            )
            raise
        except Exception as error:
            logger.exception(
                "Falló el render del editor (trabajo %s, clip %s).",
                job.id,
                job.clip_id,
            )
            self._update(
                job,
                status="failed",
                message="El render falló.",
                error=str(error).strip() or "No se pudo renderizar el clip.",
            )
