from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from . import database
from .media import MediaProcessingError, probe_media, render_clip
from .schemas import ClipEditRequest, ClipProposal, ProjectConfig, TranscriptSegment
from .settings import AppSettings


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
    if isinstance(saved, dict) and isinstance(saved.get("phrases"), list):
        return {
            "ratio": saved.get("ratio", project["config"]["ratio"]),
            "subtitle_template": saved.get(
                "subtitle_template", project["config"]["subtitle_template"]
            ),
            "phrases": saved["phrases"],
        }
    return {
        "ratio": project["config"]["ratio"],
        "subtitle_template": project["config"]["subtitle_template"],
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
        "phrases": document["phrases"],
    }


def render_editor_version(
    settings: AppSettings,
    project: dict[str, Any],
    clip: dict[str, Any],
    edit: ClipEditRequest,
    preview: bool,
    output_resolution: int | None = None,
    enhance_audio: bool | None = None,
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
                    settings.database_path, edit.subtitle_template
                ),
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
            lambda _: None,
            removed_ranges=removed_ranges,
            subtitle_overrides=subtitle_overrides,
            output_resolution=resolution,
            enhance_audio=audio_is_enhanced,
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
