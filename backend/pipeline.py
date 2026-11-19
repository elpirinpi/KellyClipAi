from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from . import database
from .integrations import IntegrationError, download_broll
from .llm import (
    propose_clips,
    resolve_llm,
)
from .media import MediaProcessingError, create_thumbnail, probe_media, render_clip, transcribe
from .schemas import ClipProposal, LLMConfig, ProjectConfig, TranscriptSegment
from .settings import AppSettings, get_settings


class ProjectNotFoundError(RuntimeError):
    pass


def _publish(
    settings: AppSettings,
    project_id: str,
    status: str,
    step: str,
    progress: int | None,
    message: str,
) -> None:
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


def _duration_bounds(choice: str) -> tuple[float, float]:
    return {
        "Cualquiera": (3.0, float("inf")),
        "<30s": (3.0, 29.9),
        "30–60s": (30.0, 60.0),
        "60–90s": (60.0, 90.0),
        "90s–3min": (90.0, 180.0),
    }[choice]


def _validate_proposals(
    proposals: list[ClipProposal],
    duration: float,
    choice: str,
) -> None:
    minimum, maximum = _duration_bounds(choice)
    if choice != "Cualquiera" and duration + 0.05 < minimum:
        raise MediaProcessingError(
            f"El video dura {duration:.1f} segundos y no alcanza la duración mínima de {choice}."
        )
    previous_end = -1.0
    for proposal in sorted(proposals, key=lambda clip: clip.inicio):
        clip_duration = proposal.fin - proposal.inicio
        if proposal.fin > duration + 0.05:
            raise MediaProcessingError("El modelo propuso un clip fuera de la duración del video.")
        if clip_duration < minimum - 0.05 or clip_duration > min(maximum, duration) + 0.05:
            raise MediaProcessingError(
                f"El modelo propuso un clip fuera del rango de duración {choice}."
            )
        if proposal.inicio < previous_end:
            raise MediaProcessingError("El modelo propuso clips que se solapan.")
        previous_end = proposal.fin


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
        if words_before_end and _ends_sentence(words_before_end[-1].text):
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

        completed_end = min(end_limit, boundary + 0.25)
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


def _render_projects(
    settings: AppSettings,
    project_id: str,
    input_path: Path,
    project_config: ProjectConfig,
    proposals: list[ClipProposal],
    transcript: list[TranscriptSegment],
    ffmpeg_path: str,
    owner_id: str,
) -> list[dict[str, Any]]:
    project_dir = input_path.parent
    output_dir = project_dir / "clips"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[dict[str, Any]] = []
    count = len(proposals)
    if not count:
        return rendered

    for index, proposal in enumerate(proposals):
        clip_id = str(uuid.uuid4())
        output_path = output_dir / f"{clip_id}.mp4"
        subtitle_path = output_dir / f"{clip_id}.ass"

        def on_encode_progress(percent: float) -> None:
            overall = 72 + round(((index + percent / 100) / count) * 26)
            _publish(
                settings,
                project_id,
                "processing",
                "editing",
                min(overall, 98),
                f"Editando clip {index + 1} de {count}…",
            )

        render_config = project_config.model_copy(
            update={
                "subtitle_template": database.resolve_subtitle_template(
                    settings.database_path,
                    project_config.subtitle_template,
                    owner_id,
                )
            }
        )
        broll_path: Path | None = None
        broll_metadata: dict[str, str] | None = None
        if project_config.options.broll:
            broll_path = project_dir / "broll" / f"{clip_id}.mp4"
            try:
                broll_metadata = download_broll(
                    settings,
                    " ".join(proposal.palabras_clave) or proposal.titulo,
                    broll_path,
                )
            except IntegrationError as error:
                raise MediaProcessingError(str(error)) from error
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
        )
        metadata = probe_media(output_path, shutil.which("ffprobe") or "ffprobe")
        rendered.append(
            {
                "id": clip_id,
                "title": proposal.titulo,
                "start": proposal.inicio,
                "end": proposal.fin,
                "duration": clip_duration,
                "score": proposal.puntuacion,
                "viral_reason": proposal.razon_viral,
                "keywords": proposal.palabras_clave,
                "emojis": proposal.emojis if project_config.options.emojis else [],
                "width": metadata["width"],
                "height": metadata["height"],
                "resolution": 720,
                "audio_enhanced": False,
                "filename": output_path.name,
                "transcript": clip_transcript,
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
    return rendered


def process_project(
    project_id: str,
    request_llm: LLMConfig | None,
    settings: AppSettings | None = None,
) -> None:
    settings = settings or get_settings()
    project_dir = settings.data_dir / "projects" / project_id
    input_path = project_dir / "source"
    output_dir = project_dir / "clips"
    try:
        stored = database.get_project(settings.database_path, project_id)
        if stored is None or not input_path.is_file():
            raise ProjectNotFoundError("No se encontró el video guardado para este proyecto.")
        project_config = ProjectConfig.model_validate(stored["config"])
        llm = resolve_llm(request_llm, settings)
        ffmpeg_path = shutil.which("ffmpeg")
        ffprobe_path = shutil.which("ffprobe")
        if not ffmpeg_path or not ffprobe_path:
            raise MediaProcessingError(
                "FFmpeg y ffprobe son necesarios. Instálalos y comprueba que estén en PATH."
            )

        stored_metadata = stored.get("metadata")
        stored_transcript = stored.get("transcript")
        if (
            isinstance(stored_metadata, dict)
            and stored_metadata.get("duration")
            and isinstance(stored_transcript, list)
            and stored_transcript
        ):
            metadata = stored_metadata
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
            _publish(
                settings,
                project_id,
                "processing",
                "process_video",
                0,
                "Leyendo metadatos y creando miniatura con ffprobe y FFmpeg…",
            )
            metadata = probe_media(input_path, ffprobe_path)
            project_dir.mkdir(parents=True, exist_ok=True)
            create_thumbnail(
                input_path,
                project_dir / "thumbnail.jpg",
                float(metadata["duration"]),
                ffmpeg_path,
            )
            database.update_project(settings.database_path, project_id, metadata=metadata)
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

            transcript, language = transcribe(
                input_path,
                project_config.whisper_model or settings.whisper_model,
                on_transcription_progress,
            )
            transcript_data = [segment.model_dump() for segment in transcript]
            metadata["language"] = language
            database.update_project(
                settings.database_path,
                project_id,
                metadata=metadata,
                transcript=transcript_data,
            )
        _progress_for_llm(settings, project_id, llm.provider)

        def on_llm_progress(current: int, total: int) -> None:
            if total > 1:
                _publish(
                    settings,
                    project_id,
                    "processing",
                    "finding",
                    None,
                    f"Analizando la transcripción con {llm.provider}… (fragmento {current} de {total})",
                )

        proposals = propose_clips(
            transcript,
            project_config,
            float(metadata["duration"]),
            llm,
            on_progress=on_llm_progress,
        )
        proposals = _complete_clip_ends(
            proposals,
            transcript,
            float(metadata["duration"]),
            project_config.duration,
        )
        _validate_proposals(
            proposals,
            float(metadata["duration"]),
            project_config.duration,
        )
        if not proposals:
            database.update_project(settings.database_path, project_id, clips=[])
            _publish(
                settings,
                project_id,
                "completed",
                "finalize",
                100,
                "El análisis terminó. No se encontraron momentos que cumplan los criterios solicitados.",
            )
            return

        _publish(
            settings,
            project_id,
            "processing",
            "editing",
            72,
            f"Editando {len(proposals)} clips con FFmpeg…",
        )
        clips = _render_projects(
            settings,
            project_id,
            input_path,
            project_config,
            proposals,
            transcript,
            ffmpeg_path,
            str(stored["owner_id"]),
        )
        database.update_project(settings.database_path, project_id, clips=clips)
        scheduled_count = 0
        if project_config.schedule is not None and clips:
            jobs = database.enqueue_schedule_jobs(
                settings.database_path,
                project_id,
                [str(clip["id"]) for clip in clips],
                project_config.schedule.platform,
                project_config.schedule.scheduled_at.isoformat(),
                project_config.schedule.visibility,
            )
            scheduled_count = len(jobs)
        completion_message = f"Procesamiento terminado: {len(clips)} clips reales listos."
        if scheduled_count:
            completion_message += f" {scheduled_count} publicaciones quedaron programadas."
        _publish(
            settings,
            project_id,
            "completed",
            "finalize",
            100,
            completion_message,
        )
    except Exception as error:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        message = str(error).strip() or "Error inesperado durante el procesamiento."
        database.update_project(settings.database_path, project_id, clips=[])
        _publish(settings, project_id, "failed", "failed", None, message)
