from __future__ import annotations

import json
import re
import subprocess
import textwrap
from pathlib import Path
from typing import Callable

from .schemas import ClipProposal, ProjectConfig, TranscriptSegment, Word

ProgressCallback = Callable[[float], None]
_SILENCE_START = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")
_CENSORED_WORDS = {
    "asshole",
    "bitch",
    "cabrón",
    "cabron",
    "coño",
    "concha",
    "damn",
    "fuck",
    "fucking",
    "gilipollas",
    "joder",
    "mierda",
    "pendejo",
    "puta",
    "puto",
    "shit",
    "verga",
}


class MediaProcessingError(RuntimeError):
    pass


def _run(arguments: list[str], purpose: str) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
    except FileNotFoundError as error:
        raise MediaProcessingError(
            f"No se encontró {arguments[0]}. Instala FFmpeg y ffprobe y vuelve a intentarlo."
        ) from error
    except subprocess.TimeoutExpired as error:
        raise MediaProcessingError(f"Se agotó el tiempo al {purpose}.") from error
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()[-8:]
        raise MediaProcessingError(f"No se pudo {purpose}: {' '.join(detail)}")
    return result


def probe_media(input_path: Path, ffprobe_path: str) -> dict[str, object]:
    result = _run(
        [
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height,avg_frame_rate",
            "-of",
            "json",
            str(input_path),
        ],
        "leer los metadatos con ffprobe",
    )
    try:
        data = json.loads(result.stdout)
        streams = data["streams"]
        video = next(stream for stream in streams if stream.get("codec_type") == "video")
        audio = next(stream for stream in streams if stream.get("codec_type") == "audio")
        duration = float(data["format"]["duration"])
        width, height = int(video["width"]), int(video["height"])
    except (KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError) as error:
        raise MediaProcessingError(
            "El archivo debe ser un video reproducible que incluya pista de audio."
        ) from error
    if duration <= 0 or width <= 0 or height <= 0:
        raise MediaProcessingError("ffprobe devolvió una duración o resolución no válida.")
    return {
        "duration": duration,
        "width": width,
        "height": height,
        "has_audio": audio is not None,
    }


def create_thumbnail(
    input_path: Path,
    output_path: Path,
    duration: float,
    ffmpeg_path: str,
) -> None:
    timestamp = min(1.0, duration / 2)
    _run(
        [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(input_path),
            "-frames:v",
            "1",
            "-vf",
            "scale=640:-2",
            "-q:v",
            "3",
            "-y",
            str(output_path),
        ],
        "crear la miniatura del video",
    )


def transcribe(
    input_path: Path,
    model_name: str,
    on_progress: ProgressCallback,
) -> tuple[list[TranscriptSegment], str]:
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        segments, info = model.transcribe(
            str(input_path),
            word_timestamps=True,
            vad_filter=True,
            beam_size=5,
        )
        transcript: list[TranscriptSegment] = []
        last_percent = -1
        duration = max(float(info.duration or 0), 0.001)
        for segment in segments:
            words = [
                Word(
                    start=max(float(word.start), 0.0),
                    end=max(float(word.end), 0.001),
                    text=word.word,
                )
                for word in (segment.words or [])
                if word.end > word.start
            ]
            transcript.append(
                TranscriptSegment(
                    start=max(float(segment.start), 0.0),
                    end=max(float(segment.end), 0.001),
                    text=segment.text.strip(),
                    words=words,
                )
            )
            percent = min(100, max(0, int(float(segment.end) / duration * 100)))
            if percent > last_percent:
                on_progress(percent)
                last_percent = percent
        if not any(segment.words for segment in transcript):
            raise MediaProcessingError(
                "Whisper no detectó voz con marcas de tiempo. Comprueba que el video tenga audio inteligible."
            )
        on_progress(100)
        return transcript, info.language or "und"
    except MediaProcessingError:
        raise
    except Exception as error:
        raise MediaProcessingError(f"Falló la transcripción con faster-whisper: {error}") from error


def _silence_intervals(
    input_path: Path,
    start: float,
    duration: float,
    ffmpeg_path: str,
) -> list[tuple[float, float]]:
    try:
        result = subprocess.run(
            [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "info",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(input_path),
                "-t",
                f"{duration:.3f}",
                "-map",
                "0:a:0",
                "-af",
                "silencedetect=noise=-35dB:d=0.45",
                "-f",
                "null",
                "-",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        raise MediaProcessingError(f"No se pudo analizar los silencios: {error}") from error
    if result.returncode:
        detail = (result.stderr or "").strip().splitlines()[-8:]
        raise MediaProcessingError(f"FFmpeg no pudo detectar silencios: {' '.join(detail)}")

    kept: list[tuple[float, float]] = []
    cursor = 0.0
    silence_open = False
    for line in (result.stderr or "").splitlines():
        start_match = _SILENCE_START.search(line)
        if start_match:
            silence_start = min(duration, max(0.0, float(start_match.group(1))))
            if silence_start > cursor:
                kept.append((cursor, silence_start))
            cursor = silence_start
            silence_open = True
            continue
        end_match = _SILENCE_END.search(line)
        if end_match and silence_open:
            cursor = min(duration, max(cursor, float(end_match.group(1))))
            silence_open = False
    if not silence_open and cursor < duration:
        kept.append((cursor, duration))
    return [(left, right) for left, right in kept if right - left >= 0.08]


def _retained_intervals(
    input_path: Path,
    proposal: ClipProposal,
    remove_silence: bool,
    ffmpeg_path: str,
) -> list[tuple[float, float]]:
    duration = proposal.fin - proposal.inicio
    if not remove_silence:
        return [(0.0, duration)]
    intervals = _silence_intervals(input_path, proposal.inicio, duration, ffmpeg_path)
    if not intervals:
        raise MediaProcessingError(
            "El análisis de silencios no encontró voz suficiente dentro de un clip propuesto."
        )
    return intervals


def _subtract_intervals(
    intervals: list[tuple[float, float]],
    exclusions: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    kept: list[tuple[float, float]] = []
    for interval_start, interval_end in intervals:
        cursor = interval_start
        for exclusion_start, exclusion_end in sorted(exclusions):
            start = max(interval_start, exclusion_start)
            end = min(interval_end, exclusion_end)
            if end <= cursor or start >= interval_end:
                continue
            if start > cursor:
                kept.append((cursor, start))
            cursor = max(cursor, end)
            if cursor >= interval_end:
                break
        if interval_end - cursor >= 0.08:
            kept.append((cursor, interval_end))
    return [(start, end) for start, end in kept if end - start >= 0.08]


def _remap_time(timestamp: float, intervals: list[tuple[float, float]]) -> float | None:
    elapsed = 0.0
    for start, end in intervals:
        if start <= timestamp <= end:
            return elapsed + timestamp - start
        elapsed += end - start
    return None


def _clip_words(
    transcript: list[TranscriptSegment],
    proposal: ClipProposal,
    intervals: list[tuple[float, float]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for segment in transcript:
        for word in segment.words:
            if word.end <= proposal.inicio or word.start >= proposal.fin:
                continue
            local_start = max(0.0, word.start - proposal.inicio)
            local_end = min(proposal.fin - proposal.inicio, word.end - proposal.inicio)
            remapped_start = _remap_time(local_start, intervals)
            remapped_end = _remap_time(local_end, intervals)
            if remapped_start is None or remapped_end is None or remapped_end <= remapped_start:
                continue
            output.append(
                {
                    "start": round(word.start, 3),
                    "end": round(word.end, 3),
                    "preview_start": round(remapped_start, 3),
                    "preview_end": round(remapped_end, 3),
                    "text": word.text,
                }
            )
    return output


def _apply_subtitle_overrides(
    words: list[dict[str, object]],
    overrides: list[tuple[float, float, str]],
    proposal: ClipProposal,
    intervals: list[tuple[float, float]],
) -> list[dict[str, object]]:
    output = list(words)
    for start, end, text in overrides:
        start = max(0.0, start)
        end = min(proposal.fin - proposal.inicio, end)
        if end <= start or not text.strip():
            continue
        preview_start = _remap_time(start, intervals)
        preview_end = _remap_time(end, intervals)
        if (
            preview_start is None
            or preview_end is None
            or preview_end <= preview_start
        ):
            continue
        absolute_start = proposal.inicio + start
        absolute_end = proposal.inicio + end
        output = [
            word
            for word in output
            if not (
                float(word["start"]) < absolute_end
                and float(word["end"]) > absolute_start
            )
        ]
        output.append(
            {
                "start": round(absolute_start, 3),
                "end": round(absolute_end, 3),
                "preview_start": round(preview_start, 3),
                "preview_end": round(preview_end, 3),
                "text": text.strip(),
            }
        )
    return sorted(output, key=lambda word: (float(word["start"]), float(word["end"])))


def _ratio_size(ratio: str, resolution: int = 720) -> tuple[int, int]:
    base_width, base_height = {
        "9:16": (720, 1280),
        "16:9": (1280, 720),
        "1:1": (720, 720),
        "4:5": (720, 900),
    }[ratio]
    scale = resolution / 720
    return round(base_width * scale), round(base_height * scale)


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{fraction:02d}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", " ")


def _censored(text: str) -> bool:
    normalized = re.sub(r"[^\w]", "", text.casefold())
    return normalized in _CENSORED_WORDS


def _caption_groups(
    words: list[dict[str, object]],
    max_chars: int = 36,
) -> list[list[dict[str, object]]]:
    groups: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    current_chars = 0
    for word in words:
        text = str(word["text"])
        if current and (
            len(current) >= 6
            or current_chars + len(text) > max_chars
            or float(word["preview_start"]) - float(current[0]["preview_start"]) > 2.5
        ):
            groups.append(current)
            current, current_chars = [], 0
        current.append(word)
        current_chars += len(text) + 1
    if current:
        groups.append(current)
    return groups


def _caption_max_chars(width: int, font_size: int, margin: int = 48) -> int:
    return max(12, int((width - 2 * margin) / (font_size * 0.68)))


def _write_ass(
    output_path: Path,
    words: list[dict[str, object]],
    proposal: ClipProposal,
    config: ProjectConfig,
    clip_duration: float,
    retained: list[tuple[float, float]],
    resolution: int = 720,
) -> list[tuple[float, float]]:
    width, height = _ratio_size(config.ratio, resolution)
    scale = resolution / 720
    style = {
        "Default": ("Arial", 58, "&H00FFFFFF", "&H0000FFFF", 3, 1, 2),
        "Modern": ("Arial", 56, "&H0000FFFF", "&H00FFFFFF", 2, 1, 2),
        "Bouncy": ("Arial Black", 60, "&H0088FF88", "&H00FFFFFF", 4, 1, 2),
        "MrBeast": ("Arial Black", 64, "&H0000FFFF", "&H000000FF", 5, 1, 2),
        "Business": ("Arial", 46, "&H00202020", "&H00FFFFFF", 1, 0, 2),
        "Esencial": ("Arial", 58, "&H00FFFFFF", "&H0000FFFF", 3, 1, 2),
        "Editorial": ("Georgia", 48, "&H00FFFFFF", "&H00FFFFFF", 2, 0, 2),
        "Neón": ("Arial Black", 60, "&H00FFFF66", "&H00FFFFFF", 4, 1, 2),
    }.get(config.subtitle_template, ("Arial", 58, "&H00FFFFFF", "&H0000FFFF", 3, 1, 2))
    font, size, primary, secondary, outline, shadow, alignment = style
    size = round(size * scale)
    caption_max_chars = _caption_max_chars(width, size)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font},{size},{primary},{secondary},&H00101010,&H99000000,-1,0,1,{outline},{shadow},{alignment},48,48,96,1
Style: Watermark,Arial,{round(20 * scale)},&H00FFFFFF,&H00FFFFFF,&H66000000,&H55000000,0,0,1,1,0,3,18,18,18,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    dialogues: list[str] = []
    censor_ranges: list[tuple[float, float]] = []
    for word in words:
        if _censored(str(word["text"])) and config.options.censor:
            censor_ranges.append((float(word["preview_start"]), float(word["preview_end"])))
            word["text"] = "****"
    groups = _caption_groups(words, caption_max_chars)
    for index, group in enumerate(groups):
        begin = float(group[0]["preview_start"])
        end = max(float(group[-1]["preview_end"]), begin + 0.08)
        if end > clip_duration:
            end = clip_duration
        parts: list[str] = []
        for word in group:
            text = _ass_escape(str(word["text"]))
            if len(text) > caption_max_chars:
                text = r"\N".join(
                    textwrap.wrap(
                        text,
                        width=caption_max_chars,
                        break_long_words=True,
                        break_on_hyphens=False,
                    )
                )
            word_duration = max(
                1,
                round((float(word["preview_end"]) - float(word["preview_start"])) * 100),
            )
            if config.options.keywords and any(
                re.search(rf"\b{re.escape(keyword)}\b", text, re.IGNORECASE)
                for keyword in proposal.palabras_clave
                if keyword
            ):
                parts.append(r"{\c&H0000FFFF&}" + text + r"{\c&H00FFFFFF&}")
            else:
                parts.append(f"{{\\kf{word_duration}}}{text}")
        caption = " ".join(parts)
        if config.options.emojis and index == len(groups) - 1 and proposal.emojis:
            caption += " " + _ass_escape("".join(proposal.emojis))
        dialogues.append(
            f"Dialogue: 0,{_ass_time(begin)},{_ass_time(end)},Caption,,0,0,0,,{caption}"
        )
    watermark_start = _ass_time(0)
    watermark_end = _ass_time(clip_duration)
    dialogues.append(
        f"Dialogue: 1,{watermark_start},{watermark_end},Watermark,,0,0,0,,hecho con kellyClipAi"
    )
    output_path.write_text(header + "\n".join(dialogues) + "\n", encoding="utf-8")
    return censor_ranges


def _filter_path(path: Path) -> str:
    value = path.resolve().as_posix().replace(":", r"\:")
    return value.replace("'", r"\'")


def _filter_time(value: float) -> str:
    return f"{max(0.0, value):.3f}"


def _build_filter(
    intervals: list[tuple[float, float]],
    width: int,
    height: int,
    subtitle_path: Path,
    clip_duration: float,
    censor_ranges: list[tuple[float, float]],
    enhance_audio: bool = False,
    include_broll: bool = False,
) -> tuple[str, str, str]:
    parts: list[str] = []
    video_labels: list[str] = []
    audio_labels: list[str] = []
    for index, (start, end) in enumerate(intervals):
        video_label, audio_label = f"v{index}", f"a{index}"
        parts.append(
            f"[0:v]trim=start={_filter_time(start)}:end={_filter_time(end)},"
            f"setpts=PTS-STARTPTS,scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1[{video_label}]"
        )
        parts.append(
            f"[0:a]atrim=start={_filter_time(start)}:end={_filter_time(end)},"
            f"asetpts=PTS-STARTPTS[{audio_label}]"
        )
        video_labels.append(f"[{video_label}]")
        audio_labels.append(f"[{audio_label}]")
    if len(intervals) == 1:
        video_input, audio_input = video_labels[0], audio_labels[0]
    else:
        parts.append(
            "".join(
                label
                for pair in zip(video_labels, audio_labels, strict=True)
                for label in pair
            )
            + f"concat=n={len(intervals)}:v=1:a=1[vcat][acat]"
        )
        video_input, audio_input = "[vcat]", "[acat]"
    parts.append(f"{video_input}subtitles=filename='{_filter_path(subtitle_path)}'[vout]")
    video_output = "[vout]"
    if include_broll:
        broll_width = max(160, width // 3)
        broll_height = max(90, (round(broll_width * 9 / 16) // 2) * 2)
        parts.append(
            f"[1:v]scale={broll_width}:{broll_height}:force_original_aspect_ratio=decrease,"
            f"pad={broll_width}:{broll_height}:(ow-iw)/2:(oh-ih)/2,"
            f"trim=duration={_filter_time(clip_duration)},setpts=PTS-STARTPTS,setsar=1[broll]"
        )
        parts.append(
            f"[vout][broll]overlay=x=W-w-24:y=24:eof_action=repeat:shortest=0[vbroll]"
        )
        video_output = "[vbroll]"
    if censor_ranges:
        enable = "+".join(
            f"between(t,{_filter_time(start)},{_filter_time(end)})"
            for start, end in censor_ranges
        )
        parts.append(f"{audio_input}volume=0:enable='{enable}'[amut]")
        parts.append(
            f"aevalsrc=0.15*sin(2*PI*1000*t):s=48000:d={_filter_time(clip_duration)},"
            f"volume=0.15:enable='{enable}'[beep]"
        )
        parts.append("[amut][beep]amix=inputs=2:duration=first:dropout_transition=0[aout]")
        audio_output = "[aout]"
    else:
        audio_output = audio_input
    if enhance_audio:
        parts.append(
            f"{audio_output}highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11,"
            "acompressor=threshold=-20dB:ratio=1.6:attack=20:release=200:makeup=1.5"
            "[aenhanced]"
        )
        audio_output = "[aenhanced]"
    return ";".join(parts), video_output, audio_output


def _render_clip(
    input_path: Path,
    output_path: Path,
    subtitle_path: Path,
    proposal: ClipProposal,
    config: ProjectConfig,
    transcript: list[TranscriptSegment],
    ffmpeg_path: str,
    on_progress: ProgressCallback,
    removed_ranges: list[tuple[float, float]] | None = None,
    subtitle_overrides: list[tuple[float, float, str]] | None = None,
    output_resolution: int = 720,
    enhance_audio: bool = False,
    broll_path: Path | None = None,
) -> tuple[float, list[dict[str, object]]]:
    source_duration = proposal.fin - proposal.inicio
    intervals = _retained_intervals(input_path, proposal, config.options.silence, ffmpeg_path)
    if removed_ranges:
        intervals = _subtract_intervals(intervals, removed_ranges)
    render_duration = sum(end - start for start, end in intervals)
    if render_duration <= 0.08:
        raise MediaProcessingError("El clip no contiene suficiente audio para exportarse.")
    words = _clip_words(transcript, proposal, intervals)
    if subtitle_overrides:
        words = _apply_subtitle_overrides(words, subtitle_overrides, proposal, intervals)
    censor_ranges = _write_ass(
        subtitle_path,
        words,
        proposal,
        config,
        render_duration,
        intervals,
        output_resolution,
    )
    width, height = _ratio_size(config.ratio, output_resolution)
    filter_graph, video_output, audio_output = _build_filter(
        intervals,
        width,
        height,
        subtitle_path,
        render_duration,
        censor_ranges,
        enhance_audio,
        broll_path is not None,
    )
    arguments = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{proposal.inicio:.3f}",
        "-i",
        str(input_path),
    ]
    if broll_path is not None:
        if not broll_path.is_file():
            raise MediaProcessingError("No se encontró el archivo de B-roll descargado.")
        arguments.extend(["-stream_loop", "-1", "-i", str(broll_path)])
    arguments.extend([
        "-t",
        f"{source_duration:.3f}",
        "-filter_complex",
        filter_graph,
        "-map",
        video_output,
        "-map",
        audio_output,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "21",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-progress",
        "pipe:1",
        "-nostats",
        "-y",
        str(output_path),
    ])
    try:
        process = subprocess.Popen(
            arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        last_lines: list[str] = []
        if process.stdout is None:
            raise MediaProcessingError("No se pudo leer el progreso de FFmpeg.")
        for line in process.stdout:
            stripped = line.strip()
            if stripped:
                last_lines.append(stripped)
                last_lines = last_lines[-12:]
            if stripped.startswith("out_time_ms="):
                try:
                    processed_seconds = int(stripped.split("=", 1)[1]) / 1_000_000
                    on_progress(min(100.0, max(0.0, processed_seconds / render_duration * 100)))
                except ValueError:
                    continue
        return_code = process.wait(timeout=600)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise MediaProcessingError(f"FFmpeg no pudo exportar el clip: {error}") from error
    if return_code:
        raise MediaProcessingError(f"FFmpeg no pudo exportar el clip: {' '.join(last_lines[-6:])}")
    clip_transcript = words
    return render_duration, clip_transcript


def render_clip(
    input_path: Path,
    output_path: Path,
    subtitle_path: Path,
    proposal: ClipProposal,
    config: ProjectConfig,
    transcript: list[TranscriptSegment],
    ffmpeg_path: str,
    on_progress: ProgressCallback,
    removed_ranges: list[tuple[float, float]] | None = None,
    subtitle_overrides: list[tuple[float, float, str]] | None = None,
    output_resolution: int = 720,
    enhance_audio: bool = False,
    broll_path: Path | None = None,
) -> tuple[float, list[dict[str, object]]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return _render_clip(
        input_path,
        output_path,
        subtitle_path,
        proposal,
        config,
        transcript,
        ffmpeg_path,
        on_progress,
        removed_ranges,
        subtitle_overrides,
        output_resolution,
        enhance_audio,
        broll_path,
    )
