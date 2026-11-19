from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from backend.media import (
    _build_filter,
    _write_ass,
    format_subtitles,
    probe_media,
    render_clip,
)
from backend.schemas import ClipProposal, ProjectConfig, SubtitleStyle


def test_srt_and_vtt_use_rendered_word_timestamps() -> None:
    words = [
        {"preview_start": 0.25, "preview_end": 0.6, "text": "Hola"},
        {"preview_start": 0.65, "preview_end": 1.0, "text": "mundo"},
    ]

    assert format_subtitles(words, "srt") == (
        "1\n00:00:00,250 --> 00:00:01,000\nHola mundo\n"
    )
    assert format_subtitles(words, "vtt") == (
        "WEBVTT\n\n00:00:00.250 --> 00:00:01.000\nHola mundo\n"
    )


def test_ass_writer_applies_user_font_color_size_and_position(
    tmp_path: Path,
) -> None:
    subtitle_path = tmp_path / "custom.ass"
    proposal = ClipProposal(
        inicio=0,
        fin=2,
        titulo="Personalización",
        puntuacion=7,
        razon_viral="Prueba de estilo.",
        palabras_clave=["idea"],
    )
    config = ProjectConfig(
        subtitle_style=SubtitleStyle(
            font_family="Georgia",
            font_size=72,
            font_color="#ff0000",
            highlight_color="#00ff00",
            position="top",
        )
    )

    _write_ass(
        subtitle_path,
        [
            {
                "start": 0.1,
                "end": 0.5,
                "preview_start": 0.1,
                "preview_end": 0.5,
                "text": "idea",
            }
        ],
        proposal,
        config,
        2,
        [(0, 2)],
    )
    content = subtitle_path.read_text(encoding="utf-8")

    assert "Style: Caption,Georgia,72,&H000000FF,&H0000FF00," in content
    assert ",8,48,48,96,1" in content
    assert r"{\c&H0000FF00&}idea{\c&H000000FF&}" in content


def _create_color_video(ffmpeg: str, output: Path, color: str) -> None:
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=320x240:r=25:d=4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=4",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-y",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_broll_renders_as_full_screen_cutaway(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    assert ffmpeg and ffprobe
    source = tmp_path / "source-red.mp4"
    broll = tmp_path / "broll-blue.mp4"
    _create_color_video(ffmpeg, source, "red")
    _create_color_video(ffmpeg, broll, "blue")
    output = tmp_path / "full-screen-broll.mp4"
    proposal = ClipProposal(
        inicio=0,
        fin=4,
        titulo="Corte de apoyo",
        puntuacion=8,
        razon_viral="Comprueba un corte B-roll real.",
    )

    render_clip(
        source,
        output,
        tmp_path / "subtitles.ass",
        proposal,
        ProjectConfig(ratio="16:9", options={"face_tracking": False}),
        [],
        ffmpeg,
        lambda _percent: None,
        broll_path=broll,
    )
    metadata = probe_media(output, ffprobe)
    sample = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            "1.8",
            "-i",
            str(output),
            "-frames:v",
            "1",
            "-vf",
            "crop=2:2:0:0,format=rgb24",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    red, green, blue = sample[:3]

    assert metadata["width"] == 1280
    assert metadata["height"] == 720
    assert metadata["has_audio"] is True
    assert blue > 180
    assert red < 80
    assert green < 80


def test_broll_filter_enables_timed_full_frame_cutaways(tmp_path: Path) -> None:
    graph, video_output, _audio_output = _build_filter(
        [(0, 12)],
        1280,
        720,
        tmp_path / "captions.ass",
        12,
        [],
        include_broll=True,
    )

    assert "scale=1280:720" in graph
    assert "overlay=shortest=0" in graph
    assert "enable='between(t," in graph
    assert video_output == "[vout]"
