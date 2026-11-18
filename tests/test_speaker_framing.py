from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from backend.face_tracking import (
    _Face,
    _SpeakerActivityTracker,
    analyze_face_crop_paths,
)
from backend.media import _build_filter, probe_media, render_clip
from backend.schemas import ClipProposal, EditOptions, ProjectConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("framing", ["speaker", "split", "blur", "center"])
def test_project_config_accepts_explicit_framing_modes(framing: str) -> None:
    config = ProjectConfig.model_validate({"framing": framing})

    assert config.framing == framing


def test_legacy_project_configuration_uses_speaker_framing_default() -> None:
    config = ProjectConfig.model_validate({"ratio": "9:16"})

    assert config.framing == "speaker"


def test_face_landmarker_asset_is_hashed_and_added_to_windows_bundle() -> None:
    model_path = ROOT / "backend" / "models" / "face_landmarker.task"
    build_script = (ROOT / "build_windows.ps1").read_text(encoding="utf-8")

    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == (
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
    )
    assert "face_landmarker.task" in build_script
    assert "--collect-all mediapipe" in build_script


def test_lip_motion_selects_a_face_during_voice_activity() -> None:
    tracker = _SpeakerActivityTracker(1280, 720)
    left = _Face(100, 100, 100, 100, 0.9)
    right = _Face(500, 100, 100, 100, 0.9)

    assert tracker.update([(left, 0.06), (right, 0.06)], 0.0, False) is None
    active = tracker.update([(left, 0.06), (right, 0.18)], 0.25, True)

    assert active == right


def test_mediapipe_landmarker_runs_on_video_frames_with_word_timestamps(
    _video_with_audio: Path,
) -> None:
    tracks = analyze_face_crop_paths(
        _video_with_audio,
        0,
        [(0.0, 1.0)],
        720,
        1280,
        speech_intervals=[(0.1, 0.8)],
    )

    assert tracks == [None]


@pytest.mark.parametrize(
    ("framing", "expected_filter"),
    [
        ("split", "vstack=inputs=2"),
        ("blur", "boxblur=20:10"),
    ],
)
def test_filter_graph_contains_the_selected_multi_person_layout(
    framing: str,
    expected_filter: str,
    tmp_path: Path,
) -> None:
    graph, video_output, _audio_output = _build_filter(
        [(0.0, 2.0)],
        720,
        1280,
        tmp_path / "clip.ass",
        2.0,
        [],
        framing=framing,
    )

    assert expected_filter in graph
    assert video_output.startswith("[")


@pytest.fixture(scope="module")
def _video_with_audio(tmp_path_factory: pytest.TempPathFactory) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("FFmpeg no está disponible.")
    path = tmp_path_factory.mktemp("speaker-framing") / "source.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:r=25:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-y",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return path


@pytest.mark.parametrize("framing", ["split", "blur"])
def test_multi_person_layouts_render_with_real_ffmpeg(
    _video_with_audio: Path,
    framing: str,
    tmp_path: Path,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    assert ffmpeg and ffprobe
    proposal = ClipProposal(
        inicio=0,
        fin=2,
        titulo="Diseño",
        puntuacion=7,
        razon_viral="Verifica el filtro de encuadre.",
    )
    output_path = tmp_path / f"{framing}.mp4"

    render_clip(
        _video_with_audio,
        output_path,
        tmp_path / f"{framing}.ass",
        proposal,
        ProjectConfig(
            framing=framing,
            options=EditOptions(face_tracking=False),
        ),
        [],
        ffmpeg,
        lambda _progress: None,
    )
    metadata = probe_media(output_path, ffprobe)

    assert metadata["width"] == 720
    assert metadata["height"] == 1280
