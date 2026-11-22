"""Regresión de la Fase 3 (selección y edición): ítems 2, 3b, 4b, 5, 6, 7 y 8."""

from __future__ import annotations

import shutil
import subprocess
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import clip_versions, database, fillers, series, speaker_framing, translate, waveform
from backend.main import create_app
from backend.media import _build_filter, _write_ass, render_clip
from backend.moments import _ScoredCandidate, _select_diverse
from backend.pipeline import _validate_and_complete_proposals
from backend.schemas import (
    ClipProposal,
    EditOptions,
    ProjectConfig,
    TranscriptSegment,
    Word,
)
from backend.settings import AppSettings
from backend.studio import hook_range

ROOT = Path(__file__).resolve().parents[1]

# PNG 1x1 válido (para la subida de logo).
_MINIMAL_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c6360000000020001e221bc330000000049454e44ae426082"
)


def _candidate(score: float, start: float, end: float, text: str) -> _ScoredCandidate:
    return _ScoredCandidate(
        score=score,
        first_idx=0,
        last_idx=0,
        start=start,
        end=end,
        breakdown={},
        labels=[],
        text=text,
    )


def _proposal(
    start: float, end: float, title: str, keywords: list[str] | None = None
) -> ClipProposal:
    return ClipProposal(
        inicio=start,
        fin=end,
        titulo=title,
        puntuacion=8.0,
        razon_viral="Momento destacado del video.",
        palabras_clave=keywords or [],
    )


# --- ítem 2: diversidad temporal ----------------------------------------------


def test_select_diverse_temporal_buckets_spread_clips() -> None:
    """Con duración, los clips se reparten por todo el video.

    Tres momentos fuertes agrupados en el primer minuto y tres
    moderados repartidos: sin franjas salen los tres primeros; con
    franjas (video de 600 s, 3 franjas de 200 s) el segundo y el
    tercer momento del primer minuto caen bajo el umbral (8.9 x 0.8
    y 8.8 x 0.8 < 8.0) y salen los repartidos.
    """
    clustered = [
        _candidate(9.0, 10.0, 30.0, "El chef revela el truco secreto del arroz perfecto."),
        _candidate(8.9, 60.0, 80.0, "El estadio entero celebró el gol del delantero final."),
        _candidate(8.8, 110.0, 130.0, "La orquesta sinfónica interpretó la novena sinfonía."),
    ]
    spread = [
        _candidate(8.5, 250.0, 270.0, "El astronauta describió la vista de la Tierra orbital."),
        _candidate(8.5, 450.0, 470.0, "El carpintero talló la mesa de roble macizo pulido."),
        _candidate(8.5, 550.0, 570.0, "El buzo encontró el arrecife de coral escondido."),
    ]
    corpus = [cand.text for cand in clustered + spread]
    kwargs = {
        "threshold": 8.0,
        "separation_seconds": 30.0,
        "max_clips": 3,
        "idf_corpus": corpus,
    }
    without_duration = _select_diverse(clustered + spread, **kwargs)
    assert [cand.start for cand in without_duration] == [10.0, 60.0, 110.0]
    with_duration = _select_diverse(clustered + spread, duration=600.0, **kwargs)
    assert [cand.start for cand in with_duration] == [10.0, 250.0, 450.0]


def test_select_diverse_temporal_factor_allows_exceptional_moment() -> None:
    """Un momento excepcional (9.9) sí puede duplicar franja."""
    first = _candidate(9.9, 10.0, 30.0, "El chef revela el truco secreto del arroz perfecto.")
    second = _candidate(9.9, 60.0, 80.0, "El estadio entero celebró el gol del delantero final.")
    selected = _select_diverse(
        [first, second],
        threshold=5.0,
        separation_seconds=30.0,
        max_clips=2,
        idf_corpus=[first.text, second.text],
        duration=600.0,
    )
    assert len(selected) == 2


# --- ítem 2: temas y series ----------------------------------------------------


def test_assign_series_groups_by_topic_and_numbers_titles() -> None:
    proposals = [
        _proposal(10.0, 30.0, "Los precios suben en 2026", ["precios", "mercado"]),
        _proposal(200.0, 220.0, "Hablamos de precios y ofertas", ["precios", "mercado", "ofertas"]),
        _proposal(300.0, 320.0, "Mi viaje a la playa", ["viaje", "playa"]),
        _proposal(500.0, 520.0, "El viaje más largo", ["viaje", "aventura"]),
    ]
    assigned = series.assign_series(proposals)
    by_start = {proposal.inicio: proposal for proposal in assigned}
    assert by_start[10.0].series == "precios"
    assert by_start[10.0].part_number == 1
    assert by_start[10.0].titulo.endswith(" · Parte 1")
    assert by_start[200.0].series == "precios"
    assert by_start[200.0].part_number == 2
    assert by_start[200.0].titulo.endswith(" · Parte 2")
    assert by_start[300.0].series == "viaje"
    assert by_start[300.0].part_number == 1
    assert by_start[500.0].part_number == 2


def test_assign_series_generic_words_do_not_group() -> None:
    """Compartir solo una palabra genérica ("clip") no crea una serie."""
    proposals = [
        _proposal(0.0, 20.0, "Clip uno"),
        _proposal(30.0, 50.0, "Clip dos"),
    ]
    grouped = series.assign_series(proposals)
    assert all(proposal.series is None for proposal in grouped)
    assert grouped[0].titulo == "Clip uno"
    assert grouped[1].titulo == "Clip dos"


def test_assign_series_singleton_gets_topic_but_no_series() -> None:
    assigned = series.assign_series(
        [_proposal(10.0, 30.0, "Un momento único", ["único"])]
    )
    assert assigned[0].topic == "único"
    assert assigned[0].series is None
    assert assigned[0].part_number is None
    assert assigned[0].titulo == "Un momento único"


def test_assign_series_is_idempotent_after_title_polish() -> None:
    proposals = [
        _proposal(10.0, 30.0, "Los precios suben", ["precios", "mercado"]),
        _proposal(200.0, 220.0, "Más sobre precios", ["precios", "mercado"]),
    ]
    once = series.assign_series(proposals)
    twice = series.assign_series(once)
    assert [proposal.titulo for proposal in once] == [
        proposal.titulo for proposal in twice
    ]
    assert twice[0].titulo.count("Parte") == 1


def test_validate_proposals_assigns_series_for_any_engine() -> None:
    """El embudo común (_validate_and_complete_proposals) numera series."""
    transcript = [
        TranscriptSegment(
            start=0.0,
            end=400.0,
            text="Texto de prueba.",
            words=[
                Word(start=10.0, end=11.0, text="hola."),
                Word(start=29.0, end=30.0, text="fin."),
                Word(start=200.0, end=201.0, text="hola."),
                Word(start=219.0, end=220.0, text="fin."),
            ],
        )
    ]
    proposals = [
        _proposal(10.0, 30.0, "Los precios suben", ["precios", "mercado"]),
        _proposal(200.0, 220.0, "Más sobre precios", ["precios", "mercado"]),
    ]
    valid, _ = _validate_and_complete_proposals(
        proposals, transcript, 600.0, "Cualquiera", source="de prueba"
    )
    assert len(valid) == 2
    assert valid[0].series == "precios"
    assert valid[0].part_number == 1
    assert valid[1].part_number == 2


# --- ítem 3b: split dinámico y punch-in ---------------------------------------


def test_choose_framing_splits_alternating_interview() -> None:
    segments = [
        {"start": 0.0, "end": 5.0, "speaker": 0},
        {"start": 5.0, "end": 10.0, "speaker": 1},
        {"start": 10.0, "end": 15.0, "speaker": 0},
        {"start": 15.0, "end": 20.0, "speaker": 1},
        {"start": 20.0, "end": 25.0, "speaker": 0},
    ]
    assert (
        speaker_framing.choose_framing(segments, 0.0, 25.0, "speaker") == "split"
    )


def test_choose_framing_keeps_speaker_for_monologue() -> None:
    segments = [
        {"start": 0.0, "end": 20.0, "speaker": 0},
        {"start": 20.0, "end": 22.0, "speaker": 1},
    ]
    assert (
        speaker_framing.choose_framing(segments, 0.0, 25.0, "speaker")
        == "speaker"
    )
    assert speaker_framing.choose_framing(None, 0.0, 25.0, "speaker") == "speaker"


def test_choose_framing_never_overrides_manual_framing() -> None:
    segments = [
        {"start": 0.0, "end": 5.0, "speaker": 0},
        {"start": 5.0, "end": 10.0, "speaker": 1},
        {"start": 10.0, "end": 15.0, "speaker": 0},
        {"start": 15.0, "end": 20.0, "speaker": 1},
    ]
    for configured in ("blur", "center", "split"):
        assert (
            speaker_framing.choose_framing(segments, 0.0, 20.0, configured)
            == configured
        )


def test_ranges_from_rms_marks_energy_burst() -> None:
    rms = np.full(40, 0.1)
    rms[10:14] = 0.9
    ranges = speaker_framing.ranges_from_rms(rms, 0.25)
    assert ranges == [(2.5, 3.5)]


def test_ranges_from_rms_constant_energy_has_no_peaks() -> None:
    rms = np.full(40, 0.5)
    assert speaker_framing.ranges_from_rms(rms, 0.25) == []


def test_build_filter_punch_in_adds_zoom_stage(tmp_path: Path) -> None:
    graph, video_output, _ = _build_filter(
        [(0.0, 2.0)],
        720,
        1280,
        tmp_path / "clip.ass",
        2.0,
        [],
        punch_in_ranges=[(0.5, 1.5)],
    )
    assert "crop=w='if(between(t,0.500,1.500)" in graph
    assert "[vzoom]" in graph
    assert video_output == "[vout]"


def test_build_filter_logo_adds_overlay(tmp_path: Path) -> None:
    graph, video_output, _ = _build_filter(
        [(0.0, 2.0)],
        720,
        1280,
        tmp_path / "clip.ass",
        2.0,
        [],
        logo_input=1,
    )
    assert "[1:v]scale=180:-1[logo]" in graph
    assert "overlay=x=W-w-24:y=24[vlogo]" in graph
    assert video_output == "[vlogo]"


# --- ítem 4b: forma de onda ----------------------------------------------------


def test_peaks_from_samples_normalizes_by_global_max() -> None:
    samples = np.array([0.0, 0.5, -0.25, 1.0, 0.0, 0.0, 0.2, -0.4])
    peaks = waveform.peaks_from_samples(samples, 4)
    assert peaks == pytest.approx([0.5, 1.0, 0.0, 0.4])


def test_peaks_from_samples_empty_is_zeros() -> None:
    assert waveform.peaks_from_samples(np.array([]), 3) == [0.0, 0.0, 0.0]


def test_compute_waveform_peaks_rejects_bad_range(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        waveform.compute_waveform_peaks(tmp_path / "x.mp4", 0.0, 0.0, "ffmpeg")
    with pytest.raises(ValueError):
        waveform.compute_waveform_peaks(
            tmp_path / "x.mp4", 0.0, 1.0, "ffmpeg", buckets=0
        )


# --- ítem 5: plantilla de marca -----------------------------------------------


def test_brand_kit_crud(tmp_path: Path) -> None:
    db_path = tmp_path / "test.sqlite"
    database.initialize(db_path)
    assert database.get_brand_kit(db_path) is None
    saved = database.save_brand_kit(
        db_path,
        {
            "name": "Mi canal",
            "primary_color": "#112233",
            "accent_color": "#aabbcc",
            "font_family": "Georgia",
        },
    )
    assert saved["name"] == "Mi canal"
    assert saved["primary_color"] == "#112233"
    assert database.get_brand_kit(db_path)["accent_color"] == "#aabbcc"
    # Guardar sin logo conserva el logo existente.
    database.set_brand_kit_logo(db_path, "logo.png")
    database.save_brand_kit(db_path, {"name": "Mi canal 2"})
    assert database.get_brand_kit(db_path)["logo_filename"] == "logo.png"
    assert database.get_brand_kit(db_path)["name"] == "Mi canal 2"
    assert database.delete_brand_kit(db_path) is True
    assert database.get_brand_kit(db_path) is None
    assert database.delete_brand_kit(db_path) is False


def test_write_ass_applies_brand_colors(tmp_path: Path) -> None:
    proposal = _proposal(0.0, 2.0, "Clip de prueba")
    config = ProjectConfig()
    words = [
        {
            "start": 0.0,
            "end": 1.0,
            "preview_start": 0.0,
            "preview_end": 1.0,
            "text": "hola",
        }
    ]
    output = tmp_path / "clip.ass"
    _write_ass(
        output,
        words,
        proposal,
        config,
        2.0,
        [(0.0, 2.0)],
        brand={
            "primary_color": "#112233",
            "accent_color": "#AABBCC",
            "font_family": "Georgia",
        },
    )
    content = output.read_text(encoding="utf-8")
    # ASS usa BGR: #112233 -> &H00332211, #AABBCC -> &H00CCBBAA.
    assert "Georgia" in content
    assert "&H00332211" in content
    assert "&H00CCBBAA" in content


# --- ítem 6: versiones de clip -------------------------------------------------


def _make_clip_file(clips_dir: Path, name: str, content: bytes) -> Path:
    path = clips_dir / name
    path.write_bytes(content)
    return path


def test_clip_versions_archive_list_restore(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    clips_dir = project_dir / "clips"
    clips_dir.mkdir(parents=True)
    _make_clip_file(clips_dir, "clip1.mp4", b"v-original")
    (clips_dir / "clip1.ass").write_bytes(b"subs-original")

    archived = clip_versions.archive_version(
        project_dir, clips_dir, "clip1", "clip1.mp4"
    )
    assert archived is not None and archived["version"] == 1
    assert archived["has_subtitles"] is True
    assert not (clips_dir / "clip1.mp4").exists()
    assert (project_dir / "versions" / "clip1" / "v1.mp4").read_bytes() == b"v-original"

    _make_clip_file(clips_dir, "clip1.mp4", b"v-nueva")
    clip_versions.archive_version(project_dir, clips_dir, "clip1", "clip1.mp4")
    versions = clip_versions.list_versions(project_dir, "clip1")
    assert [item["version"] for item in versions] == [2, 1]

    _make_clip_file(clips_dir, "clip1.mp4", b"v-actual")
    clip_versions.restore_version(project_dir, clips_dir, "clip1", "clip1.mp4", 1)
    assert (clips_dir / "clip1.mp4").read_bytes() == b"v-original"
    assert (clips_dir / "clip1.ass").read_bytes() == b"subs-original"
    # Restaurar archiva el vigente: ahora hay v1, v2 y v3.
    assert [item["version"] for item in clip_versions.list_versions(project_dir, "clip1")] == [3, 2, 1]


def test_clip_versions_archive_without_current_file_is_none(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    clips_dir = project_dir / "clips"
    clips_dir.mkdir(parents=True)
    assert (
        clip_versions.archive_version(project_dir, clips_dir, "clip1", "clip1.mp4")
        is None
    )
    with pytest.raises(FileNotFoundError):
        clip_versions.restore_version(project_dir, clips_dir, "clip1", "clip1.mp4", 1)


def test_clip_versions_prunes_oldest_beyond_max(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    clips_dir = project_dir / "clips"
    clips_dir.mkdir(parents=True)
    for index in range(clip_versions.MAX_VERSIONS + 2):
        _make_clip_file(clips_dir, "clip1.mp4", f"v{index}".encode())
        clip_versions.archive_version(project_dir, clips_dir, "clip1", "clip1.mp4")
    versions = clip_versions.list_versions(project_dir, "clip1")
    assert len(versions) == clip_versions.MAX_VERSIONS
    assert min(item["version"] for item in versions) == 3


# --- ítem 7: traducción de subtítulos ------------------------------------------


def test_clip_words_to_cues_uses_local_times() -> None:
    words = [
        {
            "start": 100.0,
            "end": 101.0,
            "preview_start": 0.0,
            "preview_end": 1.0,
            "text": "hola",
        },
        {
            "start": 101.0,
            "end": 102.0,
            "preview_start": 1.0,
            "preview_end": 2.0,
            "text": "mundo",
        },
    ]
    cues = translate.clip_words_to_cues(words)
    assert len(cues) == 1
    assert cues[0]["start"] == 0.0
    assert cues[0]["end"] == 2.0
    assert cues[0]["text"] == "hola mundo"


def test_translate_cues_maps_numbered_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        translate,
        "chat_completion_text",
        lambda llm, **kwargs: "1. Hello world\n2. How are you?",
    )
    cues = [
        {"start": 0.0, "end": 1.0, "text": "Hola mundo"},
        {"start": 1.0, "end": 2.0, "text": "Cómo estás"},
    ]
    translated = translate.translate_cues(cues, "en", llm=None)  # type: ignore[arg-type]
    assert [cue["text"] for cue in translated] == ["Hello world", "How are you?"]
    assert translated[0]["start"] == 0.0
    assert translated[1]["end"] == 2.0


def test_translate_cues_keeps_original_when_line_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        translate, "chat_completion_text", lambda llm, **kwargs: "1. Hello world"
    )
    cues = [
        {"start": 0.0, "end": 1.0, "text": "Hola mundo"},
        {"start": 1.0, "end": 2.0, "text": "Cómo estás"},
    ]
    translated = translate.translate_cues(cues, "en", llm=None)  # type: ignore[arg-type]
    assert translated[1]["text"] == "Cómo estás"


def test_translate_cues_rejects_unknown_language() -> None:
    with pytest.raises(ValueError):
        translate.translate_cues(
            [{"start": 0.0, "end": 1.0, "text": "Hola"}], "xx", llm=None  # type: ignore[arg-type]
        )


def test_translated_words_distributes_times_evenly() -> None:
    cues = [{"start": 0.0, "end": 2.0, "text": "Hello brave world"}]
    words = translate.translated_words(cues)
    assert [word["text"] for word in words] == ["Hello", "brave", "world"]
    assert words[0]["preview_start"] == 0.0
    assert words[-1]["preview_end"] == 2.0
    assert words[1]["preview_start"] == pytest.approx(0.667)


# --- ítem 8: gancho y muletillas ----------------------------------------------


def test_hook_range_is_first_three_seconds() -> None:
    assert hook_range(10.0, 30.0) == (10.0, 13.0)
    assert hook_range(10.0, 11.5) == (10.0, 11.5)
    assert hook_range(5.0, 5.0) == (5.0, 5.0)


def _transcript_with_words(
    entries: list[tuple[float, float, str]],
) -> list[TranscriptSegment]:
    return [
        TranscriptSegment(
            start=entries[0][0],
            end=entries[-1][1],
            text=" ".join(text for _, _, text in entries),
            words=[
                Word(start=start, end=end, text=text)
                for start, end, text in entries
            ],
        )
    ]


def test_fillers_detects_isolated_filler_and_micro_pause() -> None:
    transcript = _transcript_with_words(
        [
            (0.0, 0.5, "hola"),       # palabra normal
            (0.8, 1.1, "eh"),         # muletilla aislada (hueco 0.3 antes)
            (1.4, 2.0, "mundo"),      # hueco 0.3 después de "eh"
            (2.4, 3.0, "esto"),       # hueco 0.4 -> micro-pausa + "esto" aislado
            (3.0, 3.5, "es"),         # pegado a "esto"
        ]
    )
    proposal = _proposal(0.0, 3.5, "Clip")
    exclusions = fillers.detect_filler_exclusions(transcript, proposal)
    # "eh" (0.8-1.1) se quita; la micro-pausa 2.0-2.4 se recorta a
    # 2.05-2.35 (conserva 0.05 s a cada lado) y "esto" (2.4-3.0) se
    # quita por aislado (hueco 0.4 antes). Los dos últimos intervalos
    # quedan separados por el respiro conservado.
    assert (0.8, 1.1) in exclusions
    assert any(
        start == pytest.approx(2.05) and end == pytest.approx(2.35)
        for start, end in exclusions
    )
    assert (2.4, 3.0) in exclusions


def test_fillers_ignores_attached_demonstrative() -> None:
    transcript = _transcript_with_words(
        [
            (0.0, 0.5, "mira"),
            (0.55, 0.9, "este"),      # demostrativo pegado (huecos 0.05)
            (0.95, 1.5, "video"),
        ]
    )
    proposal = _proposal(0.0, 1.5, "Clip")
    assert fillers.detect_filler_exclusions(transcript, proposal) == []


def test_fillers_detects_o_sea_bigram() -> None:
    transcript = _transcript_with_words(
        [
            (0.0, 0.5, "bueno"),
            (0.8, 1.0, "o"),
            (1.0, 1.3, "sea"),
            (1.6, 2.0, "vamos"),
        ]
    )
    proposal = _proposal(0.0, 2.0, "Clip")
    exclusions = fillers.detect_filler_exclusions(transcript, proposal)
    assert (0.8, 1.3) in exclusions


def test_fillers_ignores_long_silences() -> None:
    """Un hueco de 2 s es silencio (lo cubre la otra opción), no micro-pausa."""
    transcript = _transcript_with_words(
        [(0.0, 0.5, "hola"), (2.5, 3.0, "adiós")]
    )
    proposal = _proposal(0.0, 3.0, "Clip")
    assert fillers.detect_filler_exclusions(transcript, proposal) == []


# --- render con ffmpeg (cableado de _render_clip) ------------------------------


@pytest.fixture(scope="module")
def _video_with_audio(tmp_path_factory: pytest.TempPathFactory) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("FFmpeg no está disponible.")
    path = tmp_path_factory.mktemp("phase3-selection") / "source.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:r=25:d=4",
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
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def test_waveform_and_energy_on_real_video(_video_with_audio: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    peaks = waveform.compute_waveform_peaks(
        _video_with_audio, 0.0, 4.0, ffmpeg, buckets=8
    )
    assert len(peaks) == 8
    assert max(peaks) == pytest.approx(1.0)
    # Tono constante: sin picos de energía, pero sin fallar.
    assert (
        speaker_framing.energy_peak_ranges(_video_with_audio, 0.0, 4.0, ffmpeg)
        == []
    )


def test_render_clip_with_phase3_options(
    _video_with_audio: Path, tmp_path: Path
) -> None:
    """Punch-in + muletillas + split dinámico renderizan sin error."""
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    proposal = _proposal(0.0, 4.0, "Clip de prueba")
    transcript = _transcript_with_words(
        [
            (0.0, 0.5, "hola"),
            (0.8, 1.1, "eh"),
            (1.4, 2.0, "mundo"),
            (2.2, 3.0, "esto"),
            (3.2, 3.9, "termina."),
        ]
    )
    config = ProjectConfig(
        options=EditOptions(
            face_tracking=False,
            remove_fillers=True,
            punch_in=True,
            dynamic_split=True,
        )
    )
    segments = [
        {"start": 0.0, "end": 1.0, "speaker": 0},
        {"start": 1.0, "end": 2.0, "speaker": 1},
        {"start": 2.0, "end": 3.0, "speaker": 0},
        {"start": 3.0, "end": 4.0, "speaker": 1},
    ]
    output = tmp_path / "clip.mp4"
    duration, _ = render_clip(
        _video_with_audio,
        output,
        tmp_path / "clip.ass",
        proposal,
        config,
        transcript,
        ffmpeg,
        lambda _: None,
        speaker_segments=segments,
    )
    assert output.is_file()
    # Quitados: "eh" (0.3 s), "esto" aislado (0.8 s) y dos micro-pausas
    # (0.2 s cada una); los fragmentos de 0.05 s entre huecos se
    # descartan (<0.08 s): 4.0 - 1.6 = 2.4 s.
    assert duration == pytest.approx(2.4, abs=0.1)


# --- endpoints (marca, forma de onda, versiones, traducción) ------------------


def _test_settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        data_dir=tmp_path / "data",
        cors_origins=("http://localhost:5173",),
        llm_provider="Groq",
        llm_api_key="",
        llm_model="",
        llm_base_url="",
        whisper_model="small",
    )


def _seed_completed_project(
    settings: AppSettings,
    project_id: str,
    *,
    clips: list[dict] | None = None,
    metadata: dict | None = None,
    source_bytes: bytes = b"video-bytes",
) -> None:
    project_dir = settings.data_dir / "projects" / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "source").write_bytes(source_bytes)
    database.initialize(settings.database_path)
    database.create_project(
        settings.database_path,
        project_id,
        "Proyecto de prueba",
        project_dir / "source",
        ProjectConfig().model_dump(mode="json"),
        "test-user",
    )
    database.update_project(
        settings.database_path,
        project_id,
        status="completed",
        clips=clips or [],
        metadata=metadata or {},
    )


def test_brand_kit_endpoints(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    database.initialize(settings.database_path)
    with TestClient(create_app(settings, auth_required=False)) as client:
        assert client.get("/api/brand-kit").json() == {"brand_kit": None}
        assert client.get("/api/brand-kit/logo").status_code == 404
        saved = client.put(
            "/api/brand-kit",
            json={
                "name": "Mi canal",
                "primary_color": "#112233",
                "accent_color": "#aabbcc",
                "font_family": "Georgia",
            },
        )
        assert saved.status_code == 200
        kit = saved.json()["brand_kit"]
        assert kit["name"] == "Mi canal"
        assert kit["logo_url"] is None
        uploaded = client.post(
            "/api/brand-kit/logo",
            files={"file": ("logo.png", BytesIO(_MINIMAL_PNG), "image/png")},
        )
        assert uploaded.status_code == 200
        assert uploaded.json()["brand_kit"]["logo_url"] == "/api/brand-kit/logo"
        logo = client.get("/api/brand-kit/logo")
        assert logo.status_code == 200
        assert logo.headers["content-type"] == "image/png"
        assert logo.content == _MINIMAL_PNG
        bad = client.post(
            "/api/brand-kit/logo",
            files={"file": ("logo.txt", BytesIO(b"hola"), "text/plain")},
        )
        assert bad.status_code == 415
        assert client.delete("/api/brand-kit").status_code == 204
        assert client.get("/api/brand-kit").json() == {"brand_kit": None}
        assert client.get("/api/brand-kit/logo").status_code == 404


def test_waveform_endpoint(
    _video_with_audio: Path, tmp_path: Path
) -> None:
    settings = _test_settings(tmp_path)
    project_id = str(uuid.uuid4())
    _seed_completed_project(
        settings,
        project_id,
        metadata={"duration": 4.0},
        source_bytes=_video_with_audio.read_bytes(),
    )
    with TestClient(create_app(settings, auth_required=False)) as client:
        assert client.get("/api/projects/no-existe/waveform").status_code == 404
        bad = client.get(
            f"/api/projects/{project_id}/waveform",
            params={"start": 3.0, "end": 2.0},
        )
        assert bad.status_code == 400
        response = client.get(
            f"/api/projects/{project_id}/waveform",
            params={"start": 0, "end": 4.0, "buckets": 10},
        )
        assert response.status_code == 200
        payload = response.json()
        assert len(payload["peaks"]) == 10
        assert max(payload["peaks"]) == pytest.approx(1.0)


def test_versions_endpoints_empty_and_missing(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    project_id = str(uuid.uuid4())
    _seed_completed_project(
        settings,
        project_id,
        clips=[
            {
                "id": "clip-1",
                "title": "Clip",
                "start": 0.0,
                "end": 10.0,
                "filename": "clip-1.mp4",
                "status": "ready",
            }
        ],
    )
    with TestClient(create_app(settings, auth_required=False)) as client:
        listed = client.get(f"/api/projects/{project_id}/clips/clip-1/versions")
        assert listed.status_code == 200
        assert listed.json() == {"versions": []}
        missing = client.post(
            f"/api/projects/{project_id}/clips/clip-1/versions/1/restore"
        )
        assert missing.status_code == 404
        assert (
            client.get(
                f"/api/projects/{project_id}/clips/clip-1/versions/1/file"
            ).status_code
            == 404
        )


def test_translate_requires_llm_key(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    project_id = str(uuid.uuid4())
    _seed_completed_project(
        settings,
        project_id,
        clips=[
            {
                "id": "clip-1",
                "title": "Clip",
                "start": 0.0,
                "end": 2.0,
                "filename": "clip-1.mp4",
                "status": "ready",
                "transcript": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "preview_start": 0.0,
                        "preview_end": 1.0,
                        "text": "hola",
                    },
                    {
                        "start": 1.0,
                        "end": 2.0,
                        "preview_start": 1.0,
                        "preview_end": 2.0,
                        "text": "mundo",
                    },
                ],
            }
        ],
    )
    with TestClient(create_app(settings, auth_required=False)) as client:
        response = client.post(
            f"/api/projects/{project_id}/clips/clip-1/subtitles/translate",
            json={"target_language": "en", "format_name": "srt"},
        )
        assert response.status_code == 409
        assert "clave de IA" in response.json()["detail"]
