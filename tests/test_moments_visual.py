"""Tests de los canales visuales + perfiles de contenido (Fase C, v1.2.0).

Todo sintético o con mocks: sin videos reales ni red, salvo videos
generados con ffmpeg (lavfi) para la extracción. numpy es requisito
del motor (ver tests/test_moments.py); sin numpy, el módulo entero se
salta. cv2 NO está instalado en este entorno: los tests de rostros
usan un detector simulado o se saltan cuando exigen cv2 real.

Lección v1.1.1 (NO NEGOCIABLE): los fixtures son conversacionales
realistas en español — habla normal, picos de audio 1.5–1.8x — nunca
ejemplos caricaturescos. Un video plano debe seguir dando cero clips
con los canales nuevos activos.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from backend import moments, visual_signals  # noqa: E402
from backend.moments import MomentScorer, build_phrases, detect_content_profile  # noqa: E402
from backend.schemas import ProjectConfig, TranscriptSegment, Word  # noqa: E402
from backend.visual_signals import VisualStats  # noqa: E402

_FLAT_TOPICS = [
    "El tema de hoy es el clima en la ciudad y las noticias locales.",
    "Vamos a hablar del tráfico y de los horarios del transporte público.",
    "La reunión de ayer trató varios asuntos administrativos pendientes.",
    "El informe describe el procedimiento habitual de cada departamento.",
    "Recordamos que el horario de oficina es de ocho a cinco.",
]
_EXCITING_TEXTS = [
    "¿Sabes cuál es el secreto increíble para ganar dinero gratis?",
    "Mira, esto es brutal y nadie te lo cuenta, es impresionante.",
    "Primero anotas 3 ideas, segundo las ordenas y tercero actúas ya.",
    "¡Imagínalo! En 2024 esto cambió todo para siempre, es una locura.",
]
# Frases conversacionales realistas (estilo v1.1.1): habla normal en
# español con densidad de ganchos natural — alguna pregunta, alguna
# exclamación, algún número — sin saturar.
_REALISTIC_TEXTS = [
    "Bueno, hoy vamos a hablar de un tema que me preguntan mucho en los comentarios.",
    "Lo primero que hay que entender es cómo funciona el proceso completo.",
    "Yo empecé con esto hace un par de años y al principio cometí muchos errores.",
    "La placa que tengo aquí es una de las más comunes en el mercado actual.",
    "Mucha gente piensa que es complicado, pero en realidad es bastante directo.",
    "Ahora vamos a ver qué pasa cuando encendemos el equipo por primera vez.",
    "¿Cuál es la diferencia entre estos dos modelos? Vamos a verlo en detalle.",
    "Miren este resultado, la diferencia es bastante clara en la práctica.",
    "Hay tres cosas que siempre reviso antes de instalar cualquier pieza.",
    "Primero la compatibilidad, segundo el consumo y tercero el tamaño físico.",
    "¡Ojo con esto! Si lo conectas al revés puedes dañar la placa.",
    "Para los que juegan, esto hace una diferencia notable en los cuadros por segundo.",
    "¿Vale la pena gastar el doble por el modelo superior? Depende de tu uso.",
    "En resumen, es una buena opción si buscas algo equilibrado y sin gastar mucho.",
]


def _segment(start: float, text: str, *, wpm: float = 150) -> TranscriptSegment:
    tokens = text.split()
    span = max(1.0, len(tokens) * 60.0 / wpm)
    per_word = span / max(1, len(tokens))
    words = []
    cursor = start
    for token in tokens:
        words.append(Word(start=round(cursor, 3), end=round(cursor + per_word, 3), text=token))
        cursor += per_word
    return TranscriptSegment(start=start, end=round(start + span, 3), text=text, words=words)


def _transcript(duration: float, *, exciting_at: float | None = None) -> list[TranscriptSegment]:
    """Transcripción sintética plana, con un bloque excitante opcional."""
    segments: list[TranscriptSegment] = []
    cursor = 0.0
    index = 0
    inserted = False
    while cursor < duration - 6:
        if exciting_at is not None and not inserted and cursor >= exciting_at:
            for text in _EXCITING_TEXTS:
                segment = _segment(cursor, text, wpm=210)
                segments.append(segment)
                cursor = segment.end + 0.4
            index += 4
            inserted = True
            continue
        segment = _segment(cursor, _FLAT_TOPICS[index % len(_FLAT_TOPICS)])
        segments.append(segment)
        cursor = segment.end + 0.8
        index += 1
    return segments


def _sparse_transcript(duration: float) -> list[TranscriptSegment]:
    """Habla dispersa (gameplay): un comentario corto cada 30 s."""
    segments: list[TranscriptSegment] = []
    cursor = 5.0
    index = 0
    while cursor < duration - 6:
        segment = _segment(cursor, _FLAT_TOPICS[index % len(_FLAT_TOPICS)], wpm=150)
        segments.append(segment)
        cursor += 30.0
        index += 1
    return segments


def _realistic_transcript(duration: float) -> list[TranscriptSegment]:
    rng = np.random.default_rng(42)
    segments: list[TranscriptSegment] = []
    cursor = 0.0
    index = 0
    while cursor < duration - 8:
        segment = _segment(
            cursor,
            _REALISTIC_TEXTS[index % len(_REALISTIC_TEXTS)],
            wpm=float(rng.uniform(130, 170)),
        )
        segments.append(segment)
        cursor = segment.end + float(rng.uniform(0.3, 1.0))
        index += 1
    return segments


def _audio(duration: float, *, peak_at: float | None = None, peak_rms: float = 0.6) -> Any:
    rng = np.random.default_rng(7)
    rms = 0.10 + 0.01 * rng.standard_normal(int(duration))
    if peak_at is not None:
        start = int(peak_at)
        rms[start : start + 20] = peak_rms
    return np.clip(rms, 0.001, None)


def _realistic_audio(duration: float) -> Any:
    rng = np.random.default_rng(42)
    rms = 0.10 + 0.02 * rng.standard_normal(int(duration))
    for at, length, level in [(90, 8, 0.16), (210, 6, 0.18), (350, 10, 0.15), (500, 7, 0.17)]:
        rms[at : at + length] = level
    return np.clip(rms, 0.001, None)


def _cuts_video(path: Path, segments: list[tuple[str, float]]) -> Path:
    """Video de colores sólidos con cortes duros (un comando ffmpeg)."""
    command = ["ffmpeg", "-v", "error"]
    for color, seconds in segments:
        command += [
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:size=320x180:rate=15:duration={seconds}",
        ]
    inputs = "".join(f"[{index}:v]" for index in range(len(segments)))
    command += [
        "-filter_complex",
        f"{inputs}concat=n={len(segments)}:v=1:a=0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    return path


def _static_video(path: Path, duration: float) -> Path:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:size=320x180:rate=15:duration={duration}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(path),
        ],
        check=False,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    return path


class _FakeDetector:
    """Detector YuNet simulado: rostro en los frames pares, nada en los impares.

    El rostro es un cuadrado 100x100 centrado en el lienzo 320x320 con
    confianza 0.9: valor esperado 0.9 (tamaño prominente + centrado).
    """

    def __init__(self) -> None:
        self.calls = 0

    def setInputSize(self, _size: tuple[int, int]) -> None:
        pass

    def detect(self, _frame: Any) -> tuple[int, Any]:
        self.calls += 1
        if self.calls % 2 == 1:
            face = np.zeros(15, dtype=np.float64)
            face[0], face[1], face[2], face[3] = 110, 110, 100, 100
            face[-1] = 0.9
            return 0, np.array([face])
        return 0, None


# ---------------------------------------------------------------------------
# Parsing de showinfo y cortes de escena
# ---------------------------------------------------------------------------


def test_parse_showinfo_cut_times() -> None:
    # Salida realista: líneas de config y de metadatos SEI (sin pts_time)
    # mezcladas con líneas de frame (con pts_time).
    stderr = (
        "[Parsed_showinfo_1 @ 0x1] config in time_base: 1/15360, frame_rate: 30/1\n"
        "[Parsed_showinfo_1 @ 0x1] config out time_base: 0/0, frame_rate: 0/0\n"
        "[Parsed_showinfo_1 @ 0x1] n:   0 pts:  30720 pts_time:2       duration: 512 fmt:yuv420p\n"
        "[Parsed_showinfo_1 @ 0x1]   side data - H.26[45] User Data Unregistered SEI message\n"
        "[Parsed_showinfo_1 @ 0x1] n:   1 pts:  61440 pts_time:4       duration: 512 fmt:yuv420p\n"
        "[Parsed_showinfo_1 @ 0x1] n:   2 pts:  92160 pts_time:6.5     duration: 512 fmt:yuv420p\n"
        "frame=   10 fps=0.0 q=0.0 Lsize=N/A time=00:00:10.00\n"
    )
    assert visual_signals.parse_showinfo_cut_times(stderr) == [2.0, 4.0, 6.5]


def test_parse_showinfo_cut_times_empty() -> None:
    assert visual_signals.parse_showinfo_cut_times("sin cortes aquí\n") == []
    assert visual_signals.parse_showinfo_cut_times("") == []


def test_extract_scene_cut_timeline_from_synthetic_video(tmp_path: Path) -> None:
    # Cortes duros en 2, 4, 6 y 8 s (colores de alto contraste).
    video = _cuts_video(
        tmp_path / "cuts.mp4",
        [("red", 2), ("blue", 2), ("green", 2), ("black", 2), ("white", 2)],
    )
    timeline = visual_signals.extract_scene_cut_timeline(video, "ffmpeg", 10.0)
    assert timeline is not None
    assert timeline.shape == (10,)
    for second in (2, 4, 6, 8):
        assert timeline[second] > 0.5, f"falta el corte del segundo {second}"
    assert timeline[0] == 0.0
    assert timeline[1] == 0.0
    assert timeline[9] == 0.0


def test_extract_scene_cut_timeline_missing_file_returns_none(tmp_path: Path) -> None:
    assert (
        visual_signals.extract_scene_cut_timeline(tmp_path / "no-existe.mp4", "ffmpeg", 10.0)
        is None
    )


def test_extract_scene_cut_timeline_missing_ffmpeg_returns_none(tmp_path: Path) -> None:
    video = _static_video(tmp_path / "static.mp4", 3.0)
    assert visual_signals.extract_scene_cut_timeline(video, "ffmpeg-inexistente", 3.0) is None


# ---------------------------------------------------------------------------
# Movimiento
# ---------------------------------------------------------------------------


def test_extract_motion_timeline_static_video_is_zero(tmp_path: Path) -> None:
    video = _static_video(tmp_path / "static.mp4", 10.0)
    timeline = visual_signals.extract_motion_timeline(video, "ffmpeg", 10.0)
    assert timeline is not None
    assert timeline.shape == (10,)
    assert float(np.max(timeline)) == 0.0


def test_extract_motion_timeline_cuts_video_spikes_at_cuts(tmp_path: Path) -> None:
    # Los cortes son los únicos cambios: el mayor (negro->blanco, 127.5
    # de MAD) es el pico máximo del timeline normalizado.
    video = _cuts_video(
        tmp_path / "cuts.mp4",
        [("red", 2), ("blue", 2), ("green", 2), ("black", 2), ("white", 2)],
    )
    timeline = visual_signals.extract_motion_timeline(video, "ffmpeg", 10.0)
    assert timeline is not None
    assert timeline.shape == (10,)
    assert float(np.max(timeline)) == pytest.approx(1.0)
    assert timeline[8] > 0.9
    assert timeline[0] == 0.0
    assert timeline[1] == 0.0


def test_motion_normalization_is_relative() -> None:
    # Constante (movimiento uniforme): sin contraste, todo a 0.
    timeline, mean_raw = visual_signals._normalize_motion(np.full(60, 3.6))
    assert float(np.max(timeline)) == 0.0
    assert mean_raw == pytest.approx(3.6)
    # Un pico aislado: 1.0 en el pico, 0 en el resto.
    raw = np.zeros(60)
    raw[30] = 50.0
    timeline, mean_raw = visual_signals._normalize_motion(raw)
    assert timeline[30] == pytest.approx(1.0)
    assert float(np.max(np.delete(timeline, 30))) == 0.0
    assert mean_raw == pytest.approx(50.0 / 60)
    # Ruido diminuto (por debajo del piso de sigma): no se amplifica.
    raw = np.full(60, 0.2)
    raw[10] = 0.5
    timeline, _ = visual_signals._normalize_motion(raw)
    assert float(np.max(timeline)) == 0.0


def test_extract_motion_timeline_missing_file_returns_none(tmp_path: Path) -> None:
    assert (
        visual_signals.extract_motion_timeline(tmp_path / "no-existe.mp4", "ffmpeg", 10.0) is None
    )


def test_combine_visual_timeline() -> None:
    scene = np.zeros(10)
    scene[2] = 0.63
    motion = np.zeros(10)
    motion[5] = 0.8
    combined = visual_signals.combine_visual_timeline(scene, motion)
    assert combined is not None
    assert combined[2] == pytest.approx(0.63)
    assert combined[5] == pytest.approx(0.8)
    assert combined[0] == 0.0
    # El máximo (no la media): un corte que coincide con movimiento no
    # se cuenta dos veces.
    motion[2] = 0.9
    combined = visual_signals.combine_visual_timeline(scene, motion)
    assert combined is not None
    assert combined[2] == pytest.approx(0.9)
    assert visual_signals.combine_visual_timeline(None, None) is None
    assert visual_signals.combine_visual_timeline(scene, None) is not None
    assert visual_signals.combine_visual_timeline(None, motion) is not None


# ---------------------------------------------------------------------------
# Extracción combinada
# ---------------------------------------------------------------------------


def test_extract_visual_timelines_bundle(tmp_path: Path) -> None:
    video = _cuts_video(
        tmp_path / "cuts.mp4",
        [("red", 2), ("blue", 2), ("green", 2), ("black", 2), ("white", 2)],
    )
    bundle = visual_signals.extract_visual_timelines(video, "ffmpeg", 10.0)
    assert bundle.scene_cuts is not None
    assert bundle.motion is not None
    assert bundle.cut_count == 4
    assert bundle.stats.cut_rate_per_minute == pytest.approx(24.0)  # 4 cortes / 10 s.
    assert bundle.mean_motion_raw > 0
    assert bundle.combined is not None
    assert float(np.max(bundle.combined)) > 0.5
    # Sin cv2 en este entorno no hay canal de rostros (en un PC con
    # cv2, este video sin rostros daría un timeline de ceros).
    if importlib.util.find_spec("cv2") is None:
        assert bundle.face is None


def test_extract_visual_timelines_missing_file(tmp_path: Path) -> None:
    bundle = visual_signals.extract_visual_timelines(tmp_path / "no-existe.mp4", "ffmpeg", 10.0)
    assert bundle.scene_cuts is None
    assert bundle.motion is None
    assert bundle.face is None
    assert bundle.cut_count == 0
    assert bundle.combined is None


def test_extract_visual_timelines_cancelled_returns_partial(tmp_path: Path) -> None:
    video = _static_video(tmp_path / "static.mp4", 5.0)
    bundle = visual_signals.extract_visual_timelines(
        video, "ffmpeg", 5.0, should_cancel=lambda: True
    )
    # Cancelado antes de empezar: ningún canal (mejor ausencia que datos parciales).
    assert bundle.scene_cuts is None
    assert bundle.motion is None
    assert bundle.face is None


# ---------------------------------------------------------------------------
# Rostros (sin cv2 real: detector simulado)
# ---------------------------------------------------------------------------


def test_import_visual_signals_does_not_import_cv2() -> None:
    """Importar visual_signals no debe importar cv2 (perezoso).

    Se comprueba en un intérprete nuevo, como el test de onnxruntime de
    test_moments_tier1.py.
    """
    code = (
        "import sys\n"
        "from backend import visual_signals\n"
        "assert 'cv2' not in sys.modules\n"
        "assert visual_signals._SCENE_THRESHOLD == 0.4\n"
        "assert 'cv2' not in sys.modules\n"
    )
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert result.returncode == 0, result.stderr


def test_import_moments_does_not_import_cv2() -> None:
    """Importar moments (que importa visual_signals) tampoco arrastra cv2."""
    code = (
        "import sys\n"
        "from backend import moments\n"
        "assert 'cv2' not in sys.modules\n"
        "assert moments._WEIGHT_VISUAL == 0.5\n"
        "assert 'cv2' not in sys.modules\n"
    )
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert result.returncode == 0, result.stderr


def test_face_timeline_without_cv2_returns_none(tmp_path: Path) -> None:
    if importlib.util.find_spec("cv2") is not None:
        pytest.skip("cv2 está instalado: este test exige su ausencia")
    video = _static_video(tmp_path / "static.mp4", 5.0)
    assert visual_signals.extract_face_timeline(video, "ffmpeg", 5.0) is None


def test_face_timeline_with_fake_detector(tmp_path: Path) -> None:
    video = _static_video(tmp_path / "static.mp4", 6.0)
    detector = _FakeDetector()
    timeline = visual_signals.extract_face_timeline(video, "ffmpeg", 6.0, detector=detector)
    assert timeline is not None
    assert timeline.shape == (6,)
    assert detector.calls == 6  # Un frame por segundo.
    assert timeline[0] == pytest.approx(0.9)
    assert timeline[1] == 0.0
    assert timeline[2] == pytest.approx(0.9)
    assert timeline[3] == 0.0
    assert timeline[4] == pytest.approx(0.9)
    assert timeline[5] == 0.0


def test_face_frame_value_weights_size_and_centrality() -> None:
    def detection(x: float, y: float, w: float, h: float, confidence: float) -> Any:
        row = np.zeros(15, dtype=np.float64)
        row[0], row[1], row[2], row[3] = x, y, w, h
        row[-1] = confidence
        return row

    assert visual_signals._face_frame_value(None, 320, 320) == 0.0
    assert visual_signals._face_frame_value([], 320, 320) == 0.0
    big_centered = visual_signals._face_frame_value([detection(110, 110, 100, 100, 0.9)], 320, 320)
    small_corner = visual_signals._face_frame_value([detection(0, 0, 20, 20, 0.9)], 320, 320)
    assert big_centered == pytest.approx(0.9)
    assert small_corner < big_centered
    assert small_corner > 0.0
    # La confianza escala el valor.
    assert visual_signals._face_frame_value([detection(110, 110, 100, 100, 0.6)], 320, 320) == (
        pytest.approx(0.6)
    )


def test_face_timeline_missing_model_returns_none(tmp_path: Path) -> None:
    video = _static_video(tmp_path / "static.mp4", 3.0)
    assert (
        visual_signals.extract_face_timeline(
            video, "ffmpeg", 3.0, model_path=tmp_path / "no-existe.onnx"
        )
        is None
    )


def test_face_timeline_missing_file_returns_none(tmp_path: Path) -> None:
    assert (
        visual_signals.extract_face_timeline(
            tmp_path / "no-existe.mp4", "ffmpeg", 10.0, detector=_FakeDetector()
        )
        is None
    )


# ---------------------------------------------------------------------------
# Perfiles de contenido
# ---------------------------------------------------------------------------


def test_detect_profile_talk() -> None:
    phrases = build_phrases(_transcript(600))
    assert (
        detect_content_profile(phrases, 600, cut_rate_per_minute=0.5, mean_motion_raw=2.0) == "talk"
    )


def test_detect_profile_gameplay_by_cuts() -> None:
    phrases = build_phrases(_sparse_transcript(600))
    assert (
        detect_content_profile(phrases, 600, cut_rate_per_minute=10.0, mean_motion_raw=2.0)
        == "gameplay"
    )


def test_detect_profile_gameplay_by_motion() -> None:
    phrases = build_phrases(_sparse_transcript(600))
    assert (
        detect_content_profile(phrases, 600, cut_rate_per_minute=0.5, mean_motion_raw=12.0)
        == "gameplay"
    )


def test_detect_profile_mixed_dense_speech_high_motion() -> None:
    # Habla densa + mucha actividad visual: ni talk ni gameplay.
    phrases = build_phrases(_transcript(600))
    assert (
        detect_content_profile(phrases, 600, cut_rate_per_minute=0.5, mean_motion_raw=12.0)
        == "mixed"
    )


def test_detect_profile_mixed_sparse_speech_calm_visuals() -> None:
    # Habla dispersa + imagen calma: ni talk ni gameplay.
    phrases = build_phrases(_sparse_transcript(600))
    assert (
        detect_content_profile(phrases, 600, cut_rate_per_minute=0.5, mean_motion_raw=2.0)
        == "mixed"
    )


def test_detect_profile_mixed_without_visual_stats() -> None:
    # CRÍTICO: sin estadísticas visuales no hay evidencia visual y el
    # perfil es "mixed" (pesos v1.1.1 sin cambios), aunque el habla sea
    # densa. Así los fixtures existentes mantienen su comportamiento.
    phrases = build_phrases(_transcript(600))
    assert detect_content_profile(phrases, 600) == "mixed"
    phrases = build_phrases(_sparse_transcript(600))
    assert detect_content_profile(phrases, 600) == "mixed"


def test_detect_profile_mixed_without_phrases() -> None:
    assert detect_content_profile([], 600, cut_rate_per_minute=0.5, mean_motion_raw=2.0) == "mixed"


def test_profile_weights() -> None:
    phrases = build_phrases(_transcript(120))
    audio = _audio(120)
    mixed = MomentScorer(phrases, audio, 120)
    assert mixed.profile == "mixed"
    assert mixed._weights == {
        "audio": 1.0,
        "texto": 1.5,
        "ritmo": 1.0,
        "gancho_inicial": 1.0,
        "eventos": 1.5,
        "visual": 0.5,
        "rostro": 0.5,
        "hablantes": 0.75,
    }
    talk = MomentScorer(phrases, audio, 120, profile="talk")
    assert talk.profile == "talk"
    assert talk._weights["texto"] == 2.0
    assert talk._weights["visual"] == 0.25
    assert talk._weights["hablantes"] == 0.75
    gameplay = MomentScorer(phrases, audio, 120, profile="gameplay")
    assert gameplay.profile == "gameplay"
    assert gameplay._weights["audio"] == 1.5
    assert gameplay._weights["visual"] == 1.0
    assert gameplay._weights["texto"] == 1.0
    assert gameplay._weights["hablantes"] == 0.75
    unknown = MomentScorer(phrases, audio, 120, profile="inexistente")
    assert unknown.profile == "mixed"


def test_talk_profile_boosts_text_heavy_window() -> None:
    transcript = _transcript(600, exciting_at=300)
    audio = _audio(600, peak_at=300)
    phrases = build_phrases(transcript)
    exciting_idx = next(
        idx for idx, phrase in enumerate(phrases) if phrase.text.startswith("¿Sabes")
    )
    mixed = MomentScorer(phrases, audio, 600, profile="mixed")
    talk = MomentScorer(phrases, audio, 600, profile="talk")
    score_mixed, _, _ = mixed.score_window(exciting_idx, exciting_idx + 3)
    score_talk, _, _ = talk.score_window(exciting_idx, exciting_idx + 3)
    # La ventana excitante es sobre todo texto: talk (texto 2.0) la sube.
    assert score_talk > score_mixed


# ---------------------------------------------------------------------------
# Canales visuales en el scorer
# ---------------------------------------------------------------------------


def _exciting_window() -> tuple[list[Any], Any, int]:
    transcript = _transcript(600, exciting_at=300)
    audio = _audio(600, peak_at=300)
    phrases = build_phrases(transcript)
    exciting_idx = next(
        idx for idx, phrase in enumerate(phrases) if phrase.text.startswith("¿Sabes")
    )
    return phrases, audio, exciting_idx


def test_visual_channel_boosts_window_with_cuts() -> None:
    # La ventana excitante ya está casi saturada (9.1/10): un canal
    # visual de 0.83 no puede subirla más (la media ponderada diluye).
    # La comparación con sentido es una ventana PLANA: ahí los cortes
    # sí suben la puntuación de forma clara.
    phrases, audio, exciting_idx = _exciting_window()
    plain = MomentScorer(phrases, audio, 600)
    flat_plain, breakdown_plain, _ = plain.score_window(0, 3)
    assert "visual" not in breakdown_plain
    scene = np.zeros(600)
    scene[0:20] = 0.63  # Un corte por segundo durante la ventana plana.
    scene[300:320] = 0.63  # Y también durante el momento excitante.
    motion = np.zeros(600)
    # Solo componentes (sin combinado): el scorer los combina.
    with_visual = MomentScorer(
        phrases, audio, 600, scene_cut_timeline=scene, motion_timeline=motion
    )
    assert with_visual.visual_timeline is not None
    assert float(np.max(with_visual.visual_timeline[300:320])) == pytest.approx(0.63)
    flat_visual, breakdown_visual, _ = with_visual.score_window(0, 3)
    assert flat_visual > flat_plain
    assert breakdown_visual["visual"] > 0
    _, _, labels = with_visual.score_window(exciting_idx, exciting_idx + 3)
    assert "Cambio de escena" in labels


def test_motion_channel_boosts_and_labels() -> None:
    phrases, audio, exciting_idx = _exciting_window()
    plain = MomentScorer(phrases, audio, 600)
    score_plain, _, _ = plain.score_window(exciting_idx, exciting_idx + 3)
    scene = np.zeros(600)
    motion = np.zeros(600)
    motion[300:320] = 0.9  # Pico de movimiento sostenido en el momento.
    with_motion = MomentScorer(
        phrases, audio, 600, scene_cut_timeline=scene, motion_timeline=motion
    )
    score_motion, breakdown_motion, labels = with_motion.score_window(
        exciting_idx, exciting_idx + 3
    )
    assert score_motion > score_plain
    assert breakdown_motion["visual"] > 0
    assert "Momento de mucha acción" in labels
    assert "Cambio de escena" not in labels


def test_combined_visual_fallback_label() -> None:
    phrases, audio, exciting_idx = _exciting_window()
    visual = np.zeros(600)
    visual[300:320] = 0.9
    scorer = MomentScorer(phrases, audio, 600, visual_timeline=visual)
    _, breakdown, labels = scorer.score_window(exciting_idx, exciting_idx + 3)
    assert breakdown["visual"] > 0
    # Sin componentes no se sabe qué lo disparó: etiqueta genérica.
    assert "Cambio visual brusco" in labels
    assert "Cambio de escena" not in labels
    assert "Momento de mucha acción" not in labels


def test_face_channel_boosts_window_with_face() -> None:
    phrases, audio, exciting_idx = _exciting_window()
    plain = MomentScorer(phrases, audio, 600)
    score_plain, breakdown_plain, _ = plain.score_window(exciting_idx, exciting_idx + 3)
    assert "rostro" not in breakdown_plain
    face = np.zeros(600)
    face[300:320] = 0.9  # El rostro aparece solo durante el momento.
    with_face = MomentScorer(phrases, audio, 600, face_timeline=face)
    score_face, breakdown_face, labels = with_face.score_window(exciting_idx, exciting_idx + 3)
    assert score_face > score_plain
    assert breakdown_face["rostro"] > 0
    assert "Rostro en cuadro" in labels


def test_absent_visual_channels_do_not_change_score() -> None:
    phrases, audio, exciting_idx = _exciting_window()
    baseline = MomentScorer(phrases, audio, 600)
    score_base, breakdown_base, labels_base = baseline.score_window(exciting_idx, exciting_idx + 3)
    explicit_none = MomentScorer(
        phrases,
        audio,
        600,
        visual_timeline=None,
        face_timeline=None,
        scene_cut_timeline=None,
        motion_timeline=None,
    )
    score_none, breakdown_none, labels_none = explicit_none.score_window(
        exciting_idx, exciting_idx + 3
    )
    assert score_none == score_base
    assert breakdown_none == breakdown_base
    assert labels_none == labels_base


def test_constant_visual_and_face_timelines_do_not_drag_score() -> None:
    # Un canal sin variación no aporta información y no debe arrastrar
    # la puntuación (lección v1.1.1): video estático sin cortes +
    # rostro siempre en cuadro puntúan EXACTO que sin canales.
    phrases, audio, exciting_idx = _exciting_window()
    baseline = MomentScorer(phrases, audio, 600)
    score_base, _, _ = baseline.score_window(exciting_idx, exciting_idx + 3)
    gated = MomentScorer(
        phrases,
        audio,
        600,
        visual_timeline=np.zeros(600),
        face_timeline=np.ones(600),
    )
    score_gated, breakdown_gated, _ = gated.score_window(exciting_idx, exciting_idx + 3)
    assert score_gated == score_base
    # En el desglose sí aparecen (medidos en 0.0), como el audio sin audio.
    assert breakdown_gated["visual"] == 0.0
    assert breakdown_gated["rostro"] == 0.0


def test_visual_channel_in_breakdown_only_when_present() -> None:
    phrases, audio, exciting_idx = _exciting_window()
    _, breakdown_plain, _ = MomentScorer(phrases, audio, 600).score_window(
        exciting_idx, exciting_idx + 3
    )
    assert "visual" not in breakdown_plain
    assert "rostro" not in breakdown_plain
    visual = np.zeros(600)
    visual[300:320] = 0.9
    _, breakdown_visual, _ = MomentScorer(phrases, audio, 600, visual_timeline=visual).score_window(
        exciting_idx, exciting_idx + 3
    )
    assert "visual" in breakdown_visual
    assert "rostro" not in breakdown_visual


# ---------------------------------------------------------------------------
# Cableado en detect_moments
# ---------------------------------------------------------------------------


def test_detect_moments_accepts_visual_timelines() -> None:
    transcript = _transcript(600, exciting_at=300)
    audio = _audio(600, peak_at=300)
    scene = np.zeros(600)
    scene[300:320] = 0.63
    motion = np.zeros(600)
    face = np.zeros(600)
    face[300:320] = 0.9
    clips = moments.detect_moments(
        transcript,
        ProjectConfig(),
        600,
        audio_timeline=audio,
        scene_cut_timeline=scene,
        motion_timeline=motion,
        face_timeline=face,
    )
    assert len(clips) == 1
    assert clips[0].signal_breakdown is not None
    assert clips[0].signal_breakdown["visual"] > 0
    assert clips[0].signal_breakdown["rostro"] > 0
    assert "Cambio de escena" in clips[0].razon_viral


def test_detect_moments_profile_callback() -> None:
    transcript = _transcript(600, exciting_at=300)
    audio = _audio(600, peak_at=300)
    seen: list[str] = []
    moments.detect_moments(
        transcript,
        ProjectConfig(),
        600,
        audio_timeline=audio,
        visual_stats=VisualStats(cut_rate_per_minute=0.5, mean_motion_raw=2.0),
        on_profile=seen.append,
    )
    assert seen == ["talk"]
    seen.clear()
    moments.detect_moments(
        transcript,
        ProjectConfig(),
        600,
        audio_timeline=audio,
        on_profile=seen.append,
    )
    assert seen == ["mixed"]


def test_detect_moments_cut_rate_fallback_from_scene_timeline() -> None:
    # Sin visual_stats pero con cortes inyectados, la tasa se estima de
    # los cortes: 3 cortes en 600 s = 0.3/min -> habla densa + calma = talk.
    transcript = _transcript(600, exciting_at=300)
    audio = _audio(600, peak_at=300)
    scene = np.zeros(600)
    scene[[100, 300, 500]] = 0.63
    seen: list[str] = []
    moments.detect_moments(
        transcript,
        ProjectConfig(),
        600,
        audio_timeline=audio,
        scene_cut_timeline=scene,
        on_profile=seen.append,
    )
    assert seen == ["talk"]


def test_detect_moments_extracts_visual_from_video(tmp_path: Path) -> None:
    # Video real de 60 s con cortes cada 10 s: la extracción interna
    # corre y el clip encontrado lleva el canal visual en el desglose.
    video = _cuts_video(
        tmp_path / "cuts60.mp4",
        [("red", 10), ("blue", 10), ("green", 10), ("black", 10), ("white", 10), ("red", 10)],
    )
    transcript = _transcript(60, exciting_at=25)
    audio = _audio(60, peak_at=25)
    clips = moments.detect_moments(
        transcript,
        ProjectConfig(signal_score_threshold=5.0),
        60,
        input_path=video,
        ffmpeg_path="ffmpeg",
        audio_timeline=audio,
    )
    assert len(clips) == 1
    assert clips[0].signal_breakdown is not None
    assert clips[0].signal_breakdown.get("visual", 0) > 0


def test_detect_moments_fallback_when_ffmpeg_missing(tmp_path: Path) -> None:
    # ffmpeg inexistente: la extracción de audio y la visual fallan y el
    # motor sigue con Tier 0 (solo texto). El bloque excitante se
    # encuentra igual (ver test_detect_moments_without_audio_uses_text_only).
    video = _static_video(tmp_path / "static.mp4", 10.0)
    transcript = _transcript(600, exciting_at=300)
    clips = moments.detect_moments(
        transcript,
        ProjectConfig(signal_score_threshold=5.0),
        600,
        input_path=video,
        ffmpeg_path="ffmpeg-inexistente",
    )
    assert len(clips) == 1
    assert clips[0].signal_breakdown is not None
    assert clips[0].signal_breakdown["audio"] == 0.0
    assert "visual" not in clips[0].signal_breakdown


def test_detect_moments_does_not_extract_when_timelines_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Con timelines inyectados, detect_moments NO debe llamar a la
    # extracción (el pipeline ya la hizo): si la llamara, este test cae.
    def fail_if_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("No debe extraer con timelines inyectados")

    monkeypatch.setattr(moments, "extract_visual_timelines", fail_if_called)
    transcript = _transcript(600, exciting_at=300)
    audio = _audio(600, peak_at=300)
    clips = moments.detect_moments(
        transcript,
        ProjectConfig(),
        600,
        input_path=Path("/tmp/no-existe.mp4"),
        ffmpeg_path="ffmpeg",
        audio_timeline=audio,
        visual_timeline=np.zeros(600),
        face_timeline=np.zeros(600),
    )
    assert len(clips) == 1


# ---------------------------------------------------------------------------
# Regresión / calibración (lección v1.1.1)
# ---------------------------------------------------------------------------


def test_boring_video_still_zero_clips_with_visual_channels() -> None:
    # Video plano + canales visuales planos (3 cortes en 10 min, sin
    # movimiento, rostro siempre en cuadro): CERO clips a 5.0. Lo
    # visual, por sí solo y con peso 0.5, no crea clips de la nada.
    transcript = _transcript(600)
    audio = _audio(600)
    scene = np.zeros(600)
    scene[[100, 300, 500]] = 0.63
    motion = np.zeros(600)
    face = np.ones(600)
    assert (
        moments.detect_moments(
            transcript,
            ProjectConfig(signal_score_threshold=5.0),
            600,
            audio_timeline=audio,
            scene_cut_timeline=scene,
            motion_timeline=motion,
            face_timeline=face,
        )
        == []
    )


def test_realistic_video_finds_clips_with_visual_channels() -> None:
    # Video conversacional realista con momentos claros: debe dar clips
    # con los canales visuales activos (no deben arrastrar ni inflar).
    transcript = _realistic_transcript(600)
    audio = _realistic_audio(600)
    clips_plain = moments.detect_moments(
        transcript,
        ProjectConfig(signal_score_threshold=5.0),
        600,
        audio_timeline=audio,
    )
    assert len(clips_plain) >= 2
    # Visuales realistas: algún corte, picos de movimiento en los
    # momentos de audio, rostro presente en ~2/3 del video (varía).
    scene = np.zeros(600)
    scene[[95, 215, 355, 505]] = 0.63
    motion = np.zeros(600)
    for at in (90, 210, 350, 500):
        motion[at : at + 8] = 0.8
    face = np.zeros(600)
    face[0:400] = 0.9
    clips_visual = moments.detect_moments(
        transcript,
        ProjectConfig(signal_score_threshold=5.0),
        600,
        audio_timeline=audio,
        scene_cut_timeline=scene,
        motion_timeline=motion,
        face_timeline=face,
    )
    assert len(clips_visual) >= 2
    assert all(clip.puntuacion >= 5.0 for clip in clips_visual)
