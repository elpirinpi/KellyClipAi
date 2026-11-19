from __future__ import annotations

from backend import pipeline
from backend.llm import ResolvedLLM
from backend.schemas import ClipProposal, ProjectConfig, TranscriptSegment, Word


def test_long_transcripts_use_overlapping_windows_and_restore_absolute_times(
    monkeypatch,
) -> None:
    transcript = [
        TranscriptSegment(
            start=750,
            end=760,
            text="Momento en la zona de solape",
            words=[Word(start=750, end=752, text="Momento")],
        ),
        TranscriptSegment(
            start=1210,
            end=1220,
            text="Momento en la segunda ventana",
            words=[Word(start=1210, end=1212, text="Momento")],
        ),
    ]
    calls: list[tuple[float, list[TranscriptSegment]]] = []

    def propose(
        window_transcript: list[TranscriptSegment],
        _project: ProjectConfig,
        window_duration: float,
        _llm: ResolvedLLM,
        validation_feedback: list[str] | None = None,
    ) -> list[ClipProposal]:
        assert validation_feedback is None
        calls.append((window_duration, window_transcript))
        return [
            ClipProposal(
                inicio=60,
                fin=80,
                titulo=f"Ventana {len(calls)}",
                puntuacion=8,
                razon_viral="Una idea completa.",
            )
        ]

    monkeypatch.setattr(pipeline, "propose_clips", propose)

    proposals = pipeline._propose_valid_clips(
        transcript,
        ProjectConfig(),
        1300,
        ResolvedLLM("OpenAI", "test-token", "test-model", "https://llm.example"),
    )

    # Ventanas de 5 minutos con 1 minuto de solape (stride de 4 minutos).
    assert [duration for duration, _window in calls] == [300, 300, 300, 300, 300, 100]
    # El segmento de 750s aparece en la ventana 3 (480-780) y en la 4 (720-1020).
    assert [segment.start for segment in calls[2][1]] == [270]
    assert [segment.start for segment in calls[3][1]] == [30]
    # El segmento de 1210s aparece en la ventana 5 (960-1260) y en la 6 (1200-1300).
    assert [segment.start for segment in calls[4][1]] == [250]
    assert [segment.start for segment in calls[5][1]] == [10]
    assert [(proposal.inicio, proposal.fin) for proposal in proposals] == [
        (60, 80),
        (300, 320),
        (540, 560),
        (780, 800),
        (1020, 1040),
        (1260, 1280),
    ]
