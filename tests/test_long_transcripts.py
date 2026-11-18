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

    assert [duration for duration, _window in calls] == [900, 580]
    assert [segment.start for segment in calls[1][1]] == [30, 490]
    assert [(proposal.inicio, proposal.fin) for proposal in proposals] == [
        (60, 80),
        (780, 800),
    ]
