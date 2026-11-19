from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from backend import media
from backend.schemas import ProjectConfig


def test_whisper_download_forwards_real_progress(monkeypatch, tmp_path: Path) -> None:
    import huggingface_hub

    def snapshot_download(repo_id: str, **kwargs: object) -> str:
        assert repo_id == "Systran/faster-whisper-small"
        progress_class = kwargs["tqdm_class"]
        progress = progress_class(total=100, desc="model.bin")
        progress.update(40)
        progress.update(60)
        progress.close()
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    updates: list[float] = []

    model_path = media._download_whisper_model("small", updates.append)

    assert model_path == str(tmp_path)
    assert updates[0] == 0
    assert updates[-1] == 95


def test_whisper_batched_pipeline_is_cached_per_selected_model(monkeypatch) -> None:
    created_models: list[tuple[str, str, str, int]] = []
    downloaded: list[str] = []

    class Model:
        def __init__(
            self,
            path: str,
            device: str,
            compute_type: str,
            num_workers: int,
        ) -> None:
            created_models.append((path, device, compute_type, num_workers))

    class BatchedPipeline:
        def __init__(self, model: Model) -> None:
            self.model = model

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(
            WhisperModel=Model,
            BatchedInferencePipeline=BatchedPipeline,
        ),
    )
    monkeypatch.setattr(media, "_WHISPER_PIPELINE_CACHE", None)
    def download(name: str, progress) -> str:
        assert name == "small"
        assert progress is not None
        downloaded.append(name)
        return "local-model-path"

    monkeypatch.setattr(media, "_download_whisper_model", download)

    progress: list[float] = []
    first = media._get_whisper_pipeline("small", progress.append)
    cache_miss_progress: list[float] = []
    second = media._get_whisper_pipeline("small", cache_miss_progress.append)

    assert first is second
    assert downloaded == ["small"]
    assert created_models == [("local-model-path", "cpu", "int8", 2)]
    assert progress[-1] == 100
    assert cache_miss_progress == []


def test_transcribe_uses_batch_pipeline_with_fixed_language(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    word = SimpleNamespace(start=0.1, end=0.5, word=" Hola")
    segment = SimpleNamespace(start=0, end=1, text="Hola", words=[word])
    info = SimpleNamespace(duration=1, language="es")

    class BatchedPipeline:
        def transcribe(self, path: str, **kwargs: object):
            assert path == str(tmp_path / "source.mp4")
            calls.append(kwargs)
            return iter([segment]), info

    def get_pipeline(model_name: str, download_progress):
        assert model_name == "small"
        assert download_progress is None
        return BatchedPipeline()

    monkeypatch.setattr(
        media,
        "_get_whisper_pipeline",
        get_pipeline,
    )
    progress: list[float] = []

    transcript, language = media.transcribe(
        tmp_path / "source.mp4",
        "small",
        progress.append,
        language="es",
    )

    assert calls[0]["language"] == "es"
    assert calls[0]["word_timestamps"] is True
    assert calls[0]["without_timestamps"] is False
    assert calls[0]["batch_size"] == 4
    assert transcript[0].words[0].text == " Hola"
    assert language == "es"
    assert progress[-1] == 100


def test_project_whisper_language_defaults_to_detection_and_accepts_explicit_code() -> None:
    assert ProjectConfig().whisper_language == "auto"
    assert ProjectConfig.model_validate(
        {"whisper_language": "es"}
    ).whisper_language == "es"
