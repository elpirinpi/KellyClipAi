from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Ratio = Literal["9:16", "16:9", "1:1", "4:5"]
DurationChoice = Literal["Cualquiera", "<30s", "30–60s", "60–90s", "90s–3min"]
PublishingPlatform = Literal["youtube", "tiktok", "instagram"]


class ScheduledPublishing(BaseModel):
    scheduled_at: datetime
    platform: PublishingPlatform
    visibility: Literal["private", "public", "unlisted"] = "private"

    @field_validator("scheduled_at")
    @classmethod
    def require_timezone_and_normalize(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("La fecha programada debe incluir zona horaria.")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_visibility(self) -> ScheduledPublishing:
        if self.platform == "instagram" and self.visibility != "public":
            raise ValueError("Instagram solo admite la visibilidad pública.")
        if self.platform == "tiktok" and self.visibility == "unlisted":
            raise ValueError("TikTok no admite la visibilidad no listada.")
        return self


class EditOptions(BaseModel):
    emojis: bool = True
    keywords: bool = True
    broll: bool = False
    silence: bool = False
    censor: bool = False
    face_tracking: bool = True


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ratio: Ratio = "9:16"
    duration: DurationChoice = "Cualquiera"
    subtitle_template: str = Field(default="Default", min_length=1, max_length=100)
    options: EditOptions = Field(default_factory=EditOptions)
    moment: str = Field(default="", max_length=500)
    whisper_model: Literal["tiny", "small", "medium", "large-v3"] = "small"
    schedule: ScheduledPublishing | None = None


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["Groq", "OpenAI", "Anthropic", "OpenRouter", "Servidor local compatible"] | None = None
    api_key: str | None = Field(default=None, max_length=4096)
    model: str | None = Field(default=None, max_length=200)
    base_url: str | None = Field(default=None, max_length=500)


class LLMSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal[
        "Groq",
        "OpenAI",
        "Anthropic",
        "OpenRouter",
        "Servidor local compatible",
    ]
    api_key: str = Field(default="", max_length=4096)
    model: str = Field(min_length=1, max_length=200)
    base_url: str = Field(default="", max_length=500)


class ProcessRequest(BaseModel):
    config: ProjectConfig | None = None


class ZoomImportRequest(BaseModel):
    meeting_id: str = Field(min_length=1, max_length=256)
    file_id: str = Field(min_length=1, max_length=100)
    config: ProjectConfig


class StagedUploadRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    file_size: int = Field(gt=0)
    content_type: str = Field(default="application/octet-stream", max_length=100)
    duration: float = Field(default=0, ge=0)
    width: int = Field(default=0, ge=0)
    height: int = Field(default=0, ge=0)
    kind: Literal["video", "audio"] = "video"


class StagedDownloadRequest(BaseModel):
    url: str = Field(min_length=1, max_length=4096)
    config: ProjectConfig


class StagedProjectRequest(BaseModel):
    config: ProjectConfig | None = None


class EditablePhrase(BaseModel):
    id: str = Field(min_length=1, max_length=100)
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    preview_start: float = Field(ge=0, default=0)
    text: str = Field(max_length=400)
    removed: bool = False

    @model_validator(mode="after")
    def validate_phrase(self) -> EditablePhrase:
        if self.end <= self.start:
            raise ValueError("El fin de la frase debe ser posterior al inicio.")
        if not self.removed and not self.text.strip():
            raise ValueError("La frase debe tener texto o marcarse para eliminar.")
        return self


class ClipEditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ratio: Ratio
    subtitle_template: str = Field(min_length=1, max_length=100)
    phrases: list[EditablePhrase] = Field(max_length=120)

    @model_validator(mode="after")
    def validate_phrases(self) -> ClipEditRequest:
        ids = [phrase.id for phrase in self.phrases]
        if len(ids) != len(set(ids)):
            raise ValueError("Las frases del editor deben tener identificadores únicos.")
        previous_end = -1.0
        for phrase in sorted(self.phrases, key=lambda item: item.start):
            if phrase.start < previous_end - 0.02:
                raise ValueError("Las frases del editor no pueden solaparse.")
            previous_end = phrase.end
        return self


class ClipArchiveRequest(BaseModel):
    clip_ids: list[str] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def validate_clip_ids(self) -> ClipArchiveRequest:
        if len(self.clip_ids) != len(set(self.clip_ids)):
            raise ValueError("La selección contiene clips duplicados.")
        return self


class SaveProjectFilesRequest(BaseModel):
    destination: str = Field(min_length=1, max_length=32768)
    clip_ids: list[str] = Field(default_factory=list, max_length=10)
    include_original: bool = False

    @model_validator(mode="after")
    def validate_files(self) -> SaveProjectFilesRequest:
        if not self.include_original and not self.clip_ids:
            raise ValueError("Selecciona el video original o al menos un clip.")
        if len(self.clip_ids) != len(set(self.clip_ids)):
            raise ValueError("La selección contiene clips duplicados.")
        return self


class PublishClipsRequest(ClipArchiveRequest):
    platform: PublishingPlatform
    visibility: Literal["private", "public", "unlisted"] = "private"

    @model_validator(mode="after")
    def validate_visibility(self) -> PublishClipsRequest:
        if self.platform == "instagram" and self.visibility != "public":
            raise ValueError("Instagram solo admite la visibilidad pública.")
        if self.platform == "tiktok" and self.visibility == "unlisted":
            raise ValueError("TikTok no admite la visibilidad no listada.")
        return self


class ClipResolutionRequest(BaseModel):
    resolution: Literal[720, 1080]


class CustomTemplateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    base_template: Literal[
        "Default",
        "Modern",
        "Bouncy",
        "MrBeast",
        "Business",
        "Esencial",
        "Editorial",
        "Neón",
    ]

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_name(self) -> CustomTemplateRequest:
        reserved_names = {
            "default",
            "modern",
            "bouncy",
            "mrbeast",
            "business",
            "esencial",
            "editorial",
            "neón",
        }
        if self.name.casefold() in reserved_names:
            raise ValueError("El nombre debe ser distinto al de una plantilla integrada.")
        return self


class Word(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str

    @model_validator(mode="after")
    def validate_order(self) -> Word:
        if self.end <= self.start:
            raise ValueError("El fin de una palabra debe ser posterior al inicio.")
        return self


class TranscriptSegment(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str
    words: list[Word] = Field(default_factory=list)


class ClipProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inicio: float = Field(ge=0)
    fin: float = Field(gt=0)
    titulo: str = Field(min_length=1, max_length=160)
    puntuacion: float = Field(ge=0, le=10)
    razon_viral: str = Field(min_length=1, max_length=1400)
    palabras_clave: list[str] = Field(default_factory=list, max_length=12)
    emojis: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_order(self) -> ClipProposal:
        if self.fin <= self.inicio:
            raise ValueError("El fin del clip debe ser posterior al inicio.")
        return self
