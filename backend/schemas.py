from __future__ import annotations

import ipaddress
from datetime import UTC, datetime
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Límites de entrada (Fase 1, A2): un video mayor que estos límites genera
# decenas de ventanas LLM y horas de transcripción en CPU. Otro módulo
# (main.py) aplica estos límites al subir y al procesar.
MAX_UPLOAD_BYTES = 4294967296  # 4 GiB
MAX_VIDEO_DURATION_SECONDS = 14400  # 4 h

# Paginación de GET /api/projects (Fase 2, B-R10): 20 proyectos por página
# por defecto (la vista "Mis proyectos" muestra una cuadrícula), con un
# tope de 100 para que ninguna petición devuelva megabytes de clips.
DEFAULT_PROJECTS_PAGE_SIZE = 20
MAX_PROJECTS_PAGE_SIZE = 100

Ratio = Literal["9:16", "16:9", "1:1", "4:5"]
WhisperModel = Literal["tiny", "small", "medium", "large-v3"]
DurationChoice = Literal["Cualquiera", "<30s", "30–60s", "60–90s", "90s–3min"]
PublishingPlatform = Literal["youtube", "tiktok", "instagram"]
# Motor de detección de momentos (Fase A, v1.1.0): "signals" es el motor
# por defecto (señales locales, sin API ni clave); "signals+ai" elige por
# señales y pule títulos/descripciones con una llamada corta al LLM (si
# hay clave configurada; si no, funciona igual con títulos por plantilla);
# "llm" es el camino anterior (selección 100 % por LLM, por ventanas).
MomentEngine = Literal["signals", "signals+ai", "llm"]
# Umbral por defecto de la puntuación absoluta 0–10 del motor de señales:
# por debajo no hay clip ("cero clips" es un resultado válido). Valor
# inicial del informe (patrón Chopify); la calibración con videos reales
# es la Fase E.
DEFAULT_SIGNAL_SCORE_THRESHOLD = 8.0


class ScheduledPublishing(BaseModel):
    scheduled_at: datetime
    platform: PublishingPlatform
    visibility: Literal["private", "public", "unlisted"] = "private"

    @field_validator("scheduled_at")
    @classmethod
    def require_timezone_and_normalize(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("La fecha programada debe incluir zona horaria.")
        return value.astimezone(UTC)

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


class SubtitleStyle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    font_family: (
        Literal["Arial", "Arial Black", "Georgia", "Tahoma", "Verdana"] | None
    ) = None
    font_size: int | None = Field(default=None, ge=16, le=100)
    font_color: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")
    highlight_color: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")
    position: Literal["top", "center", "bottom"] | None = None


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ratio: Ratio = "9:16"
    duration: DurationChoice = "Cualquiera"
    framing: Literal["speaker", "split", "blur", "center"] = "speaker"
    whisper_language: Literal[
        "auto",
        "ar",
        "de",
        "en",
        "es",
        "fr",
        "hi",
        "it",
        "ja",
        "ko",
        "nl",
        "pl",
        "pt",
        "ru",
        "tr",
        "uk",
        "zh",
    ] = "auto"
    subtitle_template: str = Field(default="Default", min_length=1, max_length=100)
    subtitle_style: SubtitleStyle = Field(default_factory=SubtitleStyle)
    options: EditOptions = Field(default_factory=EditOptions)
    moment: str = Field(default="", max_length=500)
    whisper_model: WhisperModel = "small"
    moment_engine: MomentEngine = "signals"
    # Umbral de puntuación del motor de señales (0–10). Solo aplica a los
    # modos "signals" y "signals+ai"; el modo "llm" lo ignora.
    signal_score_threshold: float = Field(
        default=DEFAULT_SIGNAL_SCORE_THRESHOLD, ge=0, le=10
    )
    schedule: ScheduledPublishing | None = None


_BLOCKED_LLM_HOSTS = {"metadata.google.internal"}
_LOOPBACK_HOSTNAMES = {"localhost"}


def _validate_llm_base_url(value: str | None) -> str | None:
    """Valida la URL base del proveedor LLM contra SSRF y robo de la clave.

    Reglas (Fase 1, A12/B-S1):
    - None o vacío: permitido, significa "usar el default del proveedor".
    - https: permitido salvo hosts bloqueados.
    - http: solo loopback (localhost, 127.0.0.0/8, ::1); en claro la clave
      viajaría sin cifrar.
    - Si el host se parsea como IP, se rechazan las privadas, link-local
      (169.254.0.0/16, incluido 169.254.169.254, endpoint de metadatos en la
      nube), reservadas, multicast y no especificadas.
    - Se rechaza el hostname conocido de metadatos metadata.google.internal.
    - Un nombre DNS que no es IP se permite: no se puede resolver de forma
      fiable en validación (el DNS puede cambiar entre la validación y la
      petición, y resolver aquí añadiría latencia y fallos de red al guardar
      ajustes). La protección real contra DNS rebinding queda en la capa de
      red; aquí solo se bloquean los literales IP peligrosos y el hostname
      de metadatos conocido.
    """
    if value is None or not value.strip():
        return value
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(
            "La URL base del proveedor debe empezar por https:// "
            "(http:// solo se admite en localhost)."
        )
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise ValueError("La URL base del proveedor no incluye un host válido.")
    if host in _BLOCKED_LLM_HOSTS:
        raise ValueError(
            "La URL base del proveedor apunta a un host de metadatos bloqueado."
        )
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None:
        if address.is_loopback:
            return value
        if (
            address.is_private
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            raise ValueError(
                "La URL base del proveedor apunta a una dirección IP interna "
                "o reservada, no permitida."
            )
        if parsed.scheme != "https":
            raise ValueError(
                "La URL base del proveedor debe usar https:// fuera de localhost."
            )
        return value
    if host in _LOOPBACK_HOSTNAMES:
        return value
    if parsed.scheme != "https":
        raise ValueError(
            "La URL base del proveedor debe usar https:// fuera de localhost."
        )
    return value


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["Groq", "OpenAI", "Anthropic", "OpenRouter", "Servidor local compatible"] | None = None
    api_key: str | None = Field(default=None, max_length=4096)
    model: str | None = Field(default=None, max_length=200)
    base_url: str | None = Field(default=None, max_length=500)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str | None) -> str | None:
        return _validate_llm_base_url(value)


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
    # Modelo Whisper elegido en Ajustes. None = sin cambio (el cliente no lo
    # envió); el endpoint PUT lo persiste y el GET lo devuelve.
    whisper_model: WhisperModel | None = None

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        return _validate_llm_base_url(value) or ""


class ProcessRequest(BaseModel):
    config: ProjectConfig | None = None


class ZoomImportRequest(BaseModel):
    meeting_id: str = Field(min_length=1, max_length=256)
    file_id: str = Field(min_length=1, max_length=100)
    config: ProjectConfig


class StagedUploadRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    file_size: int = Field(gt=0, le=MAX_UPLOAD_BYTES)
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
    subtitle_style: SubtitleStyle = Field(default_factory=SubtitleStyle)
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


class OverdueScheduleResolution(BaseModel):
    action: Literal["publish_now", "discard"]


class ClipResolutionRequest(BaseModel):
    resolution: Literal[720, 1080]


class EditorRenderJobResponse(BaseModel):
    """Respuesta de los endpoints de trabajos del editor (Fase 2, B-P8).

    `result` trae la misma carga que devolvía el endpoint síncrono
    (preview_url/duration/width/height/… o clip_id/duration/…) una vez
    el trabajo termina; mientras tanto es None y el frontend consulta el
    progreso por polling o por SSE.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    job_id: str
    project_id: str
    clip_id: str
    kind: Literal["preview", "export", "quality", "improve"]
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    progress: float = Field(ge=0, le=100)
    message: str
    result: dict | None = None
    error: str | None = None
    created_at: str
    updated_at: str


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
    descripcion: str = Field(default="", max_length=4000)
    hashtags: dict[PublishingPlatform, list[str]] = Field(default_factory=dict)
    palabras_clave: list[str] = Field(default_factory=list, max_length=12)
    emojis: list[str] = Field(default_factory=list, max_length=8)
    # Desglose de señales del motor local (Fase A): puntuación 0–10 de
    # cada canal ("audio", "texto", "ritmo", "gancho_inicial" y, desde la
    # Fase B, "eventos") más "completitud" (multiplicador 0–1). None en
    # las propuestas del LLM. La UI lo usa para el "por qué este clip".
    signal_breakdown: dict[str, float] | None = None

    @model_validator(mode="after")
    def validate_order(self) -> ClipProposal:
        if self.fin <= self.inicio:
            raise ValueError("El fin del clip debe ser posterior al inicio.")
        return self
