from __future__ import annotations

import ipaddress
from datetime import UTC, datetime
from typing import Any, Literal
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
# v1.1.1: 5.0 (antes 8.0). Con la calibración original, 8.0 era
# inalcanzable para contenido conversacional realista (Kelly reportó cero
# clips ni con el umbral en 5.0); 5.0 es "buen momento normal" con las
# curvas recalibradas. El usuario puede subirlo hasta 9.5 en la UI.
DEFAULT_SIGNAL_SCORE_THRESHOLD = 5.0


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
    # Quitar muletillas y micro-pausas (Fase 3, ítem 8): resta los
    # intervalos detectados por fillers.py, como la eliminación de
    # silencios. Apagado por defecto: recorta palabras reales.
    remove_fillers: bool = False
    # Zoom punch-in en picos de energía del audio (Fase 3, ítem 3b).
    punch_in: bool = False
    # Split dinámico en entrevistas (Fase 3, ítem 3b): con diarización
    # disponible, un clip con dos hablantes que alternan se renderiza
    # en pantalla dividida aunque el encuadre sea "speaker".
    dynamic_split: bool = False
    # Aplicar la plantilla de marca del usuario (Fase 3, ítem 5):
    # colores/fuente de marca en los subtítulos y logo superpuesto.
    brand: bool = False


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
    # Detección rápida (v1.3.1): solo Tier 0 (transcripción + timeline de
    # audio + señales de texto), sin YAMNet, diarización, visual ni
    # semántica. La fase de señales pasa de minutos a segundos; el motor
    # completo sigue siendo el predeterminado. Solo aplica a los motores
    # de señales; el modo "llm" lo ignora.
    fast_detection: bool = False
    # Edición automática (v1.3.2): al terminar la detección, renderizar
    # todos los clips con ffmpeg (fase de edición). APAGADA por defecto:
    # el Studio edita el video fuente (trim + cues de la transcripción),
    # no el clip renderizado, así que renderizar antes es trabajo y
    # tiempo perdido. Con el ajuste apagado, el pipeline termina en las
    # propuestas detectadas (metadata["clip_proposals"]) con estado
    # completed y sin encodes; el usuario abre cada propuesta en el
    # Studio desde la página de resultados.
    auto_edit: bool = False
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


class StudioEditRequest(BaseModel):
    """Especificación de edición del editor visual (ShortStudio, v1.3.0).

    El contenido es opaco para el backend: el editor lo guarda y lo
    relee tal cual en ``clip["studio_edit"]``. Solo se exige que sea un
    objeto JSON (no una lista, cadena o null).
    """

    model_config = ConfigDict(extra="forbid")

    studio_edit: dict[str, Any]


class FaceTrackKeyframe(BaseModel):
    """Un keyframe de la pista de seguimiento facial (Studio, v1.3.1).

    ``time`` es el tiempo absoluto en segundos del video fuente.
    ``x`` e ``y`` son la posición de la ventana de recorte como
    FRACCIÓN del desplazamiento máximo en cada eje (0 = un extremo,
    1 = el otro, 0.5 = centrado o sin desplazamiento posible en ese
    eje); el frontend las convierte a píxeles según su zoom.
    """

    model_config = ConfigDict(extra="forbid")

    time: float = Field(ge=0)
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class FaceTrackResponse(BaseModel):
    """Respuesta de GET /api/projects/{id}/face-track (Studio, v1.3.1).

    La pista cubre el video fuente completo. Sin rostro en todo el
    video, ``has_face`` es False y ``keyframes`` va vacío (no es un
    error). Como mucho 48 keyframes (tope de la simplificación del
    seguimiento facial).
    """

    model_config = ConfigDict(extra="forbid")

    project_id: str
    ratio: Literal["9:16", "1:1", "16:9"]
    duration: float = Field(gt=0)
    sample_fps: float = Field(gt=0)
    has_face: bool
    keyframes: list[FaceTrackKeyframe] = Field(default_factory=list)


class FaceTrackPendingResponse(BaseModel):
    """Respuesta 202 de GET /api/projects/{id}/face-track (Studio, v1.3.2).

    El análisis del video entero tarda minutos en videos largos, así
    que corre en segundo plano y el frontend sondea el mismo endpoint
    hasta recibir 200 (FaceTrackResponse) o 503 (fallo, con reintento
    en el siguiente GET). ``progress`` es el porcentaje del análisis
    (0–100).
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["pending"] = "pending"
    progress: float = Field(ge=0, le=100)


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
    # Cuenta concreta; None = la predeterminada (Fase 3, ítem 10).
    account_id: str | None = Field(default=None, max_length=256)

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


# Plantilla de marca (Fase 3, ítem 5): un kit por usuario. El logo se
# sube aparte (POST /api/brand-kit/logo) y aquí solo se guarda su
# nombre de archivo; ``logo_url`` lo calcula el endpoint GET.
class BrandKit(BaseModel):
    name: str = Field(default="", max_length=60)
    primary_color: str = Field(default="#ffffff", pattern=r"^#[0-9a-fA-F]{6}$")
    accent_color: str = Field(default="#facc15", pattern=r"^#[0-9a-fA-F]{6}$")
    font_family: (
        Literal["Arial", "Arial Black", "Georgia", "Tahoma", "Verdana"] | None
    ) = None
    logo_filename: str | None = None


class BrandKitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="", max_length=60)
    primary_color: str = Field(default="#ffffff", pattern=r"^#[0-9a-fA-F]{6}$")
    accent_color: str = Field(default="#facc15", pattern=r"^#[0-9a-fA-F]{6}$")
    font_family: (
        Literal["Arial", "Arial Black", "Georgia", "Tahoma", "Verdana"] | None
    ) = None


class TranslateSubtitlesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_language: Literal["en", "fr", "de", "it", "pt", "es"]
    format_name: Literal["srt", "vtt"] = "srt"


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
    # Tema y serie (Fase 3, ítem 2): series.py agrupa las propuestas
    # por tema; los grupos de 2+ clips forman una serie numerada
    # ("Parte 1", "Parte 2"…) en orden cronológico.
    topic: str | None = None
    series: str | None = None
    part_number: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_order(self) -> ClipProposal:
        if self.fin <= self.inicio:
            raise ValueError("El fin del clip debe ser posterior al inicio.")
        return self


# ---------------------------------------------------------------------------
# Publicación y flujo (Fase 3, ítems 9–15)
# ---------------------------------------------------------------------------


def _validate_target_visibility(
    platform: PublishingPlatform,
    visibility: str,
) -> None:
    if platform == "instagram" and visibility != "public":
        raise ValueError("Instagram solo admite la visibilidad pública.")
    if platform == "tiktok" and visibility == "unlisted":
        raise ValueError("TikTok no admite la visibilidad no listada.")


class CrossPublishTarget(BaseModel):
    """Un destino de publicación cruzada: plataforma + cuenta + hora."""

    model_config = ConfigDict(extra="forbid")

    platform: PublishingPlatform
    # Cuenta concreta; None = la predeterminada de esa plataforma.
    account_id: str | None = Field(default=None, max_length=256)
    # Hora programada con zona horaria; None = publicar cuanto antes.
    scheduled_at: datetime | None = None
    visibility: Literal["private", "public", "unlisted"] = "private"

    @field_validator("scheduled_at")
    @classmethod
    def normalize_scheduled_at(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("La fecha programada debe incluir zona horaria.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_visibility(self) -> CrossPublishTarget:
        _validate_target_visibility(self.platform, self.visibility)
        return self


class CrossPublishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_ids: list[str] = Field(min_length=1, max_length=10)
    targets: list[CrossPublishTarget] = Field(min_length=1, max_length=9)

    @model_validator(mode="after")
    def validate_targets(self) -> CrossPublishRequest:
        if len(self.clip_ids) != len(set(self.clip_ids)):
            raise ValueError("La selección contiene clips duplicados.")
        seen: set[tuple[str, str | None]] = set()
        for target in self.targets:
            key = (target.platform, target.account_id)
            if key in seen:
                raise ValueError(
                    "Hay destinos duplicados para la misma plataforma y cuenta."
                )
            seen.add(key)
        return self


class RescheduleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scheduled_at: datetime

    @field_validator("scheduled_at")
    @classmethod
    def require_timezone_and_normalize(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("La fecha programada debe incluir zona horaria.")
        return value.astimezone(UTC)


class BatchEnqueueRequest(BaseModel):
    """Encolar varios videos: staged listos + enlaces a descargar."""

    model_config = ConfigDict(extra="forbid")

    staged_ids: list[str] = Field(default_factory=list, max_length=20)
    urls: list[str] = Field(default_factory=list, max_length=20)
    config: ProjectConfig | None = None

    @model_validator(mode="after")
    def validate_not_empty(self) -> BatchEnqueueRequest:
        if not self.staged_ids and not self.urls:
            raise ValueError("El lote necesita al menos un video o un enlace.")
        if len(self.staged_ids) + len(self.urls) > 20:
            raise ValueError("Un lote admite como máximo 20 videos.")
        return self


class WatchFolderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=1024)
    enabled: bool = True
    config: ProjectConfig | None = None
    poll_seconds: int = Field(default=300, ge=60, le=86400)


class WatchFolderUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    config: ProjectConfig | None = None
    poll_seconds: int | None = Field(default=None, ge=60, le=86400)


class NightWindowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    start: str = Field(default="22:00", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    end: str = Field(default="06:00", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")


class CaptionGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # None = todos los clips listos del proyecto.
    clip_ids: list[str] | None = Field(default=None, max_length=10)


class CaptionApplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=100)
    description: str | None = Field(default=None, max_length=4000)
    hashtags: dict[PublishingPlatform, list[str]] | None = None


class BrollSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["pexels", "pixabay"] = "pexels"
    query: str = Field(min_length=1, max_length=100)
    per_page: int = Field(default=6, ge=1, le=12)


class BrollImportRequest(BaseModel):
    """Importar un resultado del proveedor a la biblioteca local."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["pexels", "pixabay"]
    external_id: str = Field(default="", max_length=100)
    title: str = Field(min_length=1, max_length=200)
    query: str = Field(default="", max_length=100)
    creator: str = Field(default="", max_length=200)
    attribution_url: str = Field(default="", max_length=1000)
    preview_url: str = Field(default="", max_length=1000)
    download_url: str = Field(min_length=1, max_length=2000)
    duration: float = Field(default=0, ge=0)
    width: int = Field(default=0, ge=0)
    height: int = Field(default=0, ge=0)


class BrollUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=200)
    favorite: bool | None = None


class BrollAssignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    broll_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    clip_id: str = Field(min_length=1)
    # start=end=0 → B-roll de todo el clip; si no, segmento en segundos.
    start: float = Field(default=0, ge=0)
    end: float = Field(default=0, ge=0)
    label: str = Field(default="", max_length=200)

    @model_validator(mode="after")
    def validate_range(self) -> BrollAssignRequest:
        if (self.start == 0) != (self.end == 0):
            raise ValueError(
                "El segmento necesita inicio y fin, o ambos en cero para todo el clip."
            )
        if self.end and self.end <= self.start:
            raise ValueError("El fin del segmento debe ser posterior al inicio.")
        return self


class SavedMomentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=1000)
    project_id: str | None = Field(default=None, max_length=64)
    clip_id: str | None = Field(default=None, max_length=64)
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, ge=0)
    excerpt: str = Field(default="", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=10)


class SavedMomentUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    excerpt: str | None = Field(default=None, max_length=2000)
    tags: list[str] | None = Field(default=None, max_length=10)


class ApplySavedMomentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(min_length=1)


class DuplicateProjectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=200)
    ratio: str | None = None
    duration: str | None = None
    subtitle_template: str | None = None
    # True = encolar el duplicado para procesarlo de inmediato.
    process: bool = False


# --- Escritorio y distribución (Fase 3, ítems 17–20) -----------------------


class UpdateChannelRequest(BaseModel):
    """Canal de actualizaciones (C.6.40): "stable" solo acepta
    releases estables; "beta" también acepta prereleases."""

    model_config = ConfigDict(extra="forbid")

    channel: Literal["stable", "beta"]


class AutostartRequest(BaseModel):
    """Auto-inicio con Windows (C.6.39)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool


class DesktopPreferencesRequest(BaseModel):
    """Preferencias de la bandeja del sistema (C.6.38). Los campos
    ausentes (None) no cambian: la actualización es parcial."""

    model_config = ConfigDict(extra="forbid")

    minimize_to_tray: bool | None = None
    start_minimized: bool | None = None


class SetupCompleteRequest(BaseModel):
    """Cierre del asistente de primer arranque (C.6.42)."""

    model_config = ConfigDict(extra="forbid")

    whisper_model: WhisperModel | None = None
    skipped: bool = False
