import type { StudioEditSpec } from './studio/types/editor';

export type UploadInfo = {
  id: string;
  name: string;
  size: number;
  duration: number;
  width: number;
  height: number;
  kind: 'video' | 'audio';
};

export type StagedMedia = {
  id: string;
  source_type: 'upload' | 'url';
  status: 'receiving' | 'downloading' | 'ready' | 'failed' | 'finalizing' | 'completed';
  filename: string;
  title: string | null;
  file_size: number;
  duration: number;
  width: number;
  height: number;
  kind: 'video' | 'audio';
  bytes_received: number;
  total_bytes: number | null;
  speed: number | null;
  eta: number | null;
  error: string | null;
  project_id: string | null;
};

export type ClipWord = { start: number; end: number; preview_start: number; preview_end: number; text: string };

export type EditablePhrase = {
  id: string;
  start: number;
  end: number;
  preview_start: number;
  text: string;
  removed: boolean;
};

export type SubtitleStyleState = {
  font_family: string | null;
  font_size: number | null;
  font_color: string | null;
  highlight_color: string | null;
  position: 'top' | 'center' | 'bottom' | null;
};

export const DEFAULT_SUBTITLE_STYLE: SubtitleStyleState = {
  font_family: null,
  font_size: null,
  font_color: null,
  highlight_color: null,
  position: null,
};

export type ClipEditorState = {
  ratio: string;
  subtitle_template: string;
  subtitle_style: SubtitleStyleState;
  phrases: EditablePhrase[];
};

export type ClipResult = {
  id: string;
  status?: 'ready' | 'failed';
  error?: string;
  title: string;
  start: number;
  end: number;
  duration: number;
  score: number;
  viral_reason: string;
  keywords: string[];
  emojis: string[];
  // Desglose "por qué este clip" del motor de señales (canal → 0–10);
  // ausente en los clips elegidos por el LLM.
  signal_breakdown?: Record<string, number>;
  width: number;
  height: number;
  resolution?: 720 | 1080;
  audio_enhanced?: boolean;
  filename: string;
  video_url: string | null;
  download_url: string | null;
  // Especificación guardada del editor visual (Studio, v1.3.0): ajustes
  // y cues de subtítulos; ausente en los clips sin edición visual.
  studio_edit?: StudioEditSpec | null;
  broll_provider?: string;
  broll_creator?: string;
  broll_attribution_url?: string;
  // Título/descripción/hashtags editados antes de publicar (Fase 3,
  // ítem 12): el backend los guarda en el clip y los usa al publicar.
  description?: string;
  hashtags?: Record<string, string[]>;
  transcript: ClipWord[];
};

// Propuesta de clip detectada por el pipeline (backend/schemas.py:
// ClipProposal), guardada en metadata["clip_proposals"] con su id.
// Con la edición automática apagada (v1.3.2), el proyecto completa
// con estas propuestas y sin clips renderizados; cada una se abre en
// el Studio, que edita el video fuente con trim inicio/fin y los cues
// de la transcripción en ese rango.
export type ClipProposal = {
  inicio: number;
  fin: number;
  titulo: string;
  puntuacion: number;
  razon_viral: string;
  descripcion?: string;
  palabras_clave?: string[];
  signal_breakdown?: Record<string, number> | null;
};

export type ClipProposalRecord = {
  id: string;
  proposal: ClipProposal;
};

export type ProjectResult = {
  id: string;
  title: string;
  status: string;
  step: string;
  progress: number | null;
  message?: string;
  error: string | null;
  config: {
    ratio: string;
    duration: string;
    framing?: 'speaker' | 'split' | 'blur' | 'center';
    whisper_language?:
      | 'auto'
      | 'ar'
      | 'de'
      | 'en'
      | 'es'
      | 'fr'
      | 'hi'
      | 'it'
      | 'ja'
      | 'ko'
      | 'nl'
      | 'pl'
      | 'pt'
      | 'ru'
      | 'tr'
      | 'uk'
      | 'zh';
    subtitle_template: string;
    subtitle_style?: SubtitleStyleState;
    options: Record<string, boolean>;
    moment: string;
    // Motor de detección de momentos (Fase A, v1.1.0): "signals" es el
    // motor por defecto (señales locales, sin API); "signals+ai" pule los
    // títulos con una llamada corta al LLM; "llm" es el camino anterior.
    // Opcionales: los proyectos anteriores a v1.1.0 no los traen.
    moment_engine?: 'signals' | 'signals+ai' | 'llm';
    signal_score_threshold?: number;
    // Detección rápida (v1.3.1): solo Tier 0 (audio + texto). Opcional:
    // los proyectos anteriores a v1.3.1 no la traen (motor completo).
    fast_detection?: boolean;
    // Edición automática (v1.3.2): renderizar los clips con ffmpeg al
    // terminar la detección. Opcional: los proyectos anteriores a
    // v1.3.2 no la traen (apagada por defecto).
    auto_edit?: boolean;
    schedule?: { scheduled_at: string; platform: string; visibility: string } | null;
  };
  metadata: { duration: number; width: number; height: number; language?: string; moment_engine?: string; signal_score_threshold?: number; content_profile?: string; clip_proposals?: ClipProposalRecord[] } | null;
  transcript: Array<{ start: number; end: number; text: string; words: Array<{ start: number; end: number; text: string }> }> | null;
  clips: ClipResult[];
  thumbnail_url: string | null;
  created_at: string;
  updated_at: string;
};

export type SharedClip = {
  project_title: string;
  clip_id: string;
  title: string;
  duration: number;
  width: number;
  height: number;
  video_url: string;
};

export type IntegrationStatus = {
  configured: boolean;
  missing: string[];
  connected: boolean;
  display_name: string | null;
};

export type ZoomRecording = {
  meeting_id: string;
  topic: string;
  start_time: string | null;
  files: Array<{ file_id: string; file_type: string; file_size: number | null }>;
};

export type ScheduleJob = {
  id: string;
  clip_id: string;
  platform: string;
  scheduled_at: string;
  status: string;
  confirmation_required: boolean;
  visibility: string;
  external_id?: string | null;
  external_url: string | null;
  error: string | null;
  download_url: string | null;
  // Fase 3 (ítems 9–10): reintento por job, cuenta y grupo cruzado.
  attempts?: number;
  next_attempt_at?: string | null;
  account_id?: string | null;
  group_id?: string | null;
};

// Fase 3 (ítems 9–15): publicación y flujo.
export type PublishAccount = {
  provider: string;
  account_id: string;
  display_name: string;
  is_primary: boolean;
  expires_at: string | null;
};

export type BestTimesGuide = {
  platforms?: Record<string, { weekday: number[]; weekend: number[] }>;
  platform?: string;
  label?: string;
  weekday_hours?: number[];
  weekend_hours?: number[];
  note: string;
  next_slots: Record<string, string[]>;
};

export type QueueJob = {
  id: string;
  project_id: string;
  project_title: string | null;
  project_status: string | null;
  project_progress: number | null;
  retry_failed_only: boolean;
  attempts: number;
  status: string;
  error: string | null;
  created_at: string;
  updated_at: string;
};

export type BatchItem = {
  id: string;
  batch_id: string;
  staged_id: string | null;
  url: string | null;
  status: string;
  project_id: string | null;
  error: string | null;
  created_at: string;
};

export type WatchFolder = {
  id: string;
  path: string;
  enabled: boolean;
  config: Record<string, unknown>;
  poll_seconds: number;
  last_scan_at: string | null;
  created_at: string;
};

export type NightWindow = {
  enabled: boolean;
  start: string;
  end: string;
};

export type CaptionVariant = {
  clip_id: string;
  titles: string[];
  description: string;
  hashtags: Record<string, string[]>;
};

export type BrollItem = {
  id: string;
  title: string;
  query: string;
  provider: string;
  external_id: string | null;
  creator: string;
  attribution_url: string;
  preview_url: string;
  local_path: string | null;
  duration: number;
  width: number;
  height: number;
  favorite: boolean;
  created_at: string;
};

export type BrollSearchResult = {
  provider: string;
  external_id: string;
  title: string;
  creator: string;
  attribution_url: string;
  preview_url: string;
  download_url: string;
  duration: number;
  width: number;
  height: number;
};

export type BrollSegment = {
  id: string;
  broll_id: string;
  project_id: string;
  clip_id: string;
  start: number;
  end: number;
  label: string;
  broll_title: string;
  broll_provider: string;
};

export type SavedMoment = {
  id: string;
  title: string;
  description: string;
  project_id: string | null;
  clip_id: string | null;
  start: number | null;
  end: number | null;
  excerpt: string;
  tags: string[];
  created_at: string;
};
