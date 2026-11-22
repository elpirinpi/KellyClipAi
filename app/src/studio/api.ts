// Cliente de los endpoints del Studio (contrato v1.3.0). La autenticación
// viaja igual que en el resto de la app: apiFetch añade el Bearer y la
// cookie de medios cubre el <video> del video fuente.
import { apiFetch } from '../auth';
import { API_BASE } from '../context/constants';
import type { ClipResult } from '../types';
import { responseError } from '../utils';
import type {
  FaceTrack,
  FaceTrackKeyframe,
  StudioEditSpec,
  SubtitleCue,
  TargetRatio,
} from './types/editor';

export type ApiCue = { id: string; start: number; end: number; text: string };

function parseCues(value: unknown): ApiCue[] {
  if (typeof value !== 'object' || value === null) return [];
  const cues = (value as { cues?: unknown }).cues;
  if (!Array.isArray(cues)) return [];
  const parsed: ApiCue[] = [];
  cues.forEach((cue) => {
    if (typeof cue !== 'object' || cue === null) return;
    const candidate = cue as Partial<ApiCue>;
    if (
      typeof candidate.start !== 'number'
      || typeof candidate.end !== 'number'
      || typeof candidate.text !== 'string'
      || !Number.isFinite(candidate.start)
      || !Number.isFinite(candidate.end)
      || candidate.end <= candidate.start
    ) return;
    parsed.push({
      id: typeof candidate.id === 'string' ? candidate.id : String(parsed.length),
      start: candidate.start,
      end: candidate.end,
      text: candidate.text,
    });
  });
  return parsed.sort((a, b) => a.start - b.start);
}

// Los cues del backend traen id de cadena; el editor usa ids numéricos
// locales. nextId asigna ids únicos y crecientes dentro de la sesión.
export function apiCuesToLocal(cues: ApiCue[], nextId: () => number): SubtitleCue[] {
  return cues.map((cue) => ({
    id: nextId(),
    start: cue.start,
    end: cue.end,
    text: cue.text,
  }));
}

export async function fetchClipCues(projectId: string, clipId: string): Promise<ApiCue[]> {
  const response = await apiFetch(`${API_BASE}/api/projects/${projectId}/clips/${clipId}/cues`);
  if (!response.ok) throw new Error(await responseError(response));
  return parseCues(await response.json() as unknown);
}

export async function fetchRangeCues(
  projectId: string,
  start: number,
  end: number,
): Promise<ApiCue[]> {
  const params = new URLSearchParams({ start: String(start), end: String(end) });
  const response = await apiFetch(`${API_BASE}/api/projects/${projectId}/cues?${params}`);
  if (!response.ok) throw new Error(await responseError(response));
  return parseCues(await response.json() as unknown);
}

// Valida la respuesta de face-track: descarta keyframes inválidos,
// sujeta las fracciones a [0, 1] y ordena por tiempo ascendente (el
// render interpola por búsqueda binaria y asume ese orden). Una
// respuesta sin forma aprovechable lanza: el llamador la trata como
// fallo de petición (toast + encuadre centrado).
function parseFaceTrack(value: unknown, projectId: string, ratio: TargetRatio): FaceTrack {
  if (typeof value !== 'object' || value === null) {
    throw new Error('Respuesta de seguimiento facial inválida.');
  }
  const candidate = value as {
    project_id?: unknown;
    ratio?: unknown;
    duration?: unknown;
    sample_fps?: unknown;
    has_face?: unknown;
    keyframes?: unknown;
  };
  if (typeof candidate.has_face !== 'boolean' || !Array.isArray(candidate.keyframes)) {
    throw new Error('Respuesta de seguimiento facial inválida.');
  }
  const keyframes: FaceTrackKeyframe[] = [];
  candidate.keyframes.forEach((keyframe) => {
    if (typeof keyframe !== 'object' || keyframe === null) return;
    const point = keyframe as Partial<FaceTrackKeyframe>;
    if (
      typeof point.time !== 'number'
      || typeof point.x !== 'number'
      || typeof point.y !== 'number'
      || !Number.isFinite(point.time)
      || !Number.isFinite(point.x)
      || !Number.isFinite(point.y)
      || point.time < 0
    ) return;
    keyframes.push({
      time: point.time,
      x: Math.max(0, Math.min(1, point.x)),
      y: Math.max(0, Math.min(1, point.y)),
    });
  });
  keyframes.sort((a, b) => a.time - b.time);
  return {
    projectId: typeof candidate.project_id === 'string' ? candidate.project_id : projectId,
    ratio: candidate.ratio === '9:16' || candidate.ratio === '1:1' || candidate.ratio === '16:9'
      ? candidate.ratio
      : ratio,
    duration: typeof candidate.duration === 'number' && Number.isFinite(candidate.duration)
      ? candidate.duration
      : 0,
    sampleFps: typeof candidate.sample_fps === 'number' && Number.isFinite(candidate.sample_fps)
      ? candidate.sample_fps
      : 0,
    hasFace: candidate.has_face,
    keyframes,
  };
}

// Intervalo de sondeo de face-track mientras el backend analiza el
// video en segundo plano (v1.3.2): cada GET es barato (solo lee el
// estado del trabajo), así que 1 s da un progreso fluido sin cargar
// el servidor local.
const FACE_TRACK_POLL_MS = 1000;

// Pista de seguimiento facial del video fuente (v1.3.1; sondeo con
// progreso desde v1.3.2). La primera llamada recorre el video entero
// en el backend y tarda minutos en videos largos: el servidor
// responde 202 mientras el trabajo corre y 200 cuando está lista, así
// que esta función sondea el mismo endpoint cada segundo e informa
// del progreso con onProgress. Las siguientes llamadas, mismo
// proyecto y ratio, son 200 inmediatas por la caché del servidor. Sin
// rostro devuelve hasFace=false y keyframes=[] (no es error).
// Proyecto inexistente: 404; video sin duración o fallo del detector:
// 503 — ambos se propagan como Error. signal cancela el sondeo (al
// desmontar el Studio o al desactivar el seguimiento, p. ej.); el
// trabajo del servidor sigue y la próxima llamada lo reutiliza.
export async function fetchFaceTrack(
  projectId: string,
  ratio: TargetRatio,
  options: {
    onProgress?: (progress: number) => void;
    signal?: AbortSignal;
  } = {},
): Promise<FaceTrack> {
  const params = new URLSearchParams({ ratio });
  const url = `${API_BASE}/api/projects/${projectId}/face-track?${params}`;
  for (;;) {
    options.signal?.throwIfAborted();
    const response = await apiFetch(url, { signal: options.signal });
    if (response.status === 202) {
      const pending = (await response.json().catch(() => null)) as {
        progress?: unknown;
      } | null;
      if (typeof pending?.progress === 'number' && Number.isFinite(pending.progress)) {
        options.onProgress?.(Math.max(0, Math.min(100, pending.progress)));
      }
      await new Promise<void>((resolve, reject) => {
        const timer = setTimeout(resolve, FACE_TRACK_POLL_MS);
        options.signal?.addEventListener('abort', () => {
          clearTimeout(timer);
          reject(new Error('Petición cancelada.'));
        }, { once: true });
      });
      continue;
    }
    if (!response.ok) throw new Error(await responseError(response));
    return parseFaceTrack(await response.json() as unknown, projectId, ratio);
  }
}

export async function saveStudioEdit(
  projectId: string,
  clipId: string,
  spec: StudioEditSpec,
): Promise<ClipResult> {
  const response = await apiFetch(`${API_BASE}/api/projects/${projectId}/clips/${clipId}/studio`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ studio_edit: spec }),
  });
  if (!response.ok) throw new Error(await responseError(response));
  return response.json() as Promise<ClipResult>;
}

export async function exportStudioVideo(
  projectId: string,
  blob: Blob,
  filename: string,
  fields: {
    clipId?: string;
    start: number;
    end: number;
    title?: string;
    studioEdit: StudioEditSpec;
  },
  signal?: AbortSignal,
): Promise<ClipResult> {
  const form = new FormData();
  form.append('file', blob, filename);
  if (fields.clipId) form.append('clip_id', fields.clipId);
  form.append('start', String(fields.start));
  form.append('end', String(fields.end));
  if (fields.title) form.append('title', fields.title);
  form.append('studio_edit', JSON.stringify(fields.studioEdit));
  const response = await apiFetch(`${API_BASE}/api/projects/${projectId}/studio/export`, {
    method: 'POST',
    body: form,
    signal,
  });
  if (!response.ok) throw new Error(await responseError(response));
  return response.json() as Promise<ClipResult>;
}

// --- Forma de onda (Fase 3, ítem 4b) ---------------------------------------
// Picos normalizados 0-1 del video fuente; la línea de tiempo dibuja
// las barras en un <canvas> sin decodificar el audio en el navegador.

export interface Waveform {
  start: number;
  end: number;
  buckets: number;
  peaks: number[];
}

export async function fetchWaveform(
  projectId: string,
  start: number,
  end: number,
  buckets = 1200,
): Promise<Waveform> {
  const params = new URLSearchParams({
    start: String(start),
    end: String(end),
    buckets: String(buckets),
  });
  const response = await apiFetch(`${API_BASE}/api/projects/${projectId}/waveform?${params}`);
  if (!response.ok) throw new Error(await responseError(response));
  const value = (await response.json()) as {
    start?: unknown;
    end?: unknown;
    buckets?: unknown;
    peaks?: unknown;
  };
  const peaks = Array.isArray(value.peaks)
    ? value.peaks.filter((peak): peak is number => typeof peak === 'number' && Number.isFinite(peak))
    : [];
  return {
    start: typeof value.start === 'number' ? value.start : start,
    end: typeof value.end === 'number' ? value.end : end,
    buckets: typeof value.buckets === 'number' ? value.buckets : peaks.length,
    peaks: peaks.map((peak) => Math.max(0, Math.min(1, peak))),
  };
}

// --- Plantilla de marca (Fase 3, ítem 5) ------------------------------------

export interface BrandKit {
  name: string;
  primary_color: string;
  accent_color: string;
  font_family: string | null;
  logo_filename: string | null;
  logo_url: string | null;
}

function parseBrandKit(value: unknown): BrandKit | null {
  if (typeof value !== 'object' || value === null) return null;
  const candidate = value as Partial<BrandKit>;
  if (
    typeof candidate.primary_color !== 'string'
    || typeof candidate.accent_color !== 'string'
  ) return null;
  return {
    name: typeof candidate.name === 'string' ? candidate.name : '',
    primary_color: candidate.primary_color,
    accent_color: candidate.accent_color,
    font_family: typeof candidate.font_family === 'string' ? candidate.font_family : null,
    logo_filename: typeof candidate.logo_filename === 'string' ? candidate.logo_filename : null,
    logo_url: typeof candidate.logo_url === 'string' ? candidate.logo_url : null,
  };
}

export async function fetchBrandKit(): Promise<BrandKit | null> {
  const response = await apiFetch(`${API_BASE}/api/brand-kit`);
  if (!response.ok) throw new Error(await responseError(response));
  const value = (await response.json()) as { brand_kit?: unknown };
  return parseBrandKit(value.brand_kit);
}

export async function saveBrandKit(kit: {
  name: string;
  primary_color: string;
  accent_color: string;
  font_family: string | null;
}): Promise<BrandKit | null> {
  const response = await apiFetch(`${API_BASE}/api/brand-kit`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(kit),
  });
  if (!response.ok) throw new Error(await responseError(response));
  const value = (await response.json()) as { brand_kit?: unknown };
  return parseBrandKit(value.brand_kit);
}

export async function uploadBrandLogo(file: File): Promise<BrandKit | null> {
  const form = new FormData();
  form.append('file', file);
  const response = await apiFetch(`${API_BASE}/api/brand-kit/logo`, {
    method: 'POST',
    body: form,
  });
  if (!response.ok) throw new Error(await responseError(response));
  const value = (await response.json()) as { brand_kit?: unknown };
  return parseBrandKit(value.brand_kit);
}

export async function deleteBrandKit(): Promise<void> {
  const response = await apiFetch(`${API_BASE}/api/brand-kit`, { method: 'DELETE' });
  if (!response.ok) throw new Error(await responseError(response));
}

// --- Versiones de clip (Fase 3, ítem 6) --------------------------------------

export interface ClipVersion {
  version: number;
  filename: string;
  size_bytes: number;
  created_at: string;
  has_subtitles: boolean;
}

export async function fetchClipVersions(
  projectId: string,
  clipId: string,
): Promise<ClipVersion[]> {
  const response = await apiFetch(
    `${API_BASE}/api/projects/${projectId}/clips/${clipId}/versions`,
  );
  if (!response.ok) throw new Error(await responseError(response));
  const value = (await response.json()) as { versions?: unknown };
  if (!Array.isArray(value.versions)) return [];
  return value.versions.filter((item): item is ClipVersion => (
    typeof item === 'object'
    && item !== null
    && typeof (item as ClipVersion).version === 'number'
    && typeof (item as ClipVersion).filename === 'string'
  ));
}

export async function restoreClipVersion(
  projectId: string,
  clipId: string,
  version: number,
): Promise<ClipVersion[]> {
  const response = await apiFetch(
    `${API_BASE}/api/projects/${projectId}/clips/${clipId}/versions/${version}/restore`,
    { method: 'POST' },
  );
  if (!response.ok) throw new Error(await responseError(response));
  const value = (await response.json()) as { versions?: unknown };
  if (!Array.isArray(value.versions)) return [];
  return value.versions.filter((item): item is ClipVersion => (
    typeof item === 'object'
    && item !== null
    && typeof (item as ClipVersion).version === 'number'
  ));
}

// --- Traducción de subtítulos (Fase 3, ítem 7) -------------------------------
// Devuelve el archivo traducido como Blob para descargarlo; sin clave
// de IA el backend responde 409 con un mensaje claro.

export async function downloadTranslatedSubtitles(
  projectId: string,
  clipId: string,
  targetLanguage: string,
  formatName: 'srt' | 'vtt',
): Promise<{ blob: Blob; filename: string }> {
  const response = await apiFetch(
    `${API_BASE}/api/projects/${projectId}/clips/${clipId}/subtitles/translate`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_language: targetLanguage, format_name: formatName }),
    },
  );
  if (!response.ok) throw new Error(await responseError(response));
  const disposition = response.headers.get('Content-Disposition') || '';
  const match = /filename="([^"]+)"/.exec(disposition);
  return {
    blob: await response.blob(),
    filename: match ? match[1] : `${clipId}-${targetLanguage}.${formatName}`,
  };
}
