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
