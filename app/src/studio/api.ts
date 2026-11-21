// Cliente de los endpoints del Studio (contrato v1.3.0). La autenticación
// viaja igual que en el resto de la app: apiFetch añade el Bearer y la
// cookie de medios cubre el <video> del video fuente.
import { apiFetch } from '../auth';
import { API_BASE } from '../context/constants';
import type { ClipResult } from '../types';
import { responseError } from '../utils';
import type { StudioEditSpec, SubtitleCue } from './types/editor';

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
