// Tipos del editor visual (Studio), adaptados de ShortStudio para
// kellyClipAi v1.3.0. La especificación de edición (StudioEditSpec) es
// lo que se guarda en `clip.studio_edit` y lo que acompaña a la
// exportación; es JSON plano y versionada para poder migrarla.

export type FitMode = 'cover' | 'blur' | 'fit';
export type SafeZoneMode = 'none' | 'tiktok' | 'shorts' | 'grid';
export type SubMode = 'karaoke' | 'reveal' | 'bounce' | 'classic';
export type SubBgMode = 'none' | 'word-pill' | 'line-box' | 'glass-card';
export type ProgressPos = 'top' | 'bottom';
export type LeftTab = 'format' | 'subs' | 'color';

export interface SubtitleCue {
  id: number;
  start: number;
  end: number;
  text: string;
}

export interface EditorSettings {
  fitMode: FitMode;
  zoom: number;
  panX: number;
  panY: number;
  rounded: boolean;

  trimStart: number;
  trimEnd: number;
  speed: number;

  progressBar: boolean;
  progressPos: ProgressPos;
  progressColor: string;

  subMode: SubMode;
  subBgMode: SubBgMode;
  subPop: boolean;
  subUpper: boolean;
  subFont: string;
  subSize: number;
  subY: number;
  subStroke: number;
  subShadow: number;
  subMaxWidth: number;
  subColor: string;
  subHighlight: string;
  subStrokeColor: string;
  subBgColor: string;

  bright: number;
  contrast: number;
  sat: number;
  volVideo: number;
  volBgm: number;
}

export interface StudioEditSpec {
  version: 1;
  settings: EditorSettings;
  cues: SubtitleCue[];
}

export const DEFAULT_SETTINGS: EditorSettings = {
  fitMode: 'cover',
  zoom: 100,
  panX: 0,
  panY: 0,
  rounded: true,

  trimStart: 0,
  trimEnd: 12,
  speed: 1,

  progressBar: true,
  progressPos: 'top',
  progressColor: '#f43f5e',

  subMode: 'karaoke',
  subBgMode: 'none',
  subPop: true,
  subUpper: true,
  subFont: "'Montserrat', Impact, sans-serif",
  subSize: 68,
  subY: 68,
  subStroke: 13,
  subShadow: 18,
  subMaxWidth: 860,
  subColor: '#ffffff',
  subHighlight: '#facc15',
  subStrokeColor: '#000000',
  subBgColor: '#0f172a',

  bright: 100,
  contrast: 100,
  sat: 100,
  volVideo: 100,
  volBgm: 35,
};

// Valida una especificación guardada (puede venir de una versión
// anterior o corrupta) y la mezcla sobre los valores por defecto para
// que ningún campo falte. Devuelve null si no hay nada aprovechable.
export function normalizeStudioEdit(value: unknown): StudioEditSpec | null {
  if (typeof value !== 'object' || value === null) return null;
  const spec = value as Partial<StudioEditSpec>;
  if (typeof spec.settings !== 'object' || spec.settings === null) return null;
  if (!Array.isArray(spec.cues)) return null;
  const settings: EditorSettings = { ...DEFAULT_SETTINGS };
  for (const key of Object.keys(DEFAULT_SETTINGS) as Array<keyof EditorSettings>) {
    const candidate = (spec.settings as Partial<EditorSettings>)[key];
    if (typeof candidate === typeof DEFAULT_SETTINGS[key]) {
      (settings as unknown as Record<string, unknown>)[key] = candidate;
    }
  }
  const cues: SubtitleCue[] = [];
  spec.cues.forEach((cue, index) => {
    if (typeof cue !== 'object' || cue === null) return;
    const candidate = cue as Partial<SubtitleCue>;
    if (
      typeof candidate.start !== 'number'
      || typeof candidate.end !== 'number'
      || typeof candidate.text !== 'string'
      || !Number.isFinite(candidate.start)
      || !Number.isFinite(candidate.end)
      || candidate.end <= candidate.start
    ) return;
    cues.push({
      id: typeof candidate.id === 'number' && Number.isFinite(candidate.id)
        ? candidate.id
        : index + 1,
      start: candidate.start,
      end: candidate.end,
      text: candidate.text,
    });
  });
  cues.sort((a, b) => a.start - b.start);
  return { version: 1, settings, cues };
}
