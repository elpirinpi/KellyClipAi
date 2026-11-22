// Tipos del editor visual (Studio), adaptados de ShortStudio para
// kellyClipAi v1.3.0. La especificación de edición (StudioEditSpec) es
// lo que se guarda en `clip.studio_edit` y lo que acompaña a la
// exportación; es JSON plano y versionada para poder migrarla.

export type FitMode = 'cover' | 'blur' | 'fit';
export type TargetRatio = '9:16' | '1:1' | '16:9';
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

// Pista de seguimiento facial (follow-cam, v1.3.1), respuesta de
// GET /api/projects/{id}/face-track ya validada por api.ts. Los
// tiempos son absolutos del video fuente, ascendentes; x/y son la
// posición de la ventana de recorte como FRACCIÓN del desplazamiento
// máximo en cada eje (0 = un extremo, 1 = el otro, 0.5 = centrado o
// sin desplazamiento posible en ese eje). Sin rostro en todo el
// video, hasFace es false y keyframes va vacío (no es un error).
// La pista es dato calculado por el backend: vive solo en memoria
// (caché en StudioPage), nunca en studio_edit.
export interface FaceTrackKeyframe {
  time: number;
  x: number;
  y: number;
}

export interface FaceTrack {
  projectId: string;
  ratio: TargetRatio;
  duration: number;
  sampleFps: number;
  hasFace: boolean;
  keyframes: FaceTrackKeyframe[];
}

// Estado de la pista para el interruptor "Seguir rostro" del
// LeftSidebar: 'idle' = interruptor apagado; 'loading' = pidiendo
// la pista (la primera llamada recorre el video entero y tarda);
// 'ready' = pista lista; 'no-face' = has_face=false; 'error' = la
// petición falló (red, 404, 503): el interruptor queda activo pero
// el encuadre sigue centrado.
export type FaceTrackStatus = 'idle' | 'loading' | 'ready' | 'no-face' | 'error';

export interface EditorSettings {
  ratio: TargetRatio;
  fitMode: FitMode;
  zoom: number;
  panX: number;
  panY: number;
  rounded: boolean;
  // Seguimiento de rostro (follow-cam, v1.3.1): solo tiene efecto en
  // modo cover; la pista la calcula el backend y NO se guarda en
  // studio_edit (se pide de nuevo al activar el interruptor).
  faceFollow: boolean;

  trimStart: number;
  trimEnd: number;
  speed: number;

  // Texto de gancho (Fase 3, ítem 8): se muestra superpuesto durante
  // los primeros 3 segundos del clip (la ventana de gancho). Vacío =
  // sin texto de gancho.
  hookText: string;
  // Logo de marca (Fase 3, ítem 5): URL del logo guardado
  // (/api/brand-kit/logo) o null. El canvas lo dibuja en la esquina
  // superior derecha; la exportación lo incluye al capturar el canvas.
  brandLogoUrl: string | null;

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
  ratio: '9:16',
  fitMode: 'cover',
  zoom: 100,
  panX: 0,
  panY: 0,
  rounded: true,
  faceFollow: false,

  trimStart: 0,
  trimEnd: 12,
  speed: 1,

  hookText: '',
  brandLogoUrl: null,

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
    // El ratio se valida aparte: el filtro por tipo dejaría pasar
    // cualquier string guardado ('4:3', 'vertical'...).
    if (key === 'ratio') continue;
    const candidate = (spec.settings as Partial<EditorSettings>)[key];
    if (typeof candidate === typeof DEFAULT_SETTINGS[key]) {
      (settings as unknown as Record<string, unknown>)[key] = candidate;
    }
  }
  const savedRatio = (spec.settings as Partial<EditorSettings>).ratio;
  settings.ratio =
    savedRatio === '9:16' || savedRatio === '1:1' || savedRatio === '16:9'
      ? savedRatio
      : '9:16';
  // brandLogoUrl es string | null y el filtro por tipo del bucle lo
  // dejaría fuera (typeof null es 'object'): se valida aparte.
  const savedLogo = (spec.settings as Partial<EditorSettings>).brandLogoUrl;
  settings.brandLogoUrl = typeof savedLogo === 'string' ? savedLogo : null;
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

// Dimensiones del canvas de render y exportación según el ratio
// (verificado contra shortstudio-v2 src/types/editor.ts:332-336).
// Cualquier valor fuera de los tres válidos cae al 9:16 por defecto.
export function getCanvasDimensions(ratio: TargetRatio): { width: number; height: number } {
  if (ratio === '1:1') return { width: 1080, height: 1080 };
  if (ratio === '16:9') return { width: 1920, height: 1080 };
  return { width: 1080, height: 1920 };
}
