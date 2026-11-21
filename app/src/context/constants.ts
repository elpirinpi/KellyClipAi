// Constantes y datos compartidos por la interfaz, extraídos de App.tsx
// (ítem 23 de la Fase 2): la base de la API, las plantillas de subtítulos,
// las plataformas de enlace reconocidas y el formateo de tamaños.

export const API_BASE = __KELLYCLIP_API_BASE__;

export type Template = { name: string; tone: string; sample: string; featured?: boolean; base_template?: string };

export const templates: Template[] = [
  { name: 'Default', tone: 'default', sample: 'Un gran momento' },
  { name: 'Modern', tone: 'modern', sample: 'Ideas que inspiran' },
  { name: 'Bouncy', tone: 'bouncy', sample: 'Mira esto ahora' },
  { name: 'MrBeast', tone: 'impact', sample: 'NO TE LO PIERDAS' },
  { name: 'Business', tone: 'business', sample: 'Una nueva perspectiva' },
];
export const customTemplates: Template[] = [
  { name: 'Esencial', tone: 'essential', sample: 'Una idea a la vez' },
  { name: 'Editorial', tone: 'editorial', sample: 'La historia comienza' },
  { name: 'Neón', tone: 'neon', sample: 'Hazlo memorable' },
];
export const platforms = [
  { label: 'YouTube', domain: 'youtube.com', glyph: '▶', color: '#f04438' },
  { label: 'YouTube', domain: 'youtu.be', glyph: '▶', color: '#f04438' },
  { label: 'Twitch', domain: 'twitch.tv', glyph: '▰', color: '#8b5cf6' },
  { label: 'X', domain: 'x.com', glyph: '𝕏', color: '#1f2430' },
  { label: 'X', domain: 'twitter.com', glyph: '𝕏', color: '#1f2430' },
  { label: 'TikTok', domain: 'tiktok.com', glyph: '♪', color: '#141414' },
  { label: 'Instagram', domain: 'instagram.com', glyph: '◎', color: '#d94684' },
  { label: 'Facebook', domain: 'facebook.com', glyph: 'f', color: '#1877f2' },
  { label: 'Vimeo', domain: 'vimeo.com', glyph: 'v', color: '#1ab7ea' },
];

// Motor de detección de momentos (Fase A, v1.1.0): la preferencia por
// defecto vive en localStorage (como el tema), se aplica a los proyectos
// nuevos y cada proyecto puede cambiarla en su configuración. El backend
// registra el motor efectivo en los metadatos del proyecto.
export type MomentEngine = 'signals' | 'signals+ai' | 'llm';

export const MOMENT_ENGINE_STORAGE_KEY = 'kellyclipai-moment-engine';
export const SIGNAL_THRESHOLD_STORAGE_KEY = 'kellyclipai-signal-threshold';
// v1.1.1: 5.0 (antes 8.0). Ver backend/schemas.py: con la calibración
// original, 8.0 era inalcanzable para contenido conversacional realista.
export const DEFAULT_SIGNAL_THRESHOLD = 5.0;

export const MOMENT_ENGINE_LABELS: Record<MomentEngine, string> = {
  signals: 'Señales locales',
  'signals+ai': 'Señales locales + IA',
  llm: 'IA',
};

// Perfil de contenido auto-detectado por el motor de señales (Fase C,
// v1.2.0): ajusta los pesos del scorer según el tipo de video.
export type ContentProfile = 'talk' | 'mixed' | 'gameplay';

export const CONTENT_PROFILE_LABELS: Record<ContentProfile, string> = {
  talk: 'Conversación',
  mixed: 'Mixto',
  gameplay: 'Gameplay / acción',
};

export function readStoredMomentEngine(): MomentEngine {
  try {
    const stored = window.localStorage.getItem(MOMENT_ENGINE_STORAGE_KEY);
    if (stored === 'signals' || stored === 'signals+ai' || stored === 'llm') return stored;
  } catch {
    // Sin almacenamiento disponible: el motor por defecto.
  }
  return 'signals';
}

export function readStoredSignalThreshold(): number {
  try {
    const stored = window.localStorage.getItem(SIGNAL_THRESHOLD_STORAGE_KEY);
    if (stored !== null) {
      const value = Number.parseFloat(stored);
      if (Number.isFinite(value) && value >= 0 && value <= 10) return value;
    }
  } catch {
    // Sin almacenamiento disponible: el umbral por defecto.
  }
  return DEFAULT_SIGNAL_THRESHOLD;
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit > 0 && value < 10 ? 1 : 0)} ${units[unit]}`;
}
