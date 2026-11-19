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
