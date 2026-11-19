import type { CSSProperties } from 'react';
import type { SubtitleStyleState } from '../types';

// Fase 2 (ítem 18): vista previa REAL de la plantilla de subtítulos.
// Antes el carrusel mostraba un degradado con texto genérico; ahora la
// muestra se renderiza con la fuente, el tamaño, el color y la posición
// que el backend aplica de verdad al renderizar (ver _write_ass en
// backend/media.py: cada plantilla define fuente, tamaño a 720p, color
// principal, color de resaltado, grosor de contorno, sombra y alineación).
// Los tamaños se escalan de 720p al ancho de la tarjeta (145 px).
// Los ajustes manuales de estilo (SubtitleStyleState) se aplican por
// encima de la plantilla, igual que en el backend.

type TemplateStyle = {
  fontFamily: string;
  fontSize720: number;
  color: string;
  highlightColor: string;
  outline: number;
  shadow: boolean;
};

// Colores ASS (&H00BBGGRR) convertidos a hex CSS.
const TEMPLATE_STYLES: Record<string, TemplateStyle> = {
  Default: { fontFamily: 'Arial, sans-serif', fontSize720: 58, color: '#ffffff', highlightColor: '#ffff00', outline: 3, shadow: true },
  Modern: { fontFamily: 'Arial, sans-serif', fontSize720: 56, color: '#ffff00', highlightColor: '#ffffff', outline: 2, shadow: true },
  Bouncy: { fontFamily: '"Arial Black", Arial, sans-serif', fontSize720: 60, color: '#88ff88', highlightColor: '#ffffff', outline: 4, shadow: true },
  MrBeast: { fontFamily: '"Arial Black", Arial, sans-serif', fontSize720: 64, color: '#ffff00', highlightColor: '#ff0000', outline: 5, shadow: true },
  Business: { fontFamily: 'Arial, sans-serif', fontSize720: 46, color: '#202020', highlightColor: '#ffffff', outline: 1, shadow: false },
  Esencial: { fontFamily: 'Arial, sans-serif', fontSize720: 58, color: '#ffffff', highlightColor: '#ffff00', outline: 3, shadow: true },
  Editorial: { fontFamily: 'Georgia, serif', fontSize720: 48, color: '#ffffff', highlightColor: '#ffffff', outline: 2, shadow: false },
  'Neón': { fontFamily: '"Arial Black", Arial, sans-serif', fontSize720: 60, color: '#66ffff', highlightColor: '#ffffff', outline: 4, shadow: true },
};

// Resolución de respaldo por tono del carrusel (para plantillas guardadas
// cuyo nombre no coincide con un preset conocido).
const TONE_TO_TEMPLATE: Record<string, string> = {
  default: 'Default',
  modern: 'Modern',
  bouncy: 'Bouncy',
  impact: 'MrBeast',
  business: 'Business',
  essential: 'Esencial',
  editorial: 'Editorial',
  neon: 'Neón',
};

export function templateStyleFor(name: string, tone?: string): TemplateStyle {
  return TEMPLATE_STYLES[name]
    ?? (tone ? TEMPLATE_STYLES[TONE_TO_TEMPLATE[tone]] : undefined)
    ?? TEMPLATE_STYLES.Default;
}

const PREVIEW_CARD_WIDTH = 145;
const RENDER_WIDTH_720P = 720;

function scaledFontSize(size720: number): number {
  return Math.max(9, Math.round((size720 * PREVIEW_CARD_WIDTH) / RENDER_WIDTH_720P));
}

function textShadowFor(style: TemplateStyle): string {
  const shadows: string[] = [];
  if (style.outline >= 2) {
    // Contorno oscuro aproximado (el backend usa #101010).
    shadows.push(
      '1px 0 0 #101010',
      '-1px 0 0 #101010',
      '0 1px 0 #101010',
      '0 -1px 0 #101010',
    );
  }
  if (style.shadow) shadows.push('0 2px 3px rgba(0, 0, 0, 0.65)');
  // Neón: resplandor cian propio de la plantilla.
  if (style.color === '#66ffff') shadows.push('0 0 8px rgba(24, 221, 209, 0.9)');
  return shadows.join(', ');
}

export function TemplatePreview({
  name,
  tone,
  sample,
  overrides,
}: {
  name: string;
  tone: string;
  sample: string;
  // Ajustes manuales del usuario; solo se aplican a la plantilla activa
  // (son los que el backend usará al renderizar).
  overrides?: SubtitleStyleState | null;
}) {
  const base = templateStyleFor(name, tone);
  const fontFamily = overrides?.font_family ?? base.fontFamily;
  const fontSize = overrides?.font_size
    ? scaledFontSize(overrides.font_size)
    : scaledFontSize(base.fontSize720);
  const color = overrides?.font_color ?? base.color;
  const highlightColor = overrides?.highlight_color ?? base.highlightColor;
  const position = overrides?.position ?? 'bottom';
  const resolved: TemplateStyle = { ...base, fontFamily, color, highlightColor };
  // Business usa texto oscuro: sobre el fondo oscuro de la tarjeta sería
  // ilegible, así que se muestra sobre una pastilla clara (el video real
  // suele ser más luminoso; la pastilla solo ayuda a leer la muestra).
  const isBusiness = (name === 'Business' || tone === 'business') && !overrides?.font_color;

  const words = sample.split(' ');
  const firstLine = words.slice(0, 2).join(' ');
  const secondLine = words.slice(2).join(' ') || 'para tus clips';

  const containerStyle: CSSProperties = {
    alignItems: position === 'top'
      ? 'flex-start'
      : position === 'center'
        ? 'center'
        : 'flex-end',
    paddingTop: position === 'top' ? 20 : undefined,
    paddingBottom: position === 'bottom' ? 26 : undefined,
  };
  const linesStyle: CSSProperties = {
    fontFamily: resolved.fontFamily,
    fontSize,
    lineHeight: 1.25,
    color: resolved.color,
    textTransform: 'none',
    textShadow: isBusiness ? 'none' : textShadowFor(resolved),
    background: isBusiness ? '#ffffff' : undefined,
    borderRadius: isBusiness ? 4 : undefined,
    padding: isBusiness ? '5px 7px' : undefined,
  };
  const highlightStyle: CSSProperties = {
    color: resolved.highlightColor,
    textShadow: isBusiness ? 'none' : textShadowFor(resolved),
  };

  return (
    <span className="template-preview template-preview-real" style={containerStyle}>
      <span className="preview-lines" style={linesStyle}>
        <b>{firstLine}</b>
        <strong style={highlightStyle}>{secondLine}</strong>
      </span>
    </span>
  );
}
