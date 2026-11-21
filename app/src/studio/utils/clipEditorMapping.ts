// Mapeo del editor por clip (ClipEditorPage) al renderizador canvas de
// ShortStudio (app/src/studio/utils/canvasRenderer.ts), para la vista
// previa EN VIVO de kellyClipAi v1.3.0.
//
// El editor viejo guarda el estilo de subtítulos como (plantilla +
// overrides) pensados para el render ffmpeg (.ass); este módulo los
// traduce a EditorSettings del canvas lo más fielmente posible.
//
// BASE DE TIEMPOS: las frases del editor guardan start/end RELATIVOS al
// inicio del clip (backend/editor.py: editor_phrases resta clip["start"]
// a los tiempos absolutos de la transcripción). El renderizador canvas
// compara cada cue contra el currentTime ABSOLUTO del <video> del video
// fuente, así que cada cue suma el inicio del clip. El campo
// preview_start NO se usa aquí: es el tiempo remapeado dentro del clip
// RENDERIZADO (con los silencios eliminados), no del fuente.
//
// ATRIBUTOS QUE EL CANVAS NO PUEDE REPRESENTAR (ver también el informe
// de la vista previa en vivo):
// - Ratio de salida (16:9, 1:1, 4:5): el canvas es fijo 1080×1920
//   (9:16). La vista previa siempre muestra encuadre vertical; el ratio
//   elegido sigue aplicándose al exportar.
// - Seguimiento de rostro / encuadre (speaker, split, blur, center):
//   el canvas usa un recorte 'cover' centrado, sin detección de rostros.
// - B-roll: la vista previa muestra solo el video fuente.
// - Eliminación de silencios: el audio/video se reproduce continuo; las
//   frases quitadas se SALTAN al reproducir (aproximación del recorte),
//   pero los tiempos visibles son los del fuente, no los remapeados.
// - Resaltado de palabras clave: el .ass tiñe las palabras clave con el
//   color secundario de forma estática; el canvas usa el modo karaoke,
//   que resalta la palabra ACTIVA. El color secundario se conserva como
//   subHighlight, pero el resaltado por palabra clave se pierde.
// - Censura (****) y emojis al final del último grupo: el render ffmpeg
//   los aplica según la configuración del proyecto; la vista previa
//   muestra el texto editado tal cual.
// - Barra de progreso: la exportación no la tiene; en la vista previa se
//   activa solo como ayuda visual del rango recortado.
// - Sombra ASS: es un desplazamiento fijo con color de fondo
//   semitransparente; el canvas solo tiene desenfoque (blur) con sombra
//   negra fija, así que se aproxima con subShadow 8 cuando la plantilla
//   trae sombra y 0 cuando no.
import type { Template } from '../../context/constants';
import type { ClipEditorState, EditablePhrase } from '../../types';
import { DEFAULT_SETTINGS, type EditorSettings, type SubtitleCue } from '../types/editor';

// El .ass del backend se escribe a 720p de base (PlayRes 720×1280) y el
// canvas es 1080×1920: los tamaños de fuente y el ancho máximo se
// escalan por 1080/720 para que la proporción sea la misma que en la
// exportación (a 720p o a 1080p, el backend escala igual).
const ASS_720_TO_CANVAS_1080 = 1080 / 720;

// Ancho máximo de texto: márgenes ASS de 48 px a cada lado sobre 720 px
// de ancho, escalados al canvas ((720 - 2*48) * 1.5 = 936).
const SUB_MAX_WIDTH = 936;

// Posición vertical del bloque de subtítulos (% de la altura del
// canvas): aproxima el MarginV 96 del .ass (a 1080p, 144 px desde el
// borde) para arriba/abajo; el centro es el 50%.
const SUB_Y_BY_POSITION: Record<'top' | 'center' | 'bottom', number> = {
  top: 12,
  center: 50,
  bottom: 88,
};

// Color del borde ASS (&H00101010) común a todas las plantillas.
const ASS_OUTLINE_COLOR = '#101010';

// Estilo base de cada plantilla, replicando la tabla de
// backend/media.py::_write_ass: (fuente, tamaño a 720p, color primario,
// color secundario, borde, sombra). Los colores ASS (&H00BBGGRR) ya
// están convertidos a #RRGGBB. Si el backend cambia una plantilla, hay
// que actualizar esta tabla también. Las plantillas personalizadas
// resuelven a su plantilla base (base_template) antes de buscar aquí;
// un nombre desconocido cae al estilo Default, como en el backend.
interface TemplateBaseStyle {
  font: string;
  size: number;
  primary: string;
  secondary: string;
  outline: number;
  shadow: number;
}

const TEMPLATE_STYLES: Record<string, TemplateBaseStyle> = {
  Default: { font: 'Arial', size: 58, primary: '#ffffff', secondary: '#ffff00', outline: 3, shadow: 1 },
  Modern: { font: 'Arial', size: 56, primary: '#ffff00', secondary: '#ffffff', outline: 2, shadow: 1 },
  Bouncy: { font: 'Arial Black', size: 60, primary: '#88ff88', secondary: '#ffffff', outline: 4, shadow: 1 },
  MrBeast: { font: 'Arial Black', size: 64, primary: '#ffff00', secondary: '#ff0000', outline: 5, shadow: 1 },
  Business: { font: 'Arial', size: 46, primary: '#202020', secondary: '#ffffff', outline: 1, shadow: 0 },
  Esencial: { font: 'Arial', size: 58, primary: '#ffffff', secondary: '#ffff00', outline: 3, shadow: 1 },
  Editorial: { font: 'Georgia', size: 48, primary: '#ffffff', secondary: '#ffffff', outline: 2, shadow: 0 },
  'Neón': { font: 'Arial Black', size: 60, primary: '#66ffff', secondary: '#ffffff', outline: 4, shadow: 1 },
};

// Pilas de fuentes CSS para las fuentes que ofrece el editor. El canvas
// dibuja siempre con peso 900 (como el .ass, que usa Bold), así que
// Arial se ve en negrita igual que en la exportación.
const FONT_STACKS: Record<string, string> = {
  Arial: 'Arial, Helvetica, sans-serif',
  'Arial Black': "'Arial Black', Arial, sans-serif",
  Georgia: "Georgia, 'Times New Roman', serif",
  Tahoma: 'Tahoma, Verdana, sans-serif',
  Verdana: 'Verdana, Geneva, sans-serif',
};

function resolveTemplateStyle(
  templateName: string,
  allTemplates: Template[],
): TemplateBaseStyle {
  const template = allTemplates.find((item) => item.name === templateName);
  // Las plantillas personalizadas guardan su base en base_template (el
  // backend resuelve igual en database.resolve_subtitle_template).
  const baseName = template?.base_template ?? template?.name ?? 'Default';
  return TEMPLATE_STYLES[baseName] ?? TEMPLATE_STYLES.Default;
}

// Traduce el estado del editor por clip a los ajustes del renderizador
// canvas. El recorte (trim) es el rango del clip dentro del video
// fuente: la vista previa en vivo solo reproduce ese rango.
export function mapClipEditorToSettings(
  editor: ClipEditorState,
  clip: { start: number; end: number },
  allTemplates: Template[],
): EditorSettings {
  const base = resolveTemplateStyle(editor.subtitle_template, allTemplates);
  const style = editor.subtitle_style;
  // Los overrides (null = "usar plantilla") se aplican sobre la base
  // exactamente como hace el backend en _write_ass.
  const fontName = style.font_family ?? base.font;
  const assSize = style.font_size ?? base.size;
  const subSize = Math.round(assSize * ASS_720_TO_CANVAS_1080);
  const position = style.position ?? 'bottom';
  return {
    ...DEFAULT_SETTINGS,
    // Encuadre: 'cover' centrado. El seguimiento de rostro de la
    // exportación no existe en el canvas (ver la cabecera del módulo).
    fitMode: 'cover',
    zoom: 100,
    panX: 0,
    panY: 0,
    rounded: false,

    trimStart: clip.start,
    trimEnd: clip.end,
    speed: 1,

    // Barra de progreso solo como ayuda visual de la vista previa (la
    // exportación no la dibuja); el renderizador la calcula sobre el
    // rango recortado (trimStart/trimEnd).
    progressBar: true,
    progressPos: 'top',
    progressColor: '#f43f5e',

    // 'karaoke' aproxima el barrido palabra a palabra (\kf) del .ass.
    // Los nombres de plantilla (p. ej. "Bouncy") no implican animación
    // en el backend: solo cambian fuente, tamaño, colores y borde.
    subMode: 'karaoke',
    // El .ass usa BorderStyle 1 (borde + sombra, sin caja de fondo).
    subBgMode: 'none',
    // La exportación no tiene animación de entrada ni mayúsculas.
    subPop: false,
    subUpper: false,
    subFont: FONT_STACKS[fontName] ?? `${fontName}, sans-serif`,
    subSize,
    subY: SUB_Y_BY_POSITION[position],
    // Borde proporcional al tamaño, como en el .ass (borde/tamaño).
    subStroke: Math.max(0, Math.round((subSize * base.outline) / assSize)),
    subShadow: base.shadow > 0 ? 8 : 0,
    subMaxWidth: SUB_MAX_WIDTH,
    subColor: style.font_color ?? base.primary,
    subHighlight: style.highlight_color ?? base.secondary,
    subStrokeColor: ASS_OUTLINE_COLOR,
    subBgColor: '#0f172a',

    bright: 100,
    contrast: 100,
    sat: 100,
    volVideo: 100,
    volBgm: 35,
  };
}

// Convierte las frases del editor a cues del canvas en tiempo ABSOLUTO
// del video fuente (frase.start/end son relativos al inicio del clip;
// se les suma clipStart). Las frases quitadas no generan cue: su texto
// no debe aparecer en la vista previa.
export function mapPhrasesToCues(
  phrases: EditablePhrase[],
  clipStart: number,
): SubtitleCue[] {
  const cues: SubtitleCue[] = [];
  phrases.forEach((phrase, index) => {
    if (phrase.removed) return;
    const start = clipStart + phrase.start;
    const end = clipStart + phrase.end;
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return;
    cues.push({ id: index + 1, start, end, text: phrase.text });
  });
  cues.sort((a, b) => a.start - b.start);
  return cues;
}

// Rangos absolutos (tiempo del video fuente) de las frases quitadas, en
// el orden del clip: el bucle de vista previa los salta al reproducir
// para aproximar el recorte que la exportación aplica al quitar frases.
export function removedRangesAbsolute(
  phrases: EditablePhrase[],
  clipStart: number,
): Array<readonly [number, number]> {
  const ranges: Array<readonly [number, number]> = [];
  phrases.forEach((phrase) => {
    if (!phrase.removed) return;
    const start = clipStart + phrase.start;
    const end = clipStart + phrase.end;
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return;
    ranges.push([start, end]);
  });
  ranges.sort((a, b) => a[0] - b[0]);
  return ranges;
}
