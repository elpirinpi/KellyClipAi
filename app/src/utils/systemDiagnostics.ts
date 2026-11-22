// Diagnóstico del sistema (v1.4.2): aceleración gráfica de la
// interfaz y codificador de video activo, visibles en Ajustes sin
// abrir las herramientas de desarrollo.
//
// La sonda de GPU crea un contexto WebGL temporal y lee el
// renderizador desenmascarado con la extensión
// WEBGL_debug_renderer_info, disponible por defecto en Chromium
// (WebView2). Es solo lectura y no cambia nada: si WebGL no está
// disponible o el navegador oculta el renderizador, el resultado es
// 'unknown' y la UI muestra "No se pudo comprobar" en vez de inventar.

export type GpuAcceleration = 'hardware' | 'software' | 'unknown';

export interface GpuDiagnostics {
  acceleration: GpuAcceleration;
  /** Renderizador desenmascarado (p. ej. "ANGLE (NVIDIA, ...)"), o null si no se pudo leer. */
  renderer: string | null;
  webglAvailable: boolean;
}

// Fragmentos que identifican un renderizador por software en el
// nombre desenmascarado: SwiftShader de Chromium, llvmpipe/softpipe
// de Mesa y el "Basic" de ANGLE sin GPU.
const SOFTWARE_RENDERER_PATTERNS = [
  'swiftshader',
  'llvmpipe',
  'softpipe',
  'software',
  'basic render',
];

export function classifyRenderer(renderer: string | null): GpuAcceleration {
  if (!renderer) return 'unknown';
  const lowered = renderer.toLowerCase();
  return SOFTWARE_RENDERER_PATTERNS.some((pattern) => lowered.includes(pattern))
    ? 'software'
    : 'hardware';
}

/** Superficie mínima de Document que usa la sonda (inyectable en tests). */
export interface GpuProbeDocument {
  createElement(tagName: string): GpuProbeCanvas;
}

export interface GpuProbeCanvas {
  getContext(kind: string): GpuProbeWebGLContext | null;
}

export interface GpuProbeWebGLContext {
  getExtension(name: string): { UNMASKED_RENDERER_WEBGL: number } | null;
  getParameter(pname: number): unknown;
}

export function detectGpuAcceleration(doc?: GpuProbeDocument): GpuDiagnostics {
  // Sin documento inyectado (la app real), se usa el documento global;
  // el cast es solo para TypeScript: el canvas real devuelve un
  // WebGLRenderingContext, que tiene getExtension/getParameter.
  const probeDoc: GpuProbeDocument | undefined =
    doc ?? (typeof document !== 'undefined'
      ? (document as unknown as GpuProbeDocument)
      : undefined);
  if (!probeDoc) {
    return { acceleration: 'unknown', renderer: null, webglAvailable: false };
  }
  let gl: GpuProbeWebGLContext | null = null;
  try {
    const canvas = probeDoc.createElement('canvas');
    gl = canvas.getContext('webgl') ?? canvas.getContext('experimental-webgl');
  } catch {
    gl = null;
  }
  if (!gl) {
    return { acceleration: 'unknown', renderer: null, webglAvailable: false };
  }
  let renderer: string | null = null;
  try {
    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    if (debugInfo) {
      const value = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
      renderer = typeof value === 'string' ? value : null;
    }
  } catch {
    renderer = null;
  }
  return { acceleration: classifyRenderer(renderer), renderer, webglAvailable: true };
}

// Codificador de video activo, según GET /api/encoder (backend):
// { encoder, kind, label, reason }. La forma se valida antes de
// mostrarla: una respuesta con otra forma (o un 404, si el backend
// es anterior a v1.4.2) se trata como "No disponible", sin romper
// la UI.
export interface EncoderInfo {
  encoder: string;
  kind: string;
  label: string;
  reason?: string;
}

export function parseEncoderInfo(value: unknown): EncoderInfo | null {
  if (typeof value !== 'object' || value === null) return null;
  const candidate = value as Record<string, unknown>;
  if (
    typeof candidate.encoder !== 'string'
    || typeof candidate.kind !== 'string'
    || typeof candidate.label !== 'string'
  ) {
    return null;
  }
  return {
    encoder: candidate.encoder,
    kind: candidate.kind,
    label: candidate.label,
    ...(typeof candidate.reason === 'string' ? { reason: candidate.reason } : {}),
  };
}
