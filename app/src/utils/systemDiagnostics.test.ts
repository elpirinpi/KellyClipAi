// Pruebas del diagnóstico del sistema (v1.4.2): clasificación del
// renderizador WebGL y parseo defensivo de GET /api/encoder.
//
// Sin navegador y sin dependencias: se ejecuta con
//   node src/utils/systemDiagnostics.test.ts
// (desde app/; Node 22.18+ ejecuta TypeScript directamente) o con
//   npm run test:system-diagnostics
// Cada comprobación que falla lanza y el proceso sale con código 1.
// El archivo no se importa desde la app, así que no entra en el
// bundle; `tsc --noEmit` (npm run build) sí lo comprueba.
import {
  classifyRenderer,
  detectGpuAcceleration,
  parseEncoderInfo,
  type GpuProbeCanvas,
  type GpuProbeDocument,
  type GpuProbeWebGLContext,
} from './systemDiagnostics.ts';

let passed = 0;

function check(name: string, actual: unknown, expected: unknown): void {
  if (actual !== expected) {
    throw new Error(`FAIL: ${name} — esperado ${String(expected)}, obtenido ${String(actual)}`);
  }
  passed += 1;
}

// --- Clasificación del renderizador ---------------------------------------

check(
  'una GPU NVIDIA es hardware',
  classifyRenderer('ANGLE (NVIDIA, NVIDIA GeForce GTX 1060 Direct3D11 vs_5_0 ps_5_0, D3D11)'),
  'hardware',
);
check(
  'una GPU Intel es hardware',
  classifyRenderer('ANGLE (Intel, Intel(R) UHD Graphics 620 Direct3D11 vs_5_0 ps_5_0, D3D11)'),
  'hardware',
);
check(
  'SwiftShader es software',
  classifyRenderer('ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)'),
  'software',
);
check(
  'la clasificación no depende de las mayúsculas',
  classifyRenderer('ANGLE (GOOGLE, SWIFTSHADER DEVICE)'),
  'software',
);
check('llvmpipe es software', classifyRenderer('llvmpipe (LLVM 15.0.7, 256 bits)'), 'software');
check('softpipe es software', classifyRenderer('softpipe (Gallium)'), 'software');
check('sin renderizador no se puede saber', classifyRenderer(null), 'unknown');
check('un renderizador vacío no se puede saber', classifyRenderer(''), 'unknown');

// --- Sonda WebGL con contexto mockeado ------------------------------------

function fakeContext(renderer: string | null, options: {
  extension?: boolean;
  throwOnParameter?: boolean;
} = {}): GpuProbeWebGLContext {
  return {
    getExtension: (name: string) =>
      (options.extension ?? true) && name === 'WEBGL_debug_renderer_info'
        ? { UNMASKED_RENDERER_WEBGL: 0x9246 }
        : null,
    getParameter: () => {
      if (options.throwOnParameter) throw new Error('getParameter falló');
      return renderer;
    },
  };
}

function fakeDoc(context: GpuProbeWebGLContext | null, options: {
  throwOnCreate?: boolean;
} = {}): GpuProbeDocument {
  const canvas: GpuProbeCanvas = {
    getContext: () => context,
  };
  return {
    createElement: () => {
      if (options.throwOnCreate) throw new Error('createElement falló');
      return canvas;
    },
  };
}

const nvidia = fakeDoc(fakeContext('ANGLE (NVIDIA, NVIDIA GeForce GTX 1060 Direct3D11 vs_5_0 ps_5_0, D3D11)'));
const nvidiaResult = detectGpuAcceleration(nvidia);
check('sonda con GTX 1060: hardware', nvidiaResult.acceleration, 'hardware');
check('sonda con GTX 1060: WebGL disponible', nvidiaResult.webglAvailable, true);
check(
  'sonda con GTX 1060: expone el renderizador',
  nvidiaResult.renderer,
  'ANGLE (NVIDIA, NVIDIA GeForce GTX 1060 Direct3D11 vs_5_0 ps_5_0, D3D11)',
);

const swiftShader = fakeDoc(fakeContext('ANGLE (Google, SwiftShader Device, SwiftShader driver)'));
const swiftResult = detectGpuAcceleration(swiftShader);
check('sonda con SwiftShader: software', swiftResult.acceleration, 'software');
check('sonda con SwiftShader: WebGL disponible', swiftResult.webglAvailable, true);

const noWebgl = detectGpuAcceleration(fakeDoc(null));
check('sin contexto WebGL: unknown', noWebgl.acceleration, 'unknown');
check('sin contexto WebGL: WebGL no disponible', noWebgl.webglAvailable, false);
check('sin contexto WebGL: sin renderizador', noWebgl.renderer, null);

const noExtension = detectGpuAcceleration(fakeDoc(fakeContext('ANGLE (NVIDIA, ...)', { extension: false })));
check('sin extensión de depuración: unknown', noExtension.acceleration, 'unknown');
check('sin extensión de depuración: WebGL disponible', noExtension.webglAvailable, true);
check('sin extensión de depuración: sin renderizador', noExtension.renderer, null);

const throwingParameter = detectGpuAcceleration(fakeDoc(fakeContext('ANGLE (NVIDIA, ...)', { throwOnParameter: true })));
check('getParameter que lanza: unknown', throwingParameter.acceleration, 'unknown');
check('getParameter que lanza: sin renderizador', throwingParameter.renderer, null);

const throwingCreate = detectGpuAcceleration(fakeDoc(fakeContext(null), { throwOnCreate: true }));
check('createElement que lanza: unknown', throwingCreate.acceleration, 'unknown');
check('createElement que lanza: WebGL no disponible', throwingCreate.webglAvailable, false);

// Sin documento inyectado y sin `document` global (Node): unknown, sin lanzar.
const noDocument = detectGpuAcceleration();
check('sin documento global: unknown', noDocument.acceleration, 'unknown');
check('sin documento global: WebGL no disponible', noDocument.webglAvailable, false);

// --- Parseo de GET /api/encoder -------------------------------------------

const nvenc = parseEncoderInfo({
  encoder: 'h264_nvenc',
  kind: 'nvenc',
  label: 'NVENC (GPU NVIDIA)',
  reason: 'GPU NVIDIA, probe OK',
});
check('respuesta NVENC válida: etiqueta', nvenc?.label, 'NVENC (GPU NVIDIA)');
check('respuesta NVENC válida: codificador', nvenc?.encoder, 'h264_nvenc');
check('respuesta NVENC válida: tipo', nvenc?.kind, 'nvenc');
check('respuesta NVENC válida: motivo', nvenc?.reason, 'GPU NVIDIA, probe OK');

const cpu = parseEncoderInfo({ encoder: 'libx264', kind: 'cpu', label: 'CPU (libx264)' });
check('respuesta CPU sin motivo: válida', cpu?.label, 'CPU (libx264)');
check('respuesta CPU sin motivo: sin motivo', cpu?.reason, undefined);

check('sin etiqueta no es válida', parseEncoderInfo({ encoder: 'h264_nvenc', kind: 'nvenc' }), null);
check('sin codificador no es válida', parseEncoderInfo({ kind: 'nvenc', label: 'NVENC (GPU NVIDIA)' }), null);
check('tipos equivocados no son válidos', parseEncoderInfo({ encoder: 42, kind: 'nvenc', label: 'NVENC (GPU NVIDIA)' }), null);
check('null no es válido', parseEncoderInfo(null), null);
check('un texto no es válido', parseEncoderInfo('h264_nvenc'), null);
check('un arreglo no es válido', parseEncoderInfo(['h264_nvenc']), null);

console.log(`systemDiagnostics.test.ts: ${passed} comprobaciones superadas.`);
