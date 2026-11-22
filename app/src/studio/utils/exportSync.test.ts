// Pruebas de la lógica de sincronía de la exportación (BUG 3, v1.3.2).
//
// Sin navegador y sin dependencias: se ejecuta con
//   node src/studio/utils/exportSync.test.ts
// (desde app/; Node 22.18+ ejecuta TypeScript directamente) o con
//   npm run test:export-sync
// Cada comprobación que falla lanza y el proceso sale con código 1.
// El archivo no se importa desde la app, así que no entra en el
// bundle; `tsc --noEmit` (npm run build) sí lo comprueba.
import {
  detectExportStall,
  DRAW_STALL_MS,
  EXPORT_STALL_ABORT_MS,
  isStallAborted,
  trackVideoClock,
  VIDEO_CLOCK_STALL_MS,
  type ExportStallSnapshot,
  type VideoClockState,
} from './exportSync.ts';

let passed = 0;

function check(name: string, actual: unknown, expected: unknown): void {
  if (actual !== expected) {
    throw new Error(`FAIL: ${name} — esperado ${String(expected)}, obtenido ${String(actual)}`);
  }
  passed += 1;
}

function snapshot(overrides: Partial<ExportStallSnapshot>): ExportStallSnapshot {
  return {
    nowMs: 10_000,
    lastVideoAdvanceAtMs: 10_000,
    lastDrawAtMs: 10_000,
    videoPaused: false,
    videoEnded: false,
    ...overrides,
  };
}

// Umbrales documentados (si cambian, que sea a conciencia).
check('umbral de atasco del reloj del video', VIDEO_CLOCK_STALL_MS, 600);
check('umbral de atasco del bucle de dibujo', DRAW_STALL_MS, 1000);
check('tope de aborto por atasco', EXPORT_STALL_ABORT_MS, 60_000);

// Sin atasco: todo fresco.
check('todo fresco no es atasco', detectExportStall(snapshot({})), null);
// Pausado o terminado no es atasco, aunque los relojes estén viejos.
check(
  'video pausado no es atasco',
  detectExportStall(snapshot({
    videoPaused: true,
    lastVideoAdvanceAtMs: 1_000,
    lastDrawAtMs: 1_000,
  })),
  null,
);
check(
  'video terminado no es atasco',
  detectExportStall(snapshot({
    videoEnded: true,
    lastVideoAdvanceAtMs: 1_000,
    lastDrawAtMs: 1_000,
  })),
  null,
);
// Reloj del video congelado: 'video', también en el límite.
check(
  'reloj congelado 601 ms es atasco de video',
  detectExportStall(snapshot({ lastVideoAdvanceAtMs: 10_000 - 601 })),
  'video',
);
check(
  'reloj congelado justo 600 ms aún no es atasco',
  detectExportStall(snapshot({ lastVideoAdvanceAtMs: 10_000 - 600 })),
  null,
);
// Dibujo congelado con el reloj vivo: 'draw', también en el límite.
check(
  'dibujo congelado 1001 ms es atasco de dibujo',
  detectExportStall(snapshot({ lastDrawAtMs: 10_000 - 1001 })),
  'draw',
);
check(
  'dibujo congelado justo 1000 ms aún no es atasco',
  detectExportStall(snapshot({ lastDrawAtMs: 10_000 - 1000 })),
  null,
);
// Con los dos relojes congelados manda el video.
check(
  'con los dos congelados manda el video',
  detectExportStall(snapshot({
    lastVideoAdvanceAtMs: 10_000 - 900,
    lastDrawAtMs: 10_000 - 5000,
  })),
  'video',
);

// Seguimiento del reloj del video.
const clock: VideoClockState = { timeS: 5.0, advanceAtMs: 10_000 };
const advanced = trackVideoClock(clock, 5.025, 10_050);
check('avance de 0.025 s refresca el instante', advanced.advanceAtMs, 10_050);
check('avance de 0.025 s actualiza el tiempo', advanced.timeS, 5.025);
const unmoved = trackVideoClock(clock, 5.005, 10_050);
check('movimiento de 0.005 s no refresca', unmoved.advanceAtMs, 10_000);
check('movimiento de 0.005 s no actualiza el tiempo', unmoved.timeS, 5.0);
const seekBack = trackVideoClock(clock, 4.5, 10_050);
check('un seek hacia atrás también es movimiento', seekBack.advanceAtMs, 10_050);
check('un seek hacia atrás actualiza el tiempo', seekBack.timeS, 4.5);
// 5.01 - 5.0 es 0.009999… en binario, por debajo del épsilon: se
// prueba justo por encima (0.011 s) en vez del valor nominal.
const exact = trackVideoClock(clock, 5.011, 10_050);
check('justo por encima del épsilon (0.011 s) cuenta como movimiento', exact.advanceAtMs, 10_050);
// A la velocidad mínima (0.5x) un tick del watchdog (50 ms) avanza
// 0.025 s: el épsilon de 0.01 s no da falsos atascos.
const minSpeed = trackVideoClock(clock, 5.0 + 0.5 * 0.05, 10_050);
check('a 0.5x un tick de 50 ms cuenta como avance', minSpeed.advanceAtMs, 10_050);

// Válvula de aborto por atasco.
check('atasco de 59 s no aborta', isStallAborted(0, 59_000), false);
check('atasco de justo 60 s no aborta', isStallAborted(0, 60_000), false);
check('atasco de 60 s y 1 ms aborta', isStallAborted(0, 60_001), true);
check('el tope es configurable', isStallAborted(0, 5_001, 5_000), true);

console.log(`exportSync.test.ts: ${passed} comprobaciones superadas.`);
