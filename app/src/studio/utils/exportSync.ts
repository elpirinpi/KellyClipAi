// Sincronía de la exportación del Studio (BUG 3, v1.3.2).
//
// La exportación graba el canvas en tiempo real con MediaRecorder: el
// reloj de la grabación es el reloj de pared, pero el contenido avanza
// con el reloj del elemento <video> y con el bucle de dibujo (rAF). Si
// el video se atasca (decodificación o buffering del fuente) o el
// bucle de dibujo se queda sin frames (pestaña oculta, máquina
// saturada), el grabador seguiría corriendo mientras el contenido está
// parado y el video saldría alargado y desincronizado ("lento con
// retraso"). La respuesta es pausar el grabador — su reloj se congela
// y al reanudar continúa como si no hubiera pasado el tiempo — junto
// con el video y la música, y reanudar todo junto cuando el contenido
// vuelve a avanzar.
//
// Este módulo es puro (sin DOM) para poder probarlo sin navegador:
// StudioPage.tsx lo usa en el watchdog de la exportación y
// exportSync.test.ts cubre las decisiones.

/** Motivo de la pausa coordinada de la exportación. */
export type ExportStallReason =
  /** El reloj del <video> dejó de avanzar (decodificación/buffering). */
  | 'video'
  /** El bucle de dibujo (rAF) se quedó sin frames (pestaña oculta, saturación). */
  | 'draw';

/** El reloj del video se considera atascado tras este tiempo sin moverse. */
export const VIDEO_CLOCK_STALL_MS = 600;
/** El bucle de dibujo se considera atascado tras este tiempo sin dibujar. */
export const DRAW_STALL_MS = 1000;
/** Un atasco que dura más que esto es un fallo real: la exportación se aborta. */
export const EXPORT_STALL_ABORT_MS = 60_000;
/**
 * Movimiento mínimo del reloj del video para considerarlo vivo. La
 * velocidad mínima del editor es 0.5x, así que en un tick del watchdog
 * (50 ms) el reloj avanza 0.025 s: 0.01 s queda por debajo con margen
 * y por encima de cualquier fluctuación de `currentTime`.
 */
export const VIDEO_TIME_ADVANCE_EPSILON_S = 0.01;

/** Estado del reloj del video entre ticks del watchdog. */
export interface VideoClockState {
  /** Último `currentTime` observado (segundos). */
  timeS: number;
  /** `performance.now()` en que el reloj se movió por última vez. */
  advanceAtMs: number;
}

/**
 * Avance del reloj del video: cualquier movimiento mayor o igual que
 * el épsilon — hacia adelante o hacia atrás (un seek) — significa que
 * el elemento está vivo y refresca el instante del último avance.
 */
export function trackVideoClock(
  prev: VideoClockState,
  currentTimeS: number,
  nowMs: number,
): VideoClockState {
  if (Math.abs(currentTimeS - prev.timeS) >= VIDEO_TIME_ADVANCE_EPSILON_S) {
    return { timeS: currentTimeS, advanceAtMs: nowMs };
  }
  return prev;
}

/** Instantánea que el watchdog de la exportación pasa al detector. */
export interface ExportStallSnapshot {
  nowMs: number;
  /** `performance.now()` en que el reloj del video se movió por última vez. */
  lastVideoAdvanceAtMs: number;
  /** `performance.now()` del último dibujo del bucle rAF durante la exportación. */
  lastDrawAtMs: number;
  /** El <video> está pausado (por el usuario o por una pausa previa): no es un atasco. */
  videoPaused: boolean;
  /** El <video> llegó al final del medio: no es un atasco. */
  videoEnded: boolean;
}

/**
 * Detecta un atasco durante la exportación. Con el video pausado o
 * terminado no hay atasco que pausar. El reloj del video manda sobre
 * el dibujo: si el contenido no avanza, da igual que el canvas siga
 * repintando el mismo frame congelado.
 */
export function detectExportStall(snapshot: ExportStallSnapshot): ExportStallReason | null {
  const { nowMs, lastVideoAdvanceAtMs, lastDrawAtMs, videoPaused, videoEnded } = snapshot;
  if (videoPaused || videoEnded) return null;
  if (nowMs - lastVideoAdvanceAtMs > VIDEO_CLOCK_STALL_MS) return 'video';
  if (nowMs - lastDrawAtMs > DRAW_STALL_MS) return 'draw';
  return null;
}

/** ¿Lleva el atasco tanto tiempo que hay que abortar la exportación? */
export function isStallAborted(
  stalledSinceMs: number,
  nowMs: number,
  abortAfterMs: number = EXPORT_STALL_ABORT_MS,
): boolean {
  return nowMs - stalledSinceMs > abortAfterMs;
}
