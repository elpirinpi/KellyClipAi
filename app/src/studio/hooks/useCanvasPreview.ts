// Bucle de vista previa en vivo con el renderizador canvas de
// ShortStudio (renderEditorFrame): un <video> oculto con el video
// fuente del proyecto se dibuja en un <canvas> 1080×1920 en cada
// requestAnimationFrame, con los subtítulos y ajustes actuales, de modo
// que cualquier cambio del editor se ve al instante, sin re-render.
//
// Lo usa ClipEditorPage (v1.3.0) para su vista previa en vivo. StudioPage
// mantiene su propio bucle porque está acoplado a la exportación con
// MediaRecorder, el grafo de audio y la música de fondo; este hook es el
// mismo patrón sin esas dependencias, listo para reutilizarse.
//
// El consumidor renderiza el <video> oculto y el <canvas>, y conecta los
// refs y los manejadores que devuelve el hook. La reproducción se limita
// al rango recortado (settings.trimStart/trimEnd): al llegar al fin
// vuelve al inicio, y los rangos de skipRanges (frases quitadas) se
// saltan al reproducir.
import { useCallback, useEffect, useRef, useState } from 'react';
import type { RefObject } from 'react';
import type { SourceState } from '../components/PreviewStage';
import type { EditorSettings, SubtitleCue } from '../types/editor';
import { renderEditorFrame } from '../utils/canvasRenderer';

export interface CanvasPreviewOptions {
  enabled: boolean;
  settings: EditorSettings;
  cues: SubtitleCue[];
  /** Rangos absolutos (tiempo del video fuente) que se saltan al reproducir. */
  skipRanges: Array<readonly [number, number]>;
  /** Cambia al cambiar de clip (sin remontar la página): reinicia el estado del fuente. */
  resetKey?: string;
}

export interface CanvasPreview {
  canvasRef: RefObject<HTMLCanvasElement>;
  videoRef: RefObject<HTMLVideoElement>;
  sourceState: SourceState;
  sourceError: string;
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  togglePlay: () => void;
  seekTo: (time: number) => void;
  playFrom: (time: number) => void;
  retrySource: () => void;
  handleLoadedMetadata: () => void;
  handleVideoError: () => void;
  handleSeeked: () => void;
}

export function useCanvasPreview({
  enabled,
  settings,
  cues,
  skipRanges,
  resetKey,
}: CanvasPreviewOptions): CanvasPreview {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const blurCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const blurCtxRef = useRef<CanvasRenderingContext2D | null>(null);

  const [sourceState, setSourceState] = useState<SourceState>('loading');
  const [sourceError, setSourceError] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const settingsRef = useRef(settings);
  const cuesRef = useRef(cues);
  const skipRangesRef = useRef(skipRanges);
  const sourceReadyRef = useRef(false);
  const isPlayingRef = useRef(false);

  settingsRef.current = settings;
  cuesRef.current = cues;
  skipRangesRef.current = skipRanges;
  sourceReadyRef.current = sourceState === 'ready';
  isPlayingRef.current = isPlaying;

  const getBlurResources = useCallback(() => {
    if (!blurCanvasRef.current) {
      const blurCanvas = document.createElement('canvas');
      blurCanvas.width = 270;
      blurCanvas.height = 480;
      blurCanvasRef.current = blurCanvas;
      blurCtxRef.current = blurCanvas.getContext('2d', { alpha: false });
    }
    return { blurCanvas: blurCanvasRef.current, blurCtx: blurCtxRef.current };
  }, []);

  const drawCurrentFrame = useCallback(
    (explicitTime?: number) => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext('2d', { alpha: false });
      const { blurCanvas, blurCtx } = getBlurResources();
      if (!ctx || !blurCtx) return;
      const t = explicitTime !== undefined
        ? explicitTime
        : videoRef.current?.currentTime || 0;
      renderEditorFrame({
        ctx,
        blurCanvas,
        blurCtx,
        video: videoRef.current,
        hasMedia: sourceReadyRef.current,
        currentTime: t,
        settings: settingsRef.current,
        subtitles: cuesRef.current,
      });
    },
    [getBlurResources],
  );

  // Bucle principal: solo dibuja mientras se reproduce; en pausa los
  // redibujos los dispara el efecto de más abajo (cambios de ajustes o
  // de cues) y el evento 'seeked'. Al llegar al fin del recorte vuelve
  // al inicio (bucle, como StudioPage); las frases quitadas se saltan.
  useEffect(() => {
    let rafId = 0;
    const tick = () => {
      if (isPlayingRef.current && sourceReadyRef.current && videoRef.current) {
        const video = videoRef.current;
        const s = settingsRef.current;
        for (const [skipStart, skipEnd] of skipRangesRef.current) {
          if (video.currentTime >= skipStart && video.currentTime < skipEnd) {
            video.currentTime = skipEnd < s.trimEnd - 0.05 ? skipEnd : s.trimStart;
            break;
          }
        }
        if (video.currentTime >= s.trimEnd || video.ended) {
          video.currentTime = s.trimStart;
          video.play().catch(() => {});
        }
        setCurrentTime(video.currentTime);
        drawCurrentFrame(video.currentTime);
      }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [drawCurrentFrame]);

  // En pausa, cualquier cambio de ajustes, cues o estado del fuente
  // redibuja el frame actual: así los cambios se ven al instante.
  useEffect(() => {
    if (!isPlaying) drawCurrentFrame();
  }, [settings, cues, sourceState, isPlaying, drawCurrentFrame]);

  // Al cambiar el recorte (otro clip) con la vista en pausa, volver al
  // inicio del nuevo rango.
  const trimStart = settings.trimStart;
  useEffect(() => {
    if (!enabled || !sourceReadyRef.current || !videoRef.current) return;
    if (isPlayingRef.current) return;
    videoRef.current.currentTime = trimStart;
    setCurrentTime(trimStart);
    drawCurrentFrame(trimStart);
  }, [trimStart, enabled, drawCurrentFrame]);

  // Al cambiar de clip sin remontar la página, el <video> se vuelve a
  // montar: volver al estado de carga hasta que lleguen los metadatos
  // del nuevo elemento (si no, la vista quedaría en "ready" con un
  // video que aún no tiene datos).
  const resetKeyRef = useRef(resetKey);
  useEffect(() => {
    if (resetKeyRef.current === resetKey) return;
    resetKeyRef.current = resetKey;
    videoRef.current?.pause();
    isPlayingRef.current = false;
    setIsPlaying(false);
    setSourceState('loading');
    setSourceError('');
    setCurrentTime(settingsRef.current.trimStart);
  }, [resetKey]);

  // Al deshabilitarse (p. ej. al salir del editor), pausar el video.
  useEffect(() => {
    if (enabled) return;
    videoRef.current?.pause();
    isPlayingRef.current = false;
    setIsPlaying(false);
  }, [enabled]);

  useEffect(() => {
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(() => drawCurrentFrame());
    }
  }, [drawCurrentFrame]);

  useEffect(() => () => {
    videoRef.current?.pause();
  }, []);

  const togglePlay = useCallback(() => {
    if (!sourceReadyRef.current || !videoRef.current) return;
    const shouldPlay = !isPlayingRef.current;
    isPlayingRef.current = shouldPlay;
    setIsPlaying(shouldPlay);
    const video = videoRef.current;
    if (shouldPlay) {
      const s = settingsRef.current;
      if (video.currentTime < s.trimStart || video.currentTime >= s.trimEnd - 0.05) {
        video.currentTime = s.trimStart;
        setCurrentTime(s.trimStart);
      }
      video.play().catch(() => {});
    } else {
      video.pause();
    }
  }, []);

  const seekTo = useCallback((time: number) => {
    const video = videoRef.current;
    if (!sourceReadyRef.current || !video) return;
    const s = settingsRef.current;
    const clamped = Math.max(s.trimStart, Math.min(s.trimEnd, time));
    video.currentTime = clamped;
    setCurrentTime(clamped);
  }, []);

  const playFrom = useCallback((time: number) => {
    const video = videoRef.current;
    if (!sourceReadyRef.current || !video) return;
    const s = settingsRef.current;
    const clamped = Math.max(s.trimStart, Math.min(s.trimEnd - 0.05, time));
    video.currentTime = clamped;
    setCurrentTime(clamped);
    isPlayingRef.current = true;
    setIsPlaying(true);
    video.play().catch(() => {});
  }, []);

  const handleLoadedMetadata = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (Number.isFinite(video.duration) && video.duration > 0) {
      setDuration(video.duration);
    }
    setSourceState('ready');
    setSourceError('');
    const start = settingsRef.current.trimStart;
    video.currentTime = start;
    setCurrentTime(start);
    drawCurrentFrame(start);
  }, [drawCurrentFrame]);

  const handleVideoError = useCallback(() => {
    setSourceState('error');
    setSourceError('El servidor no pudo servir el video original de este proyecto.');
    isPlayingRef.current = false;
    setIsPlaying(false);
  }, []);

  const handleSeeked = useCallback(() => {
    if (!isPlayingRef.current) drawCurrentFrame();
  }, [drawCurrentFrame]);

  const retrySource = useCallback(() => {
    setSourceState('loading');
    setSourceError('');
    videoRef.current?.load();
  }, []);

  return {
    canvasRef,
    videoRef,
    sourceState,
    sourceError,
    isPlaying,
    currentTime,
    duration,
    togglePlay,
    seekTo,
    playFrom,
    retrySource,
    handleLoadedMetadata,
    handleVideoError,
    handleSeeked,
  };
}
