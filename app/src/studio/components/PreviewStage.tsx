// Escenario de vista previa, adaptado del PreviewStage de ShortStudio:
// sin subida por arrastre ni clip demo — el video fuente lo sirve el
// backend del proyecto. El overlay solo cubre la carga o un error del
// video original.
import React, { useRef } from 'react';
import { AlertCircle, Film, LoaderCircle, Pause, Play, Repeat } from 'lucide-react';
import { SafeZoneMode } from '../types/editor';
import { CANVAS_HEIGHT, CANVAS_WIDTH } from '../utils/canvasRenderer';
import { formatTime } from '../utils/srt';

export type SourceState = 'loading' | 'ready' | 'error';

interface PreviewStageProps {
  canvasRef: React.RefObject<HTMLCanvasElement>;
  safeZone: SafeZoneMode;
  onSafeZoneChange: (mode: SafeZoneMode) => void;
  sourceState: SourceState;
  sourceError: string;
  onRetrySource: () => void;
  isPlaying: boolean;
  loop: boolean;
  currentTime: number;
  duration: number;
  panX: number;
  panY: number;
  onPanChange: (panX: number, panY: number) => void;
  onTogglePlay: () => void;
  onToggleLoop: () => void;
  onSetIn: () => void;
  onSetOut: () => void;
}

export function PreviewStage({
  canvasRef,
  safeZone,
  onSafeZoneChange,
  sourceState,
  sourceError,
  onRetrySource,
  isPlaying,
  loop,
  currentTime,
  duration,
  panX,
  panY,
  onPanChange,
  onTogglePlay,
  onToggleLoop,
  onSetIn,
  onSetOut,
}: PreviewStageProps) {
  const dragStartRef = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);
  const sourceReady = sourceState === 'ready';

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!sourceReady) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    dragStartRef.current = { x: e.clientX, y: e.clientY, panX, panY };
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!dragStartRef.current) return;
    const dx = e.clientX - dragStartRef.current.x;
    const dy = e.clientY - dragStartRef.current.y;
    const nextX = Math.max(-100, Math.min(100, Math.round(dragStartRef.current.panX + dx * 0.45)));
    const nextY = Math.max(-100, Math.min(100, Math.round(dragStartRef.current.panY + dy * 0.45)));
    onPanChange(nextX, nextY);
  };

  const handlePointerUp = () => {
    dragStartRef.current = null;
  };

  return (
    <main className="flex-1 min-w-0 flex flex-col items-center justify-between py-3 px-4 bg-[#0b0d11] relative">
      {/* Safe Zone Guide Switcher */}
      <div className="flex items-center gap-1 bg-[#12151c] border border-slate-800/90 p-1 rounded-lg z-10">
        <span className="text-[11px] font-medium text-slate-400 px-2 whitespace-nowrap">
          Zonas seguras:
        </span>
        {(
          [
            { id: 'none', label: 'Limpias' },
            { id: 'tiktok', label: 'TikTok' },
            { id: 'shorts', label: 'YT Shorts' },
            { id: 'grid', label: 'Rejilla 3×3' },
          ] as Array<{ id: SafeZoneMode; label: string }>
        ).map((sz) => (
          <button
            key={sz.id}
            type="button"
            onClick={() => onSafeZoneChange(sz.id)}
            className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors whitespace-nowrap cursor-pointer ${
              safeZone === sz.id
                ? 'bg-indigo-600 text-white'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {sz.label}
          </button>
        ))}
      </div>

      {/* 9:16 Vertical Viewport */}
      <div className="flex-1 min-h-0 w-full flex items-center justify-center py-3">
        <div
          className="relative h-full max-h-full aspect-[9/16] bg-black rounded-xl overflow-hidden shadow-2xl border border-slate-800"
        >
          <canvas
            ref={canvasRef}
            width={CANVAS_WIDTH}
            height={CANVAS_HEIGHT}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            className={`w-full h-full block ${sourceReady ? 'cursor-grab active:cursor-grabbing' : ''}`}
          />

          {/* Loading / Error Overlay */}
          {!sourceReady && (
            <div className="absolute inset-0 bg-[#0b0d11]/90 flex flex-col items-center justify-center text-center p-6 gap-4 z-10">
              <div className="w-12 h-12 rounded-xl bg-[#191d26] border border-slate-800 flex items-center justify-center">
                {sourceState === 'loading'
                  ? <LoaderCircle className="w-6 h-6 text-indigo-400 animate-spin" />
                  : <AlertCircle className="w-6 h-6 text-rose-400" />}
              </div>
              <div className="space-y-1.5 max-w-[230px]">
                <h3 className="text-sm font-semibold text-slate-100">
                  {sourceState === 'loading' ? 'Cargando el video original…' : 'No se pudo cargar el video original'}
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  {sourceState === 'loading'
                    ? 'El editor trabaja sobre el video fuente del proyecto, no sobre el clip renderizado.'
                    : sourceError || 'Revisa la conexión con el servidor local e inténtalo de nuevo.'}
                </p>
              </div>
              {sourceState === 'error' && (
                <button
                  type="button"
                  onClick={onRetrySource}
                  className="inline-flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 transition-colors cursor-pointer"
                >
                  <Film className="w-3.5 h-3.5" />
                  <span>Reintentar</span>
                </button>
              )}
            </div>
          )}

          {/* Safe Zones Overlay (Never Exported) */}
          {safeZone !== 'none' && (
            <div className="absolute inset-0 pointer-events-none z-20">
              {safeZone === 'grid' && (
                <>
                  <div
                    className="absolute inset-0"
                    style={{
                      backgroundImage:
                        'linear-gradient(to right, rgba(255,255,255,0.16) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,0.16) 1px, transparent 1px)',
                      backgroundSize: '33.333% 33.333%',
                    }}
                  />
                  <div className="absolute left-1/2 top-0 bottom-0 w-px bg-cyan-400/60" />
                </>
              )}

              {safeZone === 'tiktok' && (
                <>
                  <div className="absolute top-0 left-0 right-0 h-[11%] bg-rose-500/20 border-b border-dashed border-rose-400/80 flex items-center justify-center text-[10px] font-semibold text-white/90">
                    TikTok Superior
                  </div>
                  <div className="absolute right-[2%] bottom-[14%] w-[16%] h-[42%] bg-rose-500/20 border border-dashed border-rose-400/80 flex items-center justify-center text-[10px] font-semibold text-white/90 text-center px-1">
                    Botones
                  </div>
                  <div className="absolute left-0 bottom-0 right-[19%] h-[22%] bg-rose-500/20 border-t border-r border-dashed border-rose-400/80 flex items-center justify-center text-[10px] font-semibold text-white/90">
                    Descripción y Audio
                  </div>
                </>
              )}

              {safeZone === 'shorts' && (
                <>
                  <div className="absolute top-0 left-0 right-0 h-[9%] bg-rose-500/20 border-b border-dashed border-rose-400/80 flex items-center justify-center text-[10px] font-semibold text-white/90">
                    Shorts Superior
                  </div>
                  <div className="absolute right-[2%] bottom-[11%] w-[16%] h-[42%] bg-rose-500/20 border border-dashed border-rose-400/80 flex items-center justify-center text-[10px] font-semibold text-white/90 text-center px-1">
                    Acciones
                  </div>
                  <div className="absolute left-0 bottom-0 right-[19%] h-[19%] bg-rose-500/20 border-t border-r border-dashed border-rose-400/80 flex items-center justify-center text-[10px] font-semibold text-white/90">
                    Canal y Título
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Transport Controls */}
      <div className="flex items-center gap-2 bg-[#12151c] border border-slate-800/90 px-3 py-1.5 rounded-lg">
        <button
          type="button"
          onClick={onSetIn}
          disabled={!sourceReady}
          title="Marcar inicio (Tecla I)"
          className="px-2.5 py-1.5 rounded-md text-xs font-medium text-slate-300 hover:text-white hover:bg-[#1e232e] transition-colors whitespace-nowrap cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
        >
          [ In
        </button>

        <button
          type="button"
          onClick={onTogglePlay}
          disabled={!sourceReady}
          className="inline-flex items-center justify-center gap-1.5 min-w-[92px] px-3 py-1.5 rounded-md text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 transition-colors whitespace-nowrap cursor-pointer disabled:opacity-45 disabled:cursor-not-allowed"
        >
          {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
          <span>{isPlaying ? 'Pausa' : 'Play'}</span>
        </button>

        <button
          type="button"
          onClick={onSetOut}
          disabled={!sourceReady}
          title="Marcar fin (Tecla O)"
          className="px-2.5 py-1.5 rounded-md text-xs font-medium text-slate-300 hover:text-white hover:bg-[#1e232e] transition-colors whitespace-nowrap cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Out ]
        </button>

        <span className="px-2 text-xs font-mono-tabular text-slate-400 whitespace-nowrap">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>

        <button
          type="button"
          onClick={onToggleLoop}
          title="Reproducción en bucle"
          className={`inline-flex items-center gap-1 px-2.5 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap cursor-pointer ${
            loop
              ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/40'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Repeat className="w-3.5 h-3.5" />
          <span>Loop</span>
        </button>
      </div>
    </main>
  );
}
