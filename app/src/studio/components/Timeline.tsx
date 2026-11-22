import React, { useEffect, useRef } from 'react';
import { SubtitleCue } from '../types/editor';

// Ventana de gancho (Fase 3, ítem 8): los primeros 3 s del clip.
const HOOK_SECONDS = 3;

interface TimelineProps {
  duration: number;
  currentTime: number;
  trimStart: number;
  trimEnd: number;
  speed: number;
  hasBgm: boolean;
  bgmName: string;
  subtitles: SubtitleCue[];
  selectedSubId: number | null;
  // Picos de la forma de onda (Fase 3, ítem 4b), normalizados 0-1,
  // cubriendo [0, duration). Null mientras se cargan o si fallaron.
  waveform?: number[] | null;
  onSeek: (time: number) => void;
  onSelectSub: (id: number) => void;
  onMoveSub: (id: number, start: number, end: number) => void;
}

// Lienzo de la forma de onda: una barra por pico, centrada
// verticalmente. Las barras dentro del recorte van en índigo y las de
// fuera en gris, para ver de un vistazo qué parte del audio se usa.
function WaveformCanvas({
  waveform,
  duration,
  trimStart,
  trimEnd,
}: {
  waveform: number[];
  duration: number;
  trimStart: number;
  trimEnd: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    if (!waveform.length || duration <= 0) return;
    const barWidth = width / waveform.length;
    waveform.forEach((peak, index) => {
      const time = (index / waveform.length) * duration;
      const inTrim = time >= trimStart && time < trimEnd;
      const barHeight = Math.max(2, peak * (height - 4));
      const x = index * barWidth;
      const y = (height - barHeight) / 2;
      ctx.fillStyle = inTrim ? '#818cf8' : '#475569';
      ctx.fillRect(x, y, Math.max(1, barWidth - 0.5), barHeight);
    });
  }, [waveform, duration, trimStart, trimEnd]);

  return (
    <canvas
      ref={canvasRef}
      width={1200}
      height={28}
      className="absolute inset-0 h-full w-full"
      aria-hidden="true"
    />
  );
}

export function Timeline({
  duration,
  currentTime,
  trimStart,
  trimEnd,
  speed,
  hasBgm,
  bgmName,
  subtitles,
  selectedSubId,
  waveform,
  onSeek,
  onSelectSub,
  onMoveSub,
}: TimelineProps) {
  const tracksRef = useRef<HTMLDivElement | null>(null);
  const scrubbingRef = useRef(false);
  const dragCueRef = useRef<{
    id: number;
    startX: number;
    origStart: number;
    origEnd: number;
    moved: boolean;
  } | null>(null);

  const safeDur = Math.max(0.1, duration);
  const playheadPct = Math.max(0, Math.min(100, (currentTime / safeDur) * 100));
  const trimLeftPct = (trimStart / safeDur) * 100;
  const trimWidthPct = Math.max(0.5, ((trimEnd - trimStart) / safeDur) * 100);
  const exportDur = Math.max(0, trimEnd - trimStart) / speed;
  // Ventana de gancho: los primeros 3 s del recorte (o menos, si el
  // clip es más corto). Se marca en ámbar sobre la pista de video.
  const hookEnd = Math.min(trimStart + HOOK_SECONDS, trimEnd);
  const hookWidthPct = Math.max(0, ((hookEnd - trimStart) / safeDur) * 100);

  const scrubFromClientX = (clientX: number) => {
    const el = tracksRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    onSeek(ratio * duration);
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    scrubbingRef.current = true;
    scrubFromClientX(e.clientX);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!scrubbingRef.current) return;
    scrubFromClientX(e.clientX);
  };

  const handlePointerUp = () => {
    scrubbingRef.current = false;
  };

  const handleCuePointerDown = (e: React.PointerEvent<HTMLButtonElement>, sub: SubtitleCue) => {
    e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);
    onSelectSub(sub.id);
    dragCueRef.current = {
      id: sub.id,
      startX: e.clientX,
      origStart: sub.start,
      origEnd: sub.end,
      moved: false,
    };
  };

  const handleCuePointerMove = (e: React.PointerEvent<HTMLButtonElement>) => {
    const drag = dragCueRef.current;
    const el = tracksRef.current;
    if (!drag || !el) return;
    const dx = e.clientX - drag.startX;
    if (Math.abs(dx) > 3) drag.moved = true;
    if (!drag.moved) return;

    const width = Math.max(1, el.getBoundingClientRect().width);
    const deltaSec = (dx / width) * safeDur;
    const cueLen = drag.origEnd - drag.origStart;
    const nextStart = Math.max(0, Math.min(safeDur - cueLen, drag.origStart + deltaSec));
    onMoveSub(drag.id, Number(nextStart.toFixed(2)), Number((nextStart + cueLen).toFixed(2)));
  };

  const handleCuePointerUp = (_e: React.PointerEvent<HTMLButtonElement>, sub: SubtitleCue) => {
    const drag = dragCueRef.current;
    dragCueRef.current = null;
    if (drag && !drag.moved) {
      onSeek(sub.start + 0.05);
    }
  };

  return (
    <footer className="h-[188px] shrink-0 bg-[#12151c] border-t border-slate-800/90 flex flex-col">
      <div className="h-8 px-4 border-b border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
        <span>Línea de tiempo (arrastra los bloques de subtítulos para sincronizarlos)</span>
        <div className="flex items-center gap-2 font-mono-tabular">
          <span>Duración final: {exportDur.toFixed(1)}s</span>
          <span aria-hidden="true">·</span>
          <span className={exportDur > 60 ? 'text-amber-400 font-medium' : 'text-slate-400'}>
            {exportDur > 60 ? 'Supera 60s (límite Shorts)' : 'Apto para Shorts y TikTok'}
          </span>
        </div>
      </div>

      <div className="flex-1 px-4 py-2.5 overflow-hidden">
        <div
          ref={tracksRef}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          className="relative h-full w-full flex flex-col justify-center gap-2 cursor-pointer"
        >
          {/* Video Trim Track */}
          <div className="h-7 bg-[#0b0d11] border border-slate-800 rounded-md relative overflow-hidden">
            <span className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[10px] font-semibold text-slate-400 z-10 pointer-events-none">
              Video
            </span>
            <div
              className="absolute top-0 bottom-0 bg-indigo-500/20 border-l-2 border-r-2 border-indigo-400"
              style={{ left: `${trimLeftPct}%`, width: `${trimWidthPct}%` }}
            />
            {hookWidthPct > 0 && (
              <div
                className="absolute top-0 bottom-0 bg-amber-400/25 border-r border-amber-400/70 pointer-events-none"
                style={{ left: `${trimLeftPct}%`, width: `${hookWidthPct}%` }}
                title="Ventana de gancho: los primeros 3 segundos del clip"
              >
                <span className="absolute left-1 top-1/2 -translate-y-1/2 text-[9px] font-bold text-amber-300 whitespace-nowrap">
                  Gancho 3s
                </span>
              </div>
            )}
          </div>

          {/* Audio Waveform Track */}
          <div className="h-7 bg-[#0b0d11] border border-slate-800 rounded-md relative overflow-hidden">
            <span className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[10px] font-semibold text-slate-400 z-10 pointer-events-none">
              Audio
            </span>
            {waveform && waveform.length > 0 ? (
              <WaveformCanvas
                waveform={waveform}
                duration={safeDur}
                trimStart={trimStart}
                trimEnd={trimEnd}
              />
            ) : (
              <span className="absolute left-16 top-1/2 -translate-y-1/2 text-[10px] text-slate-600 pointer-events-none">
                {waveform === null ? 'Forma de onda no disponible' : 'Cargando forma de onda…'}
              </span>
            )}
          </div>

          {/* Subtitles Track */}
          <div className="h-7 bg-[#0b0d11] border border-slate-800 rounded-md relative overflow-hidden">
            <span className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[10px] font-semibold text-slate-400 z-10 pointer-events-none">
              Subtítulos
            </span>
            {subtitles.map((sub) => {
              const left = (sub.start / safeDur) * 100;
              const width = Math.max(1.4, ((sub.end - sub.start) / safeDur) * 100);
              const isSelected = sub.id === selectedSubId;
              return (
                <button
                  key={sub.id}
                  type="button"
                  onPointerDown={(e) => handleCuePointerDown(e, sub)}
                  onPointerMove={handleCuePointerMove}
                  onPointerUp={(e) => handleCuePointerUp(e, sub)}
                  style={{ left: `${Math.max(0, left)}%`, width: `${width}%` }}
                  className={`absolute top-1 bottom-1 rounded text-[10px] font-medium px-1.5 flex items-center overflow-hidden whitespace-nowrap cursor-grab active:cursor-grabbing transition-colors ${
                    isSelected
                      ? 'bg-amber-400 text-slate-950 z-20 ring-1 ring-white'
                      : 'bg-sky-600/85 text-white hover:bg-sky-500 z-10'
                  }`}
                >
                  <span className="truncate">{sub.text.replace(/\*([^*]+)\*/g, '$1')}</span>
                </button>
              );
            })}
          </div>

          {/* BGM Track */}
          <div className="h-7 bg-[#0b0d11] border border-slate-800 rounded-md relative overflow-hidden">
            <span className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[10px] font-semibold text-slate-400 z-10 pointer-events-none">
              Música
            </span>
            {hasBgm && (
              <div className="absolute inset-1 bg-emerald-500/20 border border-emerald-500/50 rounded flex items-center pl-16 pr-2 text-[10px] text-emerald-200 truncate">
                {bgmName}
              </div>
            )}
          </div>

          {/* Playhead */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-rose-500 z-30 pointer-events-none"
            style={{ left: `${playheadPct}%` }}
          >
            <div className="w-2.5 h-2.5 bg-rose-500 rotate-45 -translate-x-1 -translate-y-1" />
          </div>
        </div>
      </div>
    </footer>
  );
}
