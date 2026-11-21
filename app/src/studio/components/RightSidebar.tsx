import { AlignLeft, Download, Merge, Plus, Scissors, Trash2 } from 'lucide-react';
import { SubtitleCue } from '../types/editor';
import { formatTime } from '../utils/srt';

interface RightSidebarProps {
  subtitles: SubtitleCue[];
  selectedSubId: number | null;
  currentTime: number;
  duration: number;
  onSelectSub: (id: number) => void;
  onSeekTo: (time: number) => void;
  onAddSub: () => void;
  onOpenScriptModal: () => void;
  onExportSrt: () => void;
  onUpdateSub: (id: number, patch: Partial<SubtitleCue>) => void;
  onSplitSub: (id: number) => void;
  onMergeNextSub: (id: number) => void;
  onDeleteSub: (id: number) => void;
}

export function RightSidebar({
  subtitles,
  selectedSubId,
  currentTime,
  duration,
  onSelectSub,
  onSeekTo,
  onAddSub,
  onOpenScriptModal,
  onExportSrt,
  onUpdateSub,
  onSplitSub,
  onMergeNextSub,
  onDeleteSub,
}: RightSidebarProps) {
  const selectedIdx = subtitles.findIndex((s) => s.id === selectedSubId);
  const selectedSub = selectedIdx >= 0 ? subtitles[selectedIdx] : null;
  const hasNextSub = selectedIdx >= 0 && selectedIdx < subtitles.length - 1;

  const shiftBlock = (delta: number) => {
    if (!selectedSub) return;
    const len = selectedSub.end - selectedSub.start;
    const nextStart = Math.max(0, Math.min(duration - len, selectedSub.start + delta));
    onUpdateSub(selectedSub.id, {
      start: Number(nextStart.toFixed(2)),
      end: Number((nextStart + len).toFixed(2)),
    });
  };

  return (
    <aside className="w-[312px] shrink-0 bg-[#12151c] border-l border-slate-800/90 flex flex-col min-h-0">
      <div className="p-4 border-b border-slate-800/90 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-semibold text-slate-200">Subtítulos Dinámicos</h2>
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-mono-tabular text-slate-400">
              {subtitles.length} bloques
            </span>
            <button
              type="button"
              onClick={onExportSrt}
              disabled={subtitles.length === 0}
              title="Descargar archivo .SRT"
              className="inline-flex items-center gap-1 text-[11px] font-medium text-slate-300 hover:text-white disabled:opacity-40 cursor-pointer"
            >
              <Download className="w-3 h-3" />
              <span>.SRT</span>
            </button>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <button
            type="button"
            onClick={onAddSub}
            className="inline-flex items-center justify-center gap-1.5 py-2 px-2.5 rounded-lg text-xs font-medium text-slate-200 bg-[#191d26] border border-slate-800 hover:bg-[#222733] transition-colors whitespace-nowrap cursor-pointer"
          >
            <Plus className="w-3.5 h-3.5 text-indigo-400" />
            <span>Añadir Bloque</span>
          </button>
          <button
            type="button"
            onClick={onOpenScriptModal}
            className="inline-flex items-center justify-center gap-1.5 py-2 px-2.5 rounded-lg text-xs font-medium text-slate-200 bg-[#191d26] border border-slate-800 hover:bg-[#222733] transition-colors whitespace-nowrap cursor-pointer"
          >
            <AlignLeft className="w-3.5 h-3.5 text-indigo-400" />
            <span>Pegar Guion</span>
          </button>
        </div>
      </div>

      {selectedSub && (
        <div className="p-4 border-b border-slate-800/90 bg-[#151922] flex flex-col gap-2.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-indigo-300">Editar Bloque</span>
            <button
              type="button"
              onClick={() => onDeleteSub(selectedSub.id)}
              className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium text-rose-300 bg-rose-500/10 border border-rose-500/30 hover:bg-rose-500/20 cursor-pointer"
            >
              <Trash2 className="w-3 h-3" />
              <span>Eliminar</span>
            </button>
          </div>

          <textarea
            rows={2}
            value={selectedSub.text}
            onChange={(e) => onUpdateSub(selectedSub.id, { text: e.target.value })}
            placeholder="Escribe el texto del subtítulo..."
            className="w-full bg-[#0b0d11] border border-slate-800 rounded-lg p-2.5 text-xs text-slate-100 focus:outline-none focus:border-indigo-500 resize-y"
          />

          <p className="text-[11px] text-slate-400">
            Consejo: usa <span className="font-mono-tabular text-slate-200">*palabra*</span> para
            resaltar en color secundario.
          </p>

          <div className="grid grid-cols-2 gap-2">
            <label className="flex flex-col gap-1">
              <span className="text-[11px] text-slate-400">Inicio (seg)</span>
              <input
                type="number"
                step={0.1}
                min={0}
                max={Math.max(0, selectedSub.end - 0.2)}
                value={Number(selectedSub.start.toFixed(1))}
                onChange={(e) => {
                  const val = Math.max(
                    0,
                    Math.min(selectedSub.end - 0.2, parseFloat(e.target.value) || 0)
                  );
                  onUpdateSub(selectedSub.id, { start: val });
                }}
                className="w-full bg-[#0b0d11] border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs font-mono-tabular text-slate-100 focus:outline-none focus:border-indigo-500"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-[11px] text-slate-400">Fin (seg)</span>
              <input
                type="number"
                step={0.1}
                min={selectedSub.start + 0.2}
                max={duration}
                value={Number(selectedSub.end.toFixed(1))}
                onChange={(e) => {
                  const val = Math.max(
                    selectedSub.start + 0.2,
                    Math.min(duration, parseFloat(e.target.value) || selectedSub.start + 0.5)
                  );
                  onUpdateSub(selectedSub.id, { end: val });
                }}
                className="w-full bg-[#0b0d11] border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs font-mono-tabular text-slate-100 focus:outline-none focus:border-indigo-500"
              />
            </label>
          </div>

          <div className="grid grid-cols-4 gap-1.5 pt-0.5">
            <button
              type="button"
              onClick={() => shiftBlock(-0.2)}
              title="Adelantar bloque 0.2s"
              className="py-1.5 px-1.5 rounded-md text-[11px] font-mono-tabular bg-[#191d26] border border-slate-800 text-slate-300 hover:bg-[#222733] cursor-pointer"
            >
              -0.2s
            </button>
            <button
              type="button"
              onClick={() => shiftBlock(0.2)}
              title="Retrasar bloque 0.2s"
              className="py-1.5 px-1.5 rounded-md text-[11px] font-mono-tabular bg-[#191d26] border border-slate-800 text-slate-300 hover:bg-[#222733] cursor-pointer"
            >
              +0.2s
            </button>
            <button
              type="button"
              onClick={() => {
                if (currentTime < selectedSub.end - 0.2) {
                  onUpdateSub(selectedSub.id, { start: Number(currentTime.toFixed(2)) });
                }
              }}
              title="Fijar inicio en el cabezal actual"
              className="py-1.5 px-1.5 rounded-md text-[11px] font-medium bg-[#191d26] border border-slate-800 text-slate-300 hover:bg-[#222733] cursor-pointer truncate"
            >
              In aquí
            </button>
            <button
              type="button"
              onClick={() => {
                if (currentTime > selectedSub.start + 0.2) {
                  onUpdateSub(selectedSub.id, { end: Number(currentTime.toFixed(2)) });
                }
              }}
              title="Fijar fin en el cabezal actual"
              className="py-1.5 px-1.5 rounded-md text-[11px] font-medium bg-[#191d26] border border-slate-800 text-slate-300 hover:bg-[#222733] cursor-pointer truncate"
            >
              Out aquí
            </button>
          </div>

          <div className="grid grid-cols-2 gap-1.5">
            <button
              type="button"
              onClick={() => onSplitSub(selectedSub.id)}
              disabled={selectedSub.end - selectedSub.start < 0.6}
              className="inline-flex items-center justify-center gap-1 py-1.5 px-2 rounded-md text-[11px] font-medium bg-[#191d26] border border-slate-800 text-slate-300 hover:bg-[#222733] disabled:opacity-40 cursor-pointer"
            >
              <Scissors className="w-3 h-3" />
              <span>Dividir en 2</span>
            </button>
            <button
              type="button"
              onClick={() => onMergeNextSub(selectedSub.id)}
              disabled={!hasNextSub}
              className="inline-flex items-center justify-center gap-1 py-1.5 px-2 rounded-md text-[11px] font-medium bg-[#191d26] border border-slate-800 text-slate-300 hover:bg-[#222733] disabled:opacity-40 cursor-pointer"
            >
              <Merge className="w-3 h-3" />
              <span>Unir siguiente</span>
            </button>
          </div>
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-2">
        {subtitles.length === 0 ? (
          <div className="py-10 px-4 text-center space-y-2">
            <p className="text-xs font-medium text-slate-300">No hay subtítulos</p>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              Añade un bloque manual, importa un archivo .SRT o pega tu guion para generarlos.
            </p>
          </div>
        ) : (
          subtitles.map((sub) => {
            const isSelected = sub.id === selectedSubId;
            return (
              <button
                key={sub.id}
                type="button"
                onClick={() => {
                  onSelectSub(sub.id);
                  onSeekTo(sub.start + 0.05);
                }}
                className={`w-full text-left p-2.5 rounded-lg border transition-colors flex flex-col gap-1 cursor-pointer ${
                  isSelected
                    ? 'bg-indigo-600/15 border-indigo-500/70'
                    : 'bg-[#171b24] border-slate-800/90 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between text-[11px] font-mono-tabular text-slate-400">
                  <span>
                    {formatTime(sub.start)} → {formatTime(sub.end)}
                  </span>
                  <span>{(sub.end - sub.start).toFixed(1)}s</span>
                </div>
                <div className="text-xs font-medium text-slate-100 truncate">
                  {sub.text.replace(/\*([^*]+)\*/g, '$1')}
                </div>
              </button>
            );
          })
        )}
      </div>
    </aside>
  );
}
