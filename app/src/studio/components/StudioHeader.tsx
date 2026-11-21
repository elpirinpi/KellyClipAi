// Cabecera del Studio, adaptada del Header de ShortStudio: sin "Subir
// Video" (el video fuente lo pone el proyecto), con volver a resultados,
// título editable del clip y guardado de la edición.
import { ArrowLeft, Check, Download, FileText, LoaderCircle, Music, Save } from 'lucide-react';

export type StudioSaveState = 'idle' | 'dirty' | 'saving' | 'saved' | 'pending-manual';

interface StudioHeaderProps {
  title: string;
  onTitleChange: (title: string) => void;
  saveState: StudioSaveState;
  savedAt: Date | null;
  onBack: () => void;
  onSave: () => void;
  onUploadBgm: () => void;
  onUploadSrt: () => void;
  onExport: () => void;
  isExporting: boolean;
  sourceReady: boolean;
}

export function StudioHeader({
  title,
  onTitleChange,
  saveState,
  savedAt,
  onBack,
  onSave,
  onUploadBgm,
  onUploadSrt,
  onExport,
  isExporting,
  sourceReady,
}: StudioHeaderProps) {
  return (
    <header className="h-14 shrink-0 bg-[#12151c] border-b border-slate-800/90 px-4 flex items-center justify-between gap-4">
      <div className="flex items-center gap-3 min-w-0">
        <button
          type="button"
          onClick={onBack}
          className="inline-flex items-center gap-1.5 px-2.5 py-2 rounded-lg text-xs font-medium text-slate-300 bg-[#191d26] border border-slate-800 hover:bg-[#222733] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Resultados</span>
        </button>
        <span className="text-base font-bold tracking-tight text-slate-100 whitespace-nowrap shrink-0 hidden md:inline">
          Editor visual
        </span>
        <label className="min-w-0 flex items-center gap-2">
          <span className="sr-only">Título del clip</span>
          <input
            type="text"
            value={title}
            maxLength={120}
            onChange={(event) => onTitleChange(event.target.value)}
            placeholder="Título del clip"
            className="w-full min-w-0 max-w-72 bg-[#0b0d11] border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-100 focus:outline-none focus:border-indigo-500"
          />
        </label>
        <span className="text-[11px] text-slate-400 whitespace-nowrap shrink-0 hidden lg:inline" role="status">
          {saveState === 'saving' && <><LoaderCircle size={11} className="inline animate-spin" /> Guardando…</>}
          {saveState === 'dirty' && '● Cambios sin guardar'}
          {saveState === 'saved' && (
            <><Check size={11} className="inline" /> Guardado{savedAt ? ` · ${savedAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}` : ''}</>
          )}
          {saveState === 'pending-manual' && 'Se guardará al exportar el clip'}
        </span>
      </div>

      <div className="flex items-center gap-2 shrink-0">
        <button
          type="button"
          onClick={onUploadBgm}
          className="inline-flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium text-slate-200 bg-[#191d26] border border-slate-800 hover:bg-[#222733] hover:border-slate-700 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
        >
          <Music className="w-3.5 h-3.5 text-slate-400" />
          <span>Música</span>
        </button>

        <button
          type="button"
          onClick={onUploadSrt}
          className="inline-flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium text-slate-200 bg-[#191d26] border border-slate-800 hover:bg-[#222733] hover:border-slate-700 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
        >
          <FileText className="w-3.5 h-3.5 text-slate-400" />
          <span>Importar .SRT</span>
        </button>

        <button
          type="button"
          onClick={onSave}
          disabled={isExporting || saveState === 'saving'}
          className="inline-flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium text-slate-200 bg-[#191d26] border border-slate-800 hover:bg-[#222733] hover:border-slate-700 transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:opacity-45 disabled:cursor-not-allowed"
        >
          {saveState === 'saving' ? <LoaderCircle className="w-3.5 h-3.5 animate-spin" /> : <Save className="w-3.5 h-3.5 text-slate-400" />}
          <span>Guardar edición</span>
        </button>

        <button
          type="button"
          onClick={onExport}
          disabled={isExporting || !sourceReady}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 disabled:opacity-45 disabled:cursor-not-allowed transition-colors whitespace-nowrap shrink-0 cursor-pointer"
        >
          <Download className="w-3.5 h-3.5" />
          <span>Exportar 1080p</span>
        </button>
      </div>
    </header>
  );
}
