import { useState } from 'react';

interface ExportModalProps {
  open: boolean;
  progress: number;
  statusText: string;
  onCancel: () => void;
}

export function ExportModal({ open, progress, statusText, onCancel }: ExportModalProps) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 bg-black/75 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="w-full max-w-md bg-[#12151c] border border-slate-800 rounded-xl p-6 flex flex-col gap-4 shadow-2xl">
        <div className="space-y-1">
          <h3 className="text-base font-semibold text-slate-100">
            Renderizando Short (1080×1920)
          </h3>
          <p className="text-xs text-slate-400 leading-relaxed font-mono-tabular">{statusText}</p>
        </div>

        <div className="h-2.5 w-full bg-[#0b0d11] border border-slate-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-indigo-500 transition-[width] duration-100 ease-linear"
            style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
          />
        </div>

        <div className="flex justify-end">
          <button
            type="button"
            onClick={onCancel}
            className="px-4 py-2 rounded-lg text-xs font-medium text-rose-300 bg-rose-500/10 border border-rose-500/30 hover:bg-rose-500/20 cursor-pointer"
          >
            Cancelar
          </button>
        </div>
      </div>
    </div>
  );
}

interface ScriptModalProps {
  open: boolean;
  onClose: () => void;
  onGenerate: (scriptText: string, wordsPerBlock: number) => void;
}

export function ScriptModal({ open, onClose, onGenerate }: ScriptModalProps) {
  const [text, setText] = useState('');
  const [wordsPerBlock, setWordsPerBlock] = useState(3);

  if (!open) return null;

  return (
    <div className="fixed inset-0 bg-black/75 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="w-full max-w-lg bg-[#12151c] border border-slate-800 rounded-xl p-6 flex flex-col gap-4 shadow-2xl">
        <div className="space-y-1">
          <h3 className="text-base font-semibold text-slate-100">
            Generar subtítulos desde guion
          </h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Pega tu texto completo para dividirlo automáticamente en bloques sincronizados dentro
            del rango de recorte actual.
          </p>
        </div>

        <textarea
          rows={5}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Ej: ¿Sabías que el 80% de las personas desliza en los primeros 2 segundos? Por eso necesitas subtítulos dinámicos que retengan la atención."
          className="w-full bg-[#0b0d11] border border-slate-800 rounded-lg p-3 text-xs text-slate-100 focus:outline-none focus:border-indigo-500 resize-y"
        />

        <label className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-slate-300">Palabras por bloque</span>
          <select
            value={wordsPerBlock}
            onChange={(e) => setWordsPerBlock(Number(e.target.value))}
            className="w-full bg-[#0b0d11] border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-indigo-500"
          >
            <option value={2}>2 palabras (Ritmo rápido / Hormozi)</option>
            <option value={3}>3 palabras (Equilibrado)</option>
            <option value={4}>4 palabras (Lectura cómoda)</option>
            <option value={6}>6 palabras (Frases completas)</option>
          </select>
        </label>

        <div className="flex justify-end gap-2 pt-1">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-lg text-xs font-medium text-slate-300 bg-[#191d26] border border-slate-800 hover:bg-[#222733] cursor-pointer"
          >
            Cancelar
          </button>
          <button
            type="button"
            onClick={() => {
              if (!text.trim()) return;
              onGenerate(text, wordsPerBlock);
              setText('');
            }}
            className="px-4 py-2 rounded-lg text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 cursor-pointer"
          >
            Generar Subtítulos
          </button>
        </div>
      </div>
    </div>
  );
}
