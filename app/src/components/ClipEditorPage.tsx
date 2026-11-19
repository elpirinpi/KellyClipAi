import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertCircle, ArrowLeft, Check, LoaderCircle, Redo2, Scissors, Undo2, Video, X,
} from 'lucide-react';
import { apiFetch } from '../auth';
import { API_BASE, customTemplates, templates, type Template } from '../context/constants';
import { SubtitleStyleControls } from './SubtitleStyleControls';
import type { ClipEditorState, EditablePhrase, ProjectResult } from '../types';
import { formatDuration, responseError } from '../utils';

// Fase 2 (ítem 16): constantes del autoguardado y del historial del editor.
const EDITOR_AUTOSAVE_DELAY_MS = 2000;
const EDITOR_HISTORY_LIMIT = 50;
const EDITOR_HISTORY_COALESCE_MS = 1500;

export function ClipEditorPage({
  projectId,
  clipId,
  project,
  savedTemplates,
  onBack,
}: {
  projectId: string;
  clipId: string;
  project: ProjectResult | null;
  savedTemplates: Template[];
  onBack: () => void;
}) {
  const [editor, setEditor] = useState<ClipEditorState | null>(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [previewing, setPreviewing] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  // Fase 2 (ítem 16): cambios sin guardar, autoguardado y deshacer.
  const [savedEditor, setSavedEditor] = useState<ClipEditorState | null>(null);
  const [autosaving, setAutosaving] = useState(false);
  const [autosavedAt, setAutosavedAt] = useState<Date | null>(null);
  const [past, setPast] = useState<ClipEditorState[]>([]);
  const [future, setFuture] = useState<ClipEditorState[]>([]);
  const editorRef = useRef<ClipEditorState | null>(null);
  const lastHistoryPushRef = useRef<{ phraseId: string; at: number }>({ phraseId: '', at: 0 });
  const player = useRef<HTMLVideoElement>(null);
  const clip = project?.clips.find((item) => item.id === clipId);
  const endpoint = `${API_BASE}/api/projects/${projectId}/clips/${clipId}/edit`;

  const isDirty = useMemo(() => {
    if (!editor || !savedEditor) return false;
    return JSON.stringify(editor) !== JSON.stringify(savedEditor);
  }, [editor, savedEditor]);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError('');
    apiFetch(endpoint, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(await responseError(response));
        return response.json() as Promise<ClipEditorState>;
      })
      .then((loaded) => {
        setEditor(loaded);
        editorRef.current = loaded;
        setSavedEditor(loaded);
        setPast([]);
        setFuture([]);
        setAutosavedAt(null);
      })
      .catch((loadError: unknown) => {
        if (!controller.signal.aborted) {
          setError(loadError instanceof Error ? loadError.message : 'No se pudo cargar el editor.');
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [endpoint]);

  // Aplica un cambio al editor registrando el estado anterior en el
  // historial de deshacer (salvo que se pida lo contrario). Cualquier
  // cambio nuevo invalida el rehacer, aunque se agrupe en el historial.
  const applyEditorChange = useCallback((next: ClipEditorState, recordHistory = true) => {
    const current = editorRef.current;
    if (recordHistory && current) {
      setPast((prev) => [...prev.slice(-(EDITOR_HISTORY_LIMIT - 1)), current]);
    }
    setFuture([]);
    editorRef.current = next;
    setEditor(next);
    setPreviewUrl('');
    setMessage('');
  }, []);

  const updatePhrase = (phraseId: string, update: Partial<EditablePhrase>) => {
    const current = editorRef.current;
    if (!current) return;
    const next: ClipEditorState = {
      ...current,
      phrases: current.phrases.map((phrase) => phrase.id === phraseId ? { ...phrase, ...update } : phrase),
    };
    // Deshacer "por frase": los cambios de texto de la misma frase dentro
    // de 1,5 s se agrupan en un solo paso del historial (si no, cada
    // tecla sería un paso de deshacer).
    const now = Date.now();
    const coalesce = update.text !== undefined
      && lastHistoryPushRef.current.phraseId === phraseId
      && now - lastHistoryPushRef.current.at < EDITOR_HISTORY_COALESCE_MS;
    if (!coalesce) lastHistoryPushRef.current = { phraseId, at: now };
    applyEditorChange(next, !coalesce);
  };

  const undo = useCallback(() => {
    const current = editorRef.current;
    if (!past.length || !current) return;
    const previous = past[past.length - 1];
    setFuture((prevFuture) => [current, ...prevFuture].slice(0, EDITOR_HISTORY_LIMIT));
    setPast(past.slice(0, -1));
    editorRef.current = previous;
    setEditor(previous);
    setPreviewUrl('');
    setMessage('');
  }, [past]);

  const redo = useCallback(() => {
    const current = editorRef.current;
    if (!future.length || !current) return;
    const [next, ...rest] = future;
    setPast((prevPast) => [...prevPast.slice(-(EDITOR_HISTORY_LIMIT - 1)), current]);
    setFuture(rest);
    editorRef.current = next;
    setEditor(next);
    setPreviewUrl('');
    setMessage('');
  }, [future]);

  // Persistencia compartida por el guardado explícito y el autoguardado:
  // el estado del servidor solo sustituye al editor si el usuario no
  // siguió editando mientras se guardaba (si no, se perderían teclas).
  const persistDraft = async (state: ClipEditorState, silent: boolean): Promise<boolean> => {
    try {
      const response = await apiFetch(endpoint, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(state),
      });
      if (!response.ok) throw new Error(await responseError(response));
      const saved = await response.json() as ClipEditorState;
      setSavedEditor(saved);
      if (editorRef.current === state) {
        editorRef.current = saved;
        setEditor(saved);
      }
      if (!silent) setMessage('Borrador guardado en el proyecto.');
      return true;
    } catch (saveError) {
      if (!silent) setError(saveError instanceof Error ? saveError.message : 'No se pudo guardar el borrador.');
      return false;
    }
  };
  const persistDraftRef = useRef(persistDraft);
  persistDraftRef.current = persistDraft;

  const saveDraft = async () => {
    if (!editor || saving) return;
    setSaving(true);
    setError('');
    setMessage('');
    await persistDraft(editor, false);
    setSaving(false);
  };

  // Autoguardado con debounce: 2 s después del último cambio, si hay
  // cambios sin guardar y no hay otra operación en curso. No interfiere
  // con el guardado explícito: comparte la misma persistencia y no toca
  // los mensajes del usuario.
  useEffect(() => {
    if (!editor || !isDirty || saving || previewing || exporting) return;
    const timer = window.setTimeout(() => {
      const state = editorRef.current;
      if (!state) return;
      setAutosaving(true);
      void persistDraftRef.current(state, true).then((ok) => {
        setAutosaving(false);
        if (ok) setAutosavedAt(new Date());
      });
    }, EDITOR_AUTOSAVE_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [editor, isDirty, saving, previewing, exporting]);

  // Aviso al cerrar o recargar la pestaña con cambios sin guardar.
  useEffect(() => {
    if (!isDirty) return;
    const onBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
    };
    window.addEventListener('beforeunload', onBeforeUnload);
    return () => window.removeEventListener('beforeunload', onBeforeUnload);
  }, [isDirty]);

  // Ctrl/Cmd+S: guardar borrador sin abrir el diálogo del navegador.
  // Ctrl/Cmd+Z: deshacer por frase. Ctrl/Mayús+Z o Ctrl+Y: rehacer.
  const saveDraftRef = useRef(saveDraft);
  saveDraftRef.current = saveDraft;
  const undoRef = useRef(undo);
  undoRef.current = undo;
  const redoRef = useRef(redo);
  redoRef.current = redo;
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
        event.preventDefault();
        void saveDraftRef.current();
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
        event.preventDefault();
        if (event.shiftKey) redoRef.current();
        else undoRef.current();
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'y') {
        event.preventDefault();
        redoRef.current();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  const createPreview = async () => {
    if (!editor) return;
    setPreviewing(true);
    setError('');
    setMessage('');
    try {
      const response = await apiFetch(`${endpoint}/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(editor),
      });
      if (!response.ok) throw new Error(await responseError(response));
      const result = await response.json() as { preview_url: string };
      setPreviewUrl(`${API_BASE}${result.preview_url}`);
      setMessage('Vista previa renderizada con tus cambios.');
    } catch (previewError) {
      setError(previewError instanceof Error ? previewError.message : 'No se pudo renderizar la vista previa.');
    } finally {
      setPreviewing(false);
    }
  };

  const exportEdit = async () => {
    if (!editor) return;
    setExporting(true);
    setError('');
    setMessage('');
    try {
      const response = await apiFetch(`${endpoint}/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(editor),
      });
      if (!response.ok) throw new Error(await responseError(response));
      onBack();
    } catch (exportError) {
      setError(exportError instanceof Error ? exportError.message : 'No se pudo exportar el clip.');
    } finally {
      setExporting(false);
    }
  };

  const busy = saving || previewing || exporting;

  return (
    <main className="editor-page" id="main-content">
      <header className="editor-header">
        <button type="button" className="back-button" onClick={onBack}><ArrowLeft size={16} /> Resultados</button>
        <div>
          <span className="eyebrow">EDITOR POR TRANSCRIPCIÓN</span>
          <h1>{clip ? clip.title : loading ? 'Cargando clip…' : 'No se encontró el clip'}</h1>
          <p>Quita frases para recortar el audio y el video; ajusta subtítulos antes de exportar.</p>
          {/* Fase 2 (ítem 16): indicador de cambios sin guardar. */}
          {editor && !loading && (
            <p className="editor-dirty-status" role="status">
              {autosaving
                ? <><LoaderCircle size={13} className="spin" /> Guardando…</>
                : isDirty
                  ? <><span className="editor-dirty-dot" aria-hidden="true" /> Cambios sin guardar</>
                  : autosavedAt
                    ? <><Check size={13} /> Guardado automáticamente · {autosavedAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</>
                    : <><Check size={13} /> Sin cambios pendientes</>}
            </p>
          )}
        </div>
      </header>

      {error && <p className="error-message" role="alert"><AlertCircle size={16} />{error}</p>}
      {message && <p className="editor-message" role="status"><Check size={16} />{message}</p>}
      {loading && <p className="results-loading"><LoaderCircle size={17} className="spin" /> Cargando transcripción guardada…</p>}
      {!loading && editor && clip && (
        <>
          <section className="editor-workspace">
            <div className="editor-preview-card">
              <div className="editor-preview-frame">
                <video
                  key={previewUrl || clip.video_url || clip.id}
                  ref={player}
                  src={previewUrl || (clip.video_url ? `${API_BASE}${clip.video_url}` : undefined)}
                  controls
                  playsInline
                  preload="metadata"
                  aria-label={`Vista previa de ${clip.title}`}
                />
              </div>
              <p>{previewUrl ? 'Vista previa de los cambios' : 'Clip exportado actualmente'}</p>
            </div>

            <div className="editor-settings">
              <label className="config-field">
                <span>Ratio de salida</span>
                <select value={editor.ratio} onChange={(event) => {
                  applyEditorChange({ ...editor, ratio: event.target.value });
                }}>
                  <option value="9:16">9:16 · Vertical</option>
                  <option value="16:9">16:9 · Horizontal</option>
                  <option value="1:1">1:1 · Cuadrado</option>
                  <option value="4:5">4:5 · Retrato</option>
                </select>
              </label>
              <label className="config-field">
                <span>Plantilla de subtítulos</span>
                <select value={editor.subtitle_template} onChange={(event) => {
                  applyEditorChange({ ...editor, subtitle_template: event.target.value });
                }}>
                  {[...templates, ...customTemplates, ...savedTemplates].map((template) => (
                    <option key={template.name} value={template.name}>{template.name}</option>
                  ))}
                </select>
              </label>
              <SubtitleStyleControls
                value={editor.subtitle_style}
                onChange={(subtitle_style) => {
                  applyEditorChange({ ...editor, subtitle_style });
                }}
              />
              <div className="editor-tip">
                <strong>{editor.phrases.filter((phrase) => phrase.removed).length} frases marcadas</strong>
                <span>Los cambios se aplican al exportar desde el video original.</span>
              </div>
            </div>
          </section>

          <section className="editor-transcript-card">
            <div className="editor-section-heading">
              <div><h2>Transcripción</h2><p>Edita el texto que aparecerá como subtítulo o quita una frase completa.</p></div>
              <span>{editor.phrases.length} frases</span>
            </div>
            {editor.phrases.length ? (
              <div className="editor-phrases">
                {editor.phrases.map((phrase, index) => (
                  <article className={`editor-phrase ${phrase.removed ? 'is-removed' : ''}`} key={phrase.id}>
                    <button
                      type="button"
                      className="editor-phrase-time"
                      onClick={() => {
                        if (!phrase.removed && player.current) {
                          player.current.currentTime = phrase.preview_start;
                          void player.current.play();
                        }
                      }}
                      disabled={phrase.removed}
                      aria-label={`Reproducir frase ${index + 1} desde ${formatDuration(phrase.preview_start)}`}
                    >
                      {formatDuration(phrase.preview_start)}
                    </button>
                    <label className="editor-phrase-text">
                      <span className="visually-hidden">Texto del subtítulo {index + 1}</span>
                      <textarea
                        rows={2}
                        maxLength={400}
                        value={phrase.text}
                        disabled={phrase.removed}
                        onChange={(event) => updatePhrase(phrase.id, { text: event.target.value })}
                      />
                    </label>
                    <button
                      type="button"
                      className="editor-phrase-toggle"
                      onClick={() => updatePhrase(phrase.id, { removed: !phrase.removed })}
                    >
                      {phrase.removed ? <><ArrowLeft size={15} /> Restaurar</> : <><X size={15} /> Quitar</>}
                    </button>
                  </article>
                ))}
              </div>
            ) : (
              <p className="empty-templates">Este clip no tiene palabras con marcas de tiempo para editar.</p>
            )}
          </section>

          <footer className="editor-actions">
            <div className="editor-actions-left">
              <button type="button" className="subtle-button" onClick={() => void saveDraft()} disabled={busy}>
                {saving ? <LoaderCircle size={16} className="spin" /> : <Check size={16} />}
                Guardar borrador
              </button>
              {/* Fase 2 (ítem 16): deshacer/rehacer por frase. */}
              <button type="button" className="subtle-button" onClick={undo} disabled={!past.length || busy} aria-label="Deshacer (Ctrl+Z)" title="Deshacer (Ctrl+Z)">
                <Undo2 size={16} /> Deshacer
              </button>
              <button type="button" className="subtle-button" onClick={redo} disabled={!future.length || busy} aria-label="Rehacer (Ctrl+Mayús+Z)" title="Rehacer (Ctrl+Mayús+Z)">
                <Redo2 size={16} /> Rehacer
              </button>
            </div>
            <div>
              <button type="button" className="outline-action" onClick={() => void createPreview()} disabled={busy}>
                {previewing ? <LoaderCircle size={16} className="spin" /> : <Video size={16} />}
                Vista previa
              </button>
              <button type="button" className="create-button" onClick={() => void exportEdit()} disabled={busy}>
                {exporting ? <LoaderCircle size={17} className="spin" /> : <Scissors size={17} />}
                Exportar cambios
              </button>
            </div>
          </footer>
        </>
      )}
    </main>
  );
}
