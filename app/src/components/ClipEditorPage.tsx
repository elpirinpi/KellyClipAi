import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertCircle, ArrowLeft, Check, LoaderCircle, Pause, Play, Redo2, RotateCcw, Scissors, Undo2, X,
} from 'lucide-react';
import { apiFetch } from '../auth';
import { API_BASE, customTemplates, templates, type Template } from '../context/constants';
import { SubtitleStyleControls } from './SubtitleStyleControls';
import type { ClipEditorState, EditablePhrase, ProjectResult } from '../types';
import { formatDuration, responseError } from '../utils';
import { useCanvasPreview } from '../studio/hooks/useCanvasPreview';
import { DEFAULT_SETTINGS } from '../studio/types/editor';
import { CANVAS_HEIGHT, CANVAS_WIDTH } from '../studio/utils/canvasRenderer';
import {
  mapClipEditorToSettings,
  mapPhrasesToCues,
  removedRangesAbsolute,
} from '../studio/utils/clipEditorMapping';

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
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
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
  const clip = project?.clips.find((item) => item.id === clipId);
  const endpoint = `${API_BASE}/api/projects/${projectId}/clips/${clipId}/edit`;

  // Vista previa en vivo (v1.3.0): el estado del editor se traduce a los
  // ajustes del renderizador canvas de ShortStudio y las frases a cues en
  // tiempo absoluto del video fuente (ver clipEditorMapping.ts); el
  // canvas se redibuja en cada frame y con cada cambio, sin re-render del
  // backend. El endpoint POST .../edit/preview ya no se llama desde esta
  // página (el backend lo conserva; ver el informe de la vista previa).
  const allTemplates = useMemo(
    () => [...templates, ...customTemplates, ...savedTemplates],
    [savedTemplates],
  );
  const liveSettings = useMemo(
    () => (editor && clip ? mapClipEditorToSettings(editor, clip, allTemplates) : DEFAULT_SETTINGS),
    [editor, clip, allTemplates],
  );
  const liveCues = useMemo(
    () => (editor && clip ? mapPhrasesToCues(editor.phrases, clip.start) : []),
    [editor, clip],
  );
  const liveSkipRanges = useMemo(
    () => (editor && clip ? removedRangesAbsolute(editor.phrases, clip.start) : []),
    [editor, clip],
  );
  const preview = useCanvasPreview({
    enabled: Boolean(editor && clip),
    settings: liveSettings,
    cues: liveCues,
    skipRanges: liveSkipRanges,
    resetKey: clipId,
  });
  // Rango recortado de la vista previa: el clip dentro del video fuente.
  const trimStart = clip?.start ?? 0;
  const trimEnd = clip?.end ?? 0;
  const trimDuration = Math.max(0, trimEnd - trimStart);
  const previewProgress = Math.max(0, Math.min(trimDuration, preview.currentTime - trimStart));

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
    if (!editor || !isDirty || saving || exporting) return;
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
  }, [editor, isDirty, saving, exporting]);

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

  const busy = saving || exporting;

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
              {/* El video fuente se sirve con la cookie de medios; el
                atributo crossOrigin con credenciales mantiene el canvas
                sin contaminar (mismo patrón que StudioPage). El <video>
                queda oculto: solo alimenta el canvas y el audio. */}
              <video
                ref={preview.videoRef}
                src={`${API_BASE}/api/projects/${projectId}/source`}
                style={{ display: 'none' }}
                playsInline
                preload="auto"
                crossOrigin="use-credentials"
                onLoadedMetadata={preview.handleLoadedMetadata}
                onError={preview.handleVideoError}
                onLoadedData={preview.handleLoadedData}
                onSeeked={preview.handleSeeked}
              />
              <div className="editor-preview-frame">
                <canvas
                  ref={preview.canvasRef}
                  width={CANVAS_WIDTH}
                  height={CANVAS_HEIGHT}
                  aria-label={`Vista previa en vivo de ${clip.title}`}
                />
                {preview.sourceState !== 'ready' && (
                  <div className="editor-preview-overlay" role="status">
                    {preview.sourceState === 'loading'
                      ? <LoaderCircle size={22} className="spin" />
                      : <AlertCircle size={22} />}
                    <p>
                      {preview.sourceState === 'loading'
                        ? 'Cargando el video original…'
                        : 'No se pudo cargar el video original.'}
                    </p>
                    {preview.sourceState === 'error' && (
                      <button type="button" className="editor-preview-retry" onClick={preview.retrySource}>
                        <RotateCcw size={14} /> Reintentar
                      </button>
                    )}
                  </div>
                )}
              </div>
              <p className="editor-preview-live">
                <span className="editor-live-dot" aria-hidden="true" /> Vista previa en vivo
              </p>
              <div className="editor-preview-transport">
                <button
                  type="button"
                  className="editor-preview-play"
                  onClick={preview.togglePlay}
                  disabled={preview.sourceState !== 'ready'}
                  aria-label={preview.isPlaying ? 'Pausar vista previa' : 'Reproducir vista previa'}
                >
                  {preview.isPlaying ? <Pause size={16} /> : <Play size={16} />}
                </button>
                <input
                  type="range"
                  min={0}
                  max={trimDuration}
                  step={0.1}
                  value={previewProgress}
                  disabled={preview.sourceState !== 'ready'}
                  onChange={(event) => preview.seekTo(trimStart + Number(event.target.value))}
                  aria-label="Posición en la vista previa"
                />
                <span>{formatDuration(previewProgress)} / {formatDuration(trimDuration)}</span>
              </div>
              <p className="editor-preview-note">Aproximación en vivo: el resultado final se genera al exportar.</p>
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
                  {allTemplates.map((template) => (
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
                        // La vista previa en vivo reproduce el video
                        // FUENTE: la frase empieza en tiempo absoluto
                        // (inicio del clip + inicio relativo de la frase).
                        // Se muestra el inicio relativo, la posición de
                        // la frase dentro del clip.
                        if (!phrase.removed) preview.playFrom(clip.start + phrase.start);
                      }}
                      disabled={phrase.removed}
                      aria-label={`Reproducir frase ${index + 1} desde ${formatDuration(phrase.start)}`}
                    >
                      {formatDuration(phrase.start)}
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
