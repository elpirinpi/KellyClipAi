import { useEffect, useState } from 'react';
import { BookmarkPlus, LoaderCircle, Trash2, X } from 'lucide-react';
import { apiFetch } from '../auth';
import { API_BASE } from '../context/constants';
import type { ClipResult, ProjectResult, SavedMoment } from '../types';
import { responseError } from '../utils';

/** Biblioteca de momentos guardados (Fase 3, ítem 14): guarda un momento
 * ("cuando hablo de precios") como favorito y aplícalo a otro proyecto
 * como tema de detección. */
export function SavedMomentsPanel({ clip, projectId, onClose, onNotice }: {
  clip?: ClipResult | null;
  projectId?: string;
  onClose: () => void;
  onNotice: (message: string, kind?: 'info' | 'success' | 'error') => void;
}) {
  const [moments, setMoments] = useState<SavedMoment[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [tags, setTags] = useState('');
  const [saving, setSaving] = useState(false);
  const [applyProjectId, setApplyProjectId] = useState('');
  const [projects, setProjects] = useState<ProjectResult[]>([]);
  const [applyingId, setApplyingId] = useState('');

  const refresh = async () => {
    try {
      const response = await apiFetch(`${API_BASE}/api/saved-moments`);
      if (!response.ok) throw new Error(await responseError(response));
      setMoments(await response.json() as SavedMoment[]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'No se pudieron leer los momentos guardados.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let active = true;
    void refresh();
    apiFetch(`${API_BASE}/api/projects?limit=50`)
      .then(async (response) => {
        if (!response.ok) return;
        const result = await response.json() as { projects?: ProjectResult[] } | ProjectResult[];
        if (active) setProjects(Array.isArray(result) ? result : result.projects ?? []);
      })
      .catch(() => undefined);
    return () => { active = false; };
  }, []);

  const save = async () => {
    if (!title.trim()) {
      setError('Escribe un título para el momento (p. ej. "cuando hablo de precios").');
      return;
    }
    setError('');
    setSaving(true);
    try {
      const response = await apiFetch(`${API_BASE}/api/saved-moments`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: title.trim(),
          description: description.trim(),
          project_id: projectId ?? null,
          clip_id: clip?.id ?? null,
          start: clip?.start ?? null,
          end: clip?.end ?? null,
          excerpt: clip?.transcript?.map((word) => word.text).join(' ').slice(0, 500) ?? '',
          tags: tags.split(',').map((tag) => tag.trim()).filter(Boolean),
        }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      setTitle('');
      setDescription('');
      setTags('');
      onNotice('Momento guardado en la biblioteca.', 'success');
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'No se pudo guardar el momento.');
    } finally {
      setSaving(false);
    }
  };

  const remove = async (momentId: string) => {
    try {
      const response = await apiFetch(`${API_BASE}/api/saved-moments/${momentId}`, { method: 'DELETE' });
      if (!response.ok) throw new Error(await responseError(response));
      setMoments((current) => current.filter((moment) => moment.id !== momentId));
    } catch (caught) {
      onNotice(caught instanceof Error ? caught.message : 'No se pudo eliminar el momento.', 'error');
    }
  };

  const apply = async (momentId: string) => {
    if (!applyProjectId) {
      setError('Elige el proyecto al que aplicar el momento.');
      return;
    }
    setError('');
    setApplyingId(momentId);
    try {
      const response = await apiFetch(`${API_BASE}/api/saved-moments/${momentId}/apply`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_id: applyProjectId }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      onNotice('Momento aplicado: el proyecto buscará ese tema al procesarse.', 'success');
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'No se pudo aplicar el momento.');
    } finally {
      setApplyingId('');
    }
  };

  return (
    <div className="pf-modal-overlay" role="dialog" aria-modal="true" aria-label="Momentos guardados">
      <section className="pf-modal pf-modal-wide">
        <header className="pf-modal-heading">
          <div>
            <h2>Momentos guardados</h2>
            <p>Favoritos reutilizables entre proyectos{clip ? ` · clip: ${clip.title}` : ''}</p>
          </div>
          <button type="button" className="pf-modal-close" onClick={onClose} aria-label="Cerrar"><X size={18} /></button>
        </header>
        {error && <p className="error-message" role="alert">{error}</p>}
        <div className="pf-moment-form">
          <label className="config-field"><span>Título del momento</span>
            <input className="text-input" value={title} maxLength={200} onChange={(event) => setTitle(event.target.value)} placeholder='p. ej. "cuando hablo de precios"' />
          </label>
          <label className="config-field"><span>Descripción (opcional)</span>
            <input className="text-input" value={description} maxLength={1000} onChange={(event) => setDescription(event.target.value)} placeholder="Qué buscar en otros videos" />
          </label>
          <label className="config-field"><span>Etiquetas, separadas por comas</span>
            <input className="text-input" value={tags} onChange={(event) => setTags(event.target.value)} placeholder="precios, anécdotas" />
          </label>
          <button type="button" className="create-button" onClick={() => void save()} disabled={saving}>
            {saving ? <LoaderCircle size={15} className="spin" /> : <BookmarkPlus size={15} />}
            {saving ? 'Guardando…' : 'Guardar momento'}
          </button>
        </div>
        <label className="config-field pf-apply-field"><span>Aplicar a proyecto</span>
          <span className="select-wrap"><select value={applyProjectId} onChange={(event) => setApplyProjectId(event.target.value)}>
            <option value="">Elige un proyecto…</option>
            {projects.map((project) => (
              <option key={project.id} value={project.id}>{project.title}</option>
            ))}
          </select></span>
        </label>
        {loading && <p className="results-loading"><LoaderCircle size={16} className="spin" /> Cargando momentos…</p>}
        {!loading && moments.length === 0 && <p className="pf-empty">Aún no hay momentos guardados. Guarda el primero desde un clip.</p>}
        <ul className="pf-moment-list">
          {moments.map((moment) => (
            <li key={moment.id} className="pf-moment-row">
              <div>
                <strong>{moment.title}</strong>
                {moment.description && <p>{moment.description}</p>}
                {moment.tags.length > 0 && <small>{moment.tags.map((tag) => `#${tag}`).join(' ')}</small>}
              </div>
              <div className="pf-moment-actions">
                <button type="button" className="subtle-button" onClick={() => void apply(moment.id)} disabled={applyingId === moment.id || !applyProjectId}>
                  {applyingId === moment.id ? 'Aplicando…' : 'Aplicar'}
                </button>
                <button type="button" className="pf-icon-button" aria-label={`Eliminar ${moment.title}`} onClick={() => void remove(moment.id)}>
                  <Trash2 size={15} />
                </button>
              </div>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
