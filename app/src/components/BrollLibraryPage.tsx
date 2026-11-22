import { useCallback, useEffect, useState } from 'react';
import { Download, Film, LoaderCircle, Search, Star, Trash2, X } from 'lucide-react';
import { apiFetch } from '../auth';
import { API_BASE } from '../context/constants';
import type { BrollItem, BrollSearchResult, ClipResult, ProjectResult } from '../types';
import { responseError } from '../utils';

/** Biblioteca de B-roll (Fase 3, ítem 13): caché local con búsqueda,
 * favoritos y reutilización; B-roll por clip o por segmento; búsqueda en
 * Pexels/Pixabay detrás de claves API opcionales (sin claves, solo la
 * biblioteca local). */
export function BrollLibraryPage({ onNotice }: {
  onNotice: (message: string, kind?: 'info' | 'success' | 'error') => void;
}) {
  const [items, setItems] = useState<BrollItem[]>([]);
  const [query, setQuery] = useState('');
  const [favoritesOnly, setFavoritesOnly] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchProvider, setSearchProvider] = useState<'pexels' | 'pixabay'>('pexels');
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState('');
  const [results, setResults] = useState<BrollSearchResult[]>([]);
  const [importing, setImporting] = useState('');
  const [assignItem, setAssignItem] = useState<BrollItem | null>(null);
  const [projects, setProjects] = useState<ProjectResult[]>([]);
  const [projectId, setProjectId] = useState('');
  const [projectClips, setProjectClips] = useState<ClipResult[]>([]);
  const [clipId, setClipId] = useState('');
  const [segmentStart, setSegmentStart] = useState('');
  const [segmentEnd, setSegmentEnd] = useState('');
  const [segmentLabel, setSegmentLabel] = useState('');
  const [assigning, setAssigning] = useState(false);
  const [previewItem, setPreviewItem] = useState<BrollItem | null>(null);

  const refresh = useCallback(async () => {
    try {
      const params = new URLSearchParams();
      if (query.trim()) params.set('q', query.trim());
      if (favoritesOnly) params.set('favorites', 'true');
      const response = await apiFetch(`${API_BASE}/api/broll?${params.toString()}`);
      if (!response.ok) throw new Error(await responseError(response));
      setItems(await response.json() as BrollItem[]);
      setError('');
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'No se pudo leer la biblioteca de B-roll.');
    } finally {
      setLoading(false);
    }
  }, [query, favoritesOnly]);

  useEffect(() => { void refresh(); }, [refresh]);

  useEffect(() => {
    apiFetch(`${API_BASE}/api/projects?limit=50`)
      .then(async (response) => {
        if (!response.ok) return;
        const result = await response.json() as { projects?: ProjectResult[] } | ProjectResult[];
        setProjects(Array.isArray(result) ? result : result.projects ?? []);
      })
      .catch(() => undefined);
  }, []);

  const loadProjectClips = async (id: string) => {
    setProjectId(id);
    setClipId('');
    setProjectClips([]);
    if (!id) return;
    try {
      const response = await apiFetch(`${API_BASE}/api/projects/${id}`);
      if (!response.ok) throw new Error(await responseError(response));
      const project = await response.json() as ProjectResult;
      setProjectClips(project.clips ?? []);
    } catch (caught) {
      onNotice(caught instanceof Error ? caught.message : 'No se pudieron leer los clips del proyecto.', 'error');
    }
  };

  const runProviderSearch = async () => {
    if (!searchQuery.trim()) {
      setSearchError('Escribe qué B-roll quieres buscar.');
      return;
    }
    setSearchError('');
    setSearching(true);
    try {
      const response = await apiFetch(`${API_BASE}/api/broll/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider: searchProvider, query: searchQuery.trim(), per_page: 6 }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      const result = await response.json() as { results: BrollSearchResult[] };
      setResults(result.results);
      if (result.results.length === 0) setSearchError('El proveedor no encontró videos para esa búsqueda.');
    } catch (caught) {
      setSearchError(caught instanceof Error ? caught.message : 'No se pudo buscar en el proveedor.');
    } finally {
      setSearching(false);
    }
  };

  const importResult = async (result: BrollSearchResult) => {
    setImporting(result.download_url);
    try {
      const response = await apiFetch(`${API_BASE}/api/broll/import`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: result.provider.toLowerCase(),
          external_id: result.external_id,
          title: result.title,
          query: searchQuery.trim(),
          creator: result.creator,
          attribution_url: result.attribution_url,
          preview_url: result.preview_url,
          download_url: result.download_url,
          duration: result.duration,
          width: result.width,
          height: result.height,
        }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      onNotice('B-roll guardado en la biblioteca.', 'success');
      await refresh();
    } catch (caught) {
      onNotice(caught instanceof Error ? caught.message : 'No se pudo importar el B-roll.', 'error');
    } finally {
      setImporting('');
    }
  };

  const toggleFavorite = async (item: BrollItem) => {
    try {
      const response = await apiFetch(`${API_BASE}/api/broll/${item.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ favorite: !item.favorite }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      await refresh();
    } catch (caught) {
      onNotice(caught instanceof Error ? caught.message : 'No se pudo cambiar el favorito.', 'error');
    }
  };

  const removeItem = async (item: BrollItem) => {
    try {
      const response = await apiFetch(`${API_BASE}/api/broll/${item.id}`, { method: 'DELETE' });
      if (!response.ok) throw new Error(await responseError(response));
      onNotice('B-roll eliminado de la biblioteca.', 'success');
      await refresh();
    } catch (caught) {
      onNotice(caught instanceof Error ? caught.message : 'No se pudo eliminar el B-roll.', 'error');
    }
  };

  const assign = async () => {
    if (!assignItem || !projectId || !clipId) return;
    const start = segmentStart.trim() === '' ? 0 : Number(segmentStart);
    const end = segmentEnd.trim() === '' ? 0 : Number(segmentEnd);
    if (Number.isNaN(start) || Number.isNaN(end) || (start === 0) !== (end === 0) || (end > 0 && end <= start)) {
      onNotice('El segmento necesita inicio y fin válidos, o ambos vacíos para todo el clip.', 'error');
      return;
    }
    setAssigning(true);
    try {
      const response = await apiFetch(`${API_BASE}/api/broll/assign`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          broll_id: assignItem.id,
          project_id: projectId,
          clip_id: clipId,
          start,
          end,
          label: segmentLabel.trim(),
        }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      onNotice(end > 0 ? 'B-roll asignado al segmento del clip.' : 'B-roll asignado a todo el clip.', 'success');
      setAssignItem(null);
      setSegmentStart('');
      setSegmentEnd('');
      setSegmentLabel('');
    } catch (caught) {
      onNotice(caught instanceof Error ? caught.message : 'No se pudo asignar el B-roll.', 'error');
    } finally {
      setAssigning(false);
    }
  };

  return (
    <main className="pf-page" id="main-content">
      <header className="pf-page-heading">
        <div>
          <span className="eyebrow">BIBLIOTECA</span>
          <h1><Film size={22} /> Biblioteca de B-roll</h1>
          <p>Videos reutilizables en tus clips. Asigna uno a un clip entero o a un segmento concreto; al renderizar, el B-roll asignado gana a la descarga automática.</p>
        </div>
      </header>
      {error && <p className="error-message" role="alert">{error}</p>}
      <section className="config-card">
        <div className="card-heading"><div><h2>Buscar en Pexels / Pixabay</h2><p>Necesita la clave API del proveedor en .env (PEXELS_API_KEY o PIXABAY_API_KEY). Sin claves, la biblioteca local sigue funcionando.</p></div></div>
        <div className="pf-provider-search">
          <span className="select-wrap"><select value={searchProvider} onChange={(event) => setSearchProvider(event.target.value as 'pexels' | 'pixabay')} aria-label="Proveedor de B-roll">
            <option value="pexels">Pexels</option>
            <option value="pixabay">Pixabay</option>
          </select></span>
          <input className="text-input" value={searchQuery} onChange={(event) => setSearchQuery(event.target.value)} placeholder="p. ej. ciudad de noche" aria-label="Buscar B-roll en el proveedor" />
          <button type="button" className="create-button" onClick={() => void runProviderSearch()} disabled={searching}>
            {searching ? <LoaderCircle size={15} className="spin" /> : <Search size={15} />} Buscar
          </button>
        </div>
        {searchError && <p className="error-message" role="alert">{searchError}</p>}
        {results.length > 0 && (
          <ul className="pf-search-results">
            {results.map((result) => (
              <li key={`${result.provider}-${result.external_id}`} className="pf-search-result">
                {result.preview_url ? <img src={result.preview_url} alt="" loading="lazy" /> : <span className="pf-search-thumb"><Film size={22} /></span>}
                <div>
                  <strong>{result.title}</strong>
                  <small>{result.creator} · {result.provider}{result.duration > 0 ? ` · ${result.duration.toFixed(0)} s` : ''}</small>
                </div>
                <button type="button" className="subtle-button" onClick={() => void importResult(result)} disabled={importing === result.download_url}>
                  {importing === result.download_url ? <LoaderCircle size={15} className="spin" /> : <Download size={15} />}
                  Guardar
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
      <section className="config-card">
        <div className="card-heading"><div><h2>Mi biblioteca</h2><p>{items.length} video{items.length === 1 ? '' : 's'} guardados en este dispositivo.</p></div></div>
        <div className="pf-library-filters">
          <label className="results-search">
            <Search size={16} aria-hidden="true" />
            <span className="visually-hidden">Buscar en la biblioteca</span>
            <input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Buscar por título, tema o autor…" autoComplete="off" />
          </label>
          <button type="button" className={`subtle-button ${favoritesOnly ? 'pf-active-filter' : ''}`} onClick={() => setFavoritesOnly((value) => !value)}>
            <Star size={15} /> Favoritos
          </button>
        </div>
        {loading && <p className="results-loading"><LoaderCircle size={17} className="spin" /> Cargando biblioteca…</p>}
        {!loading && items.length === 0 && <p className="pf-empty">La biblioteca está vacía. Busca arriba en Pexels o Pixabay y guarda los videos que quieras reutilizar.</p>}
        <ul className="pf-broll-grid">
          {items.map((item) => (
            <li key={item.id} className="pf-broll-card">
              <button type="button" className="pf-broll-preview" onClick={() => setPreviewItem(item)} aria-label={`Ver ${item.title}`}>
                {item.preview_url ? <img src={item.preview_url} alt="" loading="lazy" /> : <span><Film size={26} /></span>}
              </button>
              <div className="pf-broll-info">
                <strong title={item.title}>{item.title}</strong>
                <small>{item.creator} · {item.provider}{item.duration > 0 ? ` · ${item.duration.toFixed(0)} s` : ''}</small>
                {item.attribution_url && <a href={item.attribution_url} target="_blank" rel="noreferrer">Fuente</a>}
              </div>
              <div className="pf-broll-actions">
                <button type="button" className={`pf-icon-button ${item.favorite ? 'is-favorite' : ''}`} aria-label={item.favorite ? 'Quitar de favoritos' : 'Marcar favorito'} onClick={() => void toggleFavorite(item)}>
                  <Star size={16} fill={item.favorite ? 'currentColor' : 'none'} />
                </button>
                <button type="button" className="subtle-button" onClick={() => { setAssignItem(item); setProjectId(''); setClipId(''); setProjectClips([]); }}>
                  Asignar a clip
                </button>
                <button type="button" className="pf-icon-button" aria-label={`Eliminar ${item.title}`} onClick={() => void removeItem(item)}>
                  <Trash2 size={15} />
                </button>
              </div>
            </li>
          ))}
        </ul>
      </section>
      {assignItem && (
        <div className="pf-modal-overlay" role="dialog" aria-modal="true" aria-label={`Asignar ${assignItem.title}`}>
          <section className="pf-modal">
            <header className="pf-modal-heading">
              <div>
                <h2>Asignar B-roll</h2>
                <p>{assignItem.title}</p>
              </div>
              <button type="button" className="pf-modal-close" onClick={() => setAssignItem(null)} aria-label="Cerrar"><X size={18} /></button>
            </header>
            <label className="config-field"><span>Proyecto</span>
              <span className="select-wrap"><select value={projectId} onChange={(event) => void loadProjectClips(event.target.value)}>
                <option value="">Elige un proyecto…</option>
                {projects.map((project) => <option key={project.id} value={project.id}>{project.title}</option>)}
              </select></span>
            </label>
            <label className="config-field"><span>Clip</span>
              <span className="select-wrap"><select value={clipId} onChange={(event) => setClipId(event.target.value)} disabled={!projectId}>
                <option value="">Elige un clip…</option>
                {projectClips.map((clip) => <option key={clip.id} value={clip.id}>{clip.title}</option>)}
              </select></span>
            </label>
            <div className="pf-segment-fields">
              <label className="config-field"><span>Inicio (s, vacío: todo el clip)</span>
                <input className="text-input" type="number" min={0} step={0.1} value={segmentStart} onChange={(event) => setSegmentStart(event.target.value)} />
              </label>
              <label className="config-field"><span>Fin (s)</span>
                <input className="text-input" type="number" min={0} step={0.1} value={segmentEnd} onChange={(event) => setSegmentEnd(event.target.value)} />
              </label>
            </div>
            <label className="config-field"><span>Etiqueta del segmento (opcional)</span>
              <input className="text-input" value={segmentLabel} maxLength={200} onChange={(event) => setSegmentLabel(event.target.value)} placeholder='p. ej. "cuando hablo de precios"' />
            </label>
            <div className="pf-modal-actions">
              <button type="button" className="subtle-button" onClick={() => setAssignItem(null)}>Cancelar</button>
              <button type="button" className="create-button" onClick={() => void assign()} disabled={assigning || !projectId || !clipId}>
                {assigning ? <LoaderCircle size={15} className="spin" /> : null}
                {assigning ? 'Asignando…' : 'Asignar'}
              </button>
            </div>
          </section>
        </div>
      )}
      {previewItem && (
        <div className="pf-modal-overlay" role="dialog" aria-modal="true" aria-label={`Vista previa de ${previewItem.title}`}>
          <section className="pf-modal pf-modal-wide">
            <header className="pf-modal-heading">
              <div><h2>{previewItem.title}</h2><p>{previewItem.creator} · {previewItem.provider}</p></div>
              <button type="button" className="pf-modal-close" onClick={() => setPreviewItem(null)} aria-label="Cerrar"><X size={18} /></button>
            </header>
            {previewItem.local_path ? (
              <video className="pf-broll-video" src={`${API_BASE}/api/broll/${previewItem.id}/file`} controls playsInline />
            ) : (
              <p className="pf-empty">Este B-roll no tiene archivo local para previsualizar.</p>
            )}
          </section>
        </div>
      )}
    </main>
  );
}
