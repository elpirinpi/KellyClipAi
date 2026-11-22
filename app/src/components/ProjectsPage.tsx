import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertCircle, ChevronRight, CopyPlus, FileVideo2, FolderOpen, LoaderCircle, Plus, Search } from 'lucide-react';
import { apiFetch } from '../auth';
import type { ProjectResult } from '../types';
import { formatDuration, responseError } from '../utils';

// Fase 2 (ítem 15): página "Mis proyectos" (historial).
// Hoy no existe ninguna lista de proyectos: solo se accede por URL.
// Esta vista lista los proyectos con miniatura, fecha, estado y nº de
// clips, con búsqueda por texto y carga paginada.
//
// La respuesta de GET /api/projects se interpreta de forma defensiva:
// - Lista plana (formato actual): se usa tal cual.
// - Respuesta paginada (limit/offset + total, en curso en el backend):
//   se pagina con "Cargar más" y se busca sobre todos los proyectos
//   cargando las páginas restantes al buscar.
// Los parámetros limit/offset son ignorados por el backend actual, así
// que enviarlos es seguro en ambas versiones.

const PAGE_SIZE = 20;
const MAX_SEARCH_PROJECTS = 1000;

type ProjectsResponse = ProjectResult[] | {
  projects?: unknown;
  items?: unknown;
  results?: unknown;
  total?: unknown;
  count?: unknown;
};

function isProjectArray(value: unknown): value is ProjectResult[] {
  return Array.isArray(value)
    && value.every((item) => typeof item === 'object'
      && item !== null
      && typeof (item as { id?: unknown }).id === 'string');
}

function parseProjectsResponse(body: unknown): { projects: ProjectResult[]; total: number | null } {
  if (isProjectArray(body)) return { projects: body, total: null };
  if (typeof body === 'object' && body !== null) {
    const record = body as Record<string, unknown>;
    const list = record.projects ?? record.items ?? record.results;
    if (isProjectArray(list)) {
      const total = typeof record.total === 'number'
        ? record.total
        : typeof record.count === 'number'
          ? record.count
          : null;
      return { projects: list, total };
    }
  }
  throw new Error('El servidor devolvió una lista de proyectos con un formato inesperado.');
}

const STATUS_LABELS: Record<string, string> = {
  completed: 'Completado',
  processing: 'Procesando',
  queued: 'En cola',
  failed: 'Fallido',
  cancelled: 'Cancelado',
};

export function ProjectsPage({
  apiBase,
  onOpen,
  onNew,
}: {
  apiBase: string;
  onOpen: (projectId: string) => void;
  onNew: () => void;
}) {
  const [projects, setProjects] = useState<ProjectResult[]>([]);
  const [total, setTotal] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [searchLoading, setSearchLoading] = useState(false);
  const [error, setError] = useState('');
  const [query, setQuery] = useState('');
  // Fase 3 (ítem 15): duplicar un proyecto para reprocesarlo con otra
  // configuración, sin volver a subir el video.
  const [duplicatingId, setDuplicatingId] = useState('');
  const requestRef = useRef(0);

  const duplicateProject = async (projectId: string) => {
    setDuplicatingId(projectId);
    setError('');
    try {
      const response = await apiFetch(`${apiBase}/api/projects/${projectId}/duplicate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      if (!response.ok) throw new Error(await responseError(response));
      const clone = await response.json() as ProjectResult;
      // La copia aparece arriba de la lista, lista para abrir y
      // reconfigurar.
      setProjects((current) => [clone, ...current]);
      setTotal((current) => (current === null ? null : current + 1));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'No se pudo duplicar el proyecto.');
    } finally {
      setDuplicatingId('');
    }
  };

  const fetchPage = useCallback(async (offset: number): Promise<{ page: ProjectResult[]; total: number | null }> => {
    const response = await apiFetch(
      `${apiBase}/api/projects?limit=${PAGE_SIZE}&offset=${offset}`,
    );
    if (!response.ok) throw new Error(await responseError(response));
    const parsed = parseProjectsResponse(await response.json() as ProjectsResponse);
    return { page: parsed.projects, total: parsed.total };
  }, [apiBase]);

  useEffect(() => {
    let active = true;
    const request = ++requestRef.current;
    setLoading(true);
    setError('');
    fetchPage(0)
      .then(({ page, total: responseTotal }) => {
        if (!active || request !== requestRef.current) return;
        setProjects(page);
        setTotal(responseTotal);
      })
      .catch((loadError: unknown) => {
        if (active && request === requestRef.current) {
          setError(loadError instanceof Error ? loadError.message : 'No se pudieron cargar los proyectos.');
        }
      })
      .finally(() => {
        if (active && request === requestRef.current) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [fetchPage]);

  const hasMore = total !== null && projects.length < total;

  const loadMore = useCallback(async () => {
    if (loadingMore || !hasMore) return;
    setLoadingMore(true);
    setError('');
    try {
      const { page, total: responseTotal } = await fetchPage(projects.length);
      setProjects((current) => {
        const known = new Set(current.map((item) => item.id));
        return [...current, ...page.filter((item) => !known.has(item.id))];
      });
      setTotal(responseTotal);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'No se pudieron cargar más proyectos.');
    } finally {
      setLoadingMore(false);
    }
  }, [fetchPage, hasMore, loadingMore, projects.length]);

  // Al buscar con paginación activa, cargar las páginas restantes para
  // que la búsqueda cubra toda la biblioteca (con un tope de seguridad).
  useEffect(() => {
    const trimmed = query.trim();
    if (!trimmed || total === null || projects.length >= total) return;
    let active = true;
    const request = ++requestRef.current;
    setSearchLoading(true);
    const loadRemaining = async () => {
      let loaded = projects.length;
      const collected: ProjectResult[] = [];
      while (active && request === requestRef.current && loaded < total && loaded < MAX_SEARCH_PROJECTS) {
        try {
          const { page } = await fetchPage(loaded);
          if (!page.length) break;
          collected.push(...page);
          loaded += page.length;
        } catch {
          // La búsqueda sigue funcionando sobre los proyectos ya cargados.
          break;
        }
      }
      if (!active || request !== requestRef.current) return;
      if (collected.length) {
        setProjects((current) => {
          const known = new Set(current.map((item) => item.id));
          return [...current, ...collected.filter((item) => !known.has(item.id))];
        });
      }
      setSearchLoading(false);
    };
    void loadRemaining();
    return () => {
      active = false;
    };
    // Intencionado: solo reacciona al texto de búsqueda y al total.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, total]);

  const visibleProjects = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    if (!needle) return projects;
    return projects.filter((project) => project.title.toLocaleLowerCase().includes(needle));
  }, [projects, query]);

  return (
    <main className="projects-page" id="main-content">
      <header className="projects-heading">
        <div>
          <span className="eyebrow">HISTORIAL</span>
          <h1>Mis proyectos</h1>
          <p>
            {total !== null
              ? `${total} proyecto${total === 1 ? '' : 's'} en este dispositivo`
              : `${projects.length} proyecto${projects.length === 1 ? '' : 's'} en este dispositivo`}
          </p>
        </div>
        <button type="button" className="create-button projects-new" onClick={onNew}>
          <Plus size={17} /> Nuevo proyecto
        </button>
      </header>

      <div className="projects-search">
        <Search size={17} aria-hidden="true" />
        <label className="visually-hidden" htmlFor="projects-search-input">Buscar proyectos por título</label>
        <input
          id="projects-search-input"
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Buscar por título…"
          autoComplete="off"
        />
        {searchLoading && <LoaderCircle size={16} className="spin" aria-label="Cargando proyectos para buscar" />}
      </div>

      {error && <p className="error-message" role="alert"><AlertCircle size={16} />{error}</p>}
      {loading && <p className="results-loading"><LoaderCircle size={17} className="spin" /> Cargando proyectos…</p>}

      {!loading && !error && visibleProjects.length === 0 && (
        <section className="empty-clips projects-empty">
          <FolderOpen size={23} />
          <h2>{query.trim() ? 'Sin resultados' : 'Aún no tienes proyectos'}</h2>
          <p>
            {query.trim()
              ? 'Ningún proyecto coincide con esa búsqueda. Prueba con otro texto.'
              : 'Sube un video o pega un enlace para crear tu primer proyecto de clips.'}
          </p>
          {!query.trim() && (
            <button type="button" className="retry-button" onClick={onNew}>
              <Plus size={16} /> Crear proyecto
            </button>
          )}
        </section>
      )}

      {visibleProjects.length > 0 && (
        <ul className="projects-list">
          {visibleProjects.map((project) => {
            const readyClips = project.clips.filter((clip) => clip.status !== 'failed').length;
            return (
              <li key={project.id} className="pf-project-row-wrap">
                <button
                  type="button"
                  className="project-row"
                  onClick={() => onOpen(project.id)}
                  aria-label={`Abrir proyecto ${project.title}`}
                >
                  <span className="project-thumb">
                    {project.thumbnail_url
                      ? <img src={`${apiBase}${project.thumbnail_url}`} alt="" loading="lazy" />
                      : <FileVideo2 size={22} />}
                  </span>
                  <span className="project-row-copy">
                    <strong title={project.title}>{project.title}</strong>
                    <small>
                      {new Date(project.created_at).toLocaleDateString(undefined, {
                        day: 'numeric',
                        month: 'short',
                        year: 'numeric',
                      })}
                      {project.metadata ? ` · ${formatDuration(project.metadata.duration)}` : ''}
                      {` · ${readyClips} clip${readyClips === 1 ? '' : 's'}`}
                      {project.clips.length !== readyClips ? ` (${project.clips.length - readyClips} fallido${project.clips.length - readyClips === 1 ? '' : 's'})` : ''}
                    </small>
                  </span>
                  <span className={`project-status project-status-${project.status}`}>
                    {STATUS_LABELS[project.status] ?? project.status}
                  </span>
                  <ChevronRight size={17} className="project-row-arrow" aria-hidden="true" />
                </button>
                {/* El duplicar va FUERA del botón de la fila: un botón
                    dentro de otro es HTML inválido. */}
                <button
                  type="button"
                  className="pf-duplicate-button"
                  onClick={() => void duplicateProject(project.id)}
                  disabled={duplicatingId === project.id}
                  aria-label={`Duplicar proyecto ${project.title}`}
                  title="Duplicar para reprocesar con otra configuración"
                >
                  {duplicatingId === project.id ? <LoaderCircle size={16} className="spin" /> : <CopyPlus size={16} />}
                </button>
              </li>
            );
          })}
        </ul>
      )}

      {hasMore && !query.trim() && (
        <button
          type="button"
          className="subtle-button projects-load-more"
          onClick={() => void loadMore()}
          disabled={loadingMore}
        >
          {loadingMore ? <LoaderCircle size={15} className="spin" /> : null}
          {loadingMore ? 'Cargando…' : `Cargar más (${projects.length} de ${total})`}
        </button>
      )}
      {query.trim() && visibleProjects.length > 0 && (
        <p className="projects-search-count" role="status">
          {visibleProjects.length} de {total ?? projects.length} proyectos
        </p>
      )}
    </main>
  );
}
