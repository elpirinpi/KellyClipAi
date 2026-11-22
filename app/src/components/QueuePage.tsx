import { useCallback, useEffect, useState } from 'react';
import { FolderSearch, Layers, LoaderCircle, Moon, Plus, Trash2 } from 'lucide-react';
import { apiFetch } from '../auth';
import { API_BASE } from '../context/constants';
import type { BatchItem, NightWindow, QueueJob, WatchFolder } from '../types';
import { responseError } from '../utils';

const QUEUE_STATUS_LABELS: Record<string, string> = {
  pending: 'En cola',
  running: 'Procesando',
  done: 'Terminado',
  failed: 'Fallido',
  cancelled: 'Cancelado',
};

/** Cola por lotes + carpeta observada + procesar de noche (Fase 3,
 * ítem 11): vista global de la cola, encolado de varios enlaces a la vez,
 * carpetas observadas y franja nocturna. Todo opcional y apagado por
 * defecto. */
export function QueuePage({ onNotice, onOpenProject }: {
  onNotice: (message: string, kind?: 'info' | 'success' | 'error') => void;
  onOpenProject: (projectId: string) => void;
}) {
  const [jobs, setJobs] = useState<QueueJob[]>([]);
  const [batchItems, setBatchItems] = useState<BatchItem[]>([]);
  const [folders, setFolders] = useState<WatchFolder[]>([]);
  const [night, setNight] = useState<NightWindow>({ enabled: false, start: '22:00', end: '06:00' });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [urls, setUrls] = useState('');
  const [enqueueing, setEnqueueing] = useState(false);
  const [folderPath, setFolderPath] = useState('');
  const [addingFolder, setAddingFolder] = useState(false);
  const [savingNight, setSavingNight] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [queueResponse, batchResponse, foldersResponse, nightResponse] = await Promise.all([
        apiFetch(`${API_BASE}/api/queue`),
        apiFetch(`${API_BASE}/api/queue/batch`),
        apiFetch(`${API_BASE}/api/watch-folders`),
        apiFetch(`${API_BASE}/api/preferences/night-window`),
      ]);
      if (!queueResponse.ok) throw new Error(await responseError(queueResponse));
      if (!batchResponse.ok) throw new Error(await responseError(batchResponse));
      if (!foldersResponse.ok) throw new Error(await responseError(foldersResponse));
      if (!nightResponse.ok) throw new Error(await responseError(nightResponse));
      setJobs(await queueResponse.json() as QueueJob[]);
      setBatchItems(await batchResponse.json() as BatchItem[]);
      setFolders(await foldersResponse.json() as WatchFolder[]);
      setNight(await nightResponse.json() as NightWindow);
      setError('');
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'No se pudo leer la cola.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const interval = window.setInterval(() => void refresh(), 5000);
    return () => window.clearInterval(interval);
  }, [refresh]);

  const enqueueUrls = async () => {
    const list = urls.split('\n').map((line) => line.trim()).filter(Boolean);
    if (list.length === 0) {
      setError('Pega al menos un enlace, uno por línea.');
      return;
    }
    if (list.length > 20) {
      setError('Un lote admite como máximo 20 videos.');
      return;
    }
    setError('');
    setEnqueueing(true);
    try {
      const response = await apiFetch(`${API_BASE}/api/queue/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ urls: list }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      const result = await response.json() as { items: Array<{ status: string; error?: string }> };
      const failed = result.items.filter((item) => item.status === 'failed');
      if (failed.length > 0) {
        onNotice(`${failed.length} enlace${failed.length === 1 ? '' : 's'} no se pudieron encolar: ${failed[0].error ?? ''}`, 'error');
      } else {
        onNotice(`${list.length} enlace${list.length === 1 ? '' : 's'} en cola: se descargarán y procesarán en orden.`, 'success');
      }
      setUrls('');
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'No se pudo encolar el lote.');
    } finally {
      setEnqueueing(false);
    }
  };

  const addFolder = async () => {
    if (!folderPath.trim()) {
      setError('Escribe la ruta de la carpeta a observar (p. ej. C:\\Videos\\Por procesar).');
      return;
    }
    setError('');
    setAddingFolder(true);
    try {
      const response = await apiFetch(`${API_BASE}/api/watch-folders`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: folderPath.trim() }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      setFolderPath('');
      onNotice('Carpeta observada: los videos nuevos se procesarán solos.', 'success');
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'No se pudo añadir la carpeta.');
    } finally {
      setAddingFolder(false);
    }
  };

  const toggleFolder = async (folder: WatchFolder) => {
    try {
      const response = await apiFetch(`${API_BASE}/api/watch-folders/${folder.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: !folder.enabled }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      await refresh();
    } catch (caught) {
      onNotice(caught instanceof Error ? caught.message : 'No se pudo cambiar la carpeta.', 'error');
    }
  };

  const removeFolder = async (folderId: string) => {
    try {
      const response = await apiFetch(`${API_BASE}/api/watch-folders/${folderId}`, { method: 'DELETE' });
      if (!response.ok) throw new Error(await responseError(response));
      await refresh();
    } catch (caught) {
      onNotice(caught instanceof Error ? caught.message : 'No se pudo eliminar la carpeta.', 'error');
    }
  };

  const saveNight = async () => {
    setSavingNight(true);
    try {
      const response = await apiFetch(`${API_BASE}/api/preferences/night-window`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(night),
      });
      if (!response.ok) throw new Error(await responseError(response));
      setNight(await response.json() as NightWindow);
      onNotice(night.enabled ? `Procesar de noche activado (${night.start}–${night.end}).` : 'Procesar de noche desactivado.', 'success');
    } catch (caught) {
      onNotice(caught instanceof Error ? caught.message : 'No se pudo guardar la franja nocturna.', 'error');
    } finally {
      setSavingNight(false);
    }
  };

  return (
    <main className="pf-page" id="main-content">
      <header className="pf-page-heading">
        <div>
          <span className="eyebrow">FLUJO</span>
          <h1><Layers size={22} /> Cola de procesamiento</h1>
          <p>Todos los videos en cola, en un solo lugar. Los lotes y las carpetas observadas alimentan esta cola.</p>
        </div>
      </header>
      {error && <p className="error-message" role="alert">{error}</p>}
      {loading && <p className="results-loading"><LoaderCircle size={17} className="spin" /> Cargando cola…</p>}
      {!loading && (
        <>
          <section className="config-card">
            <div className="card-heading"><div><h2>En cola ahora</h2><p>La cola procesa de uno en uno para no saturar el PC.</p></div></div>
            {jobs.length === 0 && <p className="pf-empty">La cola está vacía.</p>}
            <ul className="pf-queue-list">
              {jobs.map((job) => (
                <li key={job.id} className="pf-queue-row">
                  <button type="button" className="pf-queue-title" onClick={() => job.project_id && onOpenProject(job.project_id)}>
                    {job.project_title ?? job.project_id}
                  </button>
                  <span className={`job-status ${job.status === 'running' ? 'processing' : job.status}`}>
                    {QUEUE_STATUS_LABELS[job.status] ?? job.status}
                  </span>
                  {job.project_progress !== null && job.status === 'running' && <span>{job.project_progress}%</span>}
                  {job.attempts > 1 && <small>intento {job.attempts}</small>}
                  {job.error && <small className="pf-queue-error">{job.error}</small>}
                </li>
              ))}
            </ul>
          </section>
          <section className="config-card">
            <div className="card-heading"><div><h2>Encolar varios enlaces</h2><p>Uno por línea (máx. 20). Cada enlace se descarga y se procesa en orden.</p></div></div>
            <label className="config-field"><span>Enlaces de video</span>
              <textarea className="text-input pf-textarea" rows={4} value={urls} onChange={(event) => setUrls(event.target.value)} placeholder={'https://www.youtube.com/watch?v=…\nhttps://www.youtube.com/watch?v=…'} />
            </label>
            <button type="button" className="create-button" onClick={() => void enqueueUrls()} disabled={enqueueing}>
              {enqueueing ? <LoaderCircle size={15} className="spin" /> : <Plus size={15} />}
              {enqueueing ? 'Encolando…' : 'Encolar lote'}
            </button>
            {batchItems.length > 0 && (
              <ul className="pf-batch-list">
                {batchItems.slice(0, 10).map((item) => (
                  <li key={item.id}>
                    <span>{item.url ?? item.staged_id ?? 'lote'}</span>
                    <span className={`job-status ${item.status === 'done' ? 'published' : item.status === 'failed' ? 'failed' : 'scheduled'}`}>
                      {item.status === 'done' ? 'En cola' : item.status === 'failed' ? 'Fallido' : 'Descargando'}
                    </span>
                    {item.error && <small className="pf-queue-error">{item.error}</small>}
                  </li>
                ))}
              </ul>
            )}
          </section>
          <section className="config-card">
            <div className="card-heading"><div><h2><FolderSearch size={20} /> Carpetas observadas</h2><p>Los videos nuevos en estas carpetas crean un proyecto y se encolan solos. Solo se procesan cuando dejan de crecer (una copia a medias no se encola).</p></div></div>
            <div className="pf-folder-add">
              <input
                className="text-input"
                value={folderPath}
                onChange={(event) => setFolderPath(event.target.value)}
                placeholder="Ruta de la carpeta, p. ej. C:\Videos\Por procesar"
                aria-label="Ruta de la carpeta a observar"
              />
              <button type="button" className="create-button" onClick={() => void addFolder()} disabled={addingFolder}>
                {addingFolder ? <LoaderCircle size={15} className="spin" /> : <Plus size={15} />} Observar
              </button>
            </div>
            {folders.length === 0 && <p className="pf-empty">Ninguna carpeta observada. Añade una para procesar videos en cuanto lleguen.</p>}
            <ul className="pf-folder-list">
              {folders.map((folder) => (
                <li key={folder.id} className="pf-folder-row">
                  <div>
                    <strong>{folder.path}</strong>
                    <small>
                      {folder.enabled ? 'Activa' : 'Pausada'}
                      {` · sondeo cada ${Math.round(folder.poll_seconds / 60)} min`}
                      {folder.last_scan_at ? ` · último sondeo: ${new Date(folder.last_scan_at).toLocaleString()}` : ''}
                    </small>
                  </div>
                  <div className="pf-folder-actions">
                    <button type="button" className={`switch ${folder.enabled ? 'on' : ''}`} role="switch" aria-checked={folder.enabled} aria-label={`Observar ${folder.path}`} onClick={() => void toggleFolder(folder)}><span /></button>
                    <button type="button" className="pf-icon-button" aria-label={`Dejar de observar ${folder.path}`} onClick={() => void removeFolder(folder.id)}>
                      <Trash2 size={15} />
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          </section>
          <section className="config-card">
            <div className="card-heading"><div><h2><Moon size={20} /> Procesar de noche</h2><p>Con la franja activada, la cola solo procesa dentro de ella para no competir con el uso del PC. Apagada por defecto.</p></div></div>
            <label className="fast-detection-toggle">
              <input type="checkbox" checked={night.enabled} onChange={(event) => setNight((current) => ({ ...current, enabled: event.target.checked }))} />
              <span><strong>Solo procesar dentro de la franja</strong><small>Fuera de la franja, los trabajos esperan en cola sin procesarse.</small></span>
            </label>
            <div className="pf-night-fields">
              <label className="config-field"><span>Desde</span>
                <input className="text-input" type="time" value={night.start} onChange={(event) => setNight((current) => ({ ...current, start: event.target.value }))} />
              </label>
              <label className="config-field"><span>Hasta</span>
                <input className="text-input" type="time" value={night.end} onChange={(event) => setNight((current) => ({ ...current, end: event.target.value }))} />
              </label>
              <button type="button" className="subtle-button" onClick={() => void saveNight()} disabled={savingNight}>
                {savingNight ? 'Guardando…' : 'Guardar franja'}
              </button>
            </div>
          </section>
        </>
      )}
    </main>
  );
}
