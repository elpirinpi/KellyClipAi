import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertCircle, ArrowRight, Clapperboard, Download, FileText, FileVideo2, Film, LoaderCircle, Scissors, Search, Share2, Sparkles, Trash2, WandSparkles,
} from 'lucide-react';
import { apiFetch } from '../auth';
import { API_BASE, CONTENT_PROFILE_LABELS, MOMENT_ENGINE_LABELS, formatBytes, type ContentProfile, type MomentEngine } from '../context/constants';
import { ToastStack, useToasts, type ToastKind } from './Toasts';
import type { ClipProposalRecord, ClipResult, ProjectResult, ScheduleJob } from '../types';
import { formatDuration, responseError } from '../utils';

declare global {
  interface Window {
    pywebview?: {
      api: {
        select_download_folder: () => Promise<string | null>;
      };
    };
  }
}

// Etiquetas del desglose "por qué este clip" (Fase A, v1.1.0).
const SIGNAL_CHANNEL_LABELS: Record<string, string> = {
  audio: 'Energía de audio',
  texto: 'Ganchos de texto',
  ritmo: 'Ritmo de habla',
  gancho_inicial: 'Gancho inicial',
  eventos: 'Eventos de audio',
  visual: 'Señales visuales',
  rostro: 'Rostro en cuadro',
  hablantes: 'Turnos de hablante',
};

function resolveEngine(project: ProjectResult | null): MomentEngine {
  const engine = project?.metadata?.moment_engine ?? project?.config.moment_engine;
  if (engine === 'signals' || engine === 'signals+ai' || engine === 'llm') return engine;
  // Proyectos anteriores a v1.1.0: todos usaban el LLM.
  return 'llm';
}

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function downloadWithAuth(url: string, filename: string): Promise<void> {
  const response = await apiFetch(url);
  if (!response.ok) throw new Error(await responseError(response));
  triggerDownload(await response.blob(), filename);
}

export function ResultsPage({
  project,
  error,
  onRetry,
  onEdit,
  onStudio,
  onStudioNew,
  onStudioProposal,
  onProjectUpdated,
  onProjectDeleted,
}: {
  project: ProjectResult | null;
  error: string;
  onRetry: () => void;
  onEdit: (clipId: string) => void;
  onStudio: (clipId: string) => void;
  onStudioNew: () => void;
  onStudioProposal: (proposal: ClipProposalRecord) => void;
  onProjectUpdated: (projectId: string) => Promise<void>;
  onProjectDeleted: () => void;
}) {
  const [sort, setSort] = useState<'score' | 'time'>('score');
  const [selected, setSelected] = useState<Set<string>>(new Set());
  // Fase 2 (ítem 20): búsqueda por título o palabra de la transcripción
  // y filtros por duración y por estado.
  const [clipQuery, setClipQuery] = useState('');
  const [durationFilter, setDurationFilter] = useState<'all' | 'short' | 'medium' | 'long' | 'xlong'>('all');
  const [statusFilter, setStatusFilter] = useState<'all' | 'ready' | 'failed'>('all');
  const { toasts, pushToast, dismissToast } = useToasts();
  const notify = useCallback((message: string, kind: ToastKind = 'info') => pushToast(message, kind), [pushToast]);
  const [shareLink, setShareLink] = useState<{
    url: string;
    clipId: string;
    expiresAt: string;
  } | null>(null);
  const [busyAction, setBusyAction] = useState('');
  const [sharingClip, setSharingClip] = useState('');
  const [publishPlatform, setPublishPlatform] = useState('youtube');
  const [publishVisibility, setPublishVisibility] = useState('private');
  const [scheduleJobs, setScheduleJobs] = useState<ScheduleJob[]>([]);
  const [scheduleError, setScheduleError] = useState('');
  const [storageUsedBytes, setStorageUsedBytes] = useState<number | null>(null);
  const [storageError, setStorageError] = useState('');
  const players = useRef<Record<string, HTMLVideoElement | null>>({});
  const clips = useMemo(() => {
    const allClips = [...(project?.clips ?? [])];
    return allClips.sort((a, b) => sort === 'score' ? b.score - a.score : a.start - b.start);
  }, [project?.clips, sort]);
  const visibleClips = useMemo(() => {
    const needle = clipQuery.trim().toLocaleLowerCase();
    return clips.filter((clip) => {
      if (statusFilter === 'ready' && clip.status === 'failed') return false;
      if (statusFilter === 'failed' && clip.status !== 'failed') return false;
      if (durationFilter === 'short' && clip.duration >= 30) return false;
      if (durationFilter === 'medium' && (clip.duration < 30 || clip.duration > 60)) return false;
      if (durationFilter === 'long' && (clip.duration <= 60 || clip.duration > 90)) return false;
      if (durationFilter === 'xlong' && clip.duration <= 90) return false;
      if (!needle) return true;
      const haystack = `${clip.title} ${clip.transcript.map((word) => word.text).join(' ')}`.toLocaleLowerCase();
      return haystack.includes(needle);
    });
  }, [clips, clipQuery, durationFilter, statusFilter]);
  const filtersActive = clipQuery.trim() !== '' || durationFilter !== 'all' || statusFilter !== 'all';
  const clearClipFilters = () => {
    setClipQuery('');
    setDurationFilter('all');
    setStatusFilter('all');
  };
  const readyClips = clips.filter((clip) => clip.status !== 'failed');
  const failedClips = clips.filter((clip) => clip.status === 'failed');
  // Propuestas sin render (v1.3.2): con la edición automática apagada,
  // el proyecto completa con las propuestas en los metadatos y sin
  // clips renderizados. Se validan antes de mostrarlas: los metadatos
  // de proyectos viejos pueden traer otra forma.
  const proposals = useMemo<ClipProposalRecord[]>(() => {
    const records = project?.metadata?.clip_proposals;
    if (!Array.isArray(records)) return [];
    return records.filter((record) => {
      const proposal = (record as Partial<ClipProposalRecord> | null)?.proposal;
      return typeof record?.id === 'string'
        && proposal !== undefined
        && Number.isFinite(proposal.inicio)
        && Number.isFinite(proposal.fin)
        && proposal.fin > proposal.inicio
        && typeof proposal.titulo === 'string';
    });
  }, [project?.metadata?.clip_proposals]);
  // Solo se muestran cuando el proyecto completó con la edición
  // automática apagada (false explícito): con ella encendida, las
  // propuestas ya están renderizadas como clips.
  const showProposals = project?.status === 'completed'
    && project?.config.auto_edit === false
    && proposals.length > 0;
  const visibleReadyClips = visibleClips.filter((clip) => clip.status !== 'failed');
  // Motor efectivo que eligió los clips (registrado en metadatos).
  const engine = resolveEngine(project);
  // Perfil de contenido auto-detectado (Fase C, v1.2.0; solo en
  // proyectos procesados con el motor de señales desde la 1.2.0).
  const rawProfile = project?.metadata?.content_profile;
  const contentProfile: ContentProfile | null =
    rawProfile === 'talk' || rawProfile === 'mixed' || rawProfile === 'gameplay' ? rawProfile : null;
  const allSelected = visibleReadyClips.length > 0 && visibleReadyClips.every((clip) => selected.has(clip.id));
  const selectedClips = readyClips.filter((clip) => selected.has(clip.id));

  const publishClips = async (clipIds: string[]) => {
    if (!project || !clipIds.length) {
      notify('Selecciona al menos un clip para publicar.');
      return;
    }
    setBusyAction('publish');
    try {
      const response = await apiFetch(`${API_BASE}/api/projects/${project.id}/publish`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          clip_ids: clipIds,
          platform: publishPlatform,
          visibility: publishVisibility,
        }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      const result = await response.json() as {
        published: Array<{ clip_id: string; external_id: string; url: string }>;
        failed: { clip_id: string; error: string } | null;
        not_attempted: string[];
        jobs?: ScheduleJob[];
      };
      if (result.jobs) {
        const queuedJobs = result.jobs;
        setScheduleJobs((current) => [
          ...current.filter((job) => !queuedJobs.some((queued) => queued.id === job.id)),
          ...queuedJobs,
        ]);
        notify(`${queuedJobs.length} clip(s) añadidos a la cola real de publicación.`, 'success');
        return;
      }
      const publishedMessage = result.published.length
        ? `${result.published.length} clip(s) confirmados por ${publishPlatform}.`
        : '';
      const failureMessage = result.failed
        ? ` No se confirmó el clip ${result.failed.clip_id.slice(0, 8)}: ${result.failed.error}`
        : '';
      const pendingMessage = result.not_attempted.length
        ? ` No se intentaron ${result.not_attempted.length} clips restantes.`
        : '';
      notify(`${publishedMessage}${failureMessage}${pendingMessage}`.trim() || 'La plataforma no confirmó ninguna publicación.', result.failed ? 'error' : result.published.length ? 'success' : 'info');
    } catch (error) {
      notify(error instanceof Error ? error.message : 'No se pudo completar la publicación.', 'error');
    } finally {
      setBusyAction('');
    }
  };
  const publishSelected = () => void publishClips(selectedClips.map((clip) => clip.id));

  const refreshStorageUsage = useCallback(async () => {
    try {
      const response = await apiFetch(`${API_BASE}/api/storage`);
      if (!response.ok) throw new Error(await responseError(response));
      const result = await response.json() as { used_bytes: number };
      setStorageUsedBytes(result.used_bytes);
      setStorageError('');
    } catch (error) {
      setStorageError(error instanceof Error ? error.message : 'No se pudo consultar el almacenamiento.');
    }
  }, []);

  const deleteProject = async () => {
    if (!project || !window.confirm(
      `¿Eliminar “${project.title}” y todos sus archivos, clips, publicaciones programadas y enlaces compartidos? Esta acción no se puede deshacer.`,
    )) return;
    setBusyAction('delete-project');
    try {
      const response = await apiFetch(`${API_BASE}/api/projects/${project.id}`, { method: 'DELETE' });
      if (!response.ok) throw new Error(await responseError(response));
      onProjectDeleted();
    } catch (error) {
      notify(error instanceof Error ? error.message : 'No se pudo eliminar el proyecto.', 'error');
      setBusyAction('');
    }
  };

  const retryFailedClips = async () => {
    if (!project || !failedClips.length) return;
    setBusyAction('retry-clips');
    try {
      const response = await apiFetch(`${API_BASE}/api/projects/${project.id}/retry-failed-clips`, {
        method: 'POST',
      });
      if (!response.ok) throw new Error(await responseError(response));
      notify('Los clips fallidos se añadieron a la cola para reintentarlos.', 'success');
      await onProjectUpdated(project.id);
    } catch (error) {
      notify(error instanceof Error ? error.message : 'No se pudieron reintentar los clips fallidos.', 'error');
    } finally {
      setBusyAction('');
    }
  };

  // Fase D (ítem 6): pulir los títulos de los clips con UNA llamada a la IA.
  const improveTitles = async () => {
    if (!project || !clips.length) return;
    setBusyAction('improve-titles');
    try {
      const response = await apiFetch(`${API_BASE}/api/projects/${project.id}/improve-titles`, {
        method: 'POST',
      });
      // El 409 trae el motivo en español (sin clave de IA, sin clips o
      // procesando); se muestra tal cual, incluido el aviso de ir a Ajustes.
      if (!response.ok) throw new Error(await responseError(response));
      const result = await response.json() as { improved: number; message?: string };
      if (result.improved > 0) {
        await onProjectUpdated(project.id);
        notify(result.message ?? `Se mejoraron ${result.improved} título(s).`, 'success');
      } else {
        notify('Los títulos ya están bien, no hubo cambios.');
      }
    } catch (error) {
      notify(error instanceof Error ? error.message : 'No se pudieron mejorar los títulos.', 'error');
    } finally {
      setBusyAction('');
    }
  };

  const refreshScheduleJobs = useCallback(async () => {
    if (!project) return;
    try {
      const response = await apiFetch(`${API_BASE}/api/projects/${project.id}/scheduled`);
      if (!response.ok) throw new Error(await responseError(response));
      setScheduleJobs(await response.json() as ScheduleJob[]);
      setScheduleError('');
    } catch (error) {
      setScheduleError(error instanceof Error ? error.message : 'No se pudo consultar la cola de publicaciones.');
    }
  }, [project]);

  const resolveOverdueScheduleJob = async (
    jobId: string,
    action: 'publish_now' | 'discard',
  ) => {
    setBusyAction(`overdue:${jobId}`);
    try {
      const response = await apiFetch(`${API_BASE}/api/scheduled-jobs/${jobId}/resolve-overdue`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      await refreshScheduleJobs();
    } catch (error) {
      setScheduleError(error instanceof Error ? error.message : 'No se pudo resolver la publicación vencida.');
    } finally {
      setBusyAction('');
    }
  };

  useEffect(() => {
    setScheduleJobs([]);
    if (project) void refreshScheduleJobs();
  }, [project?.id, project?.status, refreshScheduleJobs]);

  useEffect(() => {
    void refreshStorageUsage();
  }, [refreshStorageUsage]);

  useEffect(() => {
    if (!scheduleJobs.some((job) => job.status === 'scheduled' || job.status === 'processing')) return;
    const interval = window.setInterval(() => void refreshScheduleJobs(), 5000);
    return () => window.clearInterval(interval);
  }, [scheduleJobs, refreshScheduleJobs]);

  const downloadSelected = async () => {
    if (!selectedClips.length) {
      notify('Selecciona al menos un clip para descargarlo.');
      return;
    }
    await saveProjectFiles(false, selectedClips.map((clip) => clip.id));
  };
  const saveProjectFiles = async (includeOriginal: boolean, clipIds: string[]) => {
    if (!project) return;
    setBusyAction(includeOriginal ? 'original' : 'archive');
    try {
      const desktopApi = window.pywebview?.api;
      if (desktopApi) {
        const destination = await desktopApi.select_download_folder();
        if (!destination) {
          notify('Guardado cancelado. No se eligió una carpeta.');
          return;
        }
        const response = await apiFetch(`${API_BASE}/api/projects/${project.id}/downloads/save`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ destination, include_original: includeOriginal, clip_ids: clipIds }),
        });
        if (!response.ok) throw new Error(await responseError(response));
        const result = await response.json() as { saved: string[] };
        notify(`Guardado en la carpeta elegida: ${result.saved.join(', ')}`, 'success');
        return;
      }

      const filename = includeOriginal
        ? `${project.title.replace(/[<>:"/\\|?*]/g, '_')}${/\.[^.]+$/.test(project.title) ? '' : '.mp4'}`
        : clipIds.length === 1
          ? `${clipIds[0]}.mp4`
          : `kellyClipAi-clips-${project.id.slice(0, 8)}.zip`;
      const url = includeOriginal
        ? `${API_BASE}/api/projects/${project.id}/original/download`
        : clipIds.length === 1
          ? `${API_BASE}/api/projects/${project.id}/clips/${clipIds[0]}/download`
          : `${API_BASE}/api/projects/${project.id}/clips/archive`;
      const response = clipIds.length > 1 && !includeOriginal
        ? await apiFetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ clip_ids: clipIds }),
        })
        : await apiFetch(url);
      if (!response.ok) throw new Error(await responseError(response));
      triggerDownload(await response.blob(), filename);
      notify(includeOriginal ? 'Video original descargado.' : clipIds.length > 1
        ? `ZIP creado con ${clipIds.length} clips reales.`
        : 'Clip descargado.');
    } catch (downloadError) {
      notify(downloadError instanceof Error ? downloadError.message : 'No se pudieron guardar los archivos.', 'error');
    } finally {
      setBusyAction('');
    }
  };
  const downloadSubtitles = async (clipId: string, format: 'srt' | 'vtt') => {
    if (!project) return;
    try {
      const response = await apiFetch(
        `${API_BASE}/api/projects/${project.id}/clips/${clipId}/subtitles/${format}`,
      );
      if (!response.ok) throw new Error(await responseError(response));
      triggerDownload(await response.blob(), `${clipId}.${format}`);
      notify(`Subtítulos ${format.toUpperCase()} descargados.`, 'success');
    } catch (error) {
      notify(error instanceof Error ? error.message : 'No se pudieron descargar los subtítulos.', 'error');
    }
  };
  const changeResolution = async (clip: ClipResult, resolution: 720 | 1080) => {
    if (!project || resolution === (clip.resolution ?? 720)) return;
    setBusyAction(clip.id);
    try {
      const response = await apiFetch(`${API_BASE}/api/projects/${project.id}/clips/${clip.id}/quality`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resolution }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      await onProjectUpdated(project.id);
      notify(`Clip re-renderizado a ${resolution}p.`, 'success');
    } catch (renderError) {
      notify(renderError instanceof Error ? renderError.message : 'No se pudo re-renderizar el clip.', 'error');
    } finally {
      setBusyAction('');
    }
  };
  const improveAudio = async (clip: ClipResult) => {
    if (!project || clip.audio_enhanced) return;
    setBusyAction(clip.id);
    try {
      const response = await apiFetch(`${API_BASE}/api/projects/${project.id}/clips/${clip.id}/improve`, {
        method: 'POST',
      });
      if (!response.ok) throw new Error(await responseError(response));
      await onProjectUpdated(project.id);
      notify('Audio mejorado con normalización de volumen y realce ligero.', 'success');
    } catch (improveError) {
      notify(improveError instanceof Error ? improveError.message : 'No se pudo mejorar el audio.', 'error');
    } finally {
      setBusyAction('');
    }
  };
  const shareClip = async (clipId: string) => {
    if (!project) return;
    setSharingClip(clipId);
    setShareLink(null);
    try {
      const response = await apiFetch(`${API_BASE}/api/projects/${project.id}/clips/${clipId}/share`, {
        method: 'POST',
      });
      if (!response.ok) throw new Error(await responseError(response));
      const result = await response.json() as { share_url: string; expires_at: string };
      const url = new URL(result.share_url, window.location.origin).toString();
      setShareLink({ url, clipId, expiresAt: result.expires_at });
      try {
        await navigator.clipboard.writeText(url);
        notify('Enlace local copiado al portapapeles.', 'success');
      } catch {
        notify('Enlace local listo. Puedes abrirlo o copiarlo desde el enlace mostrado.');
      }
    } catch (shareError) {
      notify(shareError instanceof Error ? shareError.message : 'No se pudo crear el enlace compartido.', 'error');
    } finally {
      setSharingClip('');
    }
  };

  const revokeShareLink = async () => {
    if (!project || !shareLink) return;
    setBusyAction('revoke-share');
    try {
      const response = await apiFetch(
        `${API_BASE}/api/projects/${project.id}/clips/${shareLink.clipId}/share`,
        { method: 'DELETE' },
      );
      if (!response.ok) throw new Error(await responseError(response));
      setShareLink(null);
      notify('El enlace compartido quedó revocado.', 'success');
    } catch (error) {
      notify(error instanceof Error ? error.message : 'No se pudo revocar el enlace.', 'error');
    } finally {
      setBusyAction('');
    }
  };
  const editSelected = () => {
    if (selectedClips.length !== 1) {
      notify('Selecciona exactamente un clip para editarlo.');
      return;
    }
    onEdit(selectedClips[0].id);
  };

  return (
    <main className="results-page" id="main-content">
      <h1 className="visually-hidden">{project ? `Resultados de ${project.title}` : 'Resultados del proyecto'}</h1>
      <header className="results-toolbar">
        <label className="select-all"><input type="checkbox" checked={allSelected} onChange={(event) => setSelected((current) => {
          const next = new Set(current);
          if (event.target.checked) visibleReadyClips.forEach((clip) => next.add(clip.id));
          else visibleReadyClips.forEach((clip) => next.delete(clip.id));
          return next;
        })} /> Seleccionar todo</label>
        <div className="results-toolbar-actions">
          <label className="publish-select"><span className="visually-hidden">Plataforma para publicar</span><select value={publishPlatform} onChange={(event) => {
            const next = event.target.value;
            setPublishPlatform(next);
            if (next === 'instagram') setPublishVisibility('public');
            else if (next === 'tiktok' && publishVisibility === 'unlisted') setPublishVisibility('private');
          }}><option value="youtube">YouTube</option><option value="tiktok">TikTok</option><option value="instagram">Instagram</option></select></label>
          <label className="publish-select"><span className="visually-hidden">Visibilidad</span><select value={publishVisibility} disabled={publishPlatform === 'instagram'} onChange={(event) => setPublishVisibility(event.target.value)}>
            {publishPlatform !== 'tiktok' && <option value="unlisted">No listada</option>}
            {publishPlatform !== 'instagram' && <option value="private">Privada</option>}
            <option value="public">Pública</option>
          </select></label>
          <button type="button" onClick={publishSelected} disabled={!selectedClips.length || busyAction === 'publish'}>
            {busyAction === 'publish' ? <LoaderCircle size={15} className="spin" /> : <ArrowRight size={15} />}
            Publicar
          </button>
          <button type="button" onClick={() => void downloadSelected()} disabled={busyAction === 'archive'}>
            {busyAction === 'archive' ? <LoaderCircle size={15} className="spin" /> : <Download size={15} />}
            Descargar
          </button>
          <button type="button" onClick={() => void saveProjectFiles(true, [])} disabled={busyAction === 'original'}>
            {busyAction === 'original' ? <LoaderCircle size={15} className="spin" /> : <Download size={15} />}
            Guardar video original
          </button>
          <button type="button" onClick={editSelected}><Scissors size={15} /> Editar clips</button>
          <button type="button" onClick={onStudioNew}><Clapperboard size={15} /> Nuevo clip manual</button>
          <button type="button" onClick={() => void improveTitles()} disabled={!clips.length || busyAction === 'improve-titles' || project?.status === 'processing' || project?.status === 'queued'}>
            {busyAction === 'improve-titles' ? <LoaderCircle size={15} className="spin" /> : <Sparkles size={15} />}
            {busyAction === 'improve-titles' ? 'Mejorando títulos…' : 'Mejorar títulos con IA'}
          </button>
          <button type="button" className="delete-project-button" onClick={() => void deleteProject()} disabled={busyAction === 'delete-project'}>
            {busyAction === 'delete-project' ? <LoaderCircle size={15} className="spin" /> : <Trash2 size={15} />}
            Eliminar proyecto
          </button>
        </div>
      </header>
      <div className="clips-heading">
        <div><span className="clips-version">V1</span><strong>CLIPS IA</strong></div>
        <span className="clips-count">{filtersActive ? `${visibleClips.length} de ${clips.length} clips` : `${clips.length} clips`}</span>
        <span className="engine-badge">Motor: {MOMENT_ENGINE_LABELS[engine]}</span>
        {contentProfile && <span className="engine-badge">Perfil: {CONTENT_PROFILE_LABELS[contentProfile]}</span>}
        <span className="storage-usage" role="status">
          Almacenamiento local: {storageUsedBytes === null ? (storageError || 'calculando…') : formatBytes(storageUsedBytes)}
        </span>
        <label className="sort-control"><span>Ordenar:</span><select value={sort} onChange={(event) => setSort(event.target.value as 'score' | 'time')}><option value="score">Mayor puntuación</option><option value="time">Cronológico</option></select></label>
      </div>
      {/* Fase 2 (ítem 20): búsqueda y filtros de clips. */}
      <div className="results-filters">
        <label className="results-search">
          <Search size={16} aria-hidden="true" />
          <span className="visually-hidden">Buscar clips por título o palabra de la transcripción</span>
          <input
            type="search"
            value={clipQuery}
            onChange={(event) => setClipQuery(event.target.value)}
            placeholder="Buscar por título o palabra de la transcripción…"
            autoComplete="off"
          />
        </label>
        <label className="results-filter-select"><span>Duración</span><select value={durationFilter} onChange={(event) => setDurationFilter(event.target.value as 'all' | 'short' | 'medium' | 'long' | 'xlong')}><option value="all">Todas</option><option value="short">Menos de 30 s</option><option value="medium">30–60 s</option><option value="long">60–90 s</option><option value="xlong">Más de 90 s</option></select></label>
        <label className="results-filter-select"><span>Estado</span><select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value as 'all' | 'ready' | 'failed')}><option value="all">Todos</option><option value="ready">Listos</option><option value="failed">Fallidos</option></select></label>
        {filtersActive && <button type="button" className="subtle-button" onClick={clearClipFilters}>Limpiar filtros</button>}
      </div>
      {error && <p className="error-message" role="alert"><AlertCircle size={16} />{error}</p>}
      {failedClips.length > 0 && project?.status !== 'processing' && project?.status !== 'queued' && (
        <div className="project-failure">
          <p className="error-message" role="alert"><AlertCircle size={16} />{failedClips.length} clip(s) fallaron; los demás se conservaron.</p>
          <button type="button" className="retry-button" onClick={() => void retryFailedClips()} disabled={busyAction === 'retry-clips'}>
            {busyAction === 'retry-clips' ? <LoaderCircle size={16} className="spin" /> : <WandSparkles size={16} />}
            Reintentar clips fallidos
          </button>
        </div>
      )}
      {scheduleError && (
        <div className="project-failure">
          <p className="error-message" role="alert"><AlertCircle size={16} />{scheduleError}</p>
          <button type="button" className="retry-button" onClick={() => void refreshScheduleJobs()}>Reintentar consulta</button>
        </div>
      )}
      {scheduleJobs.length > 0 && (
        <section className="config-card schedule-card results-schedule">
          <div className="card-heading"><div><h2>Publicaciones programadas</h2><p>El estado se actualiza desde la cola real del backend local.</p></div></div>
          <div className="scheduled-job-list">
            {scheduleJobs.map((job) => (
              <article className="scheduled-job" key={job.id}>
                <div><strong>{job.platform} · clip {job.clip_id.slice(0, 8)}</strong><small>{new Date(job.scheduled_at).toLocaleString()} · {job.visibility}</small></div>
                <span className={`job-status ${job.status}`}>{{
                  scheduled: 'En cola',
                  processing: 'Procesando',
                  published: 'Publicado',
                  exported: 'Exportado localmente',
                  failed: 'No confirmado',
                }[job.status] ?? job.status}</span>
                {job.external_url && <a className="job-link" href={job.external_url} target="_blank" rel="noreferrer">Ver publicación</a>}
                {job.download_url && <button className="job-link" type="button" onClick={() => void downloadWithAuth(
                  `${API_BASE}${job.download_url}`,
                  `${job.id}.mp4`,
                ).catch((downloadError: unknown) => notify(
                  downloadError instanceof Error ? downloadError.message : 'No se pudo descargar el video.', 'error',
                ))}>Descargar MP4</button>}
                {job.error && <p role="status">{job.error}</p>}
                {job.confirmation_required && (
                  <div className="overdue-job-actions">
                    <p role="status">La app se cerró antes de publicar. Elige si quieres publicar ahora o descartar esta tarea.</p>
                    <button
                      type="button"
                      className="job-link"
                      disabled={busyAction === `overdue:${job.id}`}
                      onClick={() => void resolveOverdueScheduleJob(job.id, 'publish_now')}
                    >
                      Publicar ahora
                    </button>
                    <button
                      type="button"
                      className="job-link"
                      disabled={busyAction === `overdue:${job.id}`}
                      onClick={() => void resolveOverdueScheduleJob(job.id, 'discard')}
                    >
                      Descartar
                    </button>
                  </div>
                )}
              </article>
            ))}
          </div>
        </section>
      )}
      {project?.status === 'failed' && (
        <div className="project-failure">
          <p className="error-message" role="alert"><AlertCircle size={16} />{project.error ?? 'El procesamiento falló.'}</p>
          <button type="button" className="retry-button" onClick={onRetry}><WandSparkles size={16} /> Reintentar</button>
        </div>
      )}
      {project?.status === 'cancelled' && (
        <div className="project-failure">
          <p className="error-message" role="alert"><AlertCircle size={16} />El procesamiento se canceló. Puedes reprocesar el proyecto con la misma configuración.</p>
          <button type="button" className="retry-button" onClick={onRetry}><WandSparkles size={16} /> Reprocesar</button>
        </div>
      )}
      <ToastStack toasts={toasts} onDismiss={dismissToast} />
      {shareLink && (
        <p className="share-link-message">
          <a href={shareLink.url} target="_blank" rel="noreferrer">{shareLink.url}</a>
          <span>Expira: {new Date(shareLink.expiresAt).toLocaleString()}</span>
          <button type="button" className="job-link" onClick={() => void revokeShareLink()} disabled={busyAction === 'revoke-share'}>
            Revocar enlace
          </button>
        </p>
      )}
      {!project && error && <button type="button" className="retry-button" onClick={() => window.location.reload()}>Volver a cargar el proyecto</button>}
      {project?.status === 'completed' && clips.length === 0 && !showProposals && (
        <section className="empty-clips"><WandSparkles size={23} /><h2>No se encontraron clips</h2><p>El análisis real no identificó momentos que cumplieran la duración y el criterio solicitados.{engine === 'signals' ? ' Puedes bajar el umbral de puntuación en la configuración del proyecto y reprocesar.' : ''}</p></section>
      )}
      {project?.status === 'completed' && clips.length > 0 && visibleClips.length === 0 && (
        <section className="empty-clips"><Search size={23} /><h2>Sin resultados</h2><p>Ningún clip coincide con la búsqueda o los filtros. <button type="button" className="job-link" onClick={clearClipFilters}>Limpiar filtros</button></p></section>
      )}
      <div className="clip-list">
        {visibleClips.map((clip, index) => (
          <article className="clip-card" key={clip.id}>
            <label className="clip-select"><input type="checkbox" checked={selected.has(clip.id)} disabled={clip.status === 'failed'} onChange={(event) => setSelected((current) => {
              const next = new Set(current);
              if (event.target.checked) next.add(clip.id);
              else next.delete(clip.id);
              return next;
            })} /><span className="visually-hidden">Seleccionar clip {index + 1}</span></label>
            <div className="clip-preview">
              {clip.status === 'failed'
                ? <div className="clip-render-failed"><AlertCircle size={22} /><span>Render fallido</span></div>
                : <video
                ref={(element) => { players.current[clip.id] = element; }}
                src={`${API_BASE}${clip.video_url ?? ''}?v=${encodeURIComponent(project?.updated_at ?? '')}`}
                controls
                preload="metadata"
                playsInline
                aria-label={`Vista previa de ${clip.title}`}
              />}
              {clip.status !== 'failed' && <select
                className="clip-quality"
                value={clip.resolution ?? 720}
                onChange={(event) => void changeResolution(clip, event.target.value === '1080' ? 1080 : 720)}
                disabled={busyAction === clip.id}
                aria-label={`Resolución de ${clip.title}`}
              >
                <option value={720}>720p</option><option value={1080}>1080p</option>
              </select>}
              <span className="clip-duration">{formatDuration(clip.duration)}</span>
            </div>
            <div className="clip-content">
              <h2>#{index + 1} {clip.title}</h2>
              {clip.status === 'failed'
                ? <>
                  <p className="error-message" role="alert">{clip.error ?? 'No se pudo renderizar este clip.'}</p>
                  <div className="clip-score-actions">
                    <div className="clip-actions">
                      <button type="button" className="clip-action" onClick={() => onStudio(clip.id)} title="El editor visual trabaja con el video fuente, así que un clip con el render fallido se puede reconstruir desde aquí"><Film size={15} /> Editar en Studio</button>
                    </div>
                  </div>
                </>
                : <div className="clip-score-actions">
                <div className="clip-score"><strong>{clip.score.toFixed(1)}</strong><span>VIRALIDAD</span></div>
                <div className="clip-actions">
                  <button type="button" className="clip-action primary" onClick={() => void saveProjectFiles(false, [clip.id])} disabled={busyAction === 'archive'}>
                    {busyAction === 'archive' ? <LoaderCircle size={16} className="spin" /> : <Download size={16} />} Descargar
                  </button>
                  <button type="button" className="clip-action" onClick={() => void downloadSubtitles(clip.id, 'srt')}>
                    <FileText size={15} /> SRT
                  </button>
                  <button type="button" className="clip-action" onClick={() => void downloadSubtitles(clip.id, 'vtt')}>
                    <FileText size={15} /> VTT
                  </button>
                  <button type="button" className="clip-action" disabled={busyAction === 'publish'} onClick={() => void publishClips([clip.id])}><ArrowRight size={15} /> Publicar</button>
                  <button
                    type="button"
                    className="clip-action"
                    onClick={() => void improveAudio(clip)}
                    disabled={busyAction === clip.id || clip.audio_enhanced}
                    title="Normaliza el volumen y aplica un realce ligero al audio."
                  >
                    {busyAction === clip.id ? <LoaderCircle size={15} className="spin" /> : <Sparkles size={15} />}
                    {clip.audio_enhanced ? 'Mejorado' : 'Mejorar con IA'}
                  </button>
                  <button
                    type="button"
                    className="clip-action icon-only"
                    aria-label="Compartir clip"
                    title="Crear enlace local de reproducción"
                    onClick={() => void shareClip(clip.id)}
                    disabled={sharingClip === clip.id}
                  >
                    {sharingClip === clip.id ? <LoaderCircle size={16} className="spin" /> : <Share2 size={16} />}
                  </button>
                  <button type="button" className="clip-action icon-only" aria-label="Cambiar plantilla" onClick={() => onEdit(clip.id)}><FileVideo2 size={16} /></button>
                  <button type="button" className="clip-action icon-only" aria-label="Editar transcripción" onClick={() => onEdit(clip.id)}><Scissors size={16} /></button>
                  <button type="button" className="clip-action" onClick={() => onStudio(clip.id)} title="Abrir el editor visual con el video fuente de este clip"><Film size={15} /> Editar en Studio</button>
                </div>
              </div>}
              {clip.status !== 'failed' && (
              <>
              <section className="viral-reason"><strong>Razón viral</strong><p>{clip.viral_reason}</p></section>
              {clip.signal_breakdown && (
                <section className="signal-breakdown" aria-label={`Por qué se eligió el clip ${index + 1}`}>
                  <strong>Por qué este clip</strong>
                  <ul>
                    {Object.entries(clip.signal_breakdown)
                      .filter(([channel, value]) => channel !== 'completitud' && value > 0)
                      .sort((a, b) => b[1] - a[1])
                      .map(([channel, value]) => (
                        <li key={channel}>
                          <span className="signal-channel">{SIGNAL_CHANNEL_LABELS[channel] ?? channel}</span>
                          <span className="signal-bar" aria-hidden="true"><span style={{ width: `${Math.min(100, value * 10)}%` }} /></span>
                          <span className="signal-value">{value.toFixed(1)}</span>
                        </li>
                      ))}
                  </ul>
                </section>
              )}
              {clip.broll_provider && <p className="broll-attribution">B-roll de {clip.broll_provider} · {clip.broll_creator}{clip.broll_attribution_url && <> · <a href={clip.broll_attribution_url} target="_blank" rel="noreferrer">Ver fuente</a></>}</p>}
              <section className="clip-transcript" aria-label={`Transcripción del clip ${index + 1}`}>
                <strong>Transcripción</strong>
                <div>{clip.transcript.map((word, wordIndex) => (
                  <button
                    className="transcript-word"
                    type="button"
                    key={`${word.start}-${wordIndex}`}
                    title={`Ir a ${formatDuration(word.start)}`}
                    onClick={() => {
                      const player = players.current[clip.id];
                      if (player) {
                        player.currentTime = word.preview_start;
                        void player.play();
                      }
                    }}
                  >
                    {wordIndex === 0 || Math.floor(word.start) !== Math.floor(clip.transcript[wordIndex - 1].start)
                      ? <span className="transcript-time">{formatDuration(word.start)} </span>
                      : null}
                    {word.text}{' '}
                  </button>
                ))}</div>
              </section>
              </>
              )}
            </div>
          </article>
        ))}
      </div>
      {showProposals && (
        <section className="proposal-section" aria-label="Propuestas sin editar">
          <div className="clips-heading">
            <div><strong>PROPUESTAS SIN EDITAR</strong></div>
            <span className="clips-count">{proposals.length} propuesta{proposals.length === 1 ? '' : 's'}</span>
          </div>
          <p className="proposal-explainer">La edición automática está apagada: estos momentos se detectaron pero no se renderizaron. Abre cada uno en el Studio para editarlo sobre el video fuente.</p>
          <div className="proposal-list">
            {proposals.map((record, index) => {
              const proposal = record.proposal;
              return (
                <article className="proposal-card" key={record.id}>
                  <div className="proposal-time">
                    <strong>{formatDuration(proposal.inicio)} – {formatDuration(proposal.fin)}</strong>
                    <span>{formatDuration(proposal.fin - proposal.inicio)}</span>
                  </div>
                  <div className="clip-content">
                    <h2>#{index + 1} {proposal.titulo}</h2>
                    <div className="clip-score-actions">
                      <div className="clip-score"><strong>{proposal.puntuacion.toFixed(1)}</strong><span>VIRALIDAD</span></div>
                      <div className="clip-actions">
                        <button type="button" className="clip-action" onClick={() => onStudioProposal(record)} title="Abrir el editor visual con el video fuente de esta propuesta"><Film size={15} /> Editar en Studio</button>
                      </div>
                    </div>
                    <section className="viral-reason"><strong>Razón viral</strong><p>{proposal.razon_viral}</p></section>
                    {proposal.signal_breakdown && (
                      <section className="signal-breakdown" aria-label={`Por qué se eligió la propuesta ${index + 1}`}>
                        <strong>Por qué este clip</strong>
                        <ul>
                          {Object.entries(proposal.signal_breakdown)
                            .filter(([channel, value]) => channel !== 'completitud' && value > 0)
                            .sort((a, b) => b[1] - a[1])
                            .map(([channel, value]) => (
                              <li key={channel}>
                                <span className="signal-channel">{SIGNAL_CHANNEL_LABELS[channel] ?? channel}</span>
                                <span className="signal-bar" aria-hidden="true"><span style={{ width: `${Math.min(100, value * 10)}%` }} /></span>
                                <span className="signal-value">{value.toFixed(1)}</span>
                              </li>
                            ))}
                        </ul>
                      </section>
                    )}
                  </div>
                </article>
              );
            })}
          </div>
        </section>
      )}
      {project?.status !== 'completed' && project?.status !== 'failed' && project?.status !== 'cancelled' && !error && <p className="results-loading"><LoaderCircle size={17} className="spin" /> Cargando el estado real del proyecto…</p>}
      <a className="back-button results-back" href="/">Volver al inicio</a>
    </main>
  );
}

