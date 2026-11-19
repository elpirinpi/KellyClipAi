import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import { apiFetch } from '../auth';
import { platforms } from '../context/constants';
import { pollStagedMedia } from './useImport';
import type { ProjectResult, StagedMedia } from '../types';
import { responseError } from '../utils';

// Importación por enlace (extraída de App.tsx, ítem 23 de la Fase 2):
// pegado del enlace, detección de plataforma, descarga con yt-dlp en el
// backend con seguimiento de progreso, cancelación y reanudación de la
// descarga local al volver a la página principal.

type UseLinkImportOptions = {
  apiBase: string;
  authReady: boolean;
  userId: string | undefined;
  route: string;
  navigate: (next: string) => void;
  projectConfig: () => Record<string, unknown>;
  setLinkProject: (project: ProjectResult | null) => void;
  setIsSubmitting: (value: boolean) => void;
};

// Las descargas canceladas por el usuario no se reanudan automáticamente
// durante la sesión (resumeDownload las omitiría en caso contrario).
function cancelledStagedKey(stagedId: string): string {
  return `kellyclipai-cancelled-staged:${stagedId}`;
}

function wasLinkDownloadCancelled(stagedId: string): boolean {
  try {
    return sessionStorage.getItem(cancelledStagedKey(stagedId)) === '1';
  } catch {
    return false;
  }
}

function markLinkDownloadCancelled(stagedId: string): void {
  try {
    sessionStorage.setItem(cancelledStagedKey(stagedId), '1');
  } catch {
    // Sin almacenamiento de sesión disponible: la cancelación local sigue vigente.
  }
}

export function useLinkImport({
  apiBase,
  authReady,
  userId,
  route,
  navigate,
  projectConfig,
  setLinkProject,
  setIsSubmitting,
}: UseLinkImportOptions) {
  const [link, setLink] = useState('');
  const [linkMessage, setLinkMessage] = useState('');
  const [downloadProgress, setDownloadProgress] = useState<StagedMedia | null>(null);
  // Controla el polling de la descarga por enlace: se detiene al navegar
  // a otra vista, al desmontar y al pulsar "Cancelar descarga".
  const linkDownloadRef = useRef({ active: false, cancelled: false, stagedId: '' });

  // Al salir de la vista principal (o al desmontar), el polling de la
  // descarga por enlace se detiene: no debe navegar por su cuenta.
  useEffect(() => {
    if (route !== '/') linkDownloadRef.current.active = false;
    return () => {
      linkDownloadRef.current.active = false;
    };
  }, [route]);

  const detectedPlatform = useMemo(() => {
    try {
      const hostname = new URL(link).hostname.toLowerCase();
      return platforms.find((platform) => hostname === platform.domain || hostname.endsWith(`.${platform.domain}`));
    } catch {
      return undefined;
    }
  }, [link]);

  const handleLink = async (event: FormEvent) => {
    event.preventDefault();
    if (!link.trim()) {
      setLinkMessage('Pega un enlace para continuar.');
      return;
    }
    if (!detectedPlatform) {
      setLinkMessage('No reconocemos esta plataforma. Comprueba el enlace.');
      return;
    }
    setIsSubmitting(true);
    setDownloadProgress(null);
    setLinkMessage(`Descargando desde ${detectedPlatform.label} con yt-dlp. Solo importa contenido propio o autorizado.`);
    try {
      const startResponse = await apiFetch(`${apiBase}/api/staged-media/downloads`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: link.trim(), config: projectConfig() }),
      });
      if (!startResponse.ok) throw new Error(await responseError(startResponse));
      const started = await startResponse.json() as StagedMedia;
      setDownloadProgress(started);
      linkDownloadRef.current = { active: true, cancelled: false, stagedId: started.id };
      const ready = started.status === 'ready'
        ? started
        : await pollStagedMedia(
            apiBase,
            started.id,
            setDownloadProgress,
            () => linkDownloadRef.current.active,
          );
      // No crear el proyecto ni navegar si el usuario canceló o ya está en otra vista.
      if (!linkDownloadRef.current.active || window.location.pathname !== '/') return;
      const projectResponse = await apiFetch(`${apiBase}/api/staged-media/${ready.id}/project`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      if (!projectResponse.ok) throw new Error(await responseError(projectResponse));
      const imported = await projectResponse.json() as ProjectResult;
      setLinkProject(imported);
      setDownloadProgress(null);
      setLinkMessage('Video descargado y guardado. Revisa la configuración antes de procesar.');
      navigate(`/p/${imported.id}/configure`);
    } catch (error) {
      // Si el usuario canceló, cancelLinkDownload ya puso el mensaje.
      if (!linkDownloadRef.current.cancelled) {
        setLinkMessage(error instanceof Error ? error.message : 'No se pudo importar el enlace.');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const cancelLinkDownload = async () => {
    const stagedId = linkDownloadRef.current.stagedId || downloadProgress?.id;
    linkDownloadRef.current.active = false;
    linkDownloadRef.current.cancelled = true;
    if (stagedId) markLinkDownloadCancelled(stagedId);
    setDownloadProgress(null);
    setIsSubmitting(false);
    setLinkMessage('Descarga cancelada. Puedes pegar otro enlace.');
    if (!stagedId) return;
    try {
      const response = await apiFetch(`${apiBase}/api/staged-media/${stagedId}`, { method: 'DELETE' });
      // 404: el staged ya no existe; nada más que limpiar.
      if (!response.ok && response.status !== 404) throw new Error(await responseError(response));
    } catch {
      // El endpoint actual no admite borrar una descarga en curso (409):
      // el servidor puede seguir descargando en segundo plano, pero el
      // seguimiento local quedó detenido y no se reanudará automáticamente.
      setLinkMessage('Descarga detenida en esta pantalla. El servidor puede seguir descargando en segundo plano; no se reanudará automáticamente.');
    }
  };

  useEffect(() => {
    if (!authReady || !userId || route !== '/') return;
    let active = true;
    const resumeDownload = async () => {
      try {
        const latestResponse = await apiFetch(`${apiBase}/api/staged-media/latest?source=url`);
        if (!latestResponse.ok) throw new Error(await responseError(latestResponse));
        const latest = await latestResponse.json() as StagedMedia | null;
        if (!active || !latest) return;
        if (wasLinkDownloadCancelled(latest.id)) return;
        if (latest.status === 'failed') {
          setLinkMessage(latest.error ?? 'La descarga no se completó. Vuelve a intentarlo.');
          return;
        }
        setIsSubmitting(true);
        setDownloadProgress(latest);
        setLinkMessage('Reanudando la descarga local del enlace…');
        linkDownloadRef.current = { active: true, cancelled: false, stagedId: latest.id };
        const ready = latest.status === 'ready'
          ? latest
          : await pollStagedMedia(apiBase, latest.id, (staged) => {
              if (active) setDownloadProgress(staged);
            }, () => active && linkDownloadRef.current.active);
        if (!active || !linkDownloadRef.current.active) return;
        const response = await apiFetch(`${apiBase}/api/staged-media/${ready.id}/project`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({}),
        });
        if (!response.ok) throw new Error(await responseError(response));
        const imported = await response.json() as ProjectResult;
        if (!active) return;
        setLinkProject(imported);
        setDownloadProgress(null);
        setLinkMessage('Video descargado y guardado. Revisa la configuración antes de procesar.');
        navigate(`/p/${imported.id}/configure`);
      } catch (error) {
        if (active && !linkDownloadRef.current.cancelled) {
          setLinkMessage(error instanceof Error ? error.message : 'No se pudo reanudar la descarga.');
          setDownloadProgress(null);
        }
      } finally {
        if (active) setIsSubmitting(false);
      }
    };
    void resumeDownload();
    return () => {
      active = false;
    };
  }, [apiBase, authReady, navigate, route, setIsSubmitting, setLinkProject, userId]);

  return {
    link,
    setLink,
    linkMessage,
    setLinkMessage,
    downloadProgress,
    handleLink,
    cancelLinkDownload,
  };
}
