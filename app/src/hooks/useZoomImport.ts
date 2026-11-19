import { useRef, useState } from 'react';
import { apiFetch } from '../auth';
import { useFocusTrap } from './useFocusTrap';
import type { ProjectResult, ZoomRecording } from '../types';
import { responseError } from '../utils';

// Importación desde Zoom (extraída de App.tsx, ítem 23 de la Fase 2):
// listado de grabaciones MP4 recientes e importación de la elegida como
// proyecto nuevo.

type UseZoomImportOptions = {
  apiBase: string;
  projectConfig: () => Record<string, unknown>;
  navigate: (next: string) => void;
  setLinkProject: (project: ProjectResult | null) => void;
  setProject: (project: ProjectResult | null) => void;
};

export function useZoomImport({ apiBase, projectConfig, navigate, setLinkProject, setProject }: UseZoomImportOptions) {
  const [zoomOpen, setZoomOpen] = useState(false);
  const [zoomRecordings, setZoomRecordings] = useState<ZoomRecording[]>([]);
  const [zoomLoading, setZoomLoading] = useState(false);
  const [zoomError, setZoomError] = useState('');
  // Fase 2 (ítem 21): trampa de foco en el modal de importar.
  const zoomModalRef = useRef<HTMLElement>(null);
  useFocusTrap(zoomModalRef, zoomOpen);

  const openZoomImport = async () => {
    setZoomOpen(true);
    setZoomError('');
    setZoomRecordings([]);
    setZoomLoading(true);
    try {
      const response = await apiFetch(`${apiBase}/api/integrations/zoom/recordings`);
      if (!response.ok) throw new Error(await responseError(response));
      setZoomRecordings(await response.json() as ZoomRecording[]);
    } catch (error) {
      setZoomError(error instanceof Error ? error.message : 'No se pudieron cargar las grabaciones de Zoom.');
    } finally {
      setZoomLoading(false);
    }
  };

  const importZoomFile = async (meetingId: string, fileId: string) => {
    setZoomLoading(true);
    setZoomError('');
    try {
      const response = await apiFetch(`${apiBase}/api/integrations/zoom/import`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ meeting_id: meetingId, file_id: fileId, config: projectConfig() }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      const imported = await response.json() as ProjectResult;
      setLinkProject(imported);
      setProject(imported);
      setZoomOpen(false);
      navigate(`/p/${imported.id}/configure`);
    } catch (error) {
      setZoomError(error instanceof Error ? error.message : 'No se pudo importar la grabación.');
    } finally {
      setZoomLoading(false);
    }
  };

  return {
    zoomOpen,
    setZoomOpen,
    zoomRecordings,
    zoomLoading,
    zoomError,
    zoomModalRef,
    openZoomImport,
    importZoomFile,
  };
}
