import { useCallback, useEffect, useState } from 'react';
import { apiFetch, streamApiEvents } from '../auth';
import type { ProjectResult } from '../types';
import { responseError } from '../utils';

type ProjectRouteMode = 'configure' | 'processing' | 'edit' | 'results';

type UseProjectsOptions = {
  apiBase: string;
  authReady: boolean;
  userId: string | undefined;
  projectId: string;
  routeMode: ProjectRouteMode;
  navigate: (path: string) => void;
  onConfigureProject: (project: ProjectResult) => void;
};

function isProjectResult(value: unknown): value is ProjectResult {
  return typeof value === 'object'
    && value !== null
    && typeof (value as { id?: unknown }).id === 'string'
    && (value as { id?: unknown }).id !== ''
    && typeof (value as { status?: unknown }).status === 'string'
    && (value as { status?: unknown }).status !== '';
}

const INVALID_PROJECT_MESSAGE = 'El servidor devolvió un proyecto con un formato inesperado.';

async function parseProject(response: Response): Promise<ProjectResult> {
  const loaded: unknown = await response.json();
  if (!isProjectResult(loaded)) throw new Error(INVALID_PROJECT_MESSAGE);
  return loaded;
}

export function useProjects({
  apiBase,
  authReady,
  userId,
  projectId,
  routeMode,
  navigate,
  onConfigureProject,
}: UseProjectsOptions) {
  const [project, setProject] = useState<ProjectResult | null>(null);
  const [projectError, setProjectError] = useState('');

  useEffect(() => {
    if (!authReady || !userId) return;
    if (!projectId) {
      setProject(null);
      setProjectError('');
      return;
    }

    let active = true;
    let eventController: AbortController | null = null;
    const loadProject = async () => {
      setProjectError('');
      try {
        const response = await apiFetch(`${apiBase}/api/projects/${projectId}`);
        if (!response.ok) throw new Error(await responseError(response));
        const loaded = await parseProject(response);
        if (!active) return;
        setProject(loaded);
        if (routeMode === 'configure') onConfigureProject(loaded);
        if (loaded.status === 'processing' || loaded.status === 'queued') {
          if (routeMode !== 'processing') navigate(`/p/${projectId}/processing`);
          eventController = new AbortController();
          void streamApiEvents(
            `${apiBase}/api/projects/${projectId}/events`,
            async (eventName, data) => {
              const update = JSON.parse(data) as Partial<ProjectResult>;
              setProject((current) => current ? { ...current, ...update } : current);
              if (eventName === 'failed' || update.status === 'completed' || update.status === 'failed') {
                eventController?.abort();
                const result = await apiFetch(`${apiBase}/api/projects/${projectId}`);
                if (!result.ok) throw new Error(await responseError(result));
                const finalProject = await parseProject(result);
                if (active) {
                  setProject(finalProject);
                  navigate(`/p/${projectId}`);
                }
              }
            },
            eventController.signal,
          ).catch((error: unknown) => {
            if (active && !eventController?.signal.aborted) {
              setProjectError(error instanceof Error ? error.message : 'Se perdió la conexión con el progreso del procesamiento.');
            }
          });
        } else if (routeMode === 'processing') {
          navigate(`/p/${projectId}`);
        }
      } catch (error) {
        if (active) setProjectError(error instanceof Error ? error.message : 'No se pudo cargar el proyecto.');
      }
    };
    void loadProject();
    return () => {
      active = false;
      eventController?.abort();
    };
  }, [apiBase, authReady, navigate, onConfigureProject, projectId, routeMode, userId]);

  const refreshProject = useCallback(async (id: string) => {
    const response = await apiFetch(`${apiBase}/api/projects/${id}`);
    if (!response.ok) throw new Error(await responseError(response));
    setProject(await parseProject(response));
  }, [apiBase]);

  return { project, setProject, projectError, setProjectError, refreshProject };
}
