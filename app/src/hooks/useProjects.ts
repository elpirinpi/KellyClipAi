import { useCallback, useEffect, useState } from 'react';
import { apiFetch, streamApiEvents } from '../auth';
import type { ProjectResult } from '../types';
import { responseError } from '../utils';

type ProjectRouteMode = 'configure' | 'processing' | 'edit' | 'results' | 'studio';

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
    // BUG 2 (v1.3.2): el flujo SSE puede cerrarse limpio sin entregar el
    // evento terminal. Antes, un cierre limpio no hacía nada: la promesa
    // se resolvía, no había .then y la vista se quedaba en 98 % hasta
    // remontar el componente. Al terminar el flujo — cierre limpio o
    // error — se relee el proyecto: si ya es terminal, se muestra el
    // resultado; si sigue en curso, se reabre el flujo.
    let finished = false;
    let restarts = 0;
    const MAX_STREAM_RESTARTS = 3;
    const revalidateProject = async (): Promise<boolean> => {
      const response = await apiFetch(`${apiBase}/api/projects/${projectId}`);
      if (!response.ok) throw new Error(await responseError(response));
      const current = await parseProject(response);
      if (!active) return true;
      setProject(current);
      if (current.status === 'completed' || current.status === 'failed' || current.status === 'cancelled') {
        navigate(`/p/${projectId}`);
        return true;
      }
      return false;
    };
    async function handleStreamEnd(error?: unknown): Promise<void> {
      if (!active || finished) return;
      try {
        if (await revalidateProject()) return;
        if (restarts < MAX_STREAM_RESTARTS) {
          restarts += 1;
          await new Promise<void>((resolve) => window.setTimeout(() => resolve(), 1000));
          if (active && !finished) subscribe();
          return;
        }
        setProjectError(error instanceof Error ? error.message : 'Se perdió la conexión con el progreso del procesamiento.');
      } catch (revalidateError) {
        if (active) setProjectError(revalidateError instanceof Error ? revalidateError.message : 'No se pudo cargar el proyecto.');
      }
    }
    function subscribe(): void {
      const controller = new AbortController();
      eventController = controller;
      void streamApiEvents(
        `${apiBase}/api/projects/${projectId}/events`,
        async (eventName, data) => {
          const update = JSON.parse(data) as Partial<ProjectResult>;
          setProject((current) => current ? { ...current, ...update } : current);
          if (eventName === 'failed' || update.status === 'completed' || update.status === 'failed' || update.status === 'cancelled') {
            finished = true;
            controller.abort();
            try {
              // Si el proyecto ya no es terminal (p. ej. se reprocesó
              // desde otra pestaña entre el evento y esta lectura), se
              // reabre el flujo en vez de quedarse sin suscripción.
              if (!(await revalidateProject()) && active) {
                finished = false;
                subscribe();
              }
            } catch (error) {
              if (active) setProjectError(error instanceof Error ? error.message : 'No se pudo cargar el proyecto.');
            }
          }
        },
        controller.signal,
      ).then(() => void handleStreamEnd()).catch((error: unknown) => {
        if (active && !finished && !controller.signal.aborted) void handleStreamEnd(error);
      });
    }
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
          subscribe();
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
