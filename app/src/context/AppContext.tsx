import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from 'react';
import type { Session } from '@supabase/supabase-js';
import { useLocation, useMatch, useNavigate } from 'react-router-dom';
import { apiFetch } from '../auth';
import { useAuth } from '../hooks/useAuth';
import { useDesktopNotifications } from '../hooks/useDesktopNotifications';
import { useGlobalShortcuts } from '../hooks/useGlobalShortcuts';
import { useImport } from '../hooks/useImport';
import { useLinkImport } from '../hooks/useLinkImport';
import { useProjectConfig } from '../hooks/useProjectConfig';
import { useProjects } from '../hooks/useProjects';
import { useRecorder } from '../hooks/useRecorder';
import { useTheme } from '../hooks/useTheme';
import { useUpdates } from '../hooks/useUpdates';
import { useZoomImport } from '../hooks/useZoomImport';
import { useToasts, type Toast, type ToastKind } from '../components/Toasts';
import type { ProjectResult } from '../types';
import { responseError } from '../utils';
import { API_BASE } from './constants';

// Estado compartido de la aplicación (ítem 23 de la Fase 2): App.tsx era
// un monolito de ~2700 líneas con ~37 useState; el estado vive ahora en
// hooks por dominio (configuración, importación, grabación, Zoom,
// actualizaciones, proyecto) y este contexto lo compone y lo expone a
// las páginas sin props-drilling. El proveedor no cambia ningún
// comportamiento: ejecuta los mismos hooks en el mismo orden relativo y
// con las mismas dependencias que el App original.

export type ProjectRouteMode = 'configure' | 'processing' | 'edit' | 'results' | 'studio';

export type AppContextValue = {
  apiBase: string;
  route: string;
  navigate: (next: string) => void;
  isSettings: boolean;
  isProjects: boolean;
  isCalendar: boolean;
  isQueue: boolean;
  isBroll: boolean;
  shareToken: string;
  projectRouteId: string;
  projectRouteMode: ProjectRouteMode;
  editClipId: string;
  studioClipId: string;
  theme: ReturnType<typeof useTheme>;
  notify: (message: string, kind?: ToastKind) => void;
  toasts: Toast[];
  dismissToast: (id: number) => void;
  authReady: boolean;
  authSession: Session | null;
  authConfigurationError: string;
  currentUserId: string | undefined;
  linkProject: ProjectResult | null;
  setLinkProject: (project: ProjectResult | null) => void;
  isSubmitting: boolean;
  setIsSubmitting: (value: boolean) => void;
  settingsReturnPath: string;
  setSettingsReturnPath: (path: string) => void;
  config: ReturnType<typeof useProjectConfig>;
  importer: ReturnType<typeof useImport>;
  linkImport: ReturnType<typeof useLinkImport>;
  recorder: ReturnType<typeof useRecorder>;
  zoom: ReturnType<typeof useZoomImport>;
  updates: ReturnType<typeof useUpdates>;
  project: ProjectResult | null;
  setProject: (project: ProjectResult | null) => void;
  projectError: string;
  setProjectError: (error: string) => void;
  refreshProject: (projectId: string) => Promise<void>;
  submitProject: () => Promise<void>;
  retryProject: () => Promise<void>;
  handleSignOut: () => Promise<void>;
};

const AppContext = createContext<AppContextValue | null>(null);

export function useApp(): AppContextValue {
  const value = useContext(AppContext);
  if (!value) throw new Error('useApp debe usarse dentro de AppProvider.');
  return value;
}

export function AppProvider({ children }: { children: ReactNode }) {
  const location = useLocation();
  const routerNavigate = useNavigate();
  const route = location.pathname;
  // Fase 2 (ítem 17): modo oscuro — sigue al sistema por defecto, con
  // interruptor manual persistido en localStorage.
  const theme = useTheme();
  const { toasts, pushToast, dismissToast } = useToasts();
  const notify = useCallback((message: string, kind: ToastKind = 'info') => pushToast(message, kind), [pushToast]);
  const onSessionError = useCallback((message: string) => notify(message, 'error'), [notify]);
  const {
    authReady,
    authSession,
    configurationError: authConfigurationError,
    signOut: signOutCurrentUser,
  } = useAuth(API_BASE, onSessionError);
  const currentUserId = authSession?.user.id;
  const [linkProject, setLinkProject] = useState<ProjectResult | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [settingsReturnPath, setSettingsReturnPath] = useState('/');
  const clearLinkProject = useCallback(() => setLinkProject(null), []);
  const config = useProjectConfig({
    apiBase: API_BASE,
    authReady,
    userId: currentUserId,
    notify,
    setLinkProject,
  });
  const importer = useImport({
    apiBase: API_BASE,
    authReady: authReady && Boolean(authSession),
    userId: currentUserId,
    onImportStart: clearLinkProject,
    onNotice: notify,
  });

  const navigate = useCallback((next: string) => {
    routerNavigate(next);
    if (next === '/') setLinkProject(null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [routerNavigate]);

  const isSettings = Boolean(useMatch('/settings'));
  // Fase 2 (ítem 15): ruta de la página "Mis proyectos".
  const isProjects = Boolean(useMatch('/projects'));
  // Fase 3 (ítems 9, 11, 13): calendario, cola y biblioteca de B-roll.
  const isCalendar = Boolean(useMatch('/calendar'));
  const isQueue = Boolean(useMatch('/queue'));
  const isBroll = Boolean(useMatch('/broll'));
  const shareToken = useMatch('/share/:token')?.params.token ?? '';
  const configureMatch = useMatch('/p/:projectId/configure');
  const processingMatch = useMatch('/p/:projectId/processing');
  const editMatch = useMatch('/p/:projectId/edit/:clipId');
  // Editor visual (Studio, v1.3.0): /projects/:id/studio/:clipId para un
  // clip existente y /projects/:id/studio/new para un clip manual nuevo.
  const studioMatch = useMatch('/projects/:projectId/studio/:clipId');
  const resultsMatch = useMatch('/p/:projectId');
  const projectRouteId = configureMatch?.params.projectId
    ?? processingMatch?.params.projectId
    ?? editMatch?.params.projectId
    ?? studioMatch?.params.projectId
    ?? resultsMatch?.params.projectId
    ?? '';
  const projectRouteMode: ProjectRouteMode = configureMatch
    ? 'configure'
    : processingMatch
      ? 'processing'
      : editMatch
        ? 'edit'
        : studioMatch
          ? 'studio'
          : 'results';
  const editClipId = editMatch?.params.clipId ?? '';
  const studioClipId = studioMatch?.params.clipId ?? '';

  const {
    project,
    setProject,
    projectError,
    setProjectError,
    refreshProject,
  } = useProjects({
    apiBase: API_BASE,
    authReady: authReady && Boolean(authSession),
    userId: currentUserId,
    projectId: projectRouteId,
    routeMode: projectRouteMode,
    navigate,
    onConfigureProject: config.applyProjectConfiguration,
  });

  // Fase 3 (ítem 18, C.4.31): aviso de escritorio cuando un
  // procesamiento largo termina con la ventana oculta.
  useDesktopNotifications({ project });

  const beginProjectProcessing = async (projectId: string) => {
    const response = await apiFetch(`${API_BASE}/api/projects/${projectId}/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config: config.projectConfig() }),
    });
    if (!response.ok) throw new Error(await responseError(response));
    navigate(`/p/${projectId}/processing`);
  };

  const submitProject = async () => {
    if (!linkProject && (!importer.fileInfo || importer.progress !== 100)) return;
    if (!linkProject && importer.fileInfo?.kind === 'audio') {
      notify('El procesamiento de clips requiere un video con pista de audio. Los archivos de solo audio no se procesan como clips de video.', 'error');
      return;
    }
    setIsSubmitting(true);
    importer.setUploadError('');
    try {
      let target = linkProject;
      if (!target) {
        if (!importer.fileInfo) throw new Error('No se encontró el video local. Vuelve a importarlo antes de procesar.');
        const response = await apiFetch(`${API_BASE}/api/staged-media/${importer.fileInfo.id}/project`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ config: config.projectConfig() }),
        });
        if (!response.ok) throw new Error(await responseError(response));
        target = await response.json() as ProjectResult;
      }
      setLinkProject(target);
      setProject(target);
      navigate(`/p/${target.id}/configure`);
      await beginProjectProcessing(target.id);
    } catch (error) {
      notify(error instanceof Error ? error.message : 'No se pudo iniciar el procesamiento.', 'error');
    } finally {
      setIsSubmitting(false);
    }
  };

  // El atajo Ctrl/Cmd+Enter llama siempre a la versión actual de submitProject.
  const submitProjectRef = useRef(submitProject);
  submitProjectRef.current = submitProject;

  const retryProject = async () => {
    if (!project) return;
    setProjectError('');
    try {
      await beginProjectProcessing(project.id);
    } catch (error) {
      setProjectError(error instanceof Error ? error.message : 'No se pudo reintentar el procesamiento.');
    }
  };

  const linkImport = useLinkImport({
    apiBase: API_BASE,
    authReady,
    userId: currentUserId,
    route,
    navigate,
    projectConfig: config.projectConfig,
    setLinkProject,
    setIsSubmitting,
  });
  const recorder = useRecorder({ importFile: importer.importFile });
  const zoom = useZoomImport({
    apiBase: API_BASE,
    projectConfig: config.projectConfig,
    navigate,
    setLinkProject,
    setProject,
  });
  const updates = useUpdates({ apiBase: API_BASE, authReady, authSession, userId: currentUserId, notify });
  useGlobalShortcuts({
    recordOpen: recorder.recordOpen,
    setRecordOpen: recorder.setRecordOpen,
    zoomOpen: zoom.zoomOpen,
    setZoomOpen: zoom.setZoomOpen,
    recording: recorder.recording,
    zoomLoading: zoom.zoomLoading,
    route,
    projectRouteMode,
    isSubmitting,
    linkProject,
    progress: importer.progress,
    submitProjectRef,
  });

  const handleSignOut = async () => {
    try {
      await signOutCurrentUser();
      importer.setFileInfo(null);
      setProject(null);
      setLinkProject(null);
      importer.setProgress(null);
      navigate('/');
    } catch (error) {
      notify(error instanceof Error ? error.message : 'No se pudo cerrar la sesión.', 'error');
    }
  };

  const value: AppContextValue = {
    apiBase: API_BASE,
    route,
    navigate,
    isSettings,
    isProjects,
    isCalendar,
    isQueue,
    isBroll,
    shareToken,
    projectRouteId,
    projectRouteMode,
    editClipId,
    studioClipId,
    theme,
    notify,
    toasts,
    dismissToast,
    authReady,
    authSession,
    authConfigurationError,
    currentUserId,
    linkProject,
    setLinkProject,
    isSubmitting,
    setIsSubmitting,
    settingsReturnPath,
    setSettingsReturnPath,
    config,
    importer,
    linkImport,
    recorder,
    zoom,
    updates,
    project,
    setProject,
    projectError,
    setProjectError,
    refreshProject,
    submitProject,
    retryProject,
    handleSignOut,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}
