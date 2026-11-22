import { useEffect, useState } from 'react';
import { useApp } from '../context/AppContext';
import { API_BASE } from '../context/constants';
import { apiFetch } from '../auth';
import { responseError } from '../utils';
import { BrollLibraryPage } from './BrollLibraryPage';
import { CalendarPage } from './CalendarPage';
import { ClipEditorPage } from './ClipEditorPage';
import { HomePage } from './HomePage';
import { AuthPage, ProcessingPage, SharedClipPage } from './Pages';
import { ProjectsPage } from './ProjectsPage';
import { QueuePage } from './QueuePage';
import { RecordModal } from './RecordModal';
import { ResultsPage } from './ResultsPage';
import { SettingsPage } from './SettingsPage';
import { SetupWizard } from './SetupWizard';
import { StudioPage } from '../studio/StudioPage';
import type { ClipProposalRecord } from '../types';
import { ToastStack } from './Toasts';
import { TopBar } from './TopBar';
import { UpdateBanner } from './UpdateBanner';
import { ZoomImportModal } from './ZoomImportModal';

// Armazón de la aplicación (extraído de App.tsx, ítem 23 de la Fase 2):
// puerta de autenticación, cabecera, banner de actualización, enrutado
// por vista, modales, avisos y pie.
export function AppShell() {
  const {
    route,
    navigate,
    shareToken,
    authReady,
    authSession,
    authConfigurationError,
    isSettings,
    isProjects,
    isCalendar,
    isQueue,
    isBroll,
    projectRouteId,
    projectRouteMode,
    editClipId,
    studioClipId,
    project,
    setProject,
    projectError,
    refreshProject,
    retryProject,
    notify,
    toasts,
    dismissToast,
    settingsReturnPath,
    config,
    theme,
  } = useApp();

  // Fase 2 (ítem 21): al cambiar de ruta, mover el foco al h1 de la
  // página (o al <main> si no lo hay) para los lectores de pantalla.
  useEffect(() => {
    const target = document.querySelector<HTMLElement>('main h1')
      ?? document.querySelector<HTMLElement>('main');
    if (!target) return;
    if (!target.hasAttribute('tabindex')) target.setAttribute('tabindex', '-1');
    target.focus({ preventScroll: true });
  }, [route]);

  // Fase 3 (ítem 19, C.6.42): asistente de primer arranque. Solo para
  // sesiones autenticadas (el asistente guarda la clave por cuenta);
  // los enlaces compartidos no lo ven. Si la comprobación falla, la
  // app sigue sin asistente en vez de bloquear el arranque.
  const [setupNeeded, setSetupNeeded] = useState<boolean | null>(null);
  useEffect(() => {
    if (shareToken || !authSession) return;
    let active = true;
    apiFetch(`${API_BASE}/api/setup`)
      .then(async (response) => {
        if (!response.ok) throw new Error(await responseError(response));
        return response.json() as Promise<{ needs_setup: boolean }>;
      })
      .then((setup) => { if (active) setSetupNeeded(setup.needs_setup); })
      .catch(() => { if (active) setSetupNeeded(false); });
    return () => { active = false; };
  }, [shareToken, authSession]);

  if (!shareToken && !authReady) {
    return <main className="auth-page"><p role="status">Conectando con el servicio de autenticación…</p></main>;
  }
  if (!shareToken && !authSession) {
    return <AuthPage configurationError={authConfigurationError} />;
  }
  if (!shareToken && setupNeeded === null) {
    return <main className="auth-page"><p role="status">Comprobando la configuración inicial…</p></main>;
  }
  if (!shareToken && setupNeeded) {
    return <SetupWizard onDone={() => setSetupNeeded(false)} />;
  }

  return (
    <div className="app-shell">
      {/* Fase 2 (ítem 21): skip link al contenido principal. */}
      <a className="skip-link" href="#main-content">Saltar al contenido principal</a>
      <TopBar />
      <UpdateBanner />
      {shareToken ? (
        <SharedClipPage apiBase={API_BASE} token={shareToken} />
      ) : isSettings ? (
        <SettingsPage
          provider={config.settingsProvider}
          setProvider={config.setSettingsProvider}
          whisper={config.whisperModel}
          setWhisper={config.setWhisperModel}
          momentEngine={config.momentEngine}
          setMomentEngine={config.setMomentEngine}
          signalThreshold={config.signalThreshold}
          setSignalThreshold={config.setSignalThreshold}
          fastDetection={config.fastDetection}
          setFastDetection={config.setFastDetection}
          autoEdit={config.autoEdit}
          setAutoEdit={config.setAutoEdit}
          theme={theme.preference}
          setTheme={theme.setPreference}
          onBack={() => navigate(settingsReturnPath)}
          onNotice={notify}
        />
      ) : isProjects ? (
        <ProjectsPage
          apiBase={API_BASE}
          onOpen={(projectId) => navigate(`/p/${projectId}`)}
          onNew={() => navigate('/')}
        />
      ) : isCalendar ? (
        <CalendarPage onNotice={notify} />
      ) : isQueue ? (
        <QueuePage
          onNotice={notify}
          onOpenProject={(projectId) => navigate(`/p/${projectId}`)}
        />
      ) : isBroll ? (
        <BrollLibraryPage onNotice={notify} />
      ) : projectRouteId && projectRouteMode !== 'configure' ? (
        projectRouteMode === 'processing'
          ? <ProcessingPage apiBase={API_BASE} project={project} error={projectError} onRetry={() => void retryProject()} onCancelled={(cancelled) => setProject(cancelled)} />
          : projectRouteMode === 'edit'
            ? <ClipEditorPage
              projectId={projectRouteId}
              clipId={editClipId}
              project={project}
              savedTemplates={config.savedTemplates}
              onBack={() => navigate(`/p/${projectRouteId}`)}
            />
            : projectRouteMode === 'studio'
              ? <StudioPage
                projectId={projectRouteId}
                clipId={studioClipId}
                project={project}
                error={projectError}
                onBack={() => navigate(`/p/${projectRouteId}`)}
                onProjectUpdated={refreshProject}
              />
              : <ResultsPage
                project={project}
                error={projectError}
                onRetry={() => void retryProject()}
                onProjectUpdated={refreshProject}
                onProjectDeleted={() => navigate('/')}
                onEdit={(clipId) => navigate(`/p/${projectRouteId}/edit/${clipId}`)}
                onStudio={(clipId) => navigate(`/projects/${projectRouteId}/studio/${clipId}`)}
                onStudioNew={() => navigate(`/projects/${projectRouteId}/studio/new`)}
                // v1.3.2: una propuesta sin render abre el modo manual
                // del Studio con el trim y el título de la propuesta
                // (el Studio lee start/end/title de la URL).
                onStudioProposal={(proposal: ClipProposalRecord) => {
                  const params = new URLSearchParams({
                    start: String(proposal.proposal.inicio),
                    end: String(proposal.proposal.fin),
                    title: proposal.proposal.titulo,
                  });
                  navigate(`/projects/${projectRouteId}/studio/new?${params}`);
                }}
              />
      ) : (
        <HomePage />
      )}
      <RecordModal />
      <ZoomImportModal />
      <ToastStack toasts={toasts} onDismiss={dismissToast} />
      <footer className="developer-credit">
        Desarrollado por <strong>KELLY BENJAMIN DE LA CRUZ</strong>
      </footer>
    </div>
  );
}
