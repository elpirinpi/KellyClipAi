import { useEffect } from 'react';
import { useApp } from '../context/AppContext';
import { API_BASE } from '../context/constants';
import { ClipEditorPage } from './ClipEditorPage';
import { HomePage } from './HomePage';
import { AuthPage, ProcessingPage, SharedClipPage } from './Pages';
import { ProjectsPage } from './ProjectsPage';
import { RecordModal } from './RecordModal';
import { ResultsPage } from './ResultsPage';
import { SettingsPage } from './SettingsPage';
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
    projectRouteId,
    projectRouteMode,
    editClipId,
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

  if (!shareToken && !authReady) {
    return <main className="auth-page"><p role="status">Conectando con el servicio de autenticación…</p></main>;
  }
  if (!shareToken && !authSession) {
    return <AuthPage configurationError={authConfigurationError} />;
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
            : <ResultsPage
              project={project}
              error={projectError}
              onRetry={() => void retryProject()}
              onProjectUpdated={refreshProject}
              onProjectDeleted={() => navigate('/')}
              onEdit={(clipId) => navigate(`/p/${projectRouteId}/edit/${clipId}`)}
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
