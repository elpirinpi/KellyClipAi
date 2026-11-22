import { ArrowLeft, CalendarDays, Download, Film, FolderOpen, Layers, LoaderCircle, LogOut, Moon, Settings2, Sun } from 'lucide-react';
import { useApp } from '../context/AppContext';

// Cabecera de la aplicación (extraída de App.tsx, ítem 23 de la Fase 2).
export function TopBar() {
  const {
    navigate,
    isProjects,
    isCalendar,
    isQueue,
    isBroll,
    isSettings,
    route,
    settingsReturnPath,
    setSettingsReturnPath,
    shareToken,
    theme: { resolved: resolvedTheme, toggleTheme },
    updates: { updateStatus, checkForUpdates },
    handleSignOut,
  } = useApp();

  return (
    <header className="topbar">
      <button className="brand" type="button" onClick={() => navigate('/')} aria-label="kellyClipAi, inicio">
        <span className="brand-mark" aria-hidden="true"><span /><span /><span /></span>
        <span className="brand-name">kelly<span>Clip</span><b>Ai</b></span>
      </button>
      <div className="topbar-actions">
      {/* Fase 2 (ítem 15): navegación a "Mis proyectos". */}
      <button className={`settings-link ${isProjects ? 'is-active' : ''}`} type="button" onClick={() => navigate('/projects')} aria-label="Mis proyectos">
        <FolderOpen size={17} />
        <span>Mis proyectos</span>
      </button>
      {/* Fase 3 (ítems 9, 11, 13): calendario, cola y biblioteca de B-roll. */}
      <button className={`settings-link ${isCalendar ? 'is-active' : ''}`} type="button" onClick={() => navigate('/calendar')} aria-label="Calendario de publicaciones">
        <CalendarDays size={17} />
        <span>Calendario</span>
      </button>
      <button className={`settings-link ${isQueue ? 'is-active' : ''}`} type="button" onClick={() => navigate('/queue')} aria-label="Cola de procesamiento">
        <Layers size={17} />
        <span>Cola</span>
      </button>
      <button className={`settings-link ${isBroll ? 'is-active' : ''}`} type="button" onClick={() => navigate('/broll')} aria-label="Biblioteca de B-roll">
        <Film size={17} />
        <span>B-roll</span>
      </button>
      {/* Fase 2 (ítem 17): interruptor manual de modo oscuro. */}
      <button className="settings-link" type="button" onClick={toggleTheme} aria-label={resolvedTheme === 'dark' ? 'Cambiar a modo claro' : 'Cambiar a modo oscuro'}>
        {resolvedTheme === 'dark' ? <Sun size={17} /> : <Moon size={17} />}
        <span>{resolvedTheme === 'dark' ? 'Claro' : 'Oscuro'}</span>
      </button>
      <button className={`settings-link ${isSettings ? 'is-active' : ''}`} type="button" onClick={() => {
        if (isSettings) navigate(settingsReturnPath);
        else {
          setSettingsReturnPath(route);
          navigate('/settings');
        }
      }} aria-label={isSettings ? 'Volver al proyecto' : 'Ajustes'}>
        {isSettings ? <ArrowLeft size={17} /> : <Settings2 size={17} />}
        <span>{isSettings ? 'Volver' : 'Ajustes'}</span>
      </button>
      {updateStatus?.enabled && <button
        className="settings-link update-check-button"
        type="button"
        onClick={() => void checkForUpdates()}
        disabled={updateStatus.state === 'checking'}
        aria-label="Comprobar actualizaciones"
      >
        {updateStatus.state === 'checking' ? <LoaderCircle size={17} className="spin" /> : <Download size={17} />}
        <span>{updateStatus.state === 'checking' ? 'Comprobando…' : 'Buscar actualizaciones'}</span>
      </button>}
      {!shareToken && <button className="settings-link" type="button" onClick={() => void handleSignOut()} aria-label="Cerrar sesión">
        <LogOut size={17} /><span>Cerrar sesión</span>
      </button>}
      </div>
    </header>
  );
}
