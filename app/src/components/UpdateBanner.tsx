import { useState } from 'react';
import { ChevronDown, Download } from 'lucide-react';
import { useApp } from '../context/AppContext';

// Banner de actualización (extraído de App.tsx, ítem 23 de la Fase 2).
// Fase 3 (ítem 20, C.6.40): muestra las notas de la versión antes de
// instalar y marca las betas. Fase 3 (ítem 17, C.6.37): el mismo
// banner cubre la restauración de la versión anterior.
export function UpdateBanner() {
  const { updates: { updateStatus, installUpdate, downloadUpdate, checkForUpdates } } = useApp();
  const [showNotes, setShowNotes] = useState(false);
  if (!updateStatus?.enabled || !(updateStatus.update_available || ['checking', 'current', 'error'].includes(updateStatus.state))) return null;

  const rollback = updateStatus.rollback;
  const versionLabel = rollback
    ? `la versión anterior ${updateStatus.latest_version}`
    : `kellyClipAi ${updateStatus.latest_version}`;
  const headline = updateStatus.state === 'checking'
    ? rollback ? `Buscando ${versionLabel}…` : 'Buscando actualizaciones…'
    : updateStatus.state === 'downloading'
      ? rollback
        ? `Restaurando ${versionLabel}… ${updateStatus.progress}%`
        : `Descargando kellyClipAi ${updateStatus.latest_version}… ${updateStatus.progress}%`
      : updateStatus.state === 'downloaded'
        ? rollback
          ? `${versionLabel} está lista para instalar`
          : `kellyClipAi ${updateStatus.latest_version} está lista para instalar`
        : updateStatus.state === 'installer_opened'
          ? rollback
            ? 'Sigue los pasos del instalador para restaurar la versión anterior'
            : 'Sigue los pasos del instalador para actualizar kellyClipAi'
          : updateStatus.state === 'error'
            ? updateStatus.error
            : updateStatus.state === 'current'
              ? 'Estás usando la última versión de kellyClipAi.'
              : `Nueva versión ${updateStatus.latest_version} disponible`;
  const showChangelog = updateStatus.release_notes
    && ['available', 'downloaded', 'installer_opened'].includes(updateStatus.state);

  return (
    <aside className="update-banner" aria-live="polite">
      <div className="update-copy">
        <strong>
          {headline}
          {updateStatus.is_prerelease && updateStatus.update_available && <span className="update-beta-badge">beta</span>}
        </strong>
        {updateStatus.state === 'downloading' && (
          <span className="update-progress"><span style={{ width: `${updateStatus.progress}%` }} /></span>
        )}
        {showChangelog && (
          <div className="update-notes">
            <button type="button" className="update-notes-toggle" onClick={() => setShowNotes((current) => !current)} aria-expanded={showNotes}>
              <ChevronDown size={14} className={showNotes ? 'update-notes-chevron open' : 'update-notes-chevron'} />
              {showNotes ? 'Ocultar novedades' : 'Ver novedades'}
            </button>
            {showNotes && (
              <div className="update-notes-body">
                <pre>{updateStatus.release_notes}</pre>
                {updateStatus.release_url && <a href={updateStatus.release_url} target="_blank" rel="noreferrer">Ver en GitHub</a>}
              </div>
            )}
          </div>
        )}
      </div>
      {updateStatus.state === 'available' && <span>La descarga comenzará automáticamente.</span>}
      {updateStatus.state === 'downloaded' && (
        <button type="button" className="subtle-button" onClick={() => void installUpdate()}>
          <Download size={15} /> {rollback ? 'Instalar versión anterior' : 'Instalar actualización'}
        </button>
      )}
      {updateStatus.state === 'installer_opened' && (
        <button type="button" className="subtle-button" onClick={() => void installUpdate()}>
          <Download size={15} /> Abrir instalador de nuevo
        </button>
      )}
      {updateStatus.state === 'error' && (
        <button type="button" className="subtle-button" onClick={() => {
          if (updateStatus.update_available) void downloadUpdate();
          else void checkForUpdates();
        }}>
          Reintentar
        </button>
      )}
    </aside>
  );
}
