import { Download } from 'lucide-react';
import { useApp } from '../context/AppContext';

// Banner de actualización (extraído de App.tsx, ítem 23 de la Fase 2).
export function UpdateBanner() {
  const { updates: { updateStatus, installUpdate, downloadUpdate, checkForUpdates } } = useApp();
  if (!updateStatus?.enabled || !(updateStatus.update_available || ['checking', 'current', 'error'].includes(updateStatus.state))) return null;

  return (
    <aside className="update-banner" aria-live="polite">
      <div className="update-copy">
        <strong>
          {updateStatus.state === 'checking'
            ? 'Buscando actualizaciones…'
            : updateStatus.state === 'downloading'
              ? `Descargando kellyClipAi ${updateStatus.latest_version}… ${updateStatus.progress}%`
              : updateStatus.state === 'downloaded'
                ? `kellyClipAi ${updateStatus.latest_version} está lista para instalar`
                : updateStatus.state === 'installer_opened'
                  ? 'Sigue los pasos del instalador para actualizar kellyClipAi'
                  : updateStatus.state === 'error'
                    ? updateStatus.error
                    : updateStatus.state === 'current'
                      ? 'Estás usando la última versión de kellyClipAi.'
                    : `Nueva versión ${updateStatus.latest_version} disponible`}
        </strong>
        {updateStatus.state === 'downloading' && (
          <span className="update-progress"><span style={{ width: `${updateStatus.progress}%` }} /></span>
        )}
      </div>
      {updateStatus.state === 'available' && <span>La descarga comenzará automáticamente.</span>}
      {updateStatus.state === 'downloaded' && (
        <button type="button" className="subtle-button" onClick={() => void installUpdate()}>
          <Download size={15} /> Instalar actualización
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
