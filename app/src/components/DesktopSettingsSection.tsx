import { useEffect, useState } from 'react';
import { Bell, Download, MonitorDown, RotateCcw } from 'lucide-react';
import { apiFetch } from '../auth';
import { API_BASE, DESKTOP_NOTIFICATIONS_STORAGE_KEY, readStoredDesktopNotifications } from '../context/constants';
import { useApp } from '../context/AppContext';
import type { ToastKind } from './Toasts';
import { responseError } from '../utils';

// Ajustes de escritorio y actualizaciones (Fase 3, ítems 17, 18 y 20):
// canal de actualizaciones beta, restauración de la versión anterior,
// auto-inicio con Windows, bandeja del sistema y notificaciones de
// escritorio. Sección propia para no engordar SettingsPage.
type AutostartStatus = { supported: boolean; enabled: boolean };
type DesktopPrefs = {
  minimize_to_tray: boolean;
  start_minimized: boolean;
  tray_available: boolean;
  tray_reason: string | null;
  portable: boolean;
};

export function DesktopSettingsSection({ onNotice }: { onNotice: (message: string, kind?: ToastKind) => void }) {
  const { updates: { updateStatus, checkForUpdates, setChannel, startRollback } } = useApp();
  const [autostart, setAutostart] = useState<AutostartStatus | null>(null);
  const [prefs, setPrefs] = useState<DesktopPrefs | null>(null);
  const [desktopBusy, setDesktopBusy] = useState(false);
  const [notificationsEnabled, setNotificationsEnabled] = useState(() => readStoredDesktopNotifications());
  const [notificationPermission, setNotificationPermission] = useState(
    () => (typeof window !== 'undefined' && 'Notification' in window ? Notification.permission : 'unsupported'),
  );

  useEffect(() => {
    let active = true;
    apiFetch(`${API_BASE}/api/desktop/autostart`)
      .then(async (response) => {
        if (!response.ok) throw new Error(await responseError(response));
        return response.json() as Promise<AutostartStatus>;
      })
      .then((status) => { if (active) setAutostart(status); })
      .catch(() => { if (active) setAutostart({ supported: false, enabled: false }); });
    apiFetch(`${API_BASE}/api/desktop/preferences`)
      .then(async (response) => {
        if (!response.ok) throw new Error(await responseError(response));
        return response.json() as Promise<DesktopPrefs>;
      })
      .then((loaded) => { if (active) setPrefs(loaded); })
      .catch(() => { if (active) setPrefs(null); });
    return () => { active = false; };
  }, []);

  const toggleAutostart = async (enabled: boolean) => {
    setDesktopBusy(true);
    try {
      const response = await apiFetch(`${API_BASE}/api/desktop/autostart`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      setAutostart({ supported: true, enabled });
      onNotice(enabled ? 'kellyClipAi se iniciará con Windows.' : 'kellyClipAi ya no se iniciará con Windows.', 'success');
    } catch (error) {
      onNotice(error instanceof Error ? error.message : 'No se pudo cambiar el auto-inicio.', 'error');
    } finally {
      setDesktopBusy(false);
    }
  };

  const savePrefs = async (updates: Partial<Pick<DesktopPrefs, 'minimize_to_tray' | 'start_minimized'>>) => {
    setDesktopBusy(true);
    try {
      const response = await apiFetch(`${API_BASE}/api/desktop/preferences`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates),
      });
      if (!response.ok) throw new Error(await responseError(response));
      setPrefs(await response.json() as DesktopPrefs);
    } catch (error) {
      onNotice(error instanceof Error ? error.message : 'No se pudieron guardar las preferencias de escritorio.', 'error');
    } finally {
      setDesktopBusy(false);
    }
  };

  const toggleNotifications = (enabled: boolean) => {
    setNotificationsEnabled(enabled);
    try {
      window.localStorage.setItem(DESKTOP_NOTIFICATIONS_STORAGE_KEY, enabled ? '1' : '0');
    } catch {
      // Sin almacenamiento disponible: la preferencia no persiste.
    }
  };

  const requestNotificationPermission = async () => {
    if (!('Notification' in window)) return;
    const permission = await Notification.requestPermission();
    setNotificationPermission(permission);
    if (permission === 'granted') onNotice('Notificaciones de escritorio activadas.', 'success');
    else onNotice('El sistema no permitió las notificaciones. Puedes activarlas después en la configuración del navegador o de Windows.', 'error');
  };

  const channel = updateStatus?.channel ?? 'stable';
  const rollbackBusy = Boolean(updateStatus?.rollback) && ['checking', 'downloading'].includes(updateStatus?.state ?? '');

  return (
    <>
      <section className="config-card">
        <div className="card-heading"><div><h2>Actualizaciones</h2><p>Canal de actualizaciones y restauración de la versión anterior</p></div><Download size={20} className="integration-icon" /></div>
        {updateStatus && (
          <p className="settings-explainer">
            Versión instalada: <strong>{updateStatus.current_version}</strong>
            {updateStatus.latest_version && updateStatus.latest_version !== updateStatus.current_version
              ? <> · Última versión: <strong>{updateStatus.latest_version}</strong>{updateStatus.is_prerelease ? ' (beta)' : ''}</>
              : ' · Estás al día.'}
          </p>
        )}
        <div className="settings-form-grid">
          <label className="config-field"><span>Canal de actualizaciones</span><span className="select-wrap"><select
            value={channel}
            disabled={!updateStatus?.enabled}
            onChange={(event) => {
              const value = event.target.value;
              if (value === 'stable' || value === 'beta') void setChannel(value);
            }}
          ><option value="stable">Estable · solo versiones publicadas</option><option value="beta">Beta · también versiones de prueba</option></select></span></label>
        </div>
        <p className="settings-explainer">El canal beta recibe las versiones de prueba antes que nadie, a cambio de posibles fallos. Al cambiar de canal se busca una actualización de inmediato.</p>
        {!updateStatus?.enabled && <p className="settings-explainer">Las actualizaciones automáticas solo están disponibles en la app instalada en Windows{updateStatus === null ? '' : '.'}</p>}
        {updateStatus?.rollback_available && updateStatus.previous_version && (
          <div className="key-actions">
            <button
              type="button"
              className="subtle-button"
              disabled={rollbackBusy || !updateStatus.enabled}
              onClick={() => {
                if (window.confirm(`Se restaurará kellyClipAi ${updateStatus.previous_version}. Se abrirá su instalador; guarda tu trabajo antes de continuar.`)) void startRollback();
              }}
            >
              <RotateCcw size={14} /> {rollbackBusy ? `Restaurando ${updateStatus.previous_version}… ${updateStatus.progress}%` : `Restaurar versión anterior (${updateStatus.previous_version})`}
            </button>
          </div>
        )}
        {updateStatus?.enabled && (
          <div className="key-actions">
            <button type="button" className="subtle-button" onClick={() => void checkForUpdates()}>Buscar actualizaciones ahora</button>
          </div>
        )}
      </section>

      <section className="config-card">
        <div className="card-heading"><div><h2>Escritorio</h2><p>Bandeja del sistema, inicio con Windows y notificaciones</p></div><MonitorDown size={20} className="integration-icon" /></div>
        {prefs?.portable && (
          <p className="settings-explainer">Modo portable: tus proyectos y ajustes viven junto a la aplicación. Las actualizaciones se instalan descargando el ZIP nuevo desde GitHub.</p>
        )}
        {!prefs?.tray_available && (
          <p className="settings-explainer">{prefs?.tray_reason ?? 'Comprobando la disponibilidad de la bandeja…'}</p>
        )}
        <label className="fast-detection-toggle">
          <input
            type="checkbox"
            checked={prefs?.minimize_to_tray ?? true}
            disabled={!prefs?.tray_available || desktopBusy}
            onChange={(event) => void savePrefs({ minimize_to_tray: event.target.checked })}
          />
          <span><strong>Cerrar la ventana la minimiza a la bandeja</strong><small>El procesamiento sigue en segundo plano al cerrar la ventana; la app solo se cierra con “Salir” en el icono de la bandeja.</small></span>
        </label>
        <label className="fast-detection-toggle">
          <input
            type="checkbox"
            checked={prefs?.start_minimized ?? false}
            disabled={!prefs?.tray_available || desktopBusy}
            onChange={(event) => void savePrefs({ start_minimized: event.target.checked })}
          />
          <span><strong>Iniciar minimizado en la bandeja</strong><small>La app arranca sin mostrar la ventana. Se aplica al reiniciar la aplicación.</small></span>
        </label>
        <label className="fast-detection-toggle">
          <input
            type="checkbox"
            checked={autostart?.enabled ?? false}
            disabled={!autostart?.supported || desktopBusy}
            onChange={(event) => void toggleAutostart(event.target.checked)}
          />
          <span><strong>Iniciar kellyClipAi con Windows</strong><small>{autostart?.supported ? 'La app se abrirá sola al iniciar sesión en Windows.' : 'Solo disponible en la app instalada en Windows.'}</small></span>
        </label>
        <label className="fast-detection-toggle">
          <input
            type="checkbox"
            checked={notificationsEnabled}
            disabled={notificationPermission === 'unsupported'}
            onChange={(event) => toggleNotifications(event.target.checked)}
          />
          <span><strong><Bell size={14} /> Notificaciones de escritorio</strong><small>Avisar cuando un procesamiento largo termine, aunque la ventana esté minimizada o en la bandeja.</small></span>
        </label>
        {notificationPermission === 'default' && (
          <div className="key-actions">
            <button type="button" className="subtle-button" onClick={() => void requestNotificationPermission()}>Permitir notificaciones del sistema</button>
          </div>
        )}
        {notificationPermission === 'denied' && (
          <p className="settings-explainer">Las notificaciones están bloqueadas en el sistema. Actívalas en la configuración de notificaciones de Windows para recibir los avisos.</p>
        )}
      </section>
    </>
  );
}
