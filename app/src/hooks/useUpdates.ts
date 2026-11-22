import { useEffect, useState } from 'react';
import type { Session } from '@supabase/supabase-js';
import { apiFetch } from '../auth';
import type { ToastKind } from '../components/Toasts';
import { responseError } from '../utils';

// Actualizaciones de la aplicación (extraídas de App.tsx, ítem 23 de la
// Fase 2): comprobación periódica, descarga automática cuando hay una
// versión disponible, seguimiento del progreso e instalación.

export type UpdateStatus = {
  enabled: boolean;
  current_version: string;
  latest_version: string | null;
  update_available: boolean;
  state: 'idle' | 'unavailable' | 'checking' | 'current' | 'available' | 'downloading' | 'downloaded' | 'installer_opened' | 'error';
  progress: number;
  error: string | null;
  // Fase 3 (ítem 20, C.6.40): canal beta y notas de la versión.
  channel: 'stable' | 'beta';
  release_notes: string | null;
  release_url: string | null;
  is_prerelease: boolean;
  // Fase 3 (ítem 17, C.6.37): rollback a la versión anterior.
  rollback: boolean;
  previous_version: string | null;
  rollback_available: boolean;
};

type UseUpdatesOptions = {
  apiBase: string;
  authReady: boolean;
  authSession: Session | null;
  userId: string | undefined;
  notify: (message: string, kind?: ToastKind) => void;
};

export function useUpdates({ apiBase, authReady, authSession, userId, notify }: UseUpdatesOptions) {
  const [updateStatus, setUpdateStatus] = useState<UpdateStatus | null>(null);

  useEffect(() => {
    if (!authReady || !authSession) return;
    let active = true;
    const checkForUpdates = async () => {
      try {
        const response = await apiFetch(`${apiBase}/api/updates/check`, { method: 'POST' });
        if (!response.ok) throw new Error(await responseError(response));
        let status = await response.json() as UpdateStatus;
        if (status.state === 'available') {
          const downloadResponse = await apiFetch(`${apiBase}/api/updates/download`, { method: 'POST' });
          if (!downloadResponse.ok) throw new Error(await responseError(downloadResponse));
          status = await downloadResponse.json() as UpdateStatus;
        }
        if (active) setUpdateStatus(status);
      } catch {
        if (active) {
          setUpdateStatus((current) => current ?? {
            enabled: true,
            current_version: '',
            latest_version: null,
            update_available: false,
            state: 'error',
            progress: 0,
            error: 'No se pudo comprobar si hay actualizaciones.',
            channel: 'stable',
            release_notes: null,
            release_url: null,
            is_prerelease: false,
            rollback: false,
            previous_version: null,
            rollback_available: false,
          });
        }
      }
    };
    void checkForUpdates();
    const interval = window.setInterval(() => void checkForUpdates(), 6 * 60 * 60 * 1000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [apiBase, authReady, userId]);

  useEffect(() => {
    if (!authSession || !updateStatus || !['checking', 'downloading'].includes(updateStatus.state)) return;
    let active = true;
    const refresh = async () => {
      try {
        const response = await apiFetch(`${apiBase}/api/updates`);
        if (!response.ok) throw new Error(await responseError(response));
        let status = await response.json() as UpdateStatus;
        if (status.state === 'available') {
          const downloadResponse = await apiFetch(`${apiBase}/api/updates/download`, { method: 'POST' });
          if (!downloadResponse.ok) throw new Error(await responseError(downloadResponse));
          status = await downloadResponse.json() as UpdateStatus;
        }
        if (active) setUpdateStatus(status);
      } catch {
        if (active) {
          setUpdateStatus((current) => current ? {
            ...current,
            state: 'error',
            error: 'Se perdió la conexión mientras se consultaba la actualización.',
          } : current);
        }
      }
    };
    const interval = window.setInterval(() => void refresh(), 1200);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [apiBase, authSession, updateStatus?.state]);

  const downloadUpdate = async () => {
    try {
      const response = await apiFetch(`${apiBase}/api/updates/download`, { method: 'POST' });
      if (!response.ok) throw new Error(await responseError(response));
      setUpdateStatus(await response.json() as UpdateStatus);
    } catch (error) {
      notify(error instanceof Error ? error.message : 'No se pudo iniciar la descarga de la actualización.', 'error');
    }
  };

  const installUpdate = async () => {
    if (!window.confirm('Se abrirá el instalador oficial de kellyClipAi. Guarda tu trabajo antes de continuar.')) return;
    try {
      const response = await apiFetch(`${apiBase}/api/updates/install`, { method: 'POST' });
      if (!response.ok) throw new Error(await responseError(response));
      setUpdateStatus((current) => current ? { ...current, state: 'installer_opened' } : current);
    } catch (error) {
      notify(error instanceof Error ? error.message : 'No se pudo abrir el instalador.', 'error');
    }
  };

  const checkForUpdates = async () => {
    try {
      const response = await apiFetch(`${apiBase}/api/updates/check`, { method: 'POST' });
      if (!response.ok) throw new Error(await responseError(response));
      setUpdateStatus(await response.json() as UpdateStatus);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'No se pudo comprobar si hay actualizaciones.';
      setUpdateStatus((current) => current ? { ...current, state: 'error', error: message } : current);
      notify(message, 'error');
    }
  };

  // Fase 3 (ítem 20, C.6.40): cambiar el canal descarta lo pendiente
  // y busca de inmediato en el canal nuevo.
  const setChannel = async (channel: 'stable' | 'beta') => {
    try {
      const response = await apiFetch(`${apiBase}/api/updates/channel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ channel }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      setUpdateStatus(await response.json() as UpdateStatus);
      await checkForUpdates();
    } catch (error) {
      notify(error instanceof Error ? error.message : 'No se pudo cambiar el canal de actualizaciones.', 'error');
    }
  };

  // Fase 3 (ítem 17, C.6.37): restaurar la versión anterior. El
  // sondeo de progreso existente cubre los estados checking y
  // downloading, así que no hace falta uno propio.
  const startRollback = async () => {
    try {
      const response = await apiFetch(`${apiBase}/api/updates/rollback`, { method: 'POST' });
      if (!response.ok) throw new Error(await responseError(response));
      setUpdateStatus(await response.json() as UpdateStatus);
    } catch (error) {
      notify(error instanceof Error ? error.message : 'No se pudo restaurar la versión anterior.', 'error');
    }
  };

  return { updateStatus, downloadUpdate, installUpdate, checkForUpdates, setChannel, startRollback };
}
