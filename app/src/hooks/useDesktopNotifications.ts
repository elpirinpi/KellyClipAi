import { useEffect, useRef } from 'react';
import { NOTIFICATIONS_ASKED_STORAGE_KEY, readStoredDesktopNotifications } from '../context/constants';
import type { ProjectResult } from '../types';

// Notificaciones de escritorio (Fase 3, ítem 18, C.4.31): cuando un
// procesamiento largo termina y la ventana no está visible (minimizada
// o en la bandeja), avisa con la Notification API del sistema en vez
// de obligar a vigilar la página. Con la ventana visible no hace
// falta: la propia app ya muestra el resultado.
export function useDesktopNotifications({ project }: { project: ProjectResult | null }) {
  const prevStatusRef = useRef<string | undefined>(undefined);
  const askedRef = useRef(false);

  useEffect(() => {
    const status = project?.status;
    const prev = prevStatusRef.current;
    prevStatusRef.current = status;
    if (!status || typeof window === 'undefined' || !('Notification' in window)) return;

    // Pedir permiso una sola vez, al ver el primer procesamiento en
    // curso: sin permiso no hay notificación que mostrar.
    if ((status === 'queued' || status === 'processing')
      && Notification.permission === 'default'
      && !askedRef.current) {
      askedRef.current = true;
      try {
        if (window.localStorage.getItem(NOTIFICATIONS_ASKED_STORAGE_KEY) !== '1') {
          window.localStorage.setItem(NOTIFICATIONS_ASKED_STORAGE_KEY, '1');
          void Notification.requestPermission();
        }
      } catch {
        // Sin almacenamiento disponible: no se insiste.
      }
    }

    if (!readStoredDesktopNotifications()) return;
    if ((prev === 'queued' || prev === 'processing')
      && (status === 'completed' || status === 'failed' || status === 'cancelled')
      && document.hidden
      && Notification.permission === 'granted') {
      const title = project?.title || 'Tu video';
      const body = status === 'completed'
        ? `${title}: clips listos para revisar.`
        : status === 'failed'
          ? `${title}: el procesamiento falló. Revisa el proyecto para ver el error.`
          : `${title}: procesamiento cancelado.`;
      const notification = new Notification('kellyClipAi', { body });
      notification.onclick = () => {
        window.focus();
        notification.close();
      };
    }
  }, [project?.status, project?.title]);
}
