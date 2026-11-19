import { useEffect, type MutableRefObject } from 'react';
import type { ProjectResult } from '../types';

// Atajos de teclado globales (extraídos de App.tsx, ítem 23 de la Fase 2):
// - Espacio: play/pausa del video (solo si el foco no está en un campo
//   editable ni en un control que ya usa Espacio, como un botón).
// - Esc: cerrar los modales abiertos (salvo grabación o carga en curso).
// - Ctrl/Cmd+Enter: generar clips desde la vista de configuración.

type UseGlobalShortcutsOptions = {
  recordOpen: boolean;
  setRecordOpen: (open: boolean) => void;
  zoomOpen: boolean;
  setZoomOpen: (open: boolean) => void;
  recording: boolean;
  zoomLoading: boolean;
  route: string;
  projectRouteMode: 'configure' | 'processing' | 'edit' | 'results';
  isSubmitting: boolean;
  linkProject: ProjectResult | null;
  progress: number | null;
  submitProjectRef: MutableRefObject<() => Promise<void>>;
};

export function useGlobalShortcuts({
  recordOpen,
  setRecordOpen,
  zoomOpen,
  setZoomOpen,
  recording,
  zoomLoading,
  route,
  projectRouteMode,
  isSubmitting,
  linkProject,
  progress,
  submitProjectRef,
}: UseGlobalShortcutsOptions) {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        if (recordOpen && !recording) {
          setRecordOpen(false);
          return;
        }
        if (zoomOpen && !zoomLoading) setZoomOpen(false);
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        const onHomeOrConfigure = route === '/' || projectRouteMode === 'configure';
        if (onHomeOrConfigure && !isSubmitting && (linkProject || progress === 100)) {
          event.preventDefault();
          void submitProjectRef.current();
        }
        return;
      }
      if (event.key !== ' ') return;
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName.toLowerCase();
      const editable = tag === 'input' || tag === 'textarea' || tag === 'select' || Boolean(target?.isContentEditable);
      if (editable) return;
      if (tag === 'button' || tag === 'a' || target?.getAttribute('role') === 'button') return;
      if (recordOpen || zoomOpen) return;
      const focused = document.activeElement;
      const videos = Array.from(document.querySelectorAll('video'));
      const video = focused instanceof HTMLVideoElement
        ? focused
        : videos.find((item) => !item.paused && !item.ended) ?? videos[0];
      if (!video) return;
      event.preventDefault();
      if (video.paused) void video.play();
      else video.pause();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [recordOpen, zoomOpen, recording, zoomLoading, route, projectRouteMode, isSubmitting, linkProject, progress, setRecordOpen, setZoomOpen, submitProjectRef]);
}
