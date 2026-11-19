import { useCallback, useEffect, useRef, useState } from 'react';
import { AlertCircle, Check, Info, X } from 'lucide-react';

export type ToastKind = 'success' | 'error' | 'info';
export type Toast = { id: number; message: string; kind: ToastKind };

const TOAST_DURATION_MS = 4000;
const MAX_VISIBLE_TOASTS = 5;

export function useToasts() {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const nextId = useRef(1);
  const timers = useRef(new Map<number, number>());

  const dismissToast = useCallback((id: number) => {
    const timer = timers.current.get(id);
    if (timer !== undefined) {
      window.clearTimeout(timer);
      timers.current.delete(id);
    }
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const pushToast = useCallback((message: string, kind: ToastKind = 'info') => {
    if (!message) return;
    const id = nextId.current;
    nextId.current += 1;
    setToasts((current) => [...current.slice(-(MAX_VISIBLE_TOASTS - 1)), { id, message, kind }]);
    timers.current.set(id, window.setTimeout(() => dismissToast(id), TOAST_DURATION_MS));
  }, [dismissToast]);

  useEffect(() => {
    const pending = timers.current;
    return () => {
      pending.forEach((timer) => window.clearTimeout(timer));
      pending.clear();
    };
  }, []);

  return { toasts, pushToast, dismissToast };
}

export function ToastStack({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: number) => void }) {
  if (!toasts.length) return null;
  return (
    <div className="toast-stack" aria-live="polite">
      {toasts.map((toast) => (
        <div key={toast.id} className={`toast toast-${toast.kind}`} role={toast.kind === 'error' ? 'alert' : 'status'}>
          {toast.kind === 'success'
            ? <Check size={16} />
            : toast.kind === 'error'
              ? <AlertCircle size={16} />
              : <Info size={16} />}
          <span>{toast.message}</span>
          <button type="button" onClick={() => onDismiss(toast.id)} aria-label="Cerrar aviso"><X size={14} /></button>
        </div>
      ))}
    </div>
  );
}
