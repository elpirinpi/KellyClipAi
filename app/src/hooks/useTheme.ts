import { useCallback, useEffect, useState } from 'react';

// Fase 2 (ítem 17): modo oscuro.
// Preferencia del usuario: 'system' sigue a prefers-color-scheme del
// sistema operativo; 'light' / 'dark' la imponen. Se persiste en
// localStorage para que sobreviva entre sesiones y se aplica con el
// atributo data-theme en <html>, que styles.css usa para redefinir los
// colores. Funciona en cualquier equipo: no depende de resolución ni de
// hardware, solo de la preferencia del sistema o de la elección manual.
export type ThemePreference = 'system' | 'light' | 'dark';
export type ResolvedTheme = 'light' | 'dark';

const THEME_STORAGE_KEY = 'kellyclipai-theme';

function readStoredPreference(): ThemePreference {
  try {
    const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (stored === 'light' || stored === 'dark' || stored === 'system') return stored;
  } catch {
    // Sin almacenamiento disponible: se sigue al sistema.
  }
  return 'system';
}

function systemIsDark(): boolean {
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}

export function useTheme() {
  const [preference, setPreferenceState] = useState<ThemePreference>(() => readStoredPreference());
  const [resolved, setResolved] = useState<ResolvedTheme>(() => (
    readStoredPreference() === 'dark'
      || (readStoredPreference() === 'system' && systemIsDark())
      ? 'dark'
      : 'light'
  ));

  useEffect(() => {
    const apply = () => {
      const next: ResolvedTheme = preference === 'dark'
        || (preference === 'system' && systemIsDark())
        ? 'dark'
        : 'light';
      setResolved(next);
      document.documentElement.dataset.theme = next;
    };
    apply();
    // Con 'system', reaccionar si el usuario cambia el tema del SO.
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    media.addEventListener('change', apply);
    return () => media.removeEventListener('change', apply);
  }, [preference]);

  const setPreference = useCallback((next: ThemePreference) => {
    setPreferenceState(next);
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, next);
    } catch {
      // Sin almacenamiento disponible: la preferencia vive solo en la sesión.
    }
  }, []);

  // El interruptor de la cabecera alterna entre claro y oscuro de forma
  // explícita (sale de 'system' al pulsarlo por primera vez).
  const toggleTheme = useCallback(() => {
    setPreference(resolved === 'dark' ? 'light' : 'dark');
  }, [resolved, setPreference]);

  return { preference, resolved, setPreference, toggleTheme };
}
