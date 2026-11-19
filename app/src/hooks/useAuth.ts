import { useCallback, useEffect, useState } from 'react';
import type { Session } from '@supabase/supabase-js';
import { initializeSupabase, signOut, syncMediaCookie } from '../auth';

type UseAuthResult = {
  authReady: boolean;
  authSession: Session | null;
  configurationError: string;
  signOut: () => Promise<void>;
};

export function useAuth(
  apiBase: string,
  onSessionError: (message: string) => void,
): UseAuthResult {
  const [authReady, setAuthReady] = useState(false);
  const [authSession, setAuthSession] = useState<Session | null>(null);
  const [configurationError, setConfigurationError] = useState('');

  useEffect(() => {
    let active = true;
    let unsubscribe: (() => void) | undefined;
    const syncSession = (session: Session | null) => {
      if (session?.access_token) {
        void syncMediaCookie(apiBase, session.access_token).catch((error: unknown) => {
          if (active) onSessionError(error instanceof Error ? error.message : 'No se pudo preparar la sesión.');
        });
      }
    };
    const startAuth = async () => {
      try {
        const { client, session } = await initializeSupabase(apiBase);
        if (!active) return;
        if (!client) {
          setConfigurationError('Falta configurar SUPABASE_URL y SUPABASE_ANON_KEY en el archivo .env local.');
        } else {
          setAuthSession(session);
          syncSession(session);
          const { data } = client.auth.onAuthStateChange((_event, nextSession) => {
            if (!active) return;
            setAuthSession(nextSession);
            syncSession(nextSession);
          });
          unsubscribe = () => data.subscription.unsubscribe();
        }
      } catch (error) {
        if (active) {
          setConfigurationError(error instanceof Error ? error.message : 'No se pudo iniciar la autenticación.');
        }
      } finally {
        if (active) setAuthReady(true);
      }
    };
    void startAuth();
    return () => {
      active = false;
      unsubscribe?.();
    };
  }, [apiBase, onSessionError]);

  const handleSignOut = useCallback(async () => {
    await signOut(apiBase);
    setAuthSession(null);
  }, [apiBase]);

  return { authReady, authSession, configurationError, signOut: handleSignOut };
}
