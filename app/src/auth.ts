import { createClient, type Session, type SupabaseClient } from '@supabase/supabase-js';

let supabase: SupabaseClient | null = null;

export async function initializeSupabase(apiBase: string): Promise<{
  client: SupabaseClient | null;
  session: Session | null;
}> {
  const response = await fetch(`${apiBase}/api/auth/config`);
  if (!response.ok) throw new Error('No se pudo leer la configuración de Supabase.');
  const config = await response.json() as {
    configured: boolean;
    supabase_url: string;
    supabase_anon_key: string;
  };
  if (!config.configured) return { client: null, session: null };
  supabase = createClient(config.supabase_url, config.supabase_anon_key, {
    auth: {
      autoRefreshToken: true,
      detectSessionInUrl: true,
      persistSession: true,
    },
  });
  const { data, error } = await supabase.auth.getSession();
  if (error) throw new Error(`No se pudo recuperar la sesión: ${error.message}`);
  return { client: supabase, session: data.session };
}

export function getSupabase(): SupabaseClient {
  if (!supabase) throw new Error('Supabase todavía no está inicializado.');
  return supabase;
}

export async function accessToken(): Promise<string | null> {
  if (!supabase) return null;
  const { data, error } = await supabase.auth.getSession();
  if (error) throw new Error(`No se pudo recuperar la sesión: ${error.message}`);
  return data.session?.access_token ?? null;
}

export async function apiFetch(
  input: RequestInfo | URL,
  init: RequestInit = {},
): Promise<Response> {
  const token = await accessToken();
  const headers = new Headers(init.headers);
  if (token) headers.set('Authorization', `Bearer ${token}`);
  return fetch(input, { ...init, headers, credentials: 'include' });
}

export async function syncMediaCookie(apiBase: string, token: string | null): Promise<void> {
  if (!token) return;
  const response = await fetch(`${apiBase}/api/auth/media-cookie`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
    credentials: 'include',
  });
  if (!response.ok) {
    throw new Error('No se pudo habilitar la reproducción autenticada de los clips.');
  }
}

export async function streamApiEvents(
  input: RequestInfo | URL,
  onEvent: (event: string, data: string) => void | Promise<void>,
  signal: AbortSignal,
): Promise<void> {
  const response = await apiFetch(input, {
    headers: { Accept: 'text/event-stream' },
    signal,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: string } | null;
    throw new Error(body?.detail ?? `El servidor respondió con error (${response.status}).`);
  }
  if (!response.body) throw new Error('El servidor no inició el flujo de progreso.');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const dispatch = async (frame: string) => {
    let event = 'message';
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data.push(line.slice(5).trimStart());
    }
    if (data.length) await onEvent(event, data.join('\n'));
  };
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done }).replace(/\r\n/g, '\n');
    let separator = buffer.indexOf('\n\n');
    while (separator >= 0) {
      await dispatch(buffer.slice(0, separator));
      buffer = buffer.slice(separator + 2);
      separator = buffer.indexOf('\n\n');
    }
    if (done) {
      if (buffer.trim()) await dispatch(buffer);
      return;
    }
  }
}

export async function signOut(apiBase: string): Promise<void> {
  const token = await accessToken();
  if (token) {
    const response = await fetch(`${apiBase}/api/auth/media-cookie`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${token}` },
      credentials: 'include',
    });
    if (!response.ok) throw new Error('No se pudo cerrar la sesión multimedia local.');
  }
  const { error } = await getSupabase().auth.signOut();
  if (error) throw new Error(`No se pudo cerrar la sesión: ${error.message}`);
}
