import { useEffect, useState } from 'react';
import { ArrowLeft, ChevronDown, Circle, LoaderCircle, Sparkles } from 'lucide-react';
import { useLocation, useNavigate } from 'react-router-dom';
import { apiFetch } from '../auth';
import { API_BASE, type MomentEngine } from '../context/constants';
import type { ThemePreference } from '../hooks/useTheme';
import type { ToastKind } from './Toasts';
import type { IntegrationStatus } from '../types';
import { responseError } from '../utils';

function defaultModelFor(provider: string): string {
  return {
    Groq: 'qwen/qwen3.8-27b',
    OpenAI: 'gpt-4o-mini',
    Anthropic: 'claude-3-7-sonnet-latest',
    OpenRouter: 'openai/gpt-4o-mini',
    'Servidor local compatible': 'local-model',
  }[provider] ?? '';
}

export function SettingsPage({ provider, setProvider, whisper, setWhisper, momentEngine, setMomentEngine, signalThreshold, setSignalThreshold, fastDetection, setFastDetection, theme, setTheme, onBack, onNotice }: {
  provider: string;
  setProvider: (value: string) => void;
  whisper: string;
  setWhisper: (value: string) => void;
  momentEngine: MomentEngine;
  setMomentEngine: (value: MomentEngine) => void;
  signalThreshold: number;
  setSignalThreshold: (value: number) => void;
  fastDetection: boolean;
  setFastDetection: (value: boolean) => void;
  theme: ThemePreference;
  setTheme: (value: ThemePreference) => void;
  onBack: () => void;
  onNotice: (message: string, kind?: ToastKind) => void;
}) {
  const settingsLocation = useLocation();
  const settingsNavigate = useNavigate();
  const [saved, setSaved] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState(defaultModelFor(provider));
  const [baseUrl, setBaseUrl] = useState('');
  const [checking, setChecking] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('');
  const [hasSavedKey, setHasSavedKey] = useState(false);
  const [envKeyConfigured, setEnvKeyConfigured] = useState(false);
  const [integrations, setIntegrations] = useState<Record<string, IntegrationStatus>>({});
  const [integrationBusy, setIntegrationBusy] = useState('');

  useEffect(() => {
    let active = true;
    apiFetch(`${API_BASE}/api/settings`)
      .then(async (response) => {
        if (!response.ok) throw new Error(await responseError(response));
        return response.json() as Promise<{
          provider: string;
          model: string;
          base_url: string;
          has_llm_key: boolean;
          has_saved_llm_key: boolean;
          has_env_llm_key: boolean;
          whisper_model: string;
          integrations: Record<string, IntegrationStatus>;
        }>;
      })
      .then((configuration) => {
        if (!active) return;
        setProvider(configuration.provider);
        setModel(configuration.model || defaultModelFor(configuration.provider));
        setBaseUrl(configuration.base_url);
        setWhisper(configuration.whisper_model);
        setHasSavedKey(configuration.has_saved_llm_key);
        setEnvKeyConfigured(configuration.has_env_llm_key);
        setIntegrations(configuration.integrations);
      })
      .catch((error: unknown) => {
        if (active) setConnectionStatus(error instanceof Error ? error.message : 'No se pudieron leer los ajustes del backend.');
      });
    return () => {
      active = false;
    };
  }, [setProvider, setWhisper]);

  useEffect(() => {
    const params = new URLSearchParams(settingsLocation.search);
    const integration = params.get('integration');
    if (!integration) return;
    const status = params.get('status');
    const message = params.get('message');
    onNotice(status === 'connected'
      ? `${integration} se conectó correctamente.`
      : message || `No se pudo conectar ${integration}.`,
    status === 'connected' ? 'success' : 'error');
    settingsNavigate(settingsLocation.pathname, { replace: true });
  }, [onNotice, settingsLocation.pathname, settingsLocation.search, settingsNavigate]);

  const connectIntegration = async (integration: string) => {
    setIntegrationBusy(integration);
    try {
      const response = await apiFetch(`${API_BASE}/api/integrations/${integration}/connect`);
      if (!response.ok) throw new Error(await responseError(response));
      const result = await response.json() as { authorization_url: string };
      window.location.assign(result.authorization_url);
    } catch (error) {
      onNotice(error instanceof Error ? error.message : 'No se pudo iniciar la conexión.', 'error');
      setIntegrationBusy('');
    }
  };

  const disconnectIntegration = async (integration: string) => {
    setIntegrationBusy(integration);
    try {
      const response = await apiFetch(`${API_BASE}/api/integrations/${integration}/disconnect`, { method: 'POST' });
      if (!response.ok) throw new Error(await responseError(response));
      setIntegrations((current) => ({
        ...current,
        [integration]: { ...current[integration], connected: false, display_name: null },
      }));
      onNotice(`Se desconectó ${integration}.`, 'success');
    } catch (error) {
      onNotice(error instanceof Error ? error.message : 'No se pudo desconectar la cuenta.', 'error');
    } finally {
      setIntegrationBusy('');
    }
  };

  const saveSettings = async () => {
    try {
      const response = await apiFetch(`${API_BASE}/api/settings/llm-key`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider,
          api_key: apiKey.trim(),
          model: model.trim(),
          base_url: baseUrl.trim(),
          whisper_model: whisper,
        }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      setApiKey('');
      setHasSavedKey(true);
      setSaved(true);
      onNotice('Ajustes guardados. La clave se cifra con DPAPI de Windows y no vuelve a mostrarse.', 'success');
    } catch (error) {
      onNotice(error instanceof Error ? error.message : 'No se pudieron guardar los ajustes.', 'error');
    }
  };

  const removeSavedKey = async () => {
    try {
      const response = await apiFetch(`${API_BASE}/api/settings/llm-key`, { method: 'DELETE' });
      if (!response.ok) throw new Error(await responseError(response));
      setHasSavedKey(false);
      setApiKey('');
      setSaved(false);
      onNotice('Se eliminó la clave guardada para esta cuenta.', 'success');
    } catch (error) {
      onNotice(error instanceof Error ? error.message : 'No se pudo eliminar la clave guardada.', 'error');
    }
  };

  const testGroqConnection = async () => {
    const key = apiKey.trim();
    if (!key) {
      setConnectionStatus('Añade tu clave de API para probar la conexión.');
      return;
    }
    setChecking(true);
    setConnectionStatus('Comprobando la clave con Groq…');
    try {
      const response = await fetch('https://api.groq.com/openai/v1/models', {
        headers: { Authorization: `Bearer ${key}` },
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(response.status === 401 ? 'Groq no aceptó la clave. Comprueba que sea correcta.' : `Groq respondió con error (${response.status}). ${detail.slice(0, 180)}`);
      }
      setConnectionStatus('Conexión con Groq correcta. Guarda los ajustes para conservar la clave cifrada en este dispositivo.');
    } catch (error) {
      setConnectionStatus(error instanceof Error ? error.message : 'No se pudo conectar con Groq. Comprueba tu conexión e inténtalo de nuevo.');
    } finally {
      setChecking(false);
    }
  };

  return (
    <main className="settings-page" id="main-content">
      <div className="settings-page-heading"><div><span className="eyebrow">KELLYCLIPAI</span><h1>Ajustes e integraciones</h1><p>Conecta tus servicios para habilitar el análisis de video y las publicaciones.</p></div><button type="button" className="back-button" onClick={onBack}><ArrowLeft size={16} /> Volver a subir</button></div>
      <section className="config-card integration-card">
        <div className="card-heading"><div><h2>Inteligencia artificial</h2><p>Elige un proveedor para encontrar los mejores momentos</p></div><Sparkles size={20} className="integration-icon" /></div>
        <div className="settings-form-grid">
          <label className="config-field"><span>Proveedor de IA</span><span className="select-wrap"><select value={provider} onChange={(event) => { setProvider(event.target.value); setModel(defaultModelFor(event.target.value)); setConnectionStatus(''); setSaved(false); }}><option>Groq</option><option>OpenAI</option><option>Anthropic</option><option>OpenRouter</option><option>Servidor local compatible</option></select><ChevronDown size={16} /></span></label>
          <label className="config-field"><span>Modelo LLM</span><input className="text-input" value={model} onChange={(event) => { setModel(event.target.value); setSaved(false); }} placeholder="Nombre del modelo del proveedor" /></label>
          {provider === 'Servidor local compatible' && <label className="config-field key-field"><span>URL del endpoint local compatible con OpenAI</span><input className="text-input" type="url" value={baseUrl} onChange={(event) => setBaseUrl(event.target.value)} placeholder="http://localhost:1234/v1" /></label>}
          <label className="config-field key-field"><span>Clave API de {provider}</span><input className="text-input" type="password" autoComplete="new-password" value={apiKey} onChange={(event) => { setApiKey(event.target.value); setConnectionStatus(''); setSaved(false); }} placeholder={hasSavedKey ? 'Clave guardada; deja vacío para conservarla' : envKeyConfigured ? 'Clave configurada en .env; pega otra para reemplazarla' : 'Pega aquí tu clave API'} /></label>
          <label className="config-field"><span>Modelo de Whisper</span><span className="select-wrap"><select value={whisper} onChange={(event) => { setWhisper(event.target.value); setSaved(false); }}><option value="tiny">tiny · rápido</option><option value="small">small · equilibrado</option><option value="medium">medium · preciso</option><option value="large-v3">large-v3 · máxima precisión</option></select><ChevronDown size={16} /></span></label>
        </div>
        <p className="settings-explainer">La clave se cifra con DPAPI de Windows y queda asociada a tu cuenta en este dispositivo. Nunca se devuelve al navegador ni se almacena en texto legible.</p>
        {provider === 'Groq' && <div className="key-actions"><button type="button" className="subtle-button" onClick={() => void testGroqConnection()} disabled={checking}>{checking && <LoaderCircle size={14} className="spin" />}{checking ? 'Probando…' : 'Probar conexión'}</button><a href="https://console.groq.com/keys" target="_blank" rel="noreferrer">Obtener clave en Groq</a></div>}
        {connectionStatus && <p className={`connection-message ${connectionStatus.startsWith('Conexión') ? 'success' : ''}`} role="status">{connectionStatus}</p>}
        <div className="key-actions">
          <button type="button" className="subtle-button save-settings" onClick={() => void saveSettings()}>Guardar ajustes</button>
          {hasSavedKey && <button type="button" className="outline-action" onClick={() => void removeSavedKey()}>Eliminar clave guardada</button>}
        </div>
        {saved && <p className="form-message" role="status">Preferencias y clave guardadas para esta cuenta.</p>}
      </section>
      {/* Fase A (v1.1.0): motor de momentos por defecto para los proyectos
      nuevos. Se guarda en este dispositivo; cada proyecto puede cambiarlo
      en su configuración. */}
      <section className="config-card">
        <div className="card-heading"><div><h2>Motor de momentos</h2><p>Cómo se eligen los mejores momentos en los proyectos nuevos</p></div></div>
        <div className="settings-form-grid">
          <label className="config-field"><span>Motor por defecto</span><span className="select-wrap"><select value={momentEngine} onChange={(event) => {
            const value = event.target.value;
            if (value === 'signals' || value === 'signals+ai' || value === 'llm') setMomentEngine(value);
          }}><option value="signals">Señales locales · sin internet ni clave</option><option value="signals+ai">Señales locales + pulido con IA</option><option value="llm">Solo IA (proveedor de arriba)</option></select><ChevronDown size={16} /></span></label>
          {momentEngine !== 'llm' && (
            <label className="config-field"><span>Umbral de puntuación: {signalThreshold.toFixed(1)} / 10</span><input
              className="threshold-slider"
              type="range"
              min={5}
              max={9.5}
              step={0.5}
              value={signalThreshold}
              onChange={(event) => setSignalThreshold(Number.parseFloat(event.target.value))}
              aria-label="Umbral de puntuación del motor de señales"
            /></label>
          )}
        </div>
        {/* v1.3.1: detección rápida (Tier 0) por defecto en los proyectos
        nuevos — solo aplica a los motores de señales. */}
        {momentEngine !== 'llm' && (
          <label className="fast-detection-toggle">
            <input
              type="checkbox"
              checked={fastDetection}
              onChange={(event) => setFastDetection(event.target.checked)}
            />
            <span><strong>Detección rápida por defecto</strong><small>Solo audio y texto (Tier 0): la búsqueda de clips tarda segundos en vez de minutos, sin análisis de video, hablantes ni semántica. La transcripción tarda lo mismo.</small></span>
          </label>
        )}
        <p className="settings-explainer">El motor de señales mide la energía del audio, el ritmo del habla y los ganchos del texto, y solo convierte en clips los momentos por encima del umbral. Si un video no da clips, baja el umbral y reprocesa. El modo “Solo IA” necesita la clave del proveedor.</p>
      </section>
      {/* Fase 2 (ítem 17): apariencia — modo oscuro con seguimiento del sistema. */}
      <section className="config-card">
        <div className="card-heading"><div><h2>Apariencia</h2><p>Elige el tema de la aplicación</p></div></div>
        <div className="theme-options" role="radiogroup" aria-label="Tema de la aplicación">
          <label className={`theme-option ${theme === 'system' ? 'selected' : ''}`}>
            <input type="radio" name="theme" value="system" checked={theme === 'system'} onChange={() => setTheme('system')} />
            <span>Seguir al sistema</span><small>Claro u oscuro según tu sistema operativo</small>
          </label>
          <label className={`theme-option ${theme === 'light' ? 'selected' : ''}`}>
            <input type="radio" name="theme" value="light" checked={theme === 'light'} onChange={() => setTheme('light')} />
            <span>Claro</span><small>Siempre con fondo claro</small>
          </label>
          <label className={`theme-option ${theme === 'dark' ? 'selected' : ''}`}>
            <input type="radio" name="theme" value="dark" checked={theme === 'dark'} onChange={() => setTheme('dark')} />
            <span>Oscuro</span><small>Siempre con fondo oscuro</small>
          </label>
        </div>
      </section>
      <section className="config-card integrations-list">
        <div className="card-heading"><div><h2>Cuentas e integraciones</h2><p>Las credenciales se configuran en `.env`; las cuentas se conectan mediante OAuth local.</p></div></div>
        {[
          { provider: 'zoom', name: 'Zoom · Importar grabaciones', oauth: true },
          { provider: 'youtube', name: 'YouTube · Publicar y programar', oauth: true },
          { provider: 'tiktok', name: 'TikTok · Publicar y programar', oauth: true },
          { provider: 'instagram', name: 'Instagram · Publicar y programar', oauth: true },
          { provider: 'pexels', name: 'Pexels · B-rolls premium', oauth: false },
          { provider: 'pixabay', name: 'Pixabay · B-rolls premium', oauth: false },
        ].map(({ provider: integration, name, oauth }) => {
          const item = integrations[integration];
          const configured = item?.configured ?? false;
          const connected = item?.connected ?? false;
          const missing = item?.missing ?? [];
          const label = connected
            ? `Conectado: ${item.display_name ?? 'cuenta'}`
            : configured
              ? oauth ? 'Credenciales listas' : 'Clave configurada'
              : `Falta: ${missing.join(', ') || 'configurar backend'}`;
          return (
            <div className="integration-row" key={integration}>
              <span className={`integration-status ${connected || (configured && !oauth) ? 'is-connected' : ''}`}><Circle size={8} fill="currentColor" /> {label}</span>
              <strong>{name}</strong>
              {oauth ? (
                connected
                  ? <button type="button" className="connect-button" disabled={integrationBusy === integration} onClick={() => void disconnectIntegration(integration)}>Desconectar</button>
                  : <button type="button" className="connect-button" disabled={!configured || integrationBusy === integration} onClick={() => void connectIntegration(integration)}>{integrationBusy === integration ? 'Conectando…' : 'Conectar cuenta'}</button>
              ) : (
                <button type="button" className="connect-button" onClick={() => onNotice(configured
                  ? `${name.split(' · ')[0]} quedó configurado. Selecciónalo en .env como BROLL_PROVIDER para usarlo.`
                  : `Añade ${missing.join(', ') || 'una clave API'} a .env y reinicia el backend. No compartas la clave en el chat.`)}>Cómo configurar</button>
              )}
            </div>
          );
        })}
      </section>
    </main>
  );
}
