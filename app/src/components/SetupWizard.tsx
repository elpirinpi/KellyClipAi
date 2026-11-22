import { useEffect, useState } from 'react';
import { ChevronDown, HardDrive, LoaderCircle, Sparkles } from 'lucide-react';
import { apiFetch } from '../auth';
import { API_BASE, formatBytes } from '../context/constants';
import { responseError } from '../utils';

// Asistente de primer arranque (Fase 3, ítem 19, C.6.42): en vez de
// copiar .env.example en silencio, una primera pantalla que pide la
// clave LLM (opcional: la app funciona sin ella con el motor por
// señales), explica la descarga del modelo Whisper, ofrece elegir el
// modelo según el equipo y comprueba el espacio libre en disco.
type WhisperModelOption = {
  name: string;
  size_bytes: number;
  label: string;
  description: string;
  downloaded: boolean;
};

type SetupStatus = {
  needs_setup: boolean;
  completed: boolean;
  has_llm_key: boolean;
  has_projects: boolean;
  whisper_model: string;
  whisper_models: WhisperModelOption[];
  disk_free_bytes: number | null;
  disk_total_bytes: number | null;
  portable: boolean;
};

const HEADROOM_BYTES = 1024 * 1024 * 1024;

function defaultModelFor(provider: string): string {
  return {
    Groq: 'qwen/qwen3.8-27b',
    OpenAI: 'gpt-4o-mini',
    Anthropic: 'claude-3-7-sonnet-latest',
    OpenRouter: 'openai/gpt-4o-mini',
    'Servidor local compatible': 'local-model',
  }[provider] ?? '';
}

export function SetupWizard({ onDone }: { onDone: () => void }) {
  const [setup, setSetup] = useState<SetupStatus | null>(null);
  const [loadError, setLoadError] = useState('');
  const [step, setStep] = useState(0);
  const [provider, setProvider] = useState('Groq');
  const [apiKey, setApiKey] = useState('');
  const [keySaved, setKeySaved] = useState(false);
  const [whisper, setWhisper] = useState('small');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;
    apiFetch(`${API_BASE}/api/setup`)
      .then(async (response) => {
        if (!response.ok) throw new Error(await responseError(response));
        return response.json() as Promise<SetupStatus>;
      })
      .then((status) => {
        if (!active) return;
        setSetup(status);
        setWhisper(status.whisper_model);
      })
      .catch((caught: unknown) => {
        if (active) setLoadError(caught instanceof Error ? caught.message : 'No se pudo leer la configuración inicial.');
      });
    return () => {
      active = false;
    };
  }, []);

  const complete = async (skipped: boolean) => {
    setBusy(true);
    setError('');
    try {
      const response = await apiFetch(`${API_BASE}/api/setup/complete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(skipped ? { skipped: true } : { whisper_model: whisper }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      onDone();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'No se pudo guardar la configuración inicial.');
    } finally {
      setBusy(false);
    }
  };

  const saveKeyAndContinue = async () => {
    const key = apiKey.trim();
    if (!key) {
      setStep(1);
      return;
    }
    setBusy(true);
    setError('');
    try {
      const response = await apiFetch(`${API_BASE}/api/settings/llm-key`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider,
          api_key: key,
          model: defaultModelFor(provider),
          base_url: '',
          whisper_model: null,
        }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      setApiKey('');
      setKeySaved(true);
      setStep(1);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'No se pudo guardar la clave.');
    } finally {
      setBusy(false);
    }
  };

  if (loadError) {
    return (
      <main className="setup-wizard-page" id="main-content">
        <section className="setup-wizard-card">
          <h1>Bienvenido a kellyClipAi</h1>
          <p className="error-message" role="alert">{loadError}</p>
          <p>Puedes omitir el asistente y configurar todo después en Ajustes.</p>
          <button type="button" className="create-button" onClick={() => void complete(true)} disabled={busy}>
            {busy ? 'Omitiendo…' : 'Omitir asistente'}
          </button>
          {error && <p className="error-message" role="alert">{error}</p>}
        </section>
      </main>
    );
  }

  if (!setup) {
    return (
      <main className="setup-wizard-page" id="main-content">
        <section className="setup-wizard-card">
          <p role="status"><LoaderCircle size={16} className="spin" /> Preparando el asistente…</p>
        </section>
      </main>
    );
  }

  const freeBytes = setup.disk_free_bytes;
  const modelFits = (sizeBytes: number) => freeBytes === null || freeBytes >= sizeBytes + HEADROOM_BYTES;

  return (
    <main className="setup-wizard-page" id="main-content">
      <section className="setup-wizard-card">
        <span className="eyebrow">KELLYCLIPAI · PRIMER ARRANQUE</span>
        <h1>Bienvenido a kellyClipAi</h1>
        <ol className="setup-steps" aria-label="Pasos del asistente">
          <li className={step === 0 ? 'current' : step > 0 ? 'done' : ''}>Clave de IA</li>
          <li className={step === 1 ? 'current' : step > 1 ? 'done' : ''}>Modelo de voz</li>
          <li className={step === 2 ? 'current' : ''}>Listo</li>
        </ol>

        {step === 0 && (
          <div className="setup-step">
            <h2>Tu clave de IA <span className="setup-optional">(opcional)</span></h2>
            <p>
              kellyClipAi encuentra los mejores momentos de tus videos con un motor de señales
              local: <strong>funciona sin clave y sin internet</strong>. La clave solo añade el
              pulido de títulos con IA. Puedes saltar este paso y añadirla después en Ajustes.
            </p>
            {setup.has_llm_key && <p className="form-message" role="status">Ya hay una clave configurada: puedes continuar sin cambiar nada.</p>}
            <div className="settings-form-grid">
              <label className="config-field"><span>Proveedor de IA</span><span className="select-wrap"><select value={provider} onChange={(event) => setProvider(event.target.value)}><option>Groq</option><option>OpenAI</option><option>Anthropic</option><option>OpenRouter</option><option>Servidor local compatible</option></select><ChevronDown size={16} /></span></label>
              <label className="config-field key-field"><span>Clave API de {provider}</span><input className="text-input" type="password" autoComplete="new-password" value={apiKey} onChange={(event) => setApiKey(event.target.value)} placeholder="Pega aquí tu clave API (puedes dejarlo vacío)" /></label>
            </div>
            <p className="settings-explainer"><Sparkles size={14} /> La clave se cifra con DPAPI de Windows y queda asociada a tu cuenta en este dispositivo. Con Groq, la clave gratuita basta para el pulido de títulos.</p>
            {provider === 'Groq' && <p><a href="https://console.groq.com/keys" target="_blank" rel="noreferrer">Obtener clave gratuita en Groq</a></p>}
            {error && <p className="error-message" role="alert">{error}</p>}
            <div className="setup-actions">
              <button type="button" className="create-button" onClick={() => void saveKeyAndContinue()} disabled={busy}>
                {busy && <LoaderCircle size={14} className="spin" />}{busy ? 'Guardando…' : apiKey.trim() ? 'Guardar y continuar' : 'Continuar sin clave'}
              </button>
              <button type="button" className="subtle-button" onClick={() => void complete(true)} disabled={busy}>Omitir asistente</button>
            </div>
          </div>
        )}

        {step === 1 && (
          <div className="setup-step">
            <h2>Modelo de voz (Whisper)</h2>
            <p>
              Para transcribir tus videos, la app descarga un modelo de voz de Whisper la primera
              vez que lo usa (unos 500 MB con el modelo recomendado). La descarga es automática;
              aquí solo eliges qué modelo según tu equipo.
            </p>
            <p className="setup-disk" role="status">
              <HardDrive size={15} />
              {freeBytes === null
                ? 'No se pudo medir el espacio libre en disco.'
                : `Espacio libre en disco: ${formatBytes(freeBytes)}. Se reserva 1 GB además del modelo para tus videos y clips.`}
            </p>
            <div className="setup-models" role="radiogroup" aria-label="Modelo de Whisper">
              {setup.whisper_models.map((model) => {
                const fits = modelFits(model.size_bytes);
                return (
                  <label key={model.name} className={`setup-model ${whisper === model.name ? 'selected' : ''} ${fits ? '' : 'disabled'}`}>
                    <input
                      type="radio"
                      name="whisper-model"
                      value={model.name}
                      checked={whisper === model.name}
                      disabled={!fits}
                      onChange={() => setWhisper(model.name)}
                    />
                    <span><strong>{model.label} · {formatBytes(model.size_bytes)}</strong><small>{model.description}</small></span>
                    {model.downloaded && <em className="setup-downloaded">Ya descargado</em>}
                    {!fits && <em className="setup-nofit">No cabe en el disco libre</em>}
                  </label>
                );
              })}
            </div>
            <div className="setup-actions">
              <button type="button" className="subtle-button" onClick={() => setStep(0)} disabled={busy}>Atrás</button>
              <button type="button" className="create-button" onClick={() => setStep(2)} disabled={busy}>Continuar</button>
            </div>
          </div>
        )}

        {step === 2 && (
          <div className="setup-step">
            <h2>Todo listo</h2>
            <p>
              Modelo de voz: <strong>{setup.whisper_models.find((model) => model.name === whisper)?.label ?? whisper}</strong>
              {setup.whisper_models.find((model) => model.name === whisper)?.downloaded
                ? ' (ya descargado).'
                : '. Se descargará automáticamente la primera vez que proceses un video.'}
            </p>
            <p>
              {setup.has_llm_key || keySaved
                ? 'Clave de IA configurada: el pulido de títulos con IA está disponible.'
                : 'Sin clave de IA: la app usará el motor de señales local. Puedes añadir una clave cuando quieras en Ajustes.'}
            </p>
            {setup.portable && <p>Estás en modo portable: tus proyectos y ajustes viven junto a la aplicación.</p>}
            {error && <p className="error-message" role="alert">{error}</p>}
            <div className="setup-actions">
              <button type="button" className="subtle-button" onClick={() => setStep(1)} disabled={busy}>Atrás</button>
              <button type="button" className="create-button" onClick={() => void complete(false)} disabled={busy}>
                {busy && <LoaderCircle size={14} className="spin" />}{busy ? 'Guardando…' : 'Empezar a usar kellyClipAi'}
              </button>
            </div>
          </div>
        )}
      </section>
    </main>
  );
}
