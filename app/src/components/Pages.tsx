import { useEffect, useState, type FormEvent } from 'react';
import { AlertCircle, Check, Circle, FileVideo2, LoaderCircle, WandSparkles } from 'lucide-react';
import { getSupabase } from '../auth';
import type { ProjectResult, SharedClip } from '../types';
import { formatDuration, responseError } from '../utils';

export function AuthPage({ configurationError }: { configurationError: string }) {
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setBusy(true);
    setError('');
    setMessage('');
    try {
      if (mode === 'register') {
        const { data, error: signUpError } = await getSupabase().auth.signUp({
          email: email.trim(),
          password,
          options: { emailRedirectTo: window.location.origin },
        });
        if (signUpError) throw signUpError;
        if (!data.session) {
          setMessage('Cuenta creada. Verifica tu correo y después inicia sesión.');
        }
      } else {
        const { error: signInError } = await getSupabase().auth.signInWithPassword({
          email: email.trim(),
          password,
        });
        if (signInError) throw signInError;
      }
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : 'No se pudo completar la solicitud.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="auth-page">
      <section className="auth-card" aria-labelledby="auth-title">
        <div className="auth-brand" aria-label="kellyClipAi">
          <span className="brand-mark" aria-hidden="true"><span /><span /><span /></span>
          <span className="brand-name">kelly<span>Clip</span><b>Ai</b></span>
        </div>
        <h1 id="auth-title">{mode === 'login' ? 'Inicia sesión' : 'Crea tu cuenta'}</h1>
        <p className="auth-intro">
          {mode === 'login'
            ? 'Accede a tus proyectos y continúa creando clips.'
            : 'Regístrate con tu correo para usar kellyClipAi.'}
        </p>
        {configurationError && <p className="error-message" role="alert">{configurationError}</p>}
        {!configurationError && (
          <form className="auth-form" onSubmit={(event) => void submit(event)}>
            <label className="config-field">
              <span>Correo electrónico</span>
              <input
                className="text-input"
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(event) => setEmail(event.target.value)}
              />
            </label>
            <label className="config-field">
              <span>Contraseña</span>
              <input
                className="text-input"
                type="password"
                autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                minLength={8}
                required
                value={password}
                onChange={(event) => setPassword(event.target.value)}
              />
            </label>
            {error && <p className="error-message" role="alert">{error}</p>}
            {message && <p className="connection-message success" role="status">{message}</p>}
            <button type="submit" className="auth-submit" disabled={busy}>
              {busy ? 'Un momento…' : mode === 'login' ? 'Entrar' : 'Registrarme'}
            </button>
          </form>
        )}
        {!configurationError && (
          <button
            type="button"
            className="auth-switch"
            onClick={() => {
              setMode((current) => current === 'login' ? 'register' : 'login');
              setMessage('');
              setError('');
            }}
          >
            {mode === 'login' ? '¿No tienes cuenta? Regístrate' : '¿Ya tienes cuenta? Inicia sesión'}
          </button>
        )}
      </section>
      <footer className="developer-credit">Desarrollado por <strong>KELLY BENJAMIN DE LA CRUZ</strong></footer>
    </main>
  );
}

export function ProcessingPage({
  apiBase,
  project,
  error,
  onRetry,
}: {
  apiBase: string;
  project: ProjectResult | null;
  error: string;
  onRetry: () => void;
}) {
  const steps = [
    { key: 'upload', label: 'Subida' },
    { key: 'project', label: 'Crear proyecto' },
    { key: 'process_video', label: 'Procesar video' },
    { key: 'finding', label: 'Buscando las mejores partes' },
    { key: 'editing', label: 'Editar clips' },
    { key: 'finalize', label: 'Finalizar' },
  ];
  const activeIndex = project?.status === 'queued'
    ? 2
    : Math.max(0, steps.findIndex((step) => step.key === project?.step));
  const details = project?.metadata;

  return (
    <main className="processing-page">
      <section className="processing-card">
        <div className="processing-video-card">
          {project?.thumbnail_url
            ? <img src={`${apiBase}${project.thumbnail_url}`} alt="Miniatura del video" />
            : <span className="processing-placeholder"><FileVideo2 size={26} /></span>}
          <div className="processing-video-copy">
            <strong title={project?.title}>{project?.title ?? 'Cargando proyecto…'}</strong>
            <div className="processing-badges">
              {details && <span className="meta-pill">{Math.max(details.width, details.height) >= 1900 ? '1080p' : `${Math.max(details.width, details.height)}px`}</span>}
              {details && <span className="meta-pill">{formatDuration(details.duration)}</span>}
              {!details && <span className="meta-pill">Procesando</span>}
            </div>
          </div>
        </div>
        <div className="processing-heading">
          <WandSparkles size={25} />
          <div>
            <h1>Analizando contenido y encontrando clips</h1>
            <p>Permanece en la página mientras el video se procesa.</p>
          </div>
        </div>
        {error && <p className="error-message" role="alert"><AlertCircle size={16} />{error}</p>}
        <ol className="processing-steps">
          {steps.map((step, index) => {
            const complete = index < activeIndex || project?.status === 'completed';
            const current = index === activeIndex && project?.status !== 'completed' && project?.status !== 'failed';
            const failed = index === activeIndex && project?.status === 'failed';
            const editingProgress = project?.progress == null || project.message?.includes('%')
              ? ''
              : ` ${project.progress}%`;
            const label = step.key === 'finding' && current
              ? `Buscando las mejores partes… ${project?.progress == null ? '' : `${project.progress}%`}`
              : step.key === 'editing' && current
                ? `${project?.message || 'Editando clips…'}${editingProgress}`
              : step.label;
            return (
              <li key={step.key} className={`${complete ? 'complete' : ''} ${current ? 'current' : ''} ${failed ? 'failed' : ''}`}>
                <span className="step-indicator">{complete ? <Check size={14} /> : current ? <LoaderCircle size={15} className="spin" /> : <Circle size={14} />}</span>
                <span>{label}</span>
              </li>
            );
          })}
        </ol>
        {(project?.status === 'failed' || error) && (
          <button type="button" className="retry-button" onClick={onRetry}><WandSparkles size={16} /> Reintentar</button>
        )}
      </section>
    </main>
  );
}

export function SharedClipPage({ apiBase, token }: { apiBase: string; token: string }) {
  const [shared, setShared] = useState<SharedClip | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${apiBase}/api/shares/${encodeURIComponent(token)}`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(await responseError(response));
        return response.json() as Promise<SharedClip>;
      })
      .then(setShared)
      .catch((loadError: unknown) => {
        if (!controller.signal.aborted) {
          setError(loadError instanceof Error ? loadError.message : 'No se pudo cargar el clip compartido.');
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [apiBase, token]);

  return (
    <main className="shared-page">
      <a className="back-button" href="/">kellyClipAi</a>
      {loading && <p className="results-loading"><LoaderCircle size={17} className="spin" /> Cargando clip compartido…</p>}
      {error && <p className="error-message" role="alert"><AlertCircle size={16} />{error}</p>}
      {shared && (
        <article className="shared-card">
          <span className="eyebrow">{shared.project_title}</span>
          <h1>{shared.title}</h1>
          <div className="shared-video-frame">
            <video
              src={`${apiBase}${shared.video_url}`}
              controls
              playsInline
              preload="metadata"
              aria-label={`Clip compartido: ${shared.title}`}
            />
          </div>
          <p>{formatDuration(shared.duration)} · {shared.width}×{shared.height}</p>
        </article>
      )}
    </main>
  );
}
