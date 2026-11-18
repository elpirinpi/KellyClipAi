import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent, type DragEvent, type FormEvent } from 'react';
import {
  AlertCircle, ArrowLeft, ArrowRight, AudioLines, CalendarDays, Check, ChevronDown,
  ChevronLeft, ChevronRight, Circle, CloudUpload, Download, FileText, FileVideo2, Link2, LoaderCircle,
  LockKeyhole, LogOut, MonitorUp, Pause, Plus, Scissors, Settings2, Share2, Sparkles, Trash2, Video, WandSparkles, X,
} from 'lucide-react';
import { accessToken, apiFetch, getSupabase, initializeSupabase, signOut, streamApiEvents, syncMediaCookie } from './auth';
import type { Session } from '@supabase/supabase-js';

const API_BASE = __KELLYCLIP_API_BASE__;

type UploadInfo = {
  id: string;
  name: string;
  size: number;
  duration: number;
  width: number;
  height: number;
  kind: 'video' | 'audio';
};

type StagedMedia = {
  id: string;
  source_type: 'upload' | 'url';
  status: 'receiving' | 'downloading' | 'ready' | 'failed' | 'finalizing' | 'completed';
  filename: string;
  title: string | null;
  file_size: number;
  duration: number;
  width: number;
  height: number;
  kind: 'video' | 'audio';
  bytes_received: number;
  total_bytes: number | null;
  speed: number | null;
  eta: number | null;
  error: string | null;
  project_id: string | null;
};
type Template = { name: string; tone: string; sample: string; featured?: boolean; base_template?: string };
type UpdateStatus = {
  enabled: boolean;
  current_version: string;
  latest_version: string | null;
  update_available: boolean;
  state: 'idle' | 'unavailable' | 'checking' | 'current' | 'available' | 'downloading' | 'downloaded' | 'installer_opened' | 'error';
  progress: number;
  error: string | null;
};
type ClipWord = { start: number; end: number; preview_start: number; preview_end: number; text: string };
type EditablePhrase = {
  id: string;
  start: number;
  end: number;
  preview_start: number;
  text: string;
  removed: boolean;
};
type SubtitleStyleState = {
  font_family: string | null;
  font_size: number | null;
  font_color: string | null;
  highlight_color: string | null;
  position: 'top' | 'center' | 'bottom' | null;
};
const DEFAULT_SUBTITLE_STYLE: SubtitleStyleState = {
  font_family: null,
  font_size: null,
  font_color: null,
  highlight_color: null,
  position: null,
};
type ClipEditorState = {
  ratio: string;
  subtitle_template: string;
  subtitle_style: SubtitleStyleState;
  phrases: EditablePhrase[];
};
type ClipResult = {
  id: string;
  status?: 'ready' | 'failed';
  error?: string;
  title: string;
  start: number;
  end: number;
  duration: number;
  score: number;
  viral_reason: string;
  keywords: string[];
  emojis: string[];
  width: number;
  height: number;
  resolution?: 720 | 1080;
  audio_enhanced?: boolean;
  filename: string;
  video_url: string | null;
  download_url: string | null;
  broll_provider?: string;
  broll_creator?: string;
  broll_attribution_url?: string;
  transcript: ClipWord[];
};
type ProjectResult = {
  id: string;
  title: string;
  status: string;
  step: string;
  progress: number | null;
  error: string | null;
  config: {
    ratio: string;
    duration: string;
    framing?: 'speaker' | 'split' | 'blur' | 'center';
    whisper_language?:
      | 'auto'
      | 'ar'
      | 'de'
      | 'en'
      | 'es'
      | 'fr'
      | 'hi'
      | 'it'
      | 'ja'
      | 'ko'
      | 'nl'
      | 'pl'
      | 'pt'
      | 'ru'
      | 'tr'
      | 'uk'
      | 'zh';
    subtitle_template: string;
    subtitle_style?: SubtitleStyleState;
    options: Record<string, boolean>;
    moment: string;
    schedule?: { scheduled_at: string; platform: string; visibility: string } | null;
  };
  metadata: { duration: number; width: number; height: number; language?: string } | null;
  transcript: Array<{ start: number; end: number; text: string; words: Array<{ start: number; end: number; text: string }> }> | null;
  clips: ClipResult[];
  thumbnail_url: string | null;
  created_at: string;
  updated_at: string;
};
type SharedClip = {
  project_title: string;
  clip_id: string;
  title: string;
  duration: number;
  width: number;
  height: number;
  video_url: string;
};
type IntegrationStatus = {
  configured: boolean;
  missing: string[];
  connected: boolean;
  display_name: string | null;
};
type ZoomRecording = {
  meeting_id: string;
  topic: string;
  start_time: string | null;
  files: Array<{ file_id: string; file_type: string; file_size: number | null }>;
};
type ScheduleJob = {
  id: string;
  clip_id: string;
  platform: string;
  scheduled_at: string;
  status: string;
  confirmation_required: boolean;
  visibility: string;
  external_id?: string | null;
  external_url: string | null;
  error: string | null;
  download_url: string | null;
};

declare global {
  interface Window {
    pywebview?: {
      api: {
        select_download_folder: () => Promise<string | null>;
      };
    };
  }
}

function defaultModelFor(provider: string): string {
  return {
    Groq: 'openai/gpt-oss-120b',
    OpenAI: 'gpt-4o-mini',
    Anthropic: 'claude-3-7-sonnet-latest',
    OpenRouter: 'openai/gpt-4o-mini',
    'Servidor local compatible': 'local-model',
  }[provider] ?? '';
}

async function responseError(response: Response): Promise<string> {
  try {
    const body = await response.json() as { detail?: string };
    return body.detail ?? `El servidor respondió con error (${response.status}).`;
  } catch {
    return `El servidor respondió con error (${response.status}).`;
  }
}

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function downloadWithAuth(url: string, filename: string): Promise<void> {
  const response = await apiFetch(url);
  if (!response.ok) throw new Error(await responseError(response));
  triggerDownload(await response.blob(), filename);
}

const ALLOWED_EXTENSIONS = new Set(['mp4', 'mov', '3gp', 'avi', 'mkv', 'webm', 'mp3', 'wav', 'm4a']);
const ACCEPT = '.mp4,.mov,.3gp,.avi,.mkv,.webm,.mp3,.wav,.m4a,video/mp4,video/quicktime,video/3gpp,video/x-msvideo,video/webm,audio/mpeg,audio/wav,audio/mp4';
const templates: Template[] = [
  { name: 'Default', tone: 'default', sample: 'Un gran momento' },
  { name: 'Modern', tone: 'modern', sample: 'Ideas que inspiran' },
  { name: 'Bouncy', tone: 'bouncy', sample: 'Mira esto ahora' },
  { name: 'MrBeast', tone: 'impact', sample: 'NO TE LO PIERDAS' },
  { name: 'Business', tone: 'business', sample: 'Una nueva perspectiva' },
];
const customTemplates: Template[] = [
  { name: 'Esencial', tone: 'essential', sample: 'Una idea a la vez' },
  { name: 'Editorial', tone: 'editorial', sample: 'La historia comienza' },
  { name: 'Neón', tone: 'neon', sample: 'Hazlo memorable' },
];
const platforms = [
  { label: 'YouTube', domain: 'youtube.com', glyph: '▶', color: '#f04438' },
  { label: 'YouTube', domain: 'youtu.be', glyph: '▶', color: '#f04438' },
  { label: 'Twitch', domain: 'twitch.tv', glyph: '▰', color: '#8b5cf6' },
  { label: 'X', domain: 'x.com', glyph: '𝕏', color: '#1f2430' },
  { label: 'X', domain: 'twitter.com', glyph: '𝕏', color: '#1f2430' },
  { label: 'TikTok', domain: 'tiktok.com', glyph: '♪', color: '#141414' },
  { label: 'Instagram', domain: 'instagram.com', glyph: '◎', color: '#d94684' },
  { label: 'Facebook', domain: 'facebook.com', glyph: 'f', color: '#1877f2' },
  { label: 'Vimeo', domain: 'vimeo.com', glyph: 'v', color: '#1ab7ea' },
];

function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '—';
  const value = Math.floor(seconds);
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const secs = value % 60;
  return hours > 0
    ? `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
    : `${minutes}:${secs.toString().padStart(2, '0')}`;
}

function stagedUploadInfo(staged: StagedMedia): UploadInfo {
  return {
    id: staged.id,
    name: staged.filename,
    size: staged.file_size,
    duration: staged.duration,
    width: staged.width,
    height: staged.height,
    kind: staged.kind,
  };
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit > 0 && value < 10 ? 1 : 0)} ${units[unit]}`;
}

async function uploadStagedFile(
  stagedId: string,
  file: File,
  onProgress: (progress: number) => void,
  onRequest: (request: XMLHttpRequest | null) => void,
): Promise<void> {
  const token = await accessToken();
  if (!token) throw new Error('Inicia sesión antes de cargar un video.');
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('POST', `${API_BASE}/api/staged-media/${stagedId}/content`);
    request.setRequestHeader('Content-Type', 'application/octet-stream');
    request.setRequestHeader('Authorization', `Bearer ${token}`);
    request.upload.onprogress = (event) => {
      if (event.lengthComputable && event.total > 0) {
        onProgress(Math.min(99, Math.round((event.loaded / event.total) * 100)));
      }
    };
    request.onload = () => {
      onRequest(null);
      if (request.status >= 200 && request.status < 300) {
        resolve();
        return;
      }
      try {
        const body = JSON.parse(request.responseText) as { detail?: string };
        reject(new Error(body.detail ?? `El servidor respondió con error (${request.status}).`));
      } catch {
        reject(new Error(`El servidor respondió con error (${request.status}).`));
      }
    };
    request.onerror = () => {
      onRequest(null);
      reject(new Error('No se pudo conectar con el almacenamiento local.'));
    };
    request.onabort = () => {
      onRequest(null);
      reject(new Error('Importación cancelada.'));
    };
    onRequest(request);
    request.send(file);
  });
}

async function pollStagedMedia(
  stagedId: string,
  onUpdate: (staged: StagedMedia) => void,
  active: () => boolean,
): Promise<StagedMedia> {
  while (active()) {
    const response = await apiFetch(`${API_BASE}/api/staged-media/${stagedId}`);
    if (!response.ok) throw new Error(await responseError(response));
    const staged = await response.json() as StagedMedia;
    onUpdate(staged);
    if (staged.status === 'ready' || staged.status === 'completed') return staged;
    if (staged.status === 'failed') {
      throw new Error(staged.error ?? 'La importación falló. Vuelve a intentarlo.');
    }
    await new Promise((resolve) => window.setTimeout(resolve, 600));
  }
  throw new Error('La importación se interrumpió.');
}

function metadataFor(file: File): Promise<{ duration: number; width: number; height: number; thumbnail: string }> {
  const url = URL.createObjectURL(file);
  return new Promise((resolve) => {
    const isAudio = file.type.startsWith('audio/') || ['mp3', 'wav', 'm4a'].includes(extensionOf(file.name));
    const media = document.createElement(isAudio ? 'audio' : 'video');
    let completed = false;
    const finish = (thumbnail = '') => {
      if (completed) return;
      completed = true;
      clearTimeout(timer);
      const result = {
        duration: Number.isFinite(media.duration) ? media.duration : 0,
        width: media instanceof HTMLVideoElement ? media.videoWidth : 0,
        height: media instanceof HTMLVideoElement ? media.videoHeight : 0,
        thumbnail,
      };
      URL.revokeObjectURL(url);
      media.removeAttribute('src');
      media.load();
      resolve(result);
    };
    const takeThumbnail = () => {
      if (!(media instanceof HTMLVideoElement) || !media.videoWidth || !media.videoHeight) {
        finish();
        return;
      }
      try {
        const canvas = document.createElement('canvas');
        canvas.width = media.videoWidth;
        canvas.height = media.videoHeight;
        canvas.getContext('2d')?.drawImage(media, 0, 0, canvas.width, canvas.height);
        finish(canvas.toDataURL('image/jpeg', 0.78));
      } catch {
        finish();
      }
    };
    const timer = window.setTimeout(() => finish(), 3500);
    media.preload = 'metadata';
    media.onloadedmetadata = () => {
      if (media instanceof HTMLVideoElement && media.videoWidth && media.duration > 0) {
        media.currentTime = Math.min(1, media.duration / 2);
        media.onseeked = takeThumbnail;
        media.onloadeddata = takeThumbnail;
      } else {
        finish();
      }
    };
    media.onerror = () => finish();
    media.src = url;
  });
}

function extensionOf(name: string): string {
  return name.split('.').pop()?.toLowerCase() ?? '';
}

function AuthPage({ configurationError }: { configurationError: string }) {
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

function App() {
  const [route, setRoute] = useState(window.location.pathname);
  const [authReady, setAuthReady] = useState(false);
  const [authSession, setAuthSession] = useState<Session | null>(null);
  const [authConfigurationError, setAuthConfigurationError] = useState('');
  const [fileInfo, setFileInfo] = useState<UploadInfo | null>(null);
  const [thumbnail, setThumbnail] = useState('');
  const [progress, setProgress] = useState<number | null>(null);
  const [uploadError, setUploadError] = useState('');
  const [dragging, setDragging] = useState(false);
  const [link, setLink] = useState('');
  const [linkMessage, setLinkMessage] = useState('');
  const [downloadProgress, setDownloadProgress] = useState<StagedMedia | null>(null);
  const [ratio, setRatio] = useState('9:16');
  const [duration, setDuration] = useState('Cualquiera');
  const [framing, setFraming] = useState<'speaker' | 'split' | 'blur' | 'center'>('speaker');
  const [activeTemplate, setActiveTemplate] = useState('Default');
  const [templateTab, setTemplateTab] = useState<'featured' | 'custom' | 'brand'>('featured');
  const [savedTemplates, setSavedTemplates] = useState<Template[]>([]);
  const [templateName, setTemplateName] = useState('');
  const [savingTemplate, setSavingTemplate] = useState(false);
  const [options, setOptions] = useState({
    emojis: true,
    keywords: true,
    broll: false,
    silence: false,
    censor: false,
    face_tracking: true,
  });
  const [moment, setMoment] = useState('');
  const [schedule, setSchedule] = useState(false);
  const [scheduleDate, setScheduleDate] = useState('');
  const [schedulePlatform, setSchedulePlatform] = useState('youtube');
  const [scheduleVisibility, setScheduleVisibility] = useState('private');
  const [notice, setNotice] = useState('');
  const [updateStatus, setUpdateStatus] = useState<UpdateStatus | null>(null);
  const [settingsProvider, setSettingsProvider] = useState('Groq');
  const [whisperModel, setWhisperModel] = useState('small');
  const [whisperLanguage, setWhisperLanguage] = useState('auto');
  const [subtitleStyle, setSubtitleStyle] = useState<SubtitleStyleState>(DEFAULT_SUBTITLE_STYLE);
  const [linkProject, setLinkProject] = useState<ProjectResult | null>(null);
  const [project, setProject] = useState<ProjectResult | null>(null);
  const [projectError, setProjectError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [settingsReturnPath, setSettingsReturnPath] = useState('/');
  const [recordOpen, setRecordOpen] = useState(false);
  const [recordMode, setRecordMode] = useState<'camera' | 'screen'>('camera');
  const [recording, setRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [recordError, setRecordError] = useState('');
  const [recordMic, setRecordMic] = useState(true);
  const [zoomOpen, setZoomOpen] = useState(false);
  const [zoomRecordings, setZoomRecordings] = useState<ZoomRecording[]>([]);
  const [zoomLoading, setZoomLoading] = useState(false);
  const [zoomError, setZoomError] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  const cancelImportRef = useRef(false);
  const importGenerationRef = useRef(0);
  const uploadRequestRef = useRef<XMLHttpRequest | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recordChunksRef = useRef<Blob[]>([]);
  const currentUserId = authSession?.user.id;

  useEffect(() => {
    let active = true;
    let unsubscribe: (() => void) | undefined;
    const startAuth = async () => {
      try {
        const { client, session } = await initializeSupabase(API_BASE);
        if (!active) return;
        if (!client) {
          setAuthConfigurationError(
            'Falta configurar SUPABASE_URL y SUPABASE_ANON_KEY en el archivo .env local.',
          );
        } else {
          setAuthSession(session);
          if (session?.access_token) {
            void syncMediaCookie(API_BASE, session.access_token).catch((error: unknown) => {
              if (active) setNotice(error instanceof Error ? error.message : 'No se pudo preparar la sesión.');
            });
          }
          const { data } = client.auth.onAuthStateChange((_event, nextSession) => {
            if (!active) return;
            setAuthSession(nextSession);
            if (nextSession?.access_token) {
              void syncMediaCookie(API_BASE, nextSession.access_token).catch((error: unknown) => {
                if (active) setNotice(error instanceof Error ? error.message : 'No se pudo preparar la sesión.');
              });
            }
          });
          unsubscribe = () => data.subscription.unsubscribe();
        }
      } catch (error) {
        if (active) {
          setAuthConfigurationError(
            error instanceof Error ? error.message : 'No se pudo iniciar la autenticación.',
          );
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
  }, []);

  useEffect(() => {
    if (!authReady || !authSession) return;
    let active = true;
    const checkForUpdates = async () => {
      try {
        const response = await apiFetch(`${API_BASE}/api/updates/check`, { method: 'POST' });
        if (!response.ok) throw new Error(await responseError(response));
        let status = await response.json() as UpdateStatus;
        if (status.state === 'available') {
          const downloadResponse = await apiFetch(`${API_BASE}/api/updates/download`, { method: 'POST' });
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
  }, [authReady, currentUserId]);

  useEffect(() => {
    if (!authSession || !updateStatus || !['checking', 'downloading'].includes(updateStatus.state)) return;
    let active = true;
    const refresh = async () => {
      try {
        const response = await apiFetch(`${API_BASE}/api/updates`);
        if (!response.ok) throw new Error(await responseError(response));
        let status = await response.json() as UpdateStatus;
        if (status.state === 'available') {
          const downloadResponse = await apiFetch(`${API_BASE}/api/updates/download`, { method: 'POST' });
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
  }, [authSession, updateStatus?.state]);

  const navigate = useCallback((next: string) => {
    window.history.pushState(null, '', next);
    setRoute(next);
    setNotice('');
    if (next === '/') setLinkProject(null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, []);

  useEffect(() => {
    const onPopState = () => setRoute(window.location.pathname);
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  useEffect(() => {
    if (!authReady || !authSession) return;
    let active = true;
    const restoreUpload = async () => {
      try {
        const response = await apiFetch(`${API_BASE}/api/staged-media/latest?source=upload`);
        if (!response.ok) throw new Error(await responseError(response));
        const latest = await response.json() as StagedMedia | null;
        if (!active || !latest) return;
        if (latest.status === 'failed') {
          setUploadError(latest.error ?? 'La última carga no se completó. Vuelve a seleccionar el archivo.');
          return;
        }
        setFileInfo(stagedUploadInfo(latest));
        setThumbnail('');
        setProgress(latest.status === 'ready'
          ? 100
          : latest.file_size > 0
            ? Math.min(99, Math.floor((latest.bytes_received / latest.file_size) * 100))
            : 0);
        if (latest.status === 'receiving') {
          await pollStagedMedia(
            latest.id,
            (staged) => {
              if (!active) return;
              setProgress(staged.status === 'ready'
                ? 100
                : staged.file_size > 0
                  ? Math.min(99, Math.floor((staged.bytes_received / staged.file_size) * 100))
                  : 0);
              if (staged.status === 'failed') {
                setUploadError(staged.error ?? 'La carga se interrumpió. Vuelve a seleccionar el archivo.');
                setProgress(null);
                setFileInfo(null);
              }
            },
            () => active,
          );
        }
      } catch (error) {
        if (active) setUploadError(error instanceof Error ? error.message : 'No se pudo recuperar la carga local.');
      }
    };
    void restoreUpload();
    return () => {
      active = false;
    };
  }, [authReady, currentUserId]);

  useEffect(() => {
    if (!authReady || !authSession) return;
    let active = true;
    apiFetch(`${API_BASE}/api/templates`)
      .then(async (response) => {
        if (!response.ok) throw new Error(await responseError(response));
        return response.json() as Promise<Array<{ name: string; base_template: string }>>;
      })
      .then((saved) => {
        if (!active) return;
        const presets = [...templates, ...customTemplates];
        setSavedTemplates(saved.map((item) => {
          const base = presets.find((preset) => preset.name === item.base_template) ?? templates[0];
          return { ...base, name: item.name, base_template: item.base_template };
        }));
      })
      .catch((error: unknown) => {
        if (active) setNotice(error instanceof Error ? error.message : 'No se pudieron cargar tus plantillas.');
      });
    return () => {
      active = false;
    };
  }, [authReady, currentUserId]);

  useEffect(() => {
    if (!recording) return;
    const interval = window.setInterval(() => setRecordingTime((value) => value + 1), 1000);
    return () => window.clearInterval(interval);
  }, [recording]);

  const importFile = useCallback(async (file: File, fromRecorder = false) => {
    setUploadError('');
    setNotice('');
    const extension = extensionOf(file.name);
    if (!fromRecorder && !ALLOWED_EXTENSIONS.has(extension)) {
      setUploadError('Formato no compatible. Usa MP4, MOV, 3GP, AVI, MP3, WAV o M4A.');
      return;
    }
    if (file.size === 0) {
      setUploadError('El archivo está vacío. Elige otro archivo.');
      return;
    }

    uploadRequestRef.current?.abort();
    cancelImportRef.current = false;
    const generation = ++importGenerationRef.current;
    setLinkProject(null);
    setFileInfo(null);
    setThumbnail('');
    setProgress(0);
    const kind = file.type.startsWith('audio/') || ['mp3', 'wav', 'm4a'].includes(extension) ? 'audio' : 'video';
    try {
      const metadata = await metadataFor(file);
      if (cancelImportRef.current || generation !== importGenerationRef.current) {
        setProgress(null);
        return;
      }
      const createResponse = await apiFetch(`${API_BASE}/api/staged-media/uploads`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filename: file.name,
          file_size: file.size,
          content_type: file.type || 'application/octet-stream',
          duration: metadata.duration,
          width: metadata.width,
          height: metadata.height,
          kind,
        }),
      });
      if (!createResponse.ok) throw new Error(await responseError(createResponse));
      const staged = await createResponse.json() as StagedMedia;
      if (cancelImportRef.current || generation !== importGenerationRef.current) {
        const deleteResponse = await apiFetch(`${API_BASE}/api/staged-media/${staged.id}`, { method: 'DELETE' });
        if (!deleteResponse.ok) throw new Error(await responseError(deleteResponse));
        setProgress(null);
        return;
      }
      const info = stagedUploadInfo(staged);
      setFileInfo(info);
      setThumbnail(metadata.thumbnail);
      await uploadStagedFile(
        staged.id,
        file,
        (value) => {
          if (generation === importGenerationRef.current) setProgress(value);
        },
        (request) => {
          if (generation === importGenerationRef.current) uploadRequestRef.current = request;
        },
      );
      if (generation === importGenerationRef.current) {
        setProgress(100);
        setUploadError('');
      }
    } catch (error) {
      if (generation !== importGenerationRef.current) return;
      if (uploadRequestRef.current) uploadRequestRef.current = null;
      setProgress(null);
      setFileInfo(null);
      setUploadError(error instanceof Error ? error.message : 'No se pudo guardar el archivo en este dispositivo.');
    }
  }, []);

  const chooseFile = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) void importFile(file);
    event.target.value = '';
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) void importFile(file);
  };

  const cancelImport = () => {
    cancelImportRef.current = true;
    uploadRequestRef.current?.abort();
    setProgress(null);
    setUploadError('Importación cancelada.');
  };

  const detectedPlatform = useMemo(() => {
    try {
      const hostname = new URL(link).hostname.toLowerCase();
      return platforms.find((platform) => hostname === platform.domain || hostname.endsWith(`.${platform.domain}`));
    } catch {
      return undefined;
    }
  }, [link]);

  const handleLink = async (event: FormEvent) => {
    event.preventDefault();
    if (!link.trim()) {
      setLinkMessage('Pega un enlace para continuar.');
      return;
    }
    if (!detectedPlatform) {
      setLinkMessage('No reconocemos esta plataforma. Comprueba el enlace.');
      return;
    }
    setIsSubmitting(true);
    setDownloadProgress(null);
    setLinkMessage(`Descargando desde ${detectedPlatform.label} con yt-dlp. Solo importa contenido propio o autorizado.`);
    try {
      const startResponse = await apiFetch(`${API_BASE}/api/staged-media/downloads`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: link.trim(), config: projectConfig() }),
      });
      if (!startResponse.ok) throw new Error(await responseError(startResponse));
      const started = await startResponse.json() as StagedMedia;
      setDownloadProgress(started);
      const ready = started.status === 'ready'
        ? started
        : await pollStagedMedia(
            started.id,
            setDownloadProgress,
            () => true,
          );
      const projectResponse = await apiFetch(`${API_BASE}/api/staged-media/${ready.id}/project`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      if (!projectResponse.ok) throw new Error(await responseError(projectResponse));
      const imported = await projectResponse.json() as ProjectResult;
      setLinkProject(imported);
      setDownloadProgress(null);
      setLinkMessage('Video descargado y guardado. Revisa la configuración antes de procesar.');
      navigate(`/p/${imported.id}/configure`);
    } catch (error) {
      setLinkMessage(error instanceof Error ? error.message : 'No se pudo importar el enlace.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const startRecording = async () => {
    setRecordError('');
    try {
      let stream: MediaStream;
      if (recordMode === 'screen') {
        stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
        if (recordMic) {
          const microphone = await navigator.mediaDevices.getUserMedia({ audio: true });
          microphone.getAudioTracks().forEach((track) => stream.addTrack(track));
        }
      } else {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: recordMic });
      }
      if (!MediaRecorder.isTypeSupported('video/webm')) throw new Error('Este navegador no permite grabar en formato WebM.');
      const recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
      recordChunksRef.current = [];
      streamRef.current = stream;
      recorderRef.current = recorder;
      recorder.ondataavailable = (event) => {
        if (event.data.size) recordChunksRef.current.push(event.data);
      };
      recorder.onerror = () => setRecordError('La grabación se interrumpió. Revisa los permisos de cámara y micrófono.');
      recorder.onstop = () => {
        const blob = new Blob(recordChunksRef.current, { type: 'video/webm' });
        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        recorderRef.current = null;
        setRecording(false);
        setRecordingTime(0);
        setRecordOpen(false);
        if (blob.size) void importFile(new File([blob], `grabacion-${new Date().toISOString().slice(0, 10)}.webm`, { type: 'video/webm' }), true);
      };
      recorder.start(1000);
      setRecording(true);
    } catch (error) {
      setRecordError(error instanceof Error ? error.message : 'No se pudo iniciar la grabación.');
    }
  };

  const stopRecording = () => recorderRef.current?.stop();

  const visibleTemplates = templateTab === 'featured'
    ? [...templates, ...customTemplates]
    : templateTab === 'custom'
      ? savedTemplates
      : [];

  const saveCurrentTemplate = async () => {
    const presets = [...templates, ...customTemplates, ...savedTemplates];
    const source = presets.find((item) => item.name === activeTemplate) ?? templates[0];
    const name = templateName.trim() || `Mi estilo ${savedTemplates.length + 1}`;
    setSavingTemplate(true);
    setNotice('');
    try {
      const response = await apiFetch(`${API_BASE}/api/templates`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name,
          base_template: source.base_template ?? source.name,
        }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      const saved = await response.json() as { name: string; base_template: string };
      const base = [...templates, ...customTemplates].find((item) => item.name === saved.base_template) ?? templates[0];
      setSavedTemplates((items) => [...items, { ...base, name: saved.name, base_template: saved.base_template }]);
      setActiveTemplate(saved.name);
      setTemplateName('');
      setNotice(`Se guardó “${saved.name}” y estará disponible después de recargar.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'No se pudo guardar la plantilla.');
    } finally {
      setSavingTemplate(false);
    }
  };

  const toggleOption = (key: keyof typeof options) => setOptions((current) => ({ ...current, [key]: !current[key] }));

  const projectConfig = () => {
    if (schedule && (!scheduleDate || !Number.isFinite(new Date(scheduleDate).getTime()))) {
      throw new Error('Selecciona una fecha y hora válidas para programar los clips.');
    }
    return {
      ratio,
      duration,
      framing,
      subtitle_template: activeTemplate,
      subtitle_style: subtitleStyle,
      options,
      moment,
      whisper_model: whisperModel,
      whisper_language: whisperLanguage,
      schedule: schedule
        ? {
            scheduled_at: new Date(scheduleDate).toISOString(),
            platform: schedulePlatform,
            visibility: scheduleVisibility,
          }
        : null,
    };
  };

  const beginProjectProcessing = async (projectId: string) => {
    const response = await apiFetch(`${API_BASE}/api/projects/${projectId}/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config: projectConfig() }),
    });
    if (!response.ok) throw new Error(await responseError(response));
    navigate(`/p/${projectId}/processing`);
  };

  const submitProject = async () => {
    if (!linkProject && (!fileInfo || progress !== 100)) return;
    if (!linkProject && fileInfo?.kind === 'audio') {
      setNotice('El procesamiento de clips requiere un video con pista de audio. Los archivos de solo audio no se procesan como clips de video.');
      return;
    }
    setIsSubmitting(true);
    setNotice('');
    setUploadError('');
    try {
      let target = linkProject;
      if (!target) {
        if (!fileInfo) throw new Error('No se encontró el video local. Vuelve a importarlo antes de procesar.');
        const response = await apiFetch(`${API_BASE}/api/staged-media/${fileInfo.id}/project`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ config: projectConfig() }),
        });
        if (!response.ok) throw new Error(await responseError(response));
        target = await response.json() as ProjectResult;
      }
      setLinkProject(target);
      setProject(target);
      navigate(`/p/${target.id}/configure`);
      await beginProjectProcessing(target.id);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'No se pudo iniciar el procesamiento.');
    } finally {
      setIsSubmitting(false);
    }
  };

  useEffect(() => {
    if (!authReady || !authSession || route !== '/') return;
    let active = true;
    const resumeDownload = async () => {
      try {
        const latestResponse = await apiFetch(`${API_BASE}/api/staged-media/latest?source=url`);
        if (!latestResponse.ok) throw new Error(await responseError(latestResponse));
        const latest = await latestResponse.json() as StagedMedia | null;
        if (!active || !latest) return;
        if (latest.status === 'failed') {
          setLinkMessage(latest.error ?? 'La descarga no se completó. Vuelve a intentarlo.');
          return;
        }
        setIsSubmitting(true);
        setDownloadProgress(latest);
        setLinkMessage('Reanudando la descarga local del enlace…');
        const ready = latest.status === 'ready'
          ? latest
          : await pollStagedMedia(latest.id, (staged) => {
              if (active) setDownloadProgress(staged);
            }, () => active);
        if (!active) return;
        const response = await apiFetch(`${API_BASE}/api/staged-media/${ready.id}/project`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({}),
        });
        if (!response.ok) throw new Error(await responseError(response));
        const imported = await response.json() as ProjectResult;
        if (!active) return;
        setLinkProject(imported);
        setDownloadProgress(null);
        setLinkMessage('Video descargado y guardado. Revisa la configuración antes de procesar.');
        navigate(`/p/${imported.id}/configure`);
      } catch (error) {
        if (active) {
          setLinkMessage(error instanceof Error ? error.message : 'No se pudo reanudar la descarga.');
          setDownloadProgress(null);
        }
      } finally {
        if (active) setIsSubmitting(false);
      }
    };
    void resumeDownload();
    return () => {
      active = false;
    };
  }, [authReady, currentUserId, navigate, route]);

  const isSettings = route === '/settings';
  const projectRoute = route.match(/^\/p\/([^/]+)(?:\/(configure|processing|edit\/([^/]+)))?$/);
  const projectRouteId = projectRoute?.[1] ?? '';
  const projectRouteMode = projectRoute?.[2]?.startsWith('edit/')
    ? 'edit'
    : projectRoute?.[2] ?? 'results';
  const editClipId = projectRoute?.[3] ?? '';

  useEffect(() => {
    if (!authReady || !authSession) return;
    if (!projectRouteId) {
      setProject(null);
      setProjectError('');
      return;
    }
    let active = true;
    let eventController: AbortController | null = null;
    const loadProject = async () => {
      setProjectError('');
      try {
        const response = await apiFetch(`${API_BASE}/api/projects/${projectRouteId}`);
        if (!response.ok) throw new Error(await responseError(response));
        const loaded = await response.json() as ProjectResult;
        if (!active) return;
        setProject(loaded);
        if (projectRouteMode === 'configure') {
          setLinkProject(loaded);
          setRatio(loaded.config.ratio);
          setDuration(loaded.config.duration);
          setFraming(loaded.config.framing ?? 'speaker');
          setWhisperLanguage(loaded.config.whisper_language ?? 'auto');
          setActiveTemplate(loaded.config.subtitle_template);
          setSubtitleStyle(loaded.config.subtitle_style ?? DEFAULT_SUBTITLE_STYLE);
          setOptions({
            emojis: true,
            keywords: true,
            broll: false,
            silence: false,
            censor: false,
            face_tracking: true,
            ...loaded.config.options,
          });
          setMoment(loaded.config.moment);
          setSchedule(Boolean(loaded.config.schedule));
          if (loaded.config.schedule) {
            const localDate = new Date(loaded.config.schedule.scheduled_at);
            setScheduleDate(new Date(localDate.getTime() - localDate.getTimezoneOffset() * 60000).toISOString().slice(0, 16));
            setSchedulePlatform(loaded.config.schedule.platform);
            setScheduleVisibility(loaded.config.schedule.visibility);
          }
        }
        if (loaded.status === 'processing' || loaded.status === 'queued') {
          if (projectRouteMode !== 'processing') navigate(`/p/${projectRouteId}/processing`);
          eventController = new AbortController();
          void streamApiEvents(
            `${API_BASE}/api/projects/${projectRouteId}/events`,
            async (eventName, data) => {
              const update = JSON.parse(data) as Partial<ProjectResult> & { message?: string };
              setProject((current) => current ? { ...current, ...update } : current);
              if (eventName === 'failed' || update.status === 'completed' || update.status === 'failed') {
                eventController?.abort();
                const result = await apiFetch(`${API_BASE}/api/projects/${projectRouteId}`);
                if (!result.ok) throw new Error(await responseError(result));
                const finalProject = await result.json() as ProjectResult;
                if (active) {
                  setProject(finalProject);
                  navigate(`/p/${projectRouteId}`);
                }
              }
            },
            eventController.signal,
          ).catch((error: unknown) => {
            if (active && !eventController?.signal.aborted) {
              setProjectError(error instanceof Error ? error.message : 'Se perdió la conexión con el progreso del procesamiento.');
            }
          });
        } else if (projectRouteMode === 'processing') {
          navigate(`/p/${projectRouteId}`);
        }
      } catch (error) {
        if (active) setProjectError(error instanceof Error ? error.message : 'No se pudo cargar el proyecto.');
      }
    };
    void loadProject();
    return () => {
      active = false;
      eventController?.abort();
    };
  }, [authReady, currentUserId, projectRouteId, projectRouteMode, navigate]);

  const retryProject = async () => {
    if (!project) return;
    setProjectError('');
    try {
      await beginProjectProcessing(project.id);
    } catch (error) {
      setProjectError(error instanceof Error ? error.message : 'No se pudo reintentar el procesamiento.');
    }
  };

  const refreshProject = async (projectId: string) => {
    const response = await apiFetch(`${API_BASE}/api/projects/${projectId}`);
    if (!response.ok) throw new Error(await responseError(response));
    setProject(await response.json() as ProjectResult);
  };

  const openZoomImport = async () => {
    setZoomOpen(true);
    setZoomError('');
    setZoomRecordings([]);
    setZoomLoading(true);
    try {
      const response = await apiFetch(`${API_BASE}/api/integrations/zoom/recordings`);
      if (!response.ok) throw new Error(await responseError(response));
      setZoomRecordings(await response.json() as ZoomRecording[]);
    } catch (error) {
      setZoomError(error instanceof Error ? error.message : 'No se pudieron cargar las grabaciones de Zoom.');
    } finally {
      setZoomLoading(false);
    }
  };

  const importZoomFile = async (meetingId: string, fileId: string) => {
    setZoomLoading(true);
    setZoomError('');
    try {
      const response = await apiFetch(`${API_BASE}/api/integrations/zoom/import`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ meeting_id: meetingId, file_id: fileId, config: projectConfig() }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      const imported = await response.json() as ProjectResult;
      setLinkProject(imported);
      setProject(imported);
      setZoomOpen(false);
      navigate(`/p/${imported.id}/configure`);
    } catch (error) {
      setZoomError(error instanceof Error ? error.message : 'No se pudo importar la grabación.');
    } finally {
      setZoomLoading(false);
    }
  };

  const shareRoute = route.match(/^\/share\/([^/]+)$/);
  const shareToken = shareRoute?.[1] ?? '';

  const handleSignOut = async () => {
    try {
      await signOut(API_BASE);
      setAuthSession(null);
      setFileInfo(null);
      setProject(null);
      setLinkProject(null);
      setProgress(null);
      navigate('/');
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'No se pudo cerrar la sesión.');
    }
  };

  const downloadUpdate = async () => {
    try {
      const response = await apiFetch(`${API_BASE}/api/updates/download`, { method: 'POST' });
      if (!response.ok) throw new Error(await responseError(response));
      setUpdateStatus(await response.json() as UpdateStatus);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'No se pudo iniciar la descarga de la actualización.');
    }
  };

  const installUpdate = async () => {
    if (!window.confirm('Se abrirá el instalador oficial de kellyClipAi. Guarda tu trabajo antes de continuar.')) return;
    try {
      const response = await apiFetch(`${API_BASE}/api/updates/install`, { method: 'POST' });
      if (!response.ok) throw new Error(await responseError(response));
      setUpdateStatus((current) => current ? { ...current, state: 'installer_opened' } : current);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'No se pudo abrir el instalador.');
    }
  };

  const checkForUpdates = async () => {
    try {
      const response = await apiFetch(`${API_BASE}/api/updates/check`, { method: 'POST' });
      if (!response.ok) throw new Error(await responseError(response));
      setUpdateStatus(await response.json() as UpdateStatus);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'No se pudo comprobar si hay actualizaciones.';
      setUpdateStatus((current) => current ? { ...current, state: 'error', error: message } : current);
      setNotice(message);
    }
  };

  if (!shareToken && !authReady) {
    return <main className="auth-page"><p role="status">Conectando con el servicio de autenticación…</p></main>;
  }
  if (!shareToken && !authSession) {
    return <AuthPage configurationError={authConfigurationError} />;
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" type="button" onClick={() => navigate('/')} aria-label="kellyClipAi, inicio">
          <span className="brand-mark" aria-hidden="true"><span /><span /><span /></span>
          <span className="brand-name">kelly<span>Clip</span><b>Ai</b></span>
        </button>
        <button className={`settings-link ${isSettings ? 'is-active' : ''}`} type="button" onClick={() => {
          if (isSettings) navigate(settingsReturnPath);
          else {
            setSettingsReturnPath(route);
            navigate('/settings');
          }
        }} aria-label={isSettings ? 'Volver al proyecto' : 'Ajustes'}>
          {isSettings ? <ArrowLeft size={17} /> : <Settings2 size={17} />}
          <span>{isSettings ? 'Volver' : 'Ajustes'}</span>
        </button>
        {updateStatus?.enabled && <button
          className="settings-link"
          type="button"
          onClick={() => void checkForUpdates()}
          disabled={updateStatus.state === 'checking'}
          aria-label="Comprobar actualizaciones"
        >
          {updateStatus.state === 'checking' ? <LoaderCircle size={17} className="spin" /> : <Download size={17} />}
          <span>{updateStatus.state === 'checking' ? 'Comprobando…' : 'Buscar actualizaciones'}</span>
        </button>}
        {!shareToken && <button className="settings-link" type="button" onClick={() => void handleSignOut()} aria-label="Cerrar sesión">
          <LogOut size={17} /><span>Cerrar sesión</span>
        </button>}
      </header>

      {updateStatus?.enabled && (updateStatus.update_available || ['checking', 'current', 'error'].includes(updateStatus.state)) && (
        <aside className="update-banner" aria-live="polite">
          <div className="update-copy">
            <strong>
              {updateStatus.state === 'checking'
                ? 'Buscando actualizaciones…'
                : updateStatus.state === 'downloading'
                  ? `Descargando kellyClipAi ${updateStatus.latest_version}… ${updateStatus.progress}%`
                  : updateStatus.state === 'downloaded'
                    ? `kellyClipAi ${updateStatus.latest_version} está lista para instalar`
                    : updateStatus.state === 'installer_opened'
                      ? 'Sigue los pasos del instalador para actualizar kellyClipAi'
                      : updateStatus.state === 'error'
                        ? updateStatus.error
                        : updateStatus.state === 'current'
                          ? 'Estás usando la última versión de kellyClipAi.'
                        : `Nueva versión ${updateStatus.latest_version} disponible`}
            </strong>
            {updateStatus.state === 'downloading' && (
              <span className="update-progress"><span style={{ width: `${updateStatus.progress}%` }} /></span>
            )}
          </div>
          {updateStatus.state === 'available' && <span>La descarga comenzará automáticamente.</span>}
          {updateStatus.state === 'downloaded' && (
            <button type="button" className="subtle-button" onClick={() => void installUpdate()}>
              <Download size={15} /> Instalar actualización
            </button>
          )}
          {updateStatus.state === 'installer_opened' && (
            <button type="button" className="subtle-button" onClick={() => void installUpdate()}>
              <Download size={15} /> Abrir instalador de nuevo
            </button>
          )}
          {updateStatus.state === 'error' && (
            <button type="button" className="subtle-button" onClick={() => {
              if (updateStatus.update_available) void downloadUpdate();
              else void checkForUpdates();
            }}>
              Reintentar
            </button>
          )}
        </aside>
      )}

      {shareToken ? (
        <SharedClipPage token={shareToken} />
      ) : isSettings ? (
        <SettingsPage
          provider={settingsProvider}
          setProvider={setSettingsProvider}
          whisper={whisperModel}
          setWhisper={setWhisperModel}
          onBack={() => navigate(settingsReturnPath)}
          onNotice={setNotice}
          notice={notice}
        />
      ) : projectRouteId && projectRouteMode !== 'configure' ? (
        projectRouteMode === 'processing'
          ? <ProcessingPage project={project} error={projectError} onRetry={() => void retryProject()} />
          : projectRouteMode === 'edit'
            ? <ClipEditorPage
              projectId={projectRouteId}
              clipId={editClipId}
              project={project}
              savedTemplates={savedTemplates}
              onBack={() => navigate(`/p/${projectRouteId}`)}
            />
            : <ResultsPage
              project={project}
              error={projectError}
              onRetry={() => void retryProject()}
              onProjectUpdated={refreshProject}
              onProjectDeleted={() => navigate('/')}
              onEdit={(clipId) => navigate(`/p/${projectRouteId}/edit/${clipId}`)}
            />
      ) : (
        <main className="main-content">
          {!projectRouteId && <section className="upload-stack" aria-label="Subir video">
            {progress !== null && (
              <div className="upload-progress-card" aria-live="polite">
                <div className="upload-thumb">
                  {thumbnail ? <img src={thumbnail} alt="Miniatura del archivo" /> : fileInfo?.kind === 'audio' ? <AudioLines size={22} /> : <FileVideo2 size={22} />}
                </div>
                <div className="upload-details">
                  <div className="upload-file-head">
                    <span className="file-name" title={fileInfo?.name}>{fileInfo?.name ?? 'Preparando archivo…'}</span>
                    {fileInfo?.width && fileInfo.height ? <span className="meta-pill">{Math.max(fileInfo.width, fileInfo.height) >= 1900 ? '1080p' : `${Math.max(fileInfo.width, fileInfo.height)}px`}</span> : null}
                    {fileInfo && <span className="meta-pill">{formatDuration(fileInfo.duration)}</span>}
                    <button className="icon-button cancel-upload" type="button" onClick={cancelImport} aria-label="Cancelar importación" disabled={progress === 100}><X size={17} /></button>
                  </div>
                  <div className="progress-line"><span style={{ width: `${progress}%` }} /></div>
                  <div className="progress-copy"><span>{progress === 100 ? 'Listo en el almacenamiento local' : `Subiendo al almacenamiento local ${progress}%`}</span><span>{progress}%</span></div>
                </div>
              </div>
            )}

            <form className="link-form" onSubmit={handleLink}>
              <Link2 size={18} className="link-leading" aria-hidden="true" />
              <input
                aria-label="Enlace de video"
                type="url"
                value={link}
                onChange={(event) => { setLink(event.target.value); setLinkMessage(''); }}
                placeholder="Pega el enlace de un video"
              />
              <button type="submit" className="continue-button" disabled={!link.trim() || isSubmitting}>{isSubmitting ? 'Descargando…' : 'Continuar'} <ArrowRight size={16} /></button>
            </form>
            {linkMessage && <p className="form-message" role="status">{linkMessage}</p>}
            {downloadProgress?.source_type === 'url' && downloadProgress.status === 'downloading' && (
              <div className="download-progress" aria-live="polite">
                {downloadProgress.total_bytes && downloadProgress.total_bytes > 0 && (
                  <div className="progress-line">
                    <span style={{ width: `${Math.min(100, Math.round((downloadProgress.bytes_received / downloadProgress.total_bytes) * 100))}%` }} />
                  </div>
                )}
                <div className="progress-copy">
                  <span>
                    {downloadProgress.total_bytes && downloadProgress.total_bytes > 0
                      ? `${formatBytes(downloadProgress.bytes_received)} de ${formatBytes(downloadProgress.total_bytes)}`
                      : `${formatBytes(downloadProgress.bytes_received)} descargados`}
                    {downloadProgress.speed ? ` · ${formatBytes(downloadProgress.speed)}/s` : ''}
                    {downloadProgress.eta !== null ? ` · quedan ${formatDuration(downloadProgress.eta)}` : ''}
                  </span>
                  <span>
                    {downloadProgress.total_bytes && downloadProgress.total_bytes > 0
                      ? `${Math.min(100, Math.round((downloadProgress.bytes_received / downloadProgress.total_bytes) * 100))}%`
                      : ''}
                  </span>
                </div>
              </div>
            )}

            <div
              className={`drop-zone ${dragging ? 'is-dragging' : ''}`}
              role="button"
              tabIndex={0}
              onClick={() => inputRef.current?.click()}
              onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); inputRef.current?.click(); } }}
              onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
            >
              <input ref={inputRef} className="visually-hidden" type="file" accept={ACCEPT} aria-label="Seleccionar video o audio" onChange={chooseFile} />
              <span className="upload-illustration"><CloudUpload size={30} strokeWidth={1.7} /></span>
              <p className="drop-title">Haz clic para explorar o arrastra y suelta</p>
              <p className="drop-subtitle">Sube un video o audio desde tu dispositivo</p>
              <div className="accepted-types">
                <span>MP4</span><span>MOV</span><span>3GP</span><span>AVI</span><span>MP3</span><span>WAV</span><span>M4A</span>
              </div>
              <p className="size-note">Tus archivos permanecen en este dispositivo</p>
            </div>

            {uploadError && <p className="error-message" role="alert"><AlertCircle size={16} />{uploadError}</p>}

            <div className="secondary-actions">
              <button type="button" className="outline-action" onClick={() => void openZoomImport()}>
                <span className="zoom-mark">Z</span> Importar desde Zoom
              </button>
              <button type="button" className="outline-action" onClick={() => { setRecordError(''); setRecordOpen(true); }}>
                <Video size={18} className="record-icon" /> Grabar video
              </button>
            </div>
            <p className="authorization-note">Importa solo videos propios o para los que tengas autorización.</p>
          </section>}

          {((!projectRouteId && fileInfo && progress === 100) || (projectRouteMode === 'configure' && project)) && (
            <section className="config-area" aria-label="Configuración de clips">
              {projectRouteId && project && <div className="project-source-card"><FileVideo2 size={20} /><span><strong>{project.title}</strong><small>Video importado y guardado en este dispositivo</small></span><span className="meta-pill">{project.metadata ? formatDuration(project.metadata.duration) : 'Listo'}</span></div>}
              <div className="settings-row">
                <label className="config-field">
                  <span>Ratio</span>
                  <span className="select-wrap"><select value={ratio} onChange={(event) => setRatio(event.target.value)}><option>9:16</option><option>16:9</option><option>1:1</option><option>4:5</option></select><ChevronDown size={16} /></span>
                </label>
                <label className="config-field">
                  <span>Duración del clip</span>
                  <span className="select-wrap"><select value={duration} onChange={(event) => setDuration(event.target.value)}><option>Cualquiera</option><option>&lt;30s</option><option>30–60s</option><option>60–90s</option><option>90s–3min</option></select><ChevronDown size={16} /></span>
                </label>
              </div>

              <section className="config-card templates-card">
                <div className="card-heading">
                  <div><h2>Plantillas de subtítulos</h2><p>Elige el estilo de tus clips</p></div>
                  <div className="template-save-controls">
                    <input
                      aria-label="Nombre de la plantilla"
                      maxLength={60}
                      placeholder={`Mi estilo ${savedTemplates.length + 1}`}
                      value={templateName}
                      onChange={(event) => setTemplateName(event.target.value)}
                    />
                    <button className="subtle-button" type="button" onClick={() => void saveCurrentTemplate()} disabled={savingTemplate}>
                      {savingTemplate ? <LoaderCircle size={15} className="spin" /> : <Plus size={15} />}
                      {savingTemplate ? 'Guardando…' : 'Guardar estilo'}
                    </button>
                  </div>
                </div>
                <div className="tabs" role="tablist" aria-label="Categorías de plantillas">
                  <button type="button" role="tab" aria-selected={templateTab === 'featured'} className={templateTab === 'featured' ? 'selected' : ''} onClick={() => setTemplateTab('featured')}>Destacadas</button>
                  <button type="button" role="tab" aria-selected={templateTab === 'custom'} className={templateTab === 'custom' ? 'selected' : ''} onClick={() => setTemplateTab('custom')}>Mis plantillas</button>
                  <button type="button" role="tab" aria-selected={templateTab === 'brand'} className={templateTab === 'brand' ? 'selected' : ''} onClick={() => setTemplateTab('brand')}>Plantilla de marca <span className="diamond">◆</span></button>
                </div>
                {templateTab === 'brand' ? (
                  <div className="locked-brand"><span className="lock-badge"><LockKeyhole size={20} /></span><div><strong>Plantilla de marca</strong><p>Personaliza clips con tu identidad visual. Esta opción premium todavía no está disponible.</p></div><span className="premium-chip">◆ Premium</span></div>
                ) : visibleTemplates.length ? (
                  <div className="template-rail-wrap">
                    <button className="rail-arrow rail-prev" type="button" aria-label="Ver plantillas anteriores" onClick={(event) => event.currentTarget.parentElement?.querySelector('.template-rail')?.scrollBy({ left: -260, behavior: 'smooth' })}><ChevronLeft size={18} /></button>
                    <div className="template-rail">
                      {visibleTemplates.map((template) => (
                        <button key={template.name} className={`template-option ${activeTemplate === template.name ? 'active' : ''}`} type="button" onClick={() => setActiveTemplate(template.name)} aria-pressed={activeTemplate === template.name}>
                          <span className={`template-preview ${template.tone}`}><span className="preview-lines"><b>{template.sample.split(' ').slice(0, 2).join(' ')}</b><strong>{template.sample.split(' ').slice(2).join(' ') || 'para tus clips'}</strong></span></span>
                          <span className="template-name">{template.name}{activeTemplate === template.name && <span className="template-check"><Check size={11} /></span>}</span>
                        </button>
                      ))}
                    </div>
                    <button className="rail-arrow rail-next" type="button" aria-label="Ver más plantillas" onClick={(event) => event.currentTarget.parentElement?.querySelector('.template-rail')?.scrollBy({ left: 260, behavior: 'smooth' })}><ChevronRight size={18} /></button>
                  </div>
                ) : <p className="empty-templates">Aún no guardaste estilos propios. Selecciona una plantilla y pulsa “Guardar estilo”.</p>}
              </section>

              <section className="config-card options-card">
                <div className="card-heading"><div><h2>Opciones de edición</h2><p>Personaliza el resultado de tus clips</p></div></div>
                <div className="option-grid">
                  <Option checked={options.emojis} onClick={() => toggleOption('emojis')} title="Añadir emojis" description="Haz que tus subtítulos destaquen" />
                  <Option checked={options.keywords} onClick={() => toggleOption('keywords')} title="Resaltar palabras clave" description="Destaca las ideas importantes" />
                  <Option checked={options.broll} onClick={() => { toggleOption('broll'); if (!options.broll) setNotice('B-rolls es una función premium y necesita una clave de Pexels o Pixabay.'); }} title="Añadir B-rolls" description="Inserta imágenes de apoyo" premium />
                  <Option checked={options.silence} onClick={() => toggleOption('silence')} title="Eliminar silencios" description="Recorta las pausas largas" />
                  <Option checked={options.censor} onClick={() => toggleOption('censor')} title="Auto-censurar" description="Silencia y oculta palabrotas" />
                  <Option checked={options.face_tracking} onClick={() => toggleOption('face_tracking')} title="Seguir rostro en el encuadre" description="Mantiene a la persona principal visible al recortar" />
                </div>
                <label className="config-field">
                  <span>Encuadre inteligente</span>
                  <select className="text-input" value={framing} onChange={(event) => {
                    const value = event.target.value;
                    if (value === 'speaker' || value === 'split' || value === 'blur' || value === 'center') setFraming(value);
                  }}>
                    <option value="speaker">Seguir al orador</option>
                    <option value="split">Pantalla dividida</option>
                    <option value="blur">Fondo desenfocado</option>
                    <option value="center">Centrado</option>
                  </select>
                  <small>El seguimiento del orador estima actividad labial durante la voz detectada, localmente.</small>
                </label>
                <label className="config-field">
                  <span>Idioma de transcripción</span>
                  <select className="text-input" value={whisperLanguage} onChange={(event) => {
                    setWhisperLanguage(event.target.value);
                  }}>
                    <option value="auto">Detectar automáticamente</option>
                    <option value="es">Español</option>
                    <option value="en">Inglés</option>
                    <option value="pt">Portugués</option>
                    <option value="fr">Francés</option>
                    <option value="de">Alemán</option>
                    <option value="it">Italiano</option>
                    <option value="ja">Japonés</option>
                    <option value="zh">Chino</option>
                    <option value="ko">Coreano</option>
                    <option value="ru">Ruso</option>
                    <option value="ar">Árabe</option>
                    <option value="hi">Hindi</option>
                    <option value="nl">Neerlandés</option>
                    <option value="pl">Polaco</option>
                    <option value="tr">Turco</option>
                    <option value="uk">Ucraniano</option>
                  </select>
                  <small>Se guarda en el proyecto; automático conserva la detección de Whisper.</small>
                </label>
                <SubtitleStyleControls value={subtitleStyle} onChange={setSubtitleStyle} />
              </section>

              <section className="config-card moment-card">
                <div className="card-heading"><div><h2>Encontrar momento</h2><p>Orienta el análisis hacia un tema concreto</p></div><span className="optional-chip">Opcional</span></div>
                <input className="text-input" value={moment} onChange={(event) => setMoment(event.target.value)} placeholder="Por ejemplo: Cuando alguien habla de una idea importante." />
              </section>

              <section className="config-card schedule-card">
                <div className="schedule-heading">
                  <div className="schedule-icon"><CalendarDays size={18} /></div>
                  <div className="schedule-title"><h2>Programar clips</h2><p>Prepara una publicación para más tarde</p></div>
                  <button className={`switch ${schedule ? 'on' : ''}`} type="button" role="switch" aria-checked={schedule} aria-label="Programar clips" onClick={() => setSchedule(!schedule)}><span /></button>
                </div>
                {schedule && (
                  <div className="schedule-fields">
                    <label className="config-field">
                      <span>Fecha y hora</span>
                      <input
                        type="datetime-local"
                        required
                        value={scheduleDate}
                        min={new Date(Date.now() + 60_000 - new Date().getTimezoneOffset() * 60_000).toISOString().slice(0, 16)}
                        onChange={(event) => setScheduleDate(event.target.value)}
                      />
                    </label>
                    <label className="config-field">
                      <span>Plataforma</span>
                      <span className="select-wrap"><select value={schedulePlatform} onChange={(event) => {
                        const next = event.target.value;
                        setSchedulePlatform(next);
                        if (next === 'instagram') setScheduleVisibility('public');
                        else if (scheduleVisibility === 'unlisted' && next === 'tiktok') setScheduleVisibility('private');
                      }}><option value="youtube">YouTube</option><option value="tiktok">TikTok</option><option value="instagram">Instagram</option></select><ChevronDown size={16} /></span>
                    </label>
                    <label className="config-field">
                      <span>Visibilidad</span>
                      <span className="select-wrap"><select value={scheduleVisibility} onChange={(event) => setScheduleVisibility(event.target.value)} disabled={schedulePlatform === 'instagram'}>
                        {schedulePlatform !== 'tiktok' && <option value="unlisted">No listada</option>}
                        {schedulePlatform !== 'instagram' && <option value="private">Privada</option>}
                        <option value="public">Pública</option>
                      </select><ChevronDown size={16} /></span>
                    </label>
                    <div className="connected-accounts"><span>Publicación real</span><p>Si no hay una cuenta conectada al llegar la fecha, el MP4 se exportará localmente sin publicarse. <button type="button" onClick={() => navigate('/settings')}>Configurar integraciones</button></p></div>
                  </div>
                )}
              </section>

              {notice && <p className="notice-message" role="status"><AlertCircle size={17} />{notice}</p>}
              {projectError && <p className="error-message" role="alert"><AlertCircle size={16} />{projectError}</p>}
              <button className="create-button" type="button" onClick={() => void submitProject()} disabled={isSubmitting || (!linkProject && progress !== 100)}>
                {isSubmitting ? <LoaderCircle size={18} className="spin" /> : <WandSparkles size={18} />}
                {isSubmitting ? 'Preparando proyecto…' : 'Obtener clips con IA'}
                {!isSubmitting && <ArrowRight size={17} />}
              </button>
              <p className="local-storage-note">El original permanece en este dispositivo; el proyecto se procesa en el backend local.</p>
            </section>
          )}
        </main>
      )}

      {recordOpen && (
        <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget && !recording) setRecordOpen(false); }}>
          <section className="record-modal" role="dialog" aria-modal="true" aria-labelledby="record-title">
            <div className="modal-heading"><div><h2 id="record-title">Grabar video</h2><p>La grabación se guardará en este dispositivo.</p></div><button type="button" className="icon-button" aria-label="Cerrar" disabled={recording} onClick={() => setRecordOpen(false)}><X size={19} /></button></div>
            <div className="record-mode-switch"><button type="button" className={recordMode === 'camera' ? 'active' : ''} onClick={() => setRecordMode('camera')} disabled={recording}><Video size={17} /> Cámara</button><button type="button" className={recordMode === 'screen' ? 'active' : ''} onClick={() => setRecordMode('screen')} disabled={recording}><MonitorUp size={17} /> Pantalla</button></div>
            <label className="record-mic"><input type="checkbox" checked={recordMic} onChange={(event) => setRecordMic(event.target.checked)} disabled={recording} /> Incluir micrófono</label>
            {recording && <p className="recording-status"><span className="recording-dot" /> Grabando · {formatDuration(recordingTime)}</p>}
            {recordError && <p className="error-message" role="alert"><AlertCircle size={16} />{recordError}</p>}
            <button type="button" className={`record-start ${recording ? 'stop' : ''}`} onClick={recording ? stopRecording : startRecording}>{recording ? <><Pause size={17} /> Detener y guardar</> : <><Video size={17} /> Comenzar grabación</>}</button>
            <p className="modal-footnote">Tu navegador solicitará permiso para acceder a la cámara, pantalla o micrófono.</p>
          </section>
        </div>
      )}
      {zoomOpen && (
        <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget && !zoomLoading) setZoomOpen(false); }}>
          <section className="record-modal zoom-modal" role="dialog" aria-modal="true" aria-labelledby="zoom-title">
            <div className="modal-heading">
              <div><h2 id="zoom-title">Importar desde Zoom</h2><p>Selecciona una grabación MP4 reciente.</p></div>
              <button type="button" className="icon-button" aria-label="Cerrar" disabled={zoomLoading} onClick={() => setZoomOpen(false)}><X size={19} /></button>
            </div>
            {zoomLoading && <p className="connection-message"><LoaderCircle size={15} className="spin" /> Consultando Zoom…</p>}
            {zoomError && <p className="error-message" role="alert"><AlertCircle size={16} />{zoomError}</p>}
            {!zoomLoading && !zoomError && zoomRecordings.length === 0 && <p className="empty-templates">No hay grabaciones MP4 completadas de los últimos 30 días.</p>}
            <div className="zoom-recordings">
              {zoomRecordings.map((recording) => (
                <article className="zoom-recording" key={recording.meeting_id}>
                  <strong>{recording.topic}</strong>
                  <small>{recording.start_time ? new Date(recording.start_time).toLocaleString() : 'Fecha no disponible'}</small>
                  {recording.files.map((file) => (
                    <button key={file.file_id} type="button" className="subtle-button" disabled={zoomLoading} onClick={() => void importZoomFile(recording.meeting_id, file.file_id)}>
                      <Download size={14} /> Importar MP4{file.file_size ? ` · ${Math.round(file.file_size / 1024 / 1024)} MB` : ''}
                    </button>
                  ))}
                </article>
              ))}
            </div>
            {zoomError && <button type="button" className="connect-button" onClick={() => { setZoomOpen(false); navigate('/settings'); }}>Revisar Zoom en Ajustes</button>}
          </section>
        </div>
      )}
      <footer className="developer-credit">
        Desarrollado por <strong>KELLY BENJAMIN DE LA CRUZ</strong>
      </footer>
    </div>
  );
}

function ProcessingPage({ project, error, onRetry }: { project: ProjectResult | null; error: string; onRetry: () => void }) {
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
            ? <img src={`${API_BASE}${project.thumbnail_url}`} alt="Miniatura del video" />
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
            const label = step.key === 'finding' && current
              ? `Buscando las mejores partes… ${project?.progress == null ? '' : `${project.progress}%`}`
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

function ResultsPage({
  project,
  error,
  onRetry,
  onEdit,
  onProjectUpdated,
  onProjectDeleted,
}: {
  project: ProjectResult | null;
  error: string;
  onRetry: () => void;
  onEdit: (clipId: string) => void;
  onProjectUpdated: (projectId: string) => Promise<void>;
  onProjectDeleted: () => void;
}) {
  const [sort, setSort] = useState<'score' | 'time'>('score');
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [notice, setNotice] = useState('');
  const [shareLink, setShareLink] = useState<{
    url: string;
    clipId: string;
    expiresAt: string;
  } | null>(null);
  const [busyAction, setBusyAction] = useState('');
  const [sharingClip, setSharingClip] = useState('');
  const [publishPlatform, setPublishPlatform] = useState('youtube');
  const [publishVisibility, setPublishVisibility] = useState('private');
  const [scheduleJobs, setScheduleJobs] = useState<ScheduleJob[]>([]);
  const [scheduleError, setScheduleError] = useState('');
  const [storageUsedBytes, setStorageUsedBytes] = useState<number | null>(null);
  const [storageError, setStorageError] = useState('');
  const players = useRef<Record<string, HTMLVideoElement | null>>({});
  const clips = useMemo(() => {
    const allClips = [...(project?.clips ?? [])];
    return allClips.sort((a, b) => sort === 'score' ? b.score - a.score : a.start - b.start);
  }, [project?.clips, sort]);
  const readyClips = clips.filter((clip) => clip.status !== 'failed');
  const failedClips = clips.filter((clip) => clip.status === 'failed');
  const allSelected = readyClips.length > 0 && readyClips.every((clip) => selected.has(clip.id));
  const selectedClips = readyClips.filter((clip) => selected.has(clip.id));

  const publishClips = async (clipIds: string[]) => {
    if (!project || !clipIds.length) {
      setNotice('Selecciona al menos un clip para publicar.');
      return;
    }
    setBusyAction('publish');
    setNotice('');
    try {
      const response = await apiFetch(`${API_BASE}/api/projects/${project.id}/publish`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          clip_ids: clipIds,
          platform: publishPlatform,
          visibility: publishVisibility,
        }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      const result = await response.json() as {
        published: Array<{ clip_id: string; external_id: string; url: string }>;
        failed: { clip_id: string; error: string } | null;
        not_attempted: string[];
        jobs?: ScheduleJob[];
      };
      if (result.jobs) {
        const queuedJobs = result.jobs;
        setScheduleJobs((current) => [
          ...current.filter((job) => !queuedJobs.some((queued) => queued.id === job.id)),
          ...queuedJobs,
        ]);
        setNotice(`${queuedJobs.length} clip(s) añadidos a la cola real de publicación.`);
        return;
      }
      const publishedMessage = result.published.length
        ? `${result.published.length} clip(s) confirmados por ${publishPlatform}.`
        : '';
      const failureMessage = result.failed
        ? ` No se confirmó el clip ${result.failed.clip_id.slice(0, 8)}: ${result.failed.error}`
        : '';
      const pendingMessage = result.not_attempted.length
        ? ` No se intentaron ${result.not_attempted.length} clips restantes.`
        : '';
      setNotice(`${publishedMessage}${failureMessage}${pendingMessage}`.trim() || 'La plataforma no confirmó ninguna publicación.');
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'No se pudo completar la publicación.');
    } finally {
      setBusyAction('');
    }
  };
  const publishSelected = () => void publishClips(selectedClips.map((clip) => clip.id));

  const refreshStorageUsage = useCallback(async () => {
    try {
      const response = await apiFetch(`${API_BASE}/api/storage`);
      if (!response.ok) throw new Error(await responseError(response));
      const result = await response.json() as { used_bytes: number };
      setStorageUsedBytes(result.used_bytes);
      setStorageError('');
    } catch (error) {
      setStorageError(error instanceof Error ? error.message : 'No se pudo consultar el almacenamiento.');
    }
  }, []);

  const deleteProject = async () => {
    if (!project || !window.confirm(
      `¿Eliminar “${project.title}” y todos sus archivos, clips, publicaciones programadas y enlaces compartidos? Esta acción no se puede deshacer.`,
    )) return;
    setBusyAction('delete-project');
    setNotice('');
    try {
      const response = await apiFetch(`${API_BASE}/api/projects/${project.id}`, { method: 'DELETE' });
      if (!response.ok) throw new Error(await responseError(response));
      onProjectDeleted();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'No se pudo eliminar el proyecto.');
      setBusyAction('');
    }
  };

  const retryFailedClips = async () => {
    if (!project || !failedClips.length) return;
    setBusyAction('retry-clips');
    setNotice('');
    try {
      const response = await apiFetch(`${API_BASE}/api/projects/${project.id}/retry-failed-clips`, {
        method: 'POST',
      });
      if (!response.ok) throw new Error(await responseError(response));
      setNotice('Los clips fallidos se añadieron a la cola para reintentarlos.');
      await onProjectUpdated(project.id);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'No se pudieron reintentar los clips fallidos.');
    } finally {
      setBusyAction('');
    }
  };

  const refreshScheduleJobs = useCallback(async () => {
    if (!project) return;
    try {
      const response = await apiFetch(`${API_BASE}/api/projects/${project.id}/scheduled`);
      if (!response.ok) throw new Error(await responseError(response));
      setScheduleJobs(await response.json() as ScheduleJob[]);
      setScheduleError('');
    } catch (error) {
      setScheduleError(error instanceof Error ? error.message : 'No se pudo consultar la cola de publicaciones.');
    }
  }, [project]);

  const resolveOverdueScheduleJob = async (
    jobId: string,
    action: 'publish_now' | 'discard',
  ) => {
    setBusyAction(`overdue:${jobId}`);
    try {
      const response = await apiFetch(`${API_BASE}/api/scheduled-jobs/${jobId}/resolve-overdue`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      await refreshScheduleJobs();
    } catch (error) {
      setScheduleError(error instanceof Error ? error.message : 'No se pudo resolver la publicación vencida.');
    } finally {
      setBusyAction('');
    }
  };

  useEffect(() => {
    setScheduleJobs([]);
    if (project) void refreshScheduleJobs();
  }, [project?.id, project?.status, refreshScheduleJobs]);

  useEffect(() => {
    void refreshStorageUsage();
  }, [refreshStorageUsage]);

  useEffect(() => {
    if (!scheduleJobs.some((job) => job.status === 'scheduled' || job.status === 'processing')) return;
    const interval = window.setInterval(() => void refreshScheduleJobs(), 5000);
    return () => window.clearInterval(interval);
  }, [scheduleJobs, refreshScheduleJobs]);

  const downloadSelected = async () => {
    if (!selectedClips.length) {
      setNotice('Selecciona al menos un clip para descargarlo.');
      return;
    }
    await saveProjectFiles(false, selectedClips.map((clip) => clip.id));
  };
  const saveProjectFiles = async (includeOriginal: boolean, clipIds: string[]) => {
    if (!project) return;
    setBusyAction(includeOriginal ? 'original' : 'archive');
    setNotice('');
    try {
      const desktopApi = window.pywebview?.api;
      if (desktopApi) {
        const destination = await desktopApi.select_download_folder();
        if (!destination) {
          setNotice('Guardado cancelado. No se eligió una carpeta.');
          return;
        }
        const response = await apiFetch(`${API_BASE}/api/projects/${project.id}/downloads/save`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ destination, include_original: includeOriginal, clip_ids: clipIds }),
        });
        if (!response.ok) throw new Error(await responseError(response));
        const result = await response.json() as { saved: string[] };
        setNotice(`Guardado en la carpeta elegida: ${result.saved.join(', ')}`);
        return;
      }

      const filename = includeOriginal
        ? `${project.title.replace(/[<>:"/\\|?*]/g, '_')}${/\.[^.]+$/.test(project.title) ? '' : '.mp4'}`
        : clipIds.length === 1
          ? `${clipIds[0]}.mp4`
          : `kellyClipAi-clips-${project.id.slice(0, 8)}.zip`;
      const url = includeOriginal
        ? `${API_BASE}/api/projects/${project.id}/original/download`
        : clipIds.length === 1
          ? `${API_BASE}/api/projects/${project.id}/clips/${clipIds[0]}/download`
          : `${API_BASE}/api/projects/${project.id}/clips/archive`;
      const response = clipIds.length > 1 && !includeOriginal
        ? await apiFetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ clip_ids: clipIds }),
        })
        : await apiFetch(url);
      if (!response.ok) throw new Error(await responseError(response));
      triggerDownload(await response.blob(), filename);
      setNotice(includeOriginal ? 'Video original descargado.' : clipIds.length > 1
        ? `ZIP creado con ${clipIds.length} clips reales.`
        : 'Clip descargado.');
    } catch (downloadError) {
      setNotice(downloadError instanceof Error ? downloadError.message : 'No se pudieron guardar los archivos.');
    } finally {
      setBusyAction('');
    }
  };
  const downloadSubtitles = async (clipId: string, format: 'srt' | 'vtt') => {
    if (!project) return;
    setNotice('');
    try {
      const response = await apiFetch(
        `${API_BASE}/api/projects/${project.id}/clips/${clipId}/subtitles/${format}`,
      );
      if (!response.ok) throw new Error(await responseError(response));
      triggerDownload(await response.blob(), `${clipId}.${format}`);
      setNotice(`Subtítulos ${format.toUpperCase()} descargados.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'No se pudieron descargar los subtítulos.');
    }
  };
  const changeResolution = async (clip: ClipResult, resolution: 720 | 1080) => {
    if (!project || resolution === (clip.resolution ?? 720)) return;
    setBusyAction(clip.id);
    setNotice('');
    try {
      const response = await apiFetch(`${API_BASE}/api/projects/${project.id}/clips/${clip.id}/quality`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resolution }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      await onProjectUpdated(project.id);
      setNotice(`Clip re-renderizado a ${resolution}p.`);
    } catch (renderError) {
      setNotice(renderError instanceof Error ? renderError.message : 'No se pudo re-renderizar el clip.');
    } finally {
      setBusyAction('');
    }
  };
  const improveAudio = async (clip: ClipResult) => {
    if (!project || clip.audio_enhanced) return;
    setBusyAction(clip.id);
    setNotice('');
    try {
      const response = await apiFetch(`${API_BASE}/api/projects/${project.id}/clips/${clip.id}/improve`, {
        method: 'POST',
      });
      if (!response.ok) throw new Error(await responseError(response));
      await onProjectUpdated(project.id);
      setNotice('Audio mejorado con normalización de volumen y realce ligero.');
    } catch (improveError) {
      setNotice(improveError instanceof Error ? improveError.message : 'No se pudo mejorar el audio.');
    } finally {
      setBusyAction('');
    }
  };
  const shareClip = async (clipId: string) => {
    if (!project) return;
    setSharingClip(clipId);
    setShareLink(null);
    setNotice('');
    try {
      const response = await apiFetch(`${API_BASE}/api/projects/${project.id}/clips/${clipId}/share`, {
        method: 'POST',
      });
      if (!response.ok) throw new Error(await responseError(response));
      const result = await response.json() as { share_url: string; expires_at: string };
      const url = new URL(result.share_url, window.location.origin).toString();
      setShareLink({ url, clipId, expiresAt: result.expires_at });
      try {
        await navigator.clipboard.writeText(url);
        setNotice('Enlace local copiado al portapapeles.');
      } catch {
        setNotice('Enlace local listo. Puedes abrirlo o copiarlo desde el enlace mostrado.');
      }
    } catch (shareError) {
      setNotice(shareError instanceof Error ? shareError.message : 'No se pudo crear el enlace compartido.');
    } finally {
      setSharingClip('');
    }
  };

  const revokeShareLink = async () => {
    if (!project || !shareLink) return;
    setBusyAction('revoke-share');
    try {
      const response = await apiFetch(
        `${API_BASE}/api/projects/${project.id}/clips/${shareLink.clipId}/share`,
        { method: 'DELETE' },
      );
      if (!response.ok) throw new Error(await responseError(response));
      setShareLink(null);
      setNotice('El enlace compartido quedó revocado.');
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'No se pudo revocar el enlace.');
    } finally {
      setBusyAction('');
    }
  };
  const editSelected = () => {
    if (selectedClips.length !== 1) {
      setNotice('Selecciona exactamente un clip para editarlo.');
      return;
    }
    onEdit(selectedClips[0].id);
  };

  return (
    <main className="results-page">
      <header className="results-toolbar">
        <label className="select-all"><input type="checkbox" checked={allSelected} onChange={(event) => setSelected(event.target.checked ? new Set(readyClips.map((clip) => clip.id)) : new Set())} /> Seleccionar todo</label>
        <div className="results-toolbar-actions">
          <label className="publish-select"><span className="visually-hidden">Plataforma para publicar</span><select value={publishPlatform} onChange={(event) => {
            const next = event.target.value;
            setPublishPlatform(next);
            if (next === 'instagram') setPublishVisibility('public');
            else if (next === 'tiktok' && publishVisibility === 'unlisted') setPublishVisibility('private');
          }}><option value="youtube">YouTube</option><option value="tiktok">TikTok</option><option value="instagram">Instagram</option></select></label>
          <label className="publish-select"><span className="visually-hidden">Visibilidad</span><select value={publishVisibility} disabled={publishPlatform === 'instagram'} onChange={(event) => setPublishVisibility(event.target.value)}>
            {publishPlatform !== 'tiktok' && <option value="unlisted">No listada</option>}
            {publishPlatform !== 'instagram' && <option value="private">Privada</option>}
            <option value="public">Pública</option>
          </select></label>
          <button type="button" onClick={publishSelected} disabled={!selectedClips.length || busyAction === 'publish'}>
            {busyAction === 'publish' ? <LoaderCircle size={15} className="spin" /> : <ArrowRight size={15} />}
            Publicar
          </button>
          <button type="button" onClick={() => void downloadSelected()} disabled={busyAction === 'archive'}>
            {busyAction === 'archive' ? <LoaderCircle size={15} className="spin" /> : <Download size={15} />}
            Descargar
          </button>
          <button type="button" onClick={() => void saveProjectFiles(true, [])} disabled={busyAction === 'original'}>
            {busyAction === 'original' ? <LoaderCircle size={15} className="spin" /> : <Download size={15} />}
            Guardar video original
          </button>
          <button type="button" onClick={editSelected}><Scissors size={15} /> Editar clips</button>
          <button type="button" className="delete-project-button" onClick={() => void deleteProject()} disabled={busyAction === 'delete-project'}>
            {busyAction === 'delete-project' ? <LoaderCircle size={15} className="spin" /> : <Trash2 size={15} />}
            Eliminar proyecto
          </button>
        </div>
      </header>
      <div className="clips-heading">
        <div><span className="clips-version">V1</span><strong>CLIPS IA</strong></div>
        <span className="clips-count">{clips.length} clips</span>
        <span className="storage-usage" role="status">
          Almacenamiento local: {storageUsedBytes === null ? (storageError || 'calculando…') : formatBytes(storageUsedBytes)}
        </span>
        <label className="sort-control"><span>Ordenar:</span><select value={sort} onChange={(event) => setSort(event.target.value as 'score' | 'time')}><option value="score">Mayor puntuación</option><option value="time">Cronológico</option></select></label>
      </div>
      {error && <p className="error-message" role="alert"><AlertCircle size={16} />{error}</p>}
      {failedClips.length > 0 && project?.status !== 'processing' && project?.status !== 'queued' && (
        <div className="project-failure">
          <p className="error-message" role="alert"><AlertCircle size={16} />{failedClips.length} clip(s) fallaron; los demás se conservaron.</p>
          <button type="button" className="retry-button" onClick={() => void retryFailedClips()} disabled={busyAction === 'retry-clips'}>
            {busyAction === 'retry-clips' ? <LoaderCircle size={16} className="spin" /> : <WandSparkles size={16} />}
            Reintentar clips fallidos
          </button>
        </div>
      )}
      {scheduleError && (
        <div className="project-failure">
          <p className="error-message" role="alert"><AlertCircle size={16} />{scheduleError}</p>
          <button type="button" className="retry-button" onClick={() => void refreshScheduleJobs()}>Reintentar consulta</button>
        </div>
      )}
      {scheduleJobs.length > 0 && (
        <section className="config-card schedule-card results-schedule">
          <div className="card-heading"><div><h2>Publicaciones programadas</h2><p>El estado se actualiza desde la cola real del backend local.</p></div></div>
          <div className="scheduled-job-list">
            {scheduleJobs.map((job) => (
              <article className="scheduled-job" key={job.id}>
                <div><strong>{job.platform} · clip {job.clip_id.slice(0, 8)}</strong><small>{new Date(job.scheduled_at).toLocaleString()} · {job.visibility}</small></div>
                <span className={`job-status ${job.status}`}>{{
                  scheduled: 'En cola',
                  processing: 'Procesando',
                  published: 'Publicado',
                  exported: 'Exportado localmente',
                  failed: 'No confirmado',
                }[job.status] ?? job.status}</span>
                {job.external_url && <a className="job-link" href={job.external_url} target="_blank" rel="noreferrer">Ver publicación</a>}
                {job.download_url && <button className="job-link" type="button" onClick={() => void downloadWithAuth(
                  `${API_BASE}${job.download_url}`,
                  `${job.id}.mp4`,
                ).catch((downloadError: unknown) => setNotice(
                  downloadError instanceof Error ? downloadError.message : 'No se pudo descargar el video.',
                ))}>Descargar MP4</button>}
                {job.error && <p role="status">{job.error}</p>}
                {job.confirmation_required && (
                  <div className="overdue-job-actions">
                    <p role="status">La app se cerró antes de publicar. Elige si quieres publicar ahora o descartar esta tarea.</p>
                    <button
                      type="button"
                      className="job-link"
                      disabled={busyAction === `overdue:${job.id}`}
                      onClick={() => void resolveOverdueScheduleJob(job.id, 'publish_now')}
                    >
                      Publicar ahora
                    </button>
                    <button
                      type="button"
                      className="job-link"
                      disabled={busyAction === `overdue:${job.id}`}
                      onClick={() => void resolveOverdueScheduleJob(job.id, 'discard')}
                    >
                      Descartar
                    </button>
                  </div>
                )}
              </article>
            ))}
          </div>
        </section>
      )}
      {project?.status === 'failed' && (
        <div className="project-failure">
          <p className="error-message" role="alert"><AlertCircle size={16} />{project.error ?? 'El procesamiento falló.'}</p>
          <button type="button" className="retry-button" onClick={onRetry}><WandSparkles size={16} /> Reintentar</button>
        </div>
      )}
      {notice && <p className="notice-message" role="status"><AlertCircle size={17} />{notice}<button type="button" onClick={() => setNotice('')} aria-label="Cerrar aviso"><X size={14} /></button></p>}
      {shareLink && (
        <p className="share-link-message">
          <a href={shareLink.url} target="_blank" rel="noreferrer">{shareLink.url}</a>
          <span>Expira: {new Date(shareLink.expiresAt).toLocaleString()}</span>
          <button type="button" className="job-link" onClick={() => void revokeShareLink()} disabled={busyAction === 'revoke-share'}>
            Revocar enlace
          </button>
        </p>
      )}
      {!project && error && <button type="button" className="retry-button" onClick={() => window.location.reload()}>Volver a cargar el proyecto</button>}
      {project?.status === 'completed' && clips.length === 0 && (
        <section className="empty-clips"><WandSparkles size={23} /><h2>No se encontraron clips</h2><p>El análisis real no identificó momentos que cumplieran la duración y el criterio solicitados.</p></section>
      )}
      <div className="clip-list">
        {clips.map((clip, index) => (
          <article className="clip-card" key={clip.id}>
            <label className="clip-select"><input type="checkbox" checked={selected.has(clip.id)} disabled={clip.status === 'failed'} onChange={(event) => setSelected((current) => {
              const next = new Set(current);
              if (event.target.checked) next.add(clip.id);
              else next.delete(clip.id);
              return next;
            })} /><span className="visually-hidden">Seleccionar clip {index + 1}</span></label>
            <div className="clip-preview">
              {clip.status === 'failed'
                ? <div className="clip-render-failed"><AlertCircle size={22} /><span>Render fallido</span></div>
                : <video
                ref={(element) => { players.current[clip.id] = element; }}
                src={`${API_BASE}${clip.video_url ?? ''}?v=${encodeURIComponent(project?.updated_at ?? '')}`}
                controls
                preload="metadata"
                playsInline
                aria-label={`Vista previa de ${clip.title}`}
              />}
              {clip.status !== 'failed' && <select
                className="clip-quality"
                value={clip.resolution ?? 720}
                onChange={(event) => void changeResolution(clip, event.target.value === '1080' ? 1080 : 720)}
                disabled={busyAction === clip.id}
                aria-label={`Resolución de ${clip.title}`}
              >
                <option value={720}>720p</option><option value={1080}>1080p</option>
              </select>}
              <span className="clip-duration">{formatDuration(clip.duration)}</span>
            </div>
            <div className="clip-content">
              <h2>#{index + 1} {clip.title}</h2>
              {clip.status === 'failed'
                ? <p className="error-message" role="alert">{clip.error ?? 'No se pudo renderizar este clip.'}</p>
                : <div className="clip-score-actions">
                <div className="clip-score"><strong>{clip.score.toFixed(1)}</strong><span>VIRALIDAD</span></div>
                <div className="clip-actions">
                  <button type="button" className="clip-action primary" onClick={() => void saveProjectFiles(false, [clip.id])} disabled={busyAction === 'archive'}>
                    {busyAction === 'archive' ? <LoaderCircle size={16} className="spin" /> : <Download size={16} />} Descargar
                  </button>
                  <button type="button" className="clip-action" onClick={() => void downloadSubtitles(clip.id, 'srt')}>
                    <FileText size={15} /> SRT
                  </button>
                  <button type="button" className="clip-action" onClick={() => void downloadSubtitles(clip.id, 'vtt')}>
                    <FileText size={15} /> VTT
                  </button>
                  <button type="button" className="clip-action" disabled={busyAction === 'publish'} onClick={() => void publishClips([clip.id])}><ArrowRight size={15} /> Publicar</button>
                  <button
                    type="button"
                    className="clip-action"
                    onClick={() => void improveAudio(clip)}
                    disabled={busyAction === clip.id || clip.audio_enhanced}
                    title="Normaliza el volumen y aplica un realce ligero al audio."
                  >
                    {busyAction === clip.id ? <LoaderCircle size={15} className="spin" /> : <Sparkles size={15} />}
                    {clip.audio_enhanced ? 'Mejorado' : 'Mejorar con IA'}
                  </button>
                  <button
                    type="button"
                    className="clip-action icon-only"
                    aria-label="Compartir clip"
                    title="Crear enlace local de reproducción"
                    onClick={() => void shareClip(clip.id)}
                    disabled={sharingClip === clip.id}
                  >
                    {sharingClip === clip.id ? <LoaderCircle size={16} className="spin" /> : <Share2 size={16} />}
                  </button>
                  <button type="button" className="clip-action icon-only" aria-label="Cambiar plantilla" onClick={() => onEdit(clip.id)}><FileVideo2 size={16} /></button>
                  <button type="button" className="clip-action icon-only" aria-label="Editar transcripción" onClick={() => onEdit(clip.id)}><Scissors size={16} /></button>
                </div>
              </div>}
              {clip.status !== 'failed' && (
              <>
              <section className="viral-reason"><strong>Razón viral</strong><p>{clip.viral_reason}</p></section>
              {clip.broll_provider && <p className="broll-attribution">B-roll de {clip.broll_provider} · {clip.broll_creator}{clip.broll_attribution_url && <> · <a href={clip.broll_attribution_url} target="_blank" rel="noreferrer">Ver fuente</a></>}</p>}
              <section className="clip-transcript" aria-label={`Transcripción del clip ${index + 1}`}>
                <strong>Transcripción</strong>
                <div>{clip.transcript.map((word, wordIndex) => (
                  <button
                    className="transcript-word"
                    type="button"
                    key={`${word.start}-${wordIndex}`}
                    title={`Ir a ${formatDuration(word.start)}`}
                    onClick={() => {
                      const player = players.current[clip.id];
                      if (player) {
                        player.currentTime = word.preview_start;
                        void player.play();
                      }
                    }}
                  >
                    {wordIndex === 0 || Math.floor(word.start) !== Math.floor(clip.transcript[wordIndex - 1].start)
                      ? <span className="transcript-time">{formatDuration(word.start)} </span>
                      : null}
                    {word.text}{' '}
                  </button>
                ))}</div>
              </section>
              </>
              )}
            </div>
          </article>
        ))}
      </div>
      {project?.status !== 'completed' && project?.status !== 'failed' && !error && <p className="results-loading"><LoaderCircle size={17} className="spin" /> Cargando el estado real del proyecto…</p>}
      <a className="back-button results-back" href="/">Volver al inicio</a>
    </main>
  );
}

function SharedClipPage({ token }: { token: string }) {
  const [shared, setShared] = useState<SharedClip | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${API_BASE}/api/shares/${encodeURIComponent(token)}`, { signal: controller.signal })
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
  }, [token]);

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
              src={`${API_BASE}${shared.video_url}`}
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

function ClipEditorPage({
  projectId,
  clipId,
  project,
  savedTemplates,
  onBack,
}: {
  projectId: string;
  clipId: string;
  project: ProjectResult | null;
  savedTemplates: Template[];
  onBack: () => void;
}) {
  const [editor, setEditor] = useState<ClipEditorState | null>(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [previewing, setPreviewing] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const player = useRef<HTMLVideoElement>(null);
  const clip = project?.clips.find((item) => item.id === clipId);
  const endpoint = `${API_BASE}/api/projects/${projectId}/clips/${clipId}/edit`;

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError('');
    apiFetch(endpoint, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(await responseError(response));
        return response.json() as Promise<ClipEditorState>;
      })
      .then((loaded) => setEditor(loaded))
      .catch((loadError: unknown) => {
        if (!controller.signal.aborted) {
          setError(loadError instanceof Error ? loadError.message : 'No se pudo cargar el editor.');
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [endpoint]);

  const updatePhrase = (phraseId: string, update: Partial<EditablePhrase>) => {
    setEditor((current) => current ? {
      ...current,
      phrases: current.phrases.map((phrase) => phrase.id === phraseId ? { ...phrase, ...update } : phrase),
    } : current);
    setPreviewUrl('');
    setMessage('');
  };

  const saveDraft = async () => {
    if (!editor) return;
    setSaving(true);
    setError('');
    setMessage('');
    try {
      const response = await apiFetch(endpoint, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(editor),
      });
      if (!response.ok) throw new Error(await responseError(response));
      setEditor(await response.json() as ClipEditorState);
      setMessage('Borrador guardado en el proyecto.');
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : 'No se pudo guardar el borrador.');
    } finally {
      setSaving(false);
    }
  };

  const createPreview = async () => {
    if (!editor) return;
    setPreviewing(true);
    setError('');
    setMessage('');
    try {
      const response = await apiFetch(`${endpoint}/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(editor),
      });
      if (!response.ok) throw new Error(await responseError(response));
      const result = await response.json() as { preview_url: string };
      setPreviewUrl(`${API_BASE}${result.preview_url}`);
      setMessage('Vista previa renderizada con tus cambios.');
    } catch (previewError) {
      setError(previewError instanceof Error ? previewError.message : 'No se pudo renderizar la vista previa.');
    } finally {
      setPreviewing(false);
    }
  };

  const exportEdit = async () => {
    if (!editor) return;
    setExporting(true);
    setError('');
    setMessage('');
    try {
      const response = await apiFetch(`${endpoint}/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(editor),
      });
      if (!response.ok) throw new Error(await responseError(response));
      onBack();
    } catch (exportError) {
      setError(exportError instanceof Error ? exportError.message : 'No se pudo exportar el clip.');
    } finally {
      setExporting(false);
    }
  };

  const busy = saving || previewing || exporting;

  return (
    <main className="editor-page">
      <header className="editor-header">
        <button type="button" className="back-button" onClick={onBack}><ArrowLeft size={16} /> Resultados</button>
        <div>
          <span className="eyebrow">EDITOR POR TRANSCRIPCIÓN</span>
          <h1>{clip ? clip.title : loading ? 'Cargando clip…' : 'No se encontró el clip'}</h1>
          <p>Quita frases para recortar el audio y el video; ajusta subtítulos antes de exportar.</p>
        </div>
      </header>

      {error && <p className="error-message" role="alert"><AlertCircle size={16} />{error}</p>}
      {message && <p className="editor-message" role="status"><Check size={16} />{message}</p>}
      {loading && <p className="results-loading"><LoaderCircle size={17} className="spin" /> Cargando transcripción guardada…</p>}
      {!loading && editor && clip && (
        <>
          <section className="editor-workspace">
            <div className="editor-preview-card">
              <div className="editor-preview-frame">
                <video
                  key={previewUrl || clip.video_url || clip.id}
                  ref={player}
                  src={previewUrl || (clip.video_url ? `${API_BASE}${clip.video_url}` : undefined)}
                  controls
                  playsInline
                  preload="metadata"
                  aria-label={`Vista previa de ${clip.title}`}
                />
              </div>
              <p>{previewUrl ? 'Vista previa de los cambios' : 'Clip exportado actualmente'}</p>
            </div>

            <div className="editor-settings">
              <label className="config-field">
                <span>Ratio de salida</span>
                <select value={editor.ratio} onChange={(event) => {
                  setEditor({ ...editor, ratio: event.target.value });
                  setPreviewUrl('');
                  setMessage('');
                }}>
                  <option value="9:16">9:16 · Vertical</option>
                  <option value="16:9">16:9 · Horizontal</option>
                  <option value="1:1">1:1 · Cuadrado</option>
                  <option value="4:5">4:5 · Retrato</option>
                </select>
              </label>
              <label className="config-field">
                <span>Plantilla de subtítulos</span>
                <select value={editor.subtitle_template} onChange={(event) => {
                  setEditor({ ...editor, subtitle_template: event.target.value });
                  setPreviewUrl('');
                  setMessage('');
                }}>
                  {[...templates, ...customTemplates, ...savedTemplates].map((template) => (
                    <option key={template.name} value={template.name}>{template.name}</option>
                  ))}
                </select>
              </label>
              <SubtitleStyleControls
                value={editor.subtitle_style}
                onChange={(subtitle_style) => {
                  setEditor({ ...editor, subtitle_style });
                  setPreviewUrl('');
                  setMessage('');
                }}
              />
              <div className="editor-tip">
                <strong>{editor.phrases.filter((phrase) => phrase.removed).length} frases marcadas</strong>
                <span>Los cambios se aplican al exportar desde el video original.</span>
              </div>
            </div>
          </section>

          <section className="editor-transcript-card">
            <div className="editor-section-heading">
              <div><h2>Transcripción</h2><p>Edita el texto que aparecerá como subtítulo o quita una frase completa.</p></div>
              <span>{editor.phrases.length} frases</span>
            </div>
            {editor.phrases.length ? (
              <div className="editor-phrases">
                {editor.phrases.map((phrase, index) => (
                  <article className={`editor-phrase ${phrase.removed ? 'is-removed' : ''}`} key={phrase.id}>
                    <button
                      type="button"
                      className="editor-phrase-time"
                      onClick={() => {
                        if (!phrase.removed && player.current) {
                          player.current.currentTime = phrase.preview_start;
                          void player.current.play();
                        }
                      }}
                      disabled={phrase.removed}
                      aria-label={`Reproducir frase ${index + 1} desde ${formatDuration(phrase.preview_start)}`}
                    >
                      {formatDuration(phrase.preview_start)}
                    </button>
                    <label className="editor-phrase-text">
                      <span className="visually-hidden">Texto del subtítulo {index + 1}</span>
                      <textarea
                        rows={2}
                        maxLength={400}
                        value={phrase.text}
                        disabled={phrase.removed}
                        onChange={(event) => updatePhrase(phrase.id, { text: event.target.value })}
                      />
                    </label>
                    <button
                      type="button"
                      className="editor-phrase-toggle"
                      onClick={() => updatePhrase(phrase.id, { removed: !phrase.removed })}
                    >
                      {phrase.removed ? <><ArrowLeft size={15} /> Restaurar</> : <><X size={15} /> Quitar</>}
                    </button>
                  </article>
                ))}
              </div>
            ) : (
              <p className="empty-templates">Este clip no tiene palabras con marcas de tiempo para editar.</p>
            )}
          </section>

          <footer className="editor-actions">
            <button type="button" className="subtle-button" onClick={() => void saveDraft()} disabled={busy}>
              {saving ? <LoaderCircle size={16} className="spin" /> : <Check size={16} />}
              Guardar borrador
            </button>
            <div>
              <button type="button" className="outline-action" onClick={() => void createPreview()} disabled={busy}>
                {previewing ? <LoaderCircle size={16} className="spin" /> : <Video size={16} />}
                Vista previa
              </button>
              <button type="button" className="create-button" onClick={() => void exportEdit()} disabled={busy}>
                {exporting ? <LoaderCircle size={17} className="spin" /> : <Scissors size={17} />}
                Exportar cambios
              </button>
            </div>
          </footer>
        </>
      )}
    </main>
  );
}

function Option({ checked, onClick, title, description, premium = false }: { checked: boolean; onClick: () => void; title: string; description: string; premium?: boolean }) {
  return (
    <button className={`option-item ${checked ? 'checked' : ''}`} type="button" role="checkbox" aria-checked={checked} onClick={onClick}>
      <span className={`fake-checkbox ${checked ? 'checked' : ''}`}>{checked && <Check size={12} strokeWidth={3} />}</span>
      <span className="option-copy"><span className="option-title">{title}{premium && <span className="diamond"> ◆</span>}</span><span className="option-description">{description}</span></span>
    </button>
  );
}

function SubtitleStyleControls({
  value,
  onChange,
}: {
  value: SubtitleStyleState;
  onChange: (style: SubtitleStyleState) => void;
}) {
  const update = <Key extends keyof SubtitleStyleState>(
    key: Key,
    nextValue: SubtitleStyleState[Key],
  ) => onChange({ ...value, [key]: nextValue });

  return (
    <div className="subtitle-style-grid">
      <label className="config-field">
        <span>Fuente de subtítulos</span>
        <select className="text-input" value={value.font_family ?? ''} onChange={(event) => {
          update('font_family', event.target.value || null);
        }}>
          <option value="">Usar fuente de plantilla</option>
          <option value="Arial">Arial</option>
          <option value="Arial Black">Arial Black</option>
          <option value="Georgia">Georgia</option>
          <option value="Tahoma">Tahoma</option>
          <option value="Verdana">Verdana</option>
        </select>
      </label>
      <label className="config-field">
        <span>Tamaño de fuente</span>
        <select className="text-input" value={value.font_size ?? ''} onChange={(event) => {
          update('font_size', event.target.value ? Number(event.target.value) : null);
        }}>
          <option value="">Usar tamaño de plantilla</option>
          <option value="32">Pequeño</option>
          <option value="48">Mediano</option>
          <option value="64">Grande</option>
          <option value="80">Muy grande</option>
        </select>
      </label>
      <div className="subtitle-color-control">
        <label htmlFor="subtitle-font-color">Color de texto</label>
        <input
          id="subtitle-font-color"
          type="color"
          value={value.font_color ?? '#ffffff'}
          onChange={(event) => update('font_color', event.target.value)}
        />
        <button type="button" onClick={() => update('font_color', null)}>Usar plantilla</button>
      </div>
      <div className="subtitle-color-control">
        <label htmlFor="subtitle-highlight-color">Color de resaltado</label>
        <input
          id="subtitle-highlight-color"
          type="color"
          value={value.highlight_color ?? '#ffff00'}
          onChange={(event) => update('highlight_color', event.target.value)}
        />
        <button type="button" onClick={() => update('highlight_color', null)}>Usar plantilla</button>
      </div>
      <label className="config-field">
        <span>Posición</span>
        <select className="text-input" value={value.position ?? ''} onChange={(event) => {
          const position = event.target.value;
          update(
            'position',
            position === 'top' || position === 'center' || position === 'bottom'
              ? position
              : null,
          );
        }}>
          <option value="">Usar posición de plantilla</option>
          <option value="top">Arriba</option>
          <option value="center">Centro</option>
          <option value="bottom">Abajo</option>
        </select>
      </label>
    </div>
  );
}

function SettingsPage({ provider, setProvider, whisper, setWhisper, onBack, onNotice, notice }: {
  provider: string;
  setProvider: (value: string) => void;
  whisper: string;
  setWhisper: (value: string) => void;
  onBack: () => void;
  onNotice: (value: string) => void;
  notice: string;
}) {
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
    const params = new URLSearchParams(window.location.search);
    const integration = params.get('integration');
    if (!integration) return;
    const status = params.get('status');
    const message = params.get('message');
    onNotice(status === 'connected'
      ? `${integration} se conectó correctamente.`
      : message || `No se pudo conectar ${integration}.`);
    window.history.replaceState(null, '', window.location.pathname);
  }, [onNotice]);

  const connectIntegration = async (integration: string) => {
    setIntegrationBusy(integration);
    onNotice('');
    try {
      const response = await apiFetch(`${API_BASE}/api/integrations/${integration}/connect`);
      if (!response.ok) throw new Error(await responseError(response));
      const result = await response.json() as { authorization_url: string };
      window.location.assign(result.authorization_url);
    } catch (error) {
      onNotice(error instanceof Error ? error.message : 'No se pudo iniciar la conexión.');
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
      onNotice(`Se desconectó ${integration}.`);
    } catch (error) {
      onNotice(error instanceof Error ? error.message : 'No se pudo desconectar la cuenta.');
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
        }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      setApiKey('');
      setHasSavedKey(true);
      setSaved(true);
      onNotice('Ajustes guardados. La clave se cifra con DPAPI de Windows y no vuelve a mostrarse.');
    } catch (error) {
      onNotice(error instanceof Error ? error.message : 'No se pudieron guardar los ajustes.');
    }
  };

  const removeSavedKey = async () => {
    try {
      const response = await apiFetch(`${API_BASE}/api/settings/llm-key`, { method: 'DELETE' });
      if (!response.ok) throw new Error(await responseError(response));
      setHasSavedKey(false);
      setApiKey('');
      setSaved(false);
      onNotice('Se eliminó la clave guardada para esta cuenta.');
    } catch (error) {
      onNotice(error instanceof Error ? error.message : 'No se pudo eliminar la clave guardada.');
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
    <main className="settings-page">
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
      {notice && <p className="notice-message" role="status"><AlertCircle size={17} />{notice}</p>}
    </main>
  );
}

export default App;
