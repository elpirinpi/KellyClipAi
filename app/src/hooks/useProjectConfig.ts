import { useCallback, useEffect, useState, type KeyboardEvent as ReactKeyboardEvent } from 'react';
import { apiFetch } from '../auth';
import {
  MOMENT_ENGINE_STORAGE_KEY, SIGNAL_THRESHOLD_STORAGE_KEY,
  customTemplates, readStoredMomentEngine, readStoredSignalThreshold, templates,
  type MomentEngine, type Template,
} from '../context/constants';
import type { ToastKind } from '../components/Toasts';
import { DEFAULT_SUBTITLE_STYLE, type ProjectResult, type SubtitleStyleState } from '../types';
import { responseError } from '../utils';

// Configuración de clips (extraída de App.tsx, ítem 23 de la Fase 2):
// ratio, duración, encuadre, plantillas de subtítulos, opciones de
// edición, momento, programación, modelo de Whisper e idioma, estilo de
// subtítulos, proveedor de IA y motor de momentos (Fase A, v1.1.0).
// Agrupa también la construcción de la configuración que se envía al
// backend (projectConfig) y la aplicación de la configuración de un
// proyecto cargado (applyProjectConfiguration).

type UseProjectConfigOptions = {
  apiBase: string;
  authReady: boolean;
  userId: string | undefined;
  notify: (message: string, kind?: ToastKind) => void;
  setLinkProject: (project: ProjectResult | null) => void;
};

export function useProjectConfig({ apiBase, authReady, userId, notify, setLinkProject }: UseProjectConfigOptions) {
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
  const [settingsProvider, setSettingsProvider] = useState('Groq');
  const [whisperModel, setWhisperModel] = useState('small');
  const [whisperLanguage, setWhisperLanguage] = useState('auto');
  const [subtitleStyle, setSubtitleStyle] = useState<SubtitleStyleState>(DEFAULT_SUBTITLE_STYLE);
  // Motor de momentos (Fase A, v1.1.0): el valor inicial es la preferencia
  // guardada en Ajustes (localStorage); cada cambio se persiste para los
  // próximos proyectos. Al cargar un proyecto se restaura su configuración.
  const [momentEngine, setMomentEngineState] = useState<MomentEngine>(() => readStoredMomentEngine());
  const [signalThreshold, setSignalThresholdState] = useState<number>(() => readStoredSignalThreshold());

  const setMomentEngine = useCallback((value: MomentEngine) => {
    setMomentEngineState(value);
    try {
      window.localStorage.setItem(MOMENT_ENGINE_STORAGE_KEY, value);
    } catch {
      // Sin almacenamiento disponible: la elección vale para esta sesión.
    }
  }, []);

  const setSignalThreshold = useCallback((value: number) => {
    setSignalThresholdState(value);
    try {
      window.localStorage.setItem(SIGNAL_THRESHOLD_STORAGE_KEY, String(value));
    } catch {
      // Sin almacenamiento disponible: la elección vale para esta sesión.
    }
  }, []);

  useEffect(() => {
    if (!authReady || !userId) return;
    let active = true;
    apiFetch(`${apiBase}/api/templates`)
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
        if (active) notify(error instanceof Error ? error.message : 'No se pudieron cargar tus plantillas.', 'error');
      });
    return () => {
      active = false;
    };
  }, [apiBase, authReady, notify, userId]);

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
    try {
      const response = await apiFetch(`${apiBase}/api/templates`, {
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
      notify(`Se guardó “${saved.name}” y estará disponible después de recargar.`, 'success');
    } catch (error) {
      notify(error instanceof Error ? error.message : 'No se pudo guardar la plantilla.', 'error');
    } finally {
      setSavingTemplate(false);
    }
  };

  const toggleOption = (key: keyof typeof options) => setOptions((current) => ({ ...current, [key]: !current[key] }));

  // Fase 2 (ítem 21): navegación por teclado en las pestañas de
  // plantillas (flechas, Inicio, Fin), patrón ARIA tabs con roving tabindex.
  const templateTabs = ['featured', 'custom', 'brand'] as const;
  const onTemplateTabKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    const keys = ['ArrowRight', 'ArrowLeft', 'ArrowDown', 'ArrowUp', 'Home', 'End'];
    if (!keys.includes(event.key)) return;
    event.preventDefault();
    const current = templateTabs.indexOf(templateTab);
    let next = current;
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (current + 1) % templateTabs.length;
    else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') next = (current - 1 + templateTabs.length) % templateTabs.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = templateTabs.length - 1;
    setTemplateTab(templateTabs[next]);
    document.getElementById(`template-tab-${templateTabs[next]}`)?.focus();
  };

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
      moment_engine: momentEngine,
      signal_score_threshold: signalThreshold,
      schedule: schedule
        ? {
            scheduled_at: new Date(scheduleDate).toISOString(),
            platform: schedulePlatform,
            visibility: scheduleVisibility,
          }
        : null,
    };
  };

  const applyProjectConfiguration = useCallback((loaded: ProjectResult) => {
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
    setMomentEngineState(loaded.config.moment_engine ?? readStoredMomentEngine());
    setSignalThresholdState(loaded.config.signal_score_threshold ?? readStoredSignalThreshold());
    setSchedule(Boolean(loaded.config.schedule));
    if (loaded.config.schedule) {
      const localDate = new Date(loaded.config.schedule.scheduled_at);
      setScheduleDate(new Date(localDate.getTime() - localDate.getTimezoneOffset() * 60000).toISOString().slice(0, 16));
      setSchedulePlatform(loaded.config.schedule.platform);
      setScheduleVisibility(loaded.config.schedule.visibility);
    }
  }, [setLinkProject]);

  return {
    ratio,
    setRatio,
    duration,
    setDuration,
    framing,
    setFraming,
    activeTemplate,
    setActiveTemplate,
    templateTab,
    setTemplateTab,
    savedTemplates,
    setSavedTemplates,
    templateName,
    setTemplateName,
    savingTemplate,
    options,
    toggleOption,
    moment,
    setMoment,
    schedule,
    setSchedule,
    scheduleDate,
    setScheduleDate,
    schedulePlatform,
    setSchedulePlatform,
    scheduleVisibility,
    setScheduleVisibility,
    settingsProvider,
    setSettingsProvider,
    whisperModel,
    setWhisperModel,
    whisperLanguage,
    setWhisperLanguage,
    momentEngine,
    setMomentEngine,
    signalThreshold,
    setSignalThreshold,
    subtitleStyle,
    setSubtitleStyle,
    visibleTemplates,
    saveCurrentTemplate,
    onTemplateTabKeyDown,
    projectConfig,
    applyProjectConfiguration,
  };
}
