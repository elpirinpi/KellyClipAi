// Editor visual (Studio) de kellyClipAi v1.3.0, adaptado del App de
// ShortStudio. Diferencias con el original:
// - El video es el FUENTE del proyecto (GET /api/projects/:id/source),
//   nunca el clip renderizado; sin subida de video ni clip demo.
// - El recorte inicial es el inicio/fin del clip; en un clip manual
//   nuevo abarca el video completo.
// - Los subtítulos vienen del backend (cues de la transcripción);
//   la edición (ajustes + cues) se guarda en `clip.studio_edit`.
// - La exportación usa el mismo camino MediaRecorder de ShortStudio,
//   pero sube el blob al backend en vez de descargarlo; si la subida
//   falla, descarga el video en el navegador como respaldo.
// - La música de fondo (BGM) es solo de la sesión: no se persiste.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertCircle, ArrowLeft, LoaderCircle, X } from 'lucide-react';
import { API_BASE } from '../context/constants';
import type { ClipResult, ProjectResult } from '../types';
import {
  apiCuesToLocal,
  exportStudioVideo,
  fetchClipCues,
  fetchRangeCues,
  saveStudioEdit,
} from './api';
import { LeftSidebar } from './components/LeftSidebar';
import { ExportModal, ScriptModal } from './components/Modals';
import { PreviewStage, type SourceState } from './components/PreviewStage';
import { RightSidebar } from './components/RightSidebar';
import { StudioHeader, type StudioSaveState } from './components/StudioHeader';
import { Timeline } from './components/Timeline';
import {
  DEFAULT_SETTINGS,
  normalizeStudioEdit,
  type EditorSettings,
  type LeftTab,
  type SafeZoneMode,
  type StudioEditSpec,
  type SubtitleCue,
} from './types/editor';
import { renderEditorFrame } from './utils/canvasRenderer';
import { exportSubtitlesToSRT, formatTime, parseSRT, splitScriptToSubtitles } from './utils/srt';
import './studio.css';

// Autoguardado como en ClipEditorPage: 2 s después del último cambio.
const STUDIO_AUTOSAVE_DELAY_MS = 2000;

function getSupportedMimeType(): string {
  if (typeof MediaRecorder === 'undefined') return '';
  const types = [
    'video/mp4;codecs=avc1,mp4a.40.2',
    'video/mp4',
    'video/webm;codecs=vp9,opus',
    'video/webm;codecs=vp8,opus',
    'video/webm',
  ];
  return types.find((t) => MediaRecorder.isTypeSupported(t)) || '';
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

export function StudioPage({
  projectId,
  clipId,
  project,
  error,
  onBack,
  onProjectUpdated,
}: {
  projectId: string;
  clipId: string;
  project: ProjectResult | null;
  error: string;
  onBack: () => void;
  onProjectUpdated: (projectId: string) => Promise<void>;
}) {
  const isManual = clipId === 'new';
  const clip: ClipResult | undefined = isManual
    ? undefined
    : project?.clips.find((item) => item.id === clipId);

  const [activeTab, setActiveTab] = useState<LeftTab>('subs');
  const [safeZone, setSafeZone] = useState<SafeZoneMode>('none');
  const [settings, setSettings] = useState<EditorSettings>(DEFAULT_SETTINGS);
  const [subtitles, setSubtitles] = useState<SubtitleCue[]>([]);
  const [selectedSubId, setSelectedSubId] = useState<number | null>(null);

  const [sourceState, setSourceState] = useState<SourceState>('loading');
  const [sourceError, setSourceError] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [loop, setLoop] = useState(true);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  const [hasBgm, setHasBgm] = useState(false);
  const [bgmName, setBgmName] = useState('');

  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [exportStatus, setExportStatus] = useState(
    'Preparando renderizado en 1080×1920 con mezcla de audio...'
  );
  const [scriptModalOpen, setScriptModalOpen] = useState(false);
  const [toastMsg, setToastMsg] = useState<string | null>(null);
  const [bannerError, setBannerError] = useState('');
  const [loadError, setLoadError] = useState('');

  // Título del clip (se aplica al exportar) y estado de guardado.
  const [title, setTitle] = useState('');
  const [saving, setSaving] = useState(false);
  const [autosaving, setAutosaving] = useState(false);
  const [autosavedAt, setAutosavedAt] = useState<Date | null>(null);
  const [manualSavePending, setManualSavePending] = useState(false);
  const [savedSpecJson, setSavedSpecJson] = useState<string | null>(null);
  const [cuesReady, setCuesReady] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const bgmRef = useRef<HTMLAudioElement | null>(null);
  const bgmInputRef = useRef<HTMLInputElement | null>(null);
  const srtInputRef = useRef<HTMLInputElement | null>(null);

  const blurCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const blurCtxRef = useRef<CanvasRenderingContext2D | null>(null);

  const settingsRef = useRef(settings);
  const subtitlesRef = useRef(subtitles);
  const sourceReadyRef = useRef(false);
  const isPlayingRef = useRef(isPlaying);
  const loopRef = useRef(loop);
  const durationRef = useRef(duration);
  const hasBgmRef = useRef(hasBgm);
  const isExportingRef = useRef(isExporting);
  const bgmUrlRef = useRef<string | null>(null);
  const toastTimerRef = useRef<number | null>(null);
  const nextCueIdRef = useRef(1);
  const initializedForRef = useRef('');
  const cuesFetchedForRef = useRef('');
  const savedSpecRef = useRef(false);
  const isManualRef = useRef(isManual);
  isManualRef.current = isManual;

  const audioCtxRef = useRef<AudioContext | null>(null);
  const videoGainRef = useRef<GainNode | null>(null);
  const bgmGainRef = useRef<GainNode | null>(null);
  const speakerGainRef = useRef<GainNode | null>(null);
  const audioDestRef = useRef<MediaStreamAudioDestinationNode | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const exportWatchdogRef = useRef<number | null>(null);
  const exportCanceledRef = useRef(false);
  const uploadControllerRef = useRef<AbortController | null>(null);

  settingsRef.current = settings;
  subtitlesRef.current = subtitles;
  sourceReadyRef.current = sourceState === 'ready';
  isPlayingRef.current = isPlaying;
  loopRef.current = loop;
  durationRef.current = duration;
  hasBgmRef.current = hasBgm;
  isExportingRef.current = isExporting;

  const nextCueId = useCallback(() => nextCueIdRef.current++, []);

  const showToast = useCallback((msg: string) => {
    setToastMsg(msg);
    if (toastTimerRef.current) window.clearTimeout(toastTimerRef.current);
    toastTimerRef.current = window.setTimeout(() => setToastMsg(null), 2600);
  }, []);

  const currentSpec: StudioEditSpec = useMemo(() => ({
    version: 1,
    settings,
    cues: subtitles,
  }), [settings, subtitles]);

  const initialized = initializedForRef.current === clipId;
  const isDirty = !isManual
    && initialized
    && cuesReady
    && (savedSpecJson === null || JSON.stringify(currentSpec) !== savedSpecJson);

  const saveState: StudioSaveState = saving || autosaving
    ? 'saving'
    : isManual
      ? (manualSavePending ? 'pending-manual' : 'idle')
      : isDirty
        ? 'dirty'
        : autosavedAt
          ? 'saved'
          : 'idle';

  // Inicialización al cargar el proyecto: ajustes (edición guardada o
  // recorte del clip), título y duración conocida por metadatos. Los
  // cues se cargan en el efecto siguiente, salvo que vengan guardados.
  useEffect(() => {
    if (!project || initializedForRef.current === clipId) return;
    if (!isManual && !clip) {
      setLoadError('No se encontró el clip en este proyecto.');
      return;
    }
    initializedForRef.current = clipId;
    const metaDuration = project.metadata?.duration ?? 0;
    if (metaDuration > 0) {
      setDuration(metaDuration);
      durationRef.current = metaDuration;
    }
    const saved = !isManual ? normalizeStudioEdit(clip?.studio_edit) : null;
    savedSpecRef.current = saved !== null;
    let initialSettings: EditorSettings = { ...DEFAULT_SETTINGS };
    if (saved) {
      initialSettings = { ...saved.settings };
    } else if (clip) {
      initialSettings.trimStart = clip.start;
      initialSettings.trimEnd = clip.end;
    }
    if (metaDuration > 0) {
      initialSettings.trimEnd = Math.min(initialSettings.trimEnd, metaDuration);
      initialSettings.trimStart = Math.min(
        initialSettings.trimStart,
        Math.max(0, initialSettings.trimEnd - 0.3),
      );
    }
    setSettings(initialSettings);
    settingsRef.current = initialSettings;
    setTitle(clip?.title ?? 'Clip manual');
    setCurrentTime(initialSettings.trimStart);
    if (saved) {
      const restored = saved.cues.map((cue) => ({ ...cue, id: nextCueIdRef.current++ }));
      setSubtitles(restored);
      setSelectedSubId(restored[0]?.id ?? null);
      setSavedSpecJson(JSON.stringify({ version: 1, settings: initialSettings, cues: restored }));
      setCuesReady(true);
    }
  }, [project, clipId, isManual, clip]);

  // Cues del backend cuando no hay edición guardada: del clip existente
  // o del rango completo en un clip manual (necesita la duración).
  useEffect(() => {
    if (!initialized || cuesReady || cuesFetchedForRef.current === clipId) return;
    if (isManual && duration <= 0) return;
    cuesFetchedForRef.current = clipId;
    const controller = new AbortController();
    const load = isManual
      ? fetchRangeCues(projectId, 0, duration)
      : fetchClipCues(projectId, clipId);
    load
      .then((apiCues) => {
        if (controller.signal.aborted) return;
        const local = apiCuesToLocal(apiCues, nextCueId);
        setSubtitles(local);
        setSelectedSubId(local[0]?.id ?? null);
        if (local.length === 0) {
          showToast('Este video no tiene transcripción: puedes añadir subtítulos a mano.');
        }
        setCuesReady(true);
      })
      .catch((loadError: unknown) => {
        if (controller.signal.aborted) return;
        showToast(loadError instanceof Error
          ? `No se pudieron cargar los subtítulos: ${loadError.message}`
          : 'No se pudieron cargar los subtítulos.');
        setCuesReady(true);
      });
    return () => controller.abort();
  }, [initialized, cuesReady, clipId, isManual, duration, projectId, nextCueId, showToast]);

  const getBlurResources = useCallback(() => {
    if (!blurCanvasRef.current) {
      const bc = document.createElement('canvas');
      bc.width = 270;
      bc.height = 480;
      blurCanvasRef.current = bc;
      blurCtxRef.current = bc.getContext('2d', { alpha: false });
    }
    return {
      blurCanvas: blurCanvasRef.current,
      blurCtx: blurCtxRef.current,
    };
  }, []);

  const drawCurrentFrame = useCallback(
    (explicitTime?: number) => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext('2d', { alpha: false });
      const { blurCanvas, blurCtx } = getBlurResources();
      if (!ctx || !blurCtx) return;

      const t = explicitTime !== undefined
        ? explicitTime
        : videoRef.current?.currentTime || 0;

      renderEditorFrame({
        ctx,
        blurCanvas,
        blurCtx,
        video: videoRef.current,
        hasMedia: sourceReadyRef.current,
        currentTime: t,
        settings: settingsRef.current,
        subtitles: subtitlesRef.current,
      });
    },
    [getBlurResources]
  );

  const applyAudioVolumes = useCallback((volVideo: number, volBgm: number, bgmActive: boolean) => {
    const audioCtx = audioCtxRef.current;
    if (!audioCtx) return;
    const now = audioCtx.currentTime;
    if (videoGainRef.current) {
      videoGainRef.current.gain.setTargetAtTime(volVideo / 100, now, 0.02);
    }
    if (bgmGainRef.current) {
      bgmGainRef.current.gain.setTargetAtTime(bgmActive ? volBgm / 100 : 0, now, 0.02);
    }
  }, []);

  const initAudio = useCallback(() => {
    if (audioCtxRef.current) {
      if (audioCtxRef.current.state === 'suspended') {
        audioCtxRef.current.resume().catch(() => {});
      }
      return;
    }

    const AudioCtxClass =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    const audioCtx = new AudioCtxClass();
    audioCtxRef.current = audioCtx;

    const masterGain = audioCtx.createGain();
    const speakerGain = audioCtx.createGain();
    const audioDest = audioCtx.createMediaStreamDestination();
    speakerGainRef.current = speakerGain;
    audioDestRef.current = audioDest;

    const osc = audioCtx.createOscillator();
    const silentGain = audioCtx.createGain();
    silentGain.gain.value = 0.00001;
    osc.connect(silentGain);
    silentGain.connect(masterGain);
    osc.start();

    const videoGain = audioCtx.createGain();
    const bgmGain = audioCtx.createGain();
    videoGainRef.current = videoGain;
    bgmGainRef.current = bgmGain;

    if (videoRef.current) {
      try {
        const videoSource = audioCtx.createMediaElementSource(videoRef.current);
        videoSource.connect(videoGain);
        videoGain.connect(masterGain);
      } catch {
        // Already connected
      }
    }

    if (bgmRef.current) {
      try {
        const bgmSource = audioCtx.createMediaElementSource(bgmRef.current);
        bgmSource.connect(bgmGain);
        bgmGain.connect(masterGain);
      } catch {
        // Already connected
      }
    }

    masterGain.connect(speakerGain);
    speakerGain.connect(audioCtx.destination);
    masterGain.connect(audioDest);

    applyAudioVolumes(settingsRef.current.volVideo, settingsRef.current.volBgm, hasBgmRef.current);
  }, [applyAudioVolumes]);

  const getCurrentPlayTime = useCallback(() => videoRef.current?.currentTime || 0, []);

  const seekTo = useCallback(
    (time: number) => {
      const clamped = Math.max(0, Math.min(durationRef.current, time));
      if (sourceReadyRef.current && videoRef.current) {
        videoRef.current.currentTime = clamped;
        if (hasBgmRef.current && bgmRef.current && bgmRef.current.duration) {
          bgmRef.current.currentTime =
            Math.max(0, clamped - settingsRef.current.trimStart) % bgmRef.current.duration;
        }
        setCurrentTime(clamped);
      }
    },
    []
  );

  const togglePlay = useCallback(
    (forceState?: boolean) => {
      if (!sourceReadyRef.current) {
        showToast('El video original todavía se está cargando');
        return;
      }
      initAudio();
      const shouldPlay = forceState !== undefined ? forceState : !isPlayingRef.current;
      isPlayingRef.current = shouldPlay;
      setIsPlaying(shouldPlay);

      if (shouldPlay) {
        const cur = getCurrentPlayTime();
        if (cur < settingsRef.current.trimStart || cur >= settingsRef.current.trimEnd - 0.05) {
          seekTo(settingsRef.current.trimStart);
        }
        if (videoRef.current) {
          videoRef.current.playbackRate = settingsRef.current.speed;
          videoRef.current.play().catch(() => {});
        }
        if (hasBgmRef.current && bgmRef.current) {
          bgmRef.current.play().catch(() => {});
        }
      } else {
        if (videoRef.current) {
          videoRef.current.pause();
        }
        if (bgmRef.current) {
          bgmRef.current.pause();
        }
      }
    },
    [getCurrentPlayTime, initAudio, seekTo, showToast]
  );

  useEffect(() => {
    let rafId = 0;

    const tick = () => {
      if (isPlayingRef.current && !isExportingRef.current) {
        const s = settingsRef.current;
        if (sourceReadyRef.current && videoRef.current) {
          const v = videoRef.current;
          if (v.currentTime >= s.trimEnd || v.ended) {
            if (loopRef.current) {
              v.currentTime = s.trimStart;
              if (hasBgmRef.current && bgmRef.current?.duration) {
                bgmRef.current.currentTime = 0;
              }
              v.play().catch(() => {});
            } else {
              togglePlay(false);
            }
          }
          setCurrentTime(v.currentTime);
          drawCurrentFrame(v.currentTime);
        }
      } else if (isExportingRef.current) {
        drawCurrentFrame();
      }

      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [drawCurrentFrame, togglePlay]);

  useEffect(() => {
    if (!isPlaying && !isExporting) {
      drawCurrentFrame();
    }
  }, [settings, subtitles, sourceState, isPlaying, isExporting, drawCurrentFrame]);

  useEffect(() => {
    applyAudioVolumes(settings.volVideo, settings.volBgm, hasBgm);
    if (videoRef.current) {
      videoRef.current.playbackRate = settings.speed;
    }
  }, [settings.volVideo, settings.volBgm, settings.speed, hasBgm, applyAudioVolumes]);

  useEffect(() => {
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(() => drawCurrentFrame());
    }
  }, [drawCurrentFrame]);

  useEffect(() => {
    return () => {
      if (bgmUrlRef.current) URL.revokeObjectURL(bgmUrlRef.current);
      if (exportWatchdogRef.current) window.clearInterval(exportWatchdogRef.current);
    };
  }, []);

  const updateSetting = useCallback(
    <K extends keyof EditorSettings>(key: K, value: EditorSettings[K]) => {
      setSettings((prev) => ({ ...prev, [key]: value }));
    },
    []
  );

  const applySubtitlePreset = useCallback((patch: Partial<EditorSettings>) => {
    setSettings((prev) => ({ ...prev, ...patch }));
  }, []);

  const handleTrimStartChange = (val: number) => {
    const next = Math.max(0, Math.min(settings.trimEnd - 0.3, val));
    updateSetting('trimStart', Number(next.toFixed(1)));
    seekTo(next);
  };

  const handleTrimEndChange = (val: number) => {
    const next = Math.max(settings.trimStart + 0.3, Math.min(duration, val));
    updateSetting('trimEnd', Number(next.toFixed(1)));
  };

  const handleSetIn = useCallback(() => {
    const cur = getCurrentPlayTime();
    if (cur < settingsRef.current.trimEnd - 0.3) {
      const next = Number(cur.toFixed(1));
      updateSetting('trimStart', next);
      showToast(`Inicio fijado en ${formatTime(next)}`);
    }
  }, [getCurrentPlayTime, showToast, updateSetting]);

  const handleSetOut = useCallback(() => {
    const cur = getCurrentPlayTime();
    if (cur > settingsRef.current.trimStart + 0.3) {
      const next = Number(cur.toFixed(1));
      updateSetting('trimEnd', next);
      showToast(`Fin fijado en ${formatTime(next)}`);
    }
  }, [getCurrentPlayTime, showToast, updateSetting]);

  // Persistencia de la edición: en un clip manual aún sin crear, la
  // especificación se guarda junto con la exportación (el botón solo
  // marca que está pendiente); en un clip existente va por PUT.
  const persistSpec = useCallback(async (spec: StudioEditSpec, silent: boolean): Promise<boolean> => {
    if (isManualRef.current) {
      setManualSavePending(true);
      if (!silent) showToast('La edición se guardará al exportar el clip manual.');
      return true;
    }
    try {
      await saveStudioEdit(projectId, clipId, spec);
      setSavedSpecJson(JSON.stringify(spec));
      setAutosavedAt(new Date());
      if (!silent) showToast('Edición guardada.');
      return true;
    } catch (saveError) {
      if (!silent) {
        setBannerError(saveError instanceof Error
          ? `No se pudo guardar la edición: ${saveError.message}`
          : 'No se pudo guardar la edición.');
      }
      return false;
    }
  }, [projectId, clipId, showToast]);
  const persistSpecRef = useRef(persistSpec);
  persistSpecRef.current = persistSpec;

  const saveEdit = useCallback(async () => {
    if (saving || isExporting) return;
    setSaving(true);
    setBannerError('');
    await persistSpecRef.current(currentSpec, false);
    setSaving(false);
  }, [saving, isExporting, currentSpec]);
  const saveEditRef = useRef(saveEdit);
  saveEditRef.current = saveEdit;

  // Autoguardado con debounce (solo clips existentes, como ClipEditorPage).
  useEffect(() => {
    if (!isDirty || saving || isExporting) return;
    const timer = window.setTimeout(() => {
      setAutosaving(true);
      void persistSpecRef.current(currentSpec, true).then((ok) => {
        setAutosaving(false);
        if (ok) setAutosavedAt(new Date());
      });
    }, STUDIO_AUTOSAVE_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [currentSpec, isDirty, saving, isExporting]);

  // Aviso al cerrar o recargar la pestaña con cambios sin guardar.
  useEffect(() => {
    if (!isDirty) return;
    const onBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
    };
    window.addEventListener('beforeunload', onBeforeUnload);
    return () => window.removeEventListener('beforeunload', onBeforeUnload);
  }, [isDirty]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (document.activeElement?.tagName || '').toUpperCase();
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
        e.preventDefault();
        void saveEditRef.current();
        return;
      }
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(tag)) return;

      if (e.code === 'Space') {
        e.preventDefault();
        togglePlay();
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        seekTo(getCurrentPlayTime() - 0.5);
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        seekTo(getCurrentPlayTime() + 0.5);
      } else if (e.key.toLowerCase() === 'i') {
        handleSetIn();
      } else if (e.key.toLowerCase() === 'o') {
        handleSetOut();
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [getCurrentPlayTime, handleSetIn, handleSetOut, seekTo, togglePlay]);

  const handleVideoLoadedMetadata = () => {
    const v = videoRef.current;
    if (!v) return;
    const dur = v.duration || 0;
    if (Number.isFinite(dur) && dur > 0) {
      const roundedDur = Number(dur.toFixed(1));
      setDuration(dur);
      durationRef.current = dur;
      setSettings((prev) => {
        // Clip manual sin edición guardada: el recorte abarca el video
        // completo; en el resto solo se recorta el fin a la duración real.
        const base = isManualRef.current && !savedSpecRef.current
          ? { ...prev, trimEnd: roundedDur }
          : prev;
        const trimEnd = Math.min(base.trimEnd, dur);
        const trimStart = Math.min(base.trimStart, Math.max(0, trimEnd - 0.3));
        return { ...base, trimStart, trimEnd };
      });
    }
    setSourceState('ready');
    setSourceError('');
    const start = settingsRef.current.trimStart;
    v.currentTime = start;
    setCurrentTime(start);
    drawCurrentFrame(start);
  };

  const handleVideoError = () => {
    setSourceState('error');
    setSourceError('El servidor no pudo servir el video original de este proyecto.');
  };

  const retrySource = () => {
    setSourceState('loading');
    setSourceError('');
    // Recargar sobre el mismo elemento para no romper la conexión del
    // grafo de audio (createMediaElementSource solo se permite una vez).
    videoRef.current?.load();
  };

  const handleBgmChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !bgmRef.current) return;
    if (bgmUrlRef.current) URL.revokeObjectURL(bgmUrlRef.current);
    const url = URL.createObjectURL(file);
    bgmUrlRef.current = url;
    bgmRef.current.src = url;
    bgmRef.current.load();
    setHasBgm(true);
    setBgmName(file.name);
    hasBgmRef.current = true;
    applyAudioVolumes(settings.volVideo, settings.volBgm, true);
    if (isPlayingRef.current) bgmRef.current.play().catch(() => {});
    showToast('Música de fondo añadida (solo para esta sesión)');
    e.target.value = '';
  };

  const handleRemoveBgm = () => {
    if (bgmRef.current) {
      bgmRef.current.pause();
      bgmRef.current.removeAttribute('src');
    }
    if (bgmUrlRef.current) {
      URL.revokeObjectURL(bgmUrlRef.current);
      bgmUrlRef.current = null;
    }
    setHasBgm(false);
    setBgmName('');
    hasBgmRef.current = false;
    applyAudioVolumes(settings.volVideo, settings.volBgm, false);
  };

  const handleSrtChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const content = typeof ev.target?.result === 'string' ? ev.target.result : '';
      const parsed = parseSRT(content).map((cue) => ({ ...cue, id: nextCueId() }));
      if (parsed.length > 0) {
        setSubtitles(parsed);
        setSelectedSubId(parsed[0].id);
        showToast(`${parsed.length} subtítulos importados`);
      } else {
        showToast('No se detectaron subtítulos válidos en el .SRT');
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  const handleExportSrt = () => {
    if (subtitles.length === 0) return;
    const srtContent = exportSubtitlesToSRT(subtitles);
    const blob = new Blob([srtContent], { type: 'text/plain;charset=utf-8' });
    triggerDownload(blob, `kellyClipAi_${clipId}_${Date.now()}.srt`);
    showToast('Subtítulos descargados en .SRT');
  };

  const handleAddSub = () => {
    const cur = getCurrentPlayTime();
    const start = Number(cur.toFixed(1));
    const end = Number(Math.min(duration, Math.max(start + 0.5, start + 2.0)).toFixed(1));
    const newCue: SubtitleCue = {
      id: nextCueId(),
      start,
      end,
      text: 'NUEVO TEXTO *VIRAL*',
    };
    setSubtitles((prev) => [...prev, newCue].sort((a, b) => a.start - b.start));
    setSelectedSubId(newCue.id);
    showToast('Bloque de subtítulo añadido');
  };

  const handleUpdateSub = (id: number, patch: Partial<SubtitleCue>) => {
    setSubtitles((prev) =>
      prev
        .map((s) => (s.id === id ? { ...s, ...patch } : s))
        .sort((a, b) => a.start - b.start)
    );
  };

  const handleMoveSub = (id: number, start: number, end: number) => {
    setSubtitles((prev) =>
      prev.map((s) => (s.id === id ? { ...s, start, end } : s))
    );
  };

  const handleSplitSub = (id: number) => {
    const cue = subtitles.find((s) => s.id === id);
    if (!cue || cue.end - cue.start < 0.6) return;

    const midTime = Number(((cue.start + cue.end) / 2).toFixed(2));
    const words = cue.text.trim().split(/\s+/);
    const splitAt = Math.max(1, Math.ceil(words.length / 2));
    const firstText = words.slice(0, splitAt).join(' ') || cue.text;
    const secondText = words.slice(splitAt).join(' ') || cue.text;

    const secondCue: SubtitleCue = {
      id: nextCueId(),
      start: Number((midTime + 0.02).toFixed(2)),
      end: cue.end,
      text: secondText,
    };

    setSubtitles((prev) =>
      prev
        .map((s) => (s.id === id ? { ...s, end: midTime, text: firstText } : s))
        .concat(secondCue)
        .sort((a, b) => a.start - b.start)
    );
    showToast('Bloque dividido en 2');
  };

  const handleMergeNextSub = (id: number) => {
    const sorted = [...subtitles].sort((a, b) => a.start - b.start);
    const idx = sorted.findIndex((s) => s.id === id);
    if (idx < 0 || idx >= sorted.length - 1) return;

    const current = sorted[idx];
    const next = sorted[idx + 1];
    const merged: SubtitleCue = {
      id: current.id,
      start: current.start,
      end: Math.max(current.end, next.end),
      text: `${current.text.trim()} ${next.text.trim()}`.trim(),
    };

    setSubtitles(
      sorted.filter((s) => s.id !== next.id).map((s) => (s.id === current.id ? merged : s))
    );
    showToast('Bloques unidos');
  };

  const handleDeleteSub = (id: number) => {
    setSubtitles((prev) => prev.filter((s) => s.id !== id));
    setSelectedSubId((prev) => (prev === id ? null : prev));
  };

  const handleGenerateScript = (scriptText: string, wordsPerBlock: number) => {
    const generated = splitScriptToSubtitles(
      scriptText,
      wordsPerBlock,
      settings.trimStart,
      settings.trimEnd
    ).map((cue) => ({ ...cue, id: nextCueId() }));
    if (generated.length > 0) {
      setSubtitles(generated);
      setSelectedSubId(generated[0].id);
      setScriptModalOpen(false);
      showToast(`${generated.length} bloques generados`);
    }
  };

  const cleanupAfterExport = useCallback(() => {
    if (exportWatchdogRef.current) {
      window.clearInterval(exportWatchdogRef.current);
      exportWatchdogRef.current = null;
    }
    setIsExporting(false);
    isExportingRef.current = false;
    if (speakerGainRef.current && audioCtxRef.current) {
      speakerGainRef.current.gain.setValueAtTime(1, audioCtxRef.current.currentTime);
    }
    drawCurrentFrame();
  }, [drawCurrentFrame]);

  const finishExport = useCallback(() => {
    if (exportWatchdogRef.current) {
      window.clearInterval(exportWatchdogRef.current);
      exportWatchdogRef.current = null;
    }
    if (videoRef.current) videoRef.current.pause();
    if (hasBgmRef.current && bgmRef.current) bgmRef.current.pause();

    const mr = mediaRecorderRef.current;
    if (mr && mr.state !== 'inactive') {
      mr.stop();
    } else {
      cleanupAfterExport();
    }
  }, [cleanupAfterExport]);

  const cancelExport = useCallback(() => {
    exportCanceledRef.current = true;
    uploadControllerRef.current?.abort();
    finishExport();
  }, [finishExport]);

  const startExport = () => {
    if (!sourceReadyRef.current) {
      showToast('El video original todavía se está cargando');
      return;
    }
    if (isExporting) return;

    const mimeType = getSupportedMimeType();
    if (!mimeType || !canvasRef.current) {
      showToast('Tu navegador no soporta MediaRecorder. Usa Chrome o Edge.');
      return;
    }

    togglePlay(false);
    initAudio();

    // La especificación que se guarda es la de este instante, la misma
    // que se renderiza; si el usuario sigue editando durante el render,
    // esos cambios no se mezclan con la exportación.
    const spec: StudioEditSpec = {
      version: 1,
      settings: settingsRef.current,
      cues: subtitlesRef.current,
    };

    exportCanceledRef.current = false;
    setIsExporting(true);
    isExportingRef.current = true;
    setExportProgress(0);
    setExportStatus('Renderizando en 1080×1920 (30fps)... No minimices esta pestaña.');
    setBannerError('');

    if (speakerGainRef.current && audioCtxRef.current) {
      speakerGainRef.current.gain.setValueAtTime(0, audioCtxRef.current.currentTime);
    }

    const canvasStream = canvasRef.current.captureStream(30);
    const audioTracks = audioDestRef.current ? audioDestRef.current.stream.getAudioTracks() : [];
    const combinedStream = new MediaStream([...canvasStream.getVideoTracks(), ...audioTracks]);

    const chunks: BlobPart[] = [];
    const recorder = new MediaRecorder(combinedStream, {
      mimeType,
      videoBitsPerSecond: 10_000_000,
    });
    mediaRecorderRef.current = recorder;

    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunks.push(e.data);
    };

    recorder.onstop = () => {
      if (exportCanceledRef.current) {
        showToast('Exportación cancelada');
        cleanupAfterExport();
        return;
      }
      if (chunks.length === 0) {
        setBannerError('La exportación no produjo video. Inténtalo de nuevo sin minimizar la pestaña.');
        cleanupAfterExport();
        return;
      }
      const ext = mimeType.includes('mp4') ? 'mp4' : 'webm';
      const blob = new Blob(chunks, { type: mimeType });
      setExportProgress(100);
      setExportStatus('Subiendo el clip exportado al proyecto…');
      const controller = new AbortController();
      uploadControllerRef.current = controller;
      void exportStudioVideo(
        projectId,
        blob,
        `studio-${isManual ? 'nuevo' : clipId}-${Date.now()}.${ext}`,
        {
          clipId: isManual ? undefined : clipId,
          start: spec.settings.trimStart,
          end: spec.settings.trimEnd,
          title: title.trim() || undefined,
          studioEdit: spec,
        },
        controller.signal,
      )
        .then(async () => {
          uploadControllerRef.current = null;
          setSavedSpecJson(JSON.stringify(spec));
          cleanupAfterExport();
          await onProjectUpdated(projectId);
          onBack();
        })
        .catch((uploadError: unknown) => {
          uploadControllerRef.current = null;
          if (exportCanceledRef.current || controller.signal.aborted) {
            showToast('Exportación cancelada');
            cleanupAfterExport();
            return;
          }
          // Respaldo: si la subida falla, el video se descarga en el
          // navegador para no perder el render.
          triggerDownload(blob, `kellyClipAi-studio-${Date.now()}.${ext}`);
          setBannerError(uploadError instanceof Error
            ? `No se pudo guardar el clip en el proyecto: ${uploadError.message} Se descargó el video en el navegador como respaldo.`
            : 'No se pudo guardar el clip en el proyecto. Se descargó el video en el navegador como respaldo.');
          cleanupAfterExport();
        });
    };

    const s = settingsRef.current;
    const segDur = Math.max(0.1, s.trimEnd - s.trimStart);

    if (videoRef.current) {
      videoRef.current.currentTime = s.trimStart;
      videoRef.current.playbackRate = s.speed;
    }
    if (hasBgmRef.current && bgmRef.current && bgmRef.current.duration) {
      bgmRef.current.currentTime = 0;
    }

    const beginRecording = () => {
      if (exportCanceledRef.current) return;
      drawCurrentFrame();
      recorder.start(250);

      if (videoRef.current) videoRef.current.play().catch(() => {});
      if (hasBgmRef.current && bgmRef.current) bgmRef.current.play().catch(() => {});

      exportWatchdogRef.current = window.setInterval(() => {
        if (!isExportingRef.current || exportCanceledRef.current) return;

        const cur = getCurrentPlayTime();
        const elapsed = Math.max(0, cur - s.trimStart);
        const pct = Math.min(100, Math.max(0, (elapsed / segDur) * 100));
        setExportProgress(pct);
        setExportStatus(
          `Renderizando 1080×1920 · ${pct.toFixed(0)}% (${formatTime(elapsed)} / ${formatTime(segDur)})`
        );

        if (cur >= s.trimEnd || videoRef.current?.ended) {
          setExportProgress(100);
          finishExport();
        }
      }, 50);
    };

    if (videoRef.current) {
      const v = videoRef.current;
      let started = false;
      const safeStart = () => {
        if (started) return;
        started = true;
        v.removeEventListener('seeked', safeStart);
        beginRecording();
      };
      if (!v.seeking && Math.abs(v.currentTime - s.trimStart) < 0.05 && v.readyState >= 2) {
        safeStart();
      } else {
        v.addEventListener('seeked', safeStart);
        window.setTimeout(safeStart, 450);
      }
    }
  };

  // El proyecto no cargó (error del contexto) o el clip no existe.
  if ((error && !project) || loadError) {
    return (
      <main className="studio-page studio-fallback" id="main-content">
        <section className="studio-fallback-card">
          <h1>Editor visual</h1>
          <p role="alert"><AlertCircle size={16} /> {loadError || error}</p>
          <button type="button" className="create-button" onClick={onBack}>
            <ArrowLeft size={16} /> Volver a resultados
          </button>
        </section>
      </main>
    );
  }

  if (!project) {
    return (
      <main className="studio-page studio-fallback" id="main-content">
        <p className="results-loading"><LoaderCircle size={17} className="spin" /> Cargando el proyecto…</p>
      </main>
    );
  }

  return (
    <div className="studio-page">
      <div className="studio-root">
        {/* El video fuente se sirve con la cookie de medios; el atributo
          crossOrigin con credenciales mantiene el canvas sin contaminar
          para poder capturarlo con MediaRecorder (en desarrollo también
          funciona porque el backend permite CORS con credenciales). */}
        <video
          ref={videoRef}
          src={`${API_BASE}/api/projects/${projectId}/source`}
          playsInline
          crossOrigin="use-credentials"
          onLoadedMetadata={handleVideoLoadedMetadata}
          onError={handleVideoError}
          onSeeked={() => {
            if (!isPlayingRef.current) drawCurrentFrame();
          }}
          className="hidden"
        />
        <audio ref={bgmRef} playsInline crossOrigin="anonymous" loop className="hidden" />

        <input
          ref={bgmInputRef}
          type="file"
          accept="audio/*"
          className="hidden"
          onChange={handleBgmChange}
        />
        <input
          ref={srtInputRef}
          type="file"
          accept=".srt,.txt"
          className="hidden"
          onChange={handleSrtChange}
        />

        <StudioHeader
          title={title}
          onTitleChange={setTitle}
          saveState={saveState}
          savedAt={autosavedAt}
          onBack={onBack}
          onSave={() => void saveEdit()}
          onUploadBgm={() => bgmInputRef.current?.click()}
          onUploadSrt={() => srtInputRef.current?.click()}
          onExport={startExport}
          isExporting={isExporting}
          sourceReady={sourceState === 'ready'}
        />

        {bannerError && (
          <div className="studio-error-banner" role="alert">
            <AlertCircle size={15} />
            <span>{bannerError}</span>
            <button type="button" onClick={() => setBannerError('')} aria-label="Cerrar aviso">
              <X size={15} />
            </button>
          </div>
        )}

        <div className="flex-1 min-h-0 flex">
          <LeftSidebar
            activeTab={activeTab}
            onTabChange={setActiveTab}
            settings={settings}
            duration={duration}
            hasBgm={hasBgm}
            bgmName={bgmName}
            onUpdateSetting={updateSetting}
            onApplySubtitlePreset={applySubtitlePreset}
            onTrimStartChange={handleTrimStartChange}
            onTrimEndChange={handleTrimEndChange}
            onResetFraming={() =>
              setSettings((prev) => ({ ...prev, zoom: 100, panX: 0, panY: 0 }))
            }
            onResetColor={() =>
              setSettings((prev) => ({ ...prev, bright: 100, contrast: 100, sat: 100 }))
            }
            onRemoveBgm={handleRemoveBgm}
          />

          <PreviewStage
            canvasRef={canvasRef}
            safeZone={safeZone}
            onSafeZoneChange={setSafeZone}
            sourceState={sourceState}
            sourceError={sourceError}
            onRetrySource={retrySource}
            isPlaying={isPlaying}
            loop={loop}
            currentTime={currentTime}
            duration={duration}
            panX={settings.panX}
            panY={settings.panY}
            onPanChange={(panX, panY) => setSettings((prev) => ({ ...prev, panX, panY }))}
            onTogglePlay={() => togglePlay()}
            onToggleLoop={() => setLoop((l) => !l)}
            onSetIn={handleSetIn}
            onSetOut={handleSetOut}
          />

          <RightSidebar
            subtitles={subtitles}
            selectedSubId={selectedSubId}
            currentTime={currentTime}
            duration={duration}
            onSelectSub={setSelectedSubId}
            onSeekTo={seekTo}
            onAddSub={handleAddSub}
            onOpenScriptModal={() => setScriptModalOpen(true)}
            onExportSrt={handleExportSrt}
            onUpdateSub={handleUpdateSub}
            onSplitSub={handleSplitSub}
            onMergeNextSub={handleMergeNextSub}
            onDeleteSub={handleDeleteSub}
          />
        </div>

        <Timeline
          duration={duration}
          currentTime={currentTime}
          trimStart={settings.trimStart}
          trimEnd={settings.trimEnd}
          speed={settings.speed}
          hasBgm={hasBgm}
          bgmName={bgmName}
          subtitles={subtitles}
          selectedSubId={selectedSubId}
          onSeek={seekTo}
          onSelectSub={setSelectedSubId}
          onMoveSub={handleMoveSub}
        />

        <ExportModal
          open={isExporting}
          progress={exportProgress}
          statusText={exportStatus}
          onCancel={cancelExport}
        />

        <ScriptModal
          open={scriptModalOpen}
          onClose={() => setScriptModalOpen(false)}
          onGenerate={handleGenerateScript}
        />

        {toastMsg && (
          <div className="fixed bottom-44 left-1/2 -translate-x-1/2 bg-[#1a1e28] border border-slate-700 text-slate-100 px-4 py-2 rounded-lg text-xs font-medium shadow-xl z-50 pointer-events-none">
            {toastMsg}
          </div>
        )}
      </div>
    </div>
  );
}
