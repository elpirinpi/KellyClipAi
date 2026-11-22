import { useCallback, useEffect, useMemo, useState } from 'react';
import { CalendarDays, ChevronLeft, ChevronRight, LoaderCircle, RotateCcw, Trash2, X } from 'lucide-react';
import { apiFetch } from '../auth';
import { API_BASE } from '../context/constants';
import type { BestTimesGuide, ScheduleJob } from '../types';
import { responseError } from '../utils';

const PLATFORM_COLORS: Record<string, string> = {
  youtube: '#c00',
  tiktok: '#111111',
  instagram: '#c13584',
};
const PLATFORM_LABELS: Record<string, string> = {
  youtube: 'YouTube',
  tiktok: 'TikTok',
  instagram: 'Instagram',
};
const WEEKDAYS = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'];

function dayKey(date: Date): string {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
}

function startOfWeek(date: Date): Date {
  const day = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  const offset = (day.getDay() + 6) % 7;
  day.setDate(day.getDate() - offset);
  return day;
}

function monthGridDays(anchor: Date): Date[] {
  const first = new Date(anchor.getFullYear(), anchor.getMonth(), 1);
  const start = startOfWeek(first);
  return Array.from({ length: 42 }, (_, index) => {
    const day = new Date(start);
    day.setDate(start.getDate() + index);
    return day;
  });
}

function weekDays(anchor: Date): Date[] {
  const start = startOfWeek(anchor);
  return Array.from({ length: 7 }, (_, index) => {
    const day = new Date(start);
    day.setDate(start.getDate() + index);
    return day;
  });
}

function formatHour(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/** Calendario de publicaciones (Fase 3, ítem 9): vista semanal/mensual
 * con drag & drop para reprogramar, colores por plataforma y guía de
 * mejor hora. */
export function CalendarPage({ onNotice }: {
  onNotice: (message: string, kind?: 'info' | 'success' | 'error') => void;
}) {
  const [jobs, setJobs] = useState<ScheduleJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [view, setView] = useState<'week' | 'month'>('week');
  const [anchor, setAnchor] = useState(() => new Date());
  const [guide, setGuide] = useState<BestTimesGuide | null>(null);
  const [guidePlatform, setGuidePlatform] = useState('youtube');
  const [selectedJob, setSelectedJob] = useState<ScheduleJob | null>(null);
  const [rescheduleValue, setRescheduleValue] = useState('');
  const [busy, setBusy] = useState(false);
  const [dragOverDay, setDragOverDay] = useState('');

  const refresh = useCallback(async () => {
    try {
      const response = await apiFetch(`${API_BASE}/api/scheduled-jobs`);
      if (!response.ok) throw new Error(await responseError(response));
      setJobs(await response.json() as ScheduleJob[]);
      setError('');
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'No se pudieron leer las publicaciones programadas.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  useEffect(() => {
    let active = true;
    apiFetch(`${API_BASE}/api/publish/best-times`)
      .then(async (response) => {
        if (!response.ok) throw new Error(await responseError(response));
        return response.json() as Promise<BestTimesGuide>;
      })
      .then((result) => { if (active) setGuide(result); })
      .catch(() => undefined);
    return () => { active = false; };
  }, []);

  const days = useMemo(
    () => (view === 'week' ? weekDays(anchor) : monthGridDays(anchor)),
    [view, anchor],
  );
  const jobsByDay = useMemo(() => {
    const grouped = new Map<string, ScheduleJob[]>();
    for (const job of jobs) {
      const date = new Date(job.scheduled_at);
      if (Number.isNaN(date.getTime())) continue;
      const key = dayKey(date);
      grouped.set(key, [...(grouped.get(key) ?? []), job]);
    }
    return grouped;
  }, [jobs]);

  const shift = (direction: number) => {
    setAnchor((current) => {
      const next = new Date(current);
      if (view === 'week') next.setDate(next.getDate() + 7 * direction);
      else next.setMonth(next.getMonth() + direction);
      return next;
    });
  };

  const moveJob = async (job: ScheduleJob, targetDay: Date) => {
    const original = new Date(job.scheduled_at);
    const moved = new Date(
      targetDay.getFullYear(),
      targetDay.getMonth(),
      targetDay.getDate(),
      original.getHours(),
      original.getMinutes(),
    );
    setBusy(true);
    try {
      const response = await apiFetch(`${API_BASE}/api/scheduled-jobs/${job.id}/reschedule`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scheduled_at: moved.toISOString() }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      onNotice('Publicación reprogramada.', 'success');
      await refresh();
    } catch (caught) {
      onNotice(caught instanceof Error ? caught.message : 'No se pudo reprogramar la publicación.', 'error');
    } finally {
      setBusy(false);
    }
  };

  const retryJob = async (job: ScheduleJob) => {
    setBusy(true);
    try {
      const response = await apiFetch(`${API_BASE}/api/scheduled-jobs/${job.id}/retry`, { method: 'POST' });
      if (!response.ok) throw new Error(await responseError(response));
      onNotice('La publicación fallida vuelve a la cola.', 'success');
      setSelectedJob(null);
      await refresh();
    } catch (caught) {
      onNotice(caught instanceof Error ? caught.message : 'No se pudo reintentar la publicación.', 'error');
    } finally {
      setBusy(false);
    }
  };

  const cancelJob = async (job: ScheduleJob) => {
    setBusy(true);
    try {
      const response = await apiFetch(`${API_BASE}/api/scheduled-jobs/${job.id}`, { method: 'DELETE' });
      if (!response.ok) throw new Error(await responseError(response));
      onNotice('Publicación cancelada.', 'success');
      setSelectedJob(null);
      await refresh();
    } catch (caught) {
      onNotice(caught instanceof Error ? caught.message : 'No se pudo cancelar la publicación.', 'error');
    } finally {
      setBusy(false);
    }
  };

  const rescheduleSelected = async () => {
    if (!selectedJob || !rescheduleValue) return;
    const date = new Date(rescheduleValue);
    if (Number.isNaN(date.getTime())) {
      onNotice('La fecha elegida no es válida.', 'error');
      return;
    }
    setBusy(true);
    try {
      const response = await apiFetch(`${API_BASE}/api/scheduled-jobs/${selectedJob.id}/reschedule`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scheduled_at: date.toISOString() }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      onNotice('Publicación reprogramada.', 'success');
      setSelectedJob(null);
      await refresh();
    } catch (caught) {
      onNotice(caught instanceof Error ? caught.message : 'No se pudo reprogramar la publicación.', 'error');
    } finally {
      setBusy(false);
    }
  };

  const title = view === 'week'
    ? `Semana del ${days[0].toLocaleDateString([], { day: 'numeric', month: 'long' })}`
    : anchor.toLocaleDateString([], { month: 'long', year: 'numeric' });
  const guideSlots = guide?.next_slots?.[guidePlatform] ?? [];

  return (
    <main className="pf-page" id="main-content">
      <header className="pf-page-heading">
        <div>
          <span className="eyebrow">PUBLICACIÓN</span>
          <h1><CalendarDays size={22} /> Calendario de publicaciones</h1>
          <p>Arrastra un clip a otro día para reprogramarlo. Los colores distinguen la plataforma.</p>
        </div>
        <div className="pf-view-switch" role="group" aria-label="Vista del calendario">
          <button type="button" className={view === 'week' ? 'selected' : ''} onClick={() => setView('week')}>Semana</button>
          <button type="button" className={view === 'month' ? 'selected' : ''} onClick={() => setView('month')}>Mes</button>
        </div>
      </header>
      {error && <p className="error-message" role="alert">{error}</p>}
      <div className="pf-calendar-nav">
        <button type="button" className="subtle-button" onClick={() => shift(-1)} aria-label="Anterior"><ChevronLeft size={16} /></button>
        <strong>{title}</strong>
        <button type="button" className="subtle-button" onClick={() => setAnchor(new Date())}>Hoy</button>
        <button type="button" className="subtle-button" onClick={() => shift(1)} aria-label="Siguiente"><ChevronRight size={16} /></button>
      </div>
      {loading && <p className="results-loading"><LoaderCircle size={17} className="spin" /> Cargando publicaciones…</p>}
      {!loading && (
        <div className={`pf-calendar-grid ${view === 'month' ? 'month' : ''}`}>
          {WEEKDAYS.map((day) => <div className="pf-calendar-weekday" key={day}>{day}</div>)}
          {days.map((day) => {
            const key = dayKey(day);
            const dayJobs = jobsByDay.get(key) ?? [];
            const isToday = key === dayKey(new Date());
            const outsideMonth = view === 'month' && day.getMonth() !== anchor.getMonth();
            return (
              <div
                key={key}
                className={`pf-calendar-day ${isToday ? 'today' : ''} ${outsideMonth ? 'outside' : ''} ${dragOverDay === key ? 'drop-target' : ''}`}
                onDragOver={(event) => { event.preventDefault(); setDragOverDay(key); }}
                onDragLeave={() => setDragOverDay((current) => (current === key ? '' : current))}
                onDrop={(event) => {
                  event.preventDefault();
                  setDragOverDay('');
                  const jobId = event.dataTransfer.getData('text/schedule-job');
                  const job = jobs.find((item) => item.id === jobId);
                  if (job && job.status === 'scheduled' && dayKey(new Date(job.scheduled_at)) !== key) {
                    void moveJob(job, day);
                  }
                }}
              >
                <span className="pf-calendar-day-number">{day.getDate()}</span>
                {dayJobs.map((job) => (
                  <button
                    type="button"
                    key={job.id}
                    className={`pf-calendar-job ${job.status}`}
                    style={{ '--pf-platform-color': PLATFORM_COLORS[job.platform] ?? '#666' } as React.CSSProperties}
                    draggable={job.status === 'scheduled' && !busy}
                    onDragStart={(event) => event.dataTransfer.setData('text/schedule-job', job.id)}
                    onClick={() => {
                      setSelectedJob(job);
                      const date = new Date(job.scheduled_at);
                      const local = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
                      setRescheduleValue(local.toISOString().slice(0, 16));
                    }}
                    title={`${PLATFORM_LABELS[job.platform] ?? job.platform} · ${job.status}${job.error ? ` · ${job.error}` : ''}`}
                  >
                    <span className="pf-calendar-job-time">{formatHour(job.scheduled_at)}</span>
                    <span>{PLATFORM_LABELS[job.platform] ?? job.platform}</span>
                    {job.status === 'failed' && <span className="pf-calendar-job-badge">falló</span>}
                    {(job.attempts ?? 0) > 0 && job.status === 'scheduled' && (
                      <span className="pf-calendar-job-badge">reintento {job.attempts}/3</span>
                    )}
                  </button>
                ))}
              </div>
            );
          })}
        </div>
      )}
      <section className="config-card pf-best-times">
        <div className="card-heading"><div><h2>Mejor hora por plataforma</h2><p>{guide?.note ?? 'Guía orientativa basada en las franjas de mayor actividad general de cada plataforma.'}</p></div></div>
        <label className="config-field"><span>Plataforma</span>
          <span className="select-wrap"><select value={guidePlatform} onChange={(event) => setGuidePlatform(event.target.value)}>
            {Object.keys(PLATFORM_LABELS).map((platform) => <option key={platform} value={platform}>{PLATFORM_LABELS[platform]}</option>)}
          </select></span>
        </label>
        {guideSlots.length > 0 ? (
          <ul className="pf-slot-list">
            {guideSlots.map((slot) => (
              <li key={slot}>{new Date(slot).toLocaleString([], { weekday: 'long', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })}</li>
            ))}
          </ul>
        ) : (
          <p className="pf-empty">Cargando sugerencias…</p>
        )}
      </section>
      {selectedJob && (
        <div className="pf-modal-overlay" role="dialog" aria-modal="true" aria-label="Detalle de la publicación">
          <section className="pf-modal">
            <header className="pf-modal-heading">
              <div>
                <h2>{PLATFORM_LABELS[selectedJob.platform] ?? selectedJob.platform} · {selectedJob.status}</h2>
                <p>{new Date(selectedJob.scheduled_at).toLocaleString()} · {selectedJob.visibility}</p>
              </div>
              <button type="button" className="pf-modal-close" onClick={() => setSelectedJob(null)} aria-label="Cerrar"><X size={18} /></button>
            </header>
            {selectedJob.error && <p className="error-message" role="alert">{selectedJob.error}</p>}
            {(selectedJob.attempts ?? 0) > 0 && (
              <p>Intentos: {selectedJob.attempts} de 3{selectedJob.next_attempt_at ? ` · siguiente reintento: ${new Date(selectedJob.next_attempt_at).toLocaleString()}` : ''}</p>
            )}
            {selectedJob.external_url && <p><a href={selectedJob.external_url} target="_blank" rel="noreferrer">Ver publicación</a></p>}
            {selectedJob.status === 'scheduled' && (
              <>
                <label className="config-field"><span>Reprogramar a</span>
                  <input className="text-input" type="datetime-local" value={rescheduleValue} onChange={(event) => setRescheduleValue(event.target.value)} />
                </label>
                <div className="pf-modal-actions">
                  <button type="button" className="subtle-button" onClick={() => void rescheduleSelected()} disabled={busy}>Reprogramar</button>
                  <button type="button" className="subtle-button pf-danger" onClick={() => void cancelJob(selectedJob)} disabled={busy}>
                    <Trash2 size={15} /> Cancelar
                  </button>
                </div>
              </>
            )}
            {selectedJob.status === 'failed' && (
              <div className="pf-modal-actions">
                <button type="button" className="create-button" onClick={() => void retryJob(selectedJob)} disabled={busy}>
                  <RotateCcw size={15} /> Reintentar ahora
                </button>
              </div>
            )}
          </section>
        </div>
      )}
    </main>
  );
}
