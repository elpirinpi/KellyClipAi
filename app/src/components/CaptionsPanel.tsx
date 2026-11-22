import { useState } from 'react';
import { Check, Copy, LoaderCircle, Sparkles, X } from 'lucide-react';
import { apiFetch } from '../auth';
import { API_BASE } from '../context/constants';
import type { CaptionVariant, ClipResult } from '../types';
import { responseError } from '../utils';

const PLATFORMS = ['youtube', 'tiktok', 'instagram'] as const;
const PLATFORM_LABELS: Record<string, string> = {
  youtube: 'YouTube',
  tiktok: 'TikTok',
  instagram: 'Instagram',
};

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

/** Generador de títulos/descripciones/hashtags (Fase 3, ítem 12): 3
 * variantes de título, descripción y hashtags por clip, con copia y
 * edición antes de publicar. Requiere LLM: sin clave, el backend
 * responde 409 y aquí se explica. */
export function CaptionsPanel({ projectId, clip, onClose, onApplied, onNotice }: {
  projectId: string;
  clip: ClipResult;
  onClose: () => void;
  onApplied: () => void;
  onNotice: (message: string, kind?: 'info' | 'success' | 'error') => void;
}) {
  const [variant, setVariant] = useState<CaptionVariant | null>(null);
  const [generating, setGenerating] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState('');
  const [title, setTitle] = useState(clip.title);
  const [description, setDescription] = useState(clip.description ?? '');
  const [hashtags, setHashtags] = useState<Record<string, string>>(() => {
    const initial: Record<string, string> = {};
    for (const platform of PLATFORMS) {
      const raw = clip.hashtags;
      const list = Array.isArray(raw)
        ? raw
        : (raw as Record<string, string[]> | undefined)?.[platform] ?? [];
      initial[platform] = list.map((tag) => `#${String(tag).replace(/^#/, '')}`).join(' ');
    }
    return initial;
  });

  const generate = async () => {
    setError('');
    setGenerating(true);
    try {
      const response = await apiFetch(`${API_BASE}/api/projects/${projectId}/generate-captions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ clip_ids: [clip.id] }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      const result = await response.json() as { variants: Record<string, CaptionVariant> };
      const found = result.variants[clip.id];
      if (!found) throw new Error('El backend no devolvió variantes para este clip.');
      setVariant(found);
      setTitle(found.titles[0] ?? clip.title);
      setDescription(found.description);
      const next: Record<string, string> = {};
      for (const platform of PLATFORMS) {
        next[platform] = (found.hashtags[platform] ?? []).map((tag) => `#${tag}`).join(' ');
      }
      setHashtags(next);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'No se pudieron generar los títulos.');
    } finally {
      setGenerating(false);
    }
  };

  const apply = async () => {
    setError('');
    setSaving(true);
    try {
      const parsed: Record<string, string[]> = {};
      for (const platform of PLATFORMS) {
        parsed[platform] = hashtags[platform]
          .split(/[\s,]+/)
          .map((tag) => tag.trim().replace(/^#+/, ''))
          .filter((tag) => /^[A-Za-z0-9_áéíóúñÁÉÍÓÚÑ]+$/.test(tag));
      }
      const response = await apiFetch(`${API_BASE}/api/projects/${projectId}/clips/${clip.id}/caption`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: title.trim(), description: description.trim(), hashtags: parsed }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      onNotice('Título, descripción y hashtags guardados en el clip.', 'success');
      onApplied();
      onClose();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'No se pudo guardar la edición.');
    } finally {
      setSaving(false);
    }
  };

  const copyField = async (key: string, text: string) => {
    if (await copyText(text)) {
      setCopied(key);
      window.setTimeout(() => setCopied(''), 1500);
    } else {
      onNotice('No se pudo copiar al portapapeles.', 'error');
    }
  };

  return (
    <div className="pf-modal-overlay" role="dialog" aria-modal="true" aria-label={`Títulos y hashtags de ${clip.title}`}>
      <section className="pf-modal pf-modal-wide">
        <header className="pf-modal-heading">
          <div>
            <h2>Títulos, descripción y hashtags</h2>
            <p>{clip.title}</p>
          </div>
          <button type="button" className="pf-modal-close" onClick={onClose} aria-label="Cerrar"><X size={18} /></button>
        </header>
        {error && <p className="error-message" role="alert">{error}</p>}
        {!variant && (
          <div className="pf-captions-intro">
            <p>La IA propone 3 variantes de título, una descripción y hashtags por plataforma a partir de lo que ya sabe del clip (título, razón viral, palabras clave y emojis). Luego eliges y editas antes de publicar.</p>
            <button type="button" className="create-button" onClick={() => void generate()} disabled={generating}>
              {generating ? <LoaderCircle size={15} className="spin" /> : <Sparkles size={15} />}
              {generating ? 'Generando…' : 'Generar con IA'}
            </button>
          </div>
        )}
        {variant && (
          <>
            <div className="pf-variant-titles">
              <span>Variantes de título</span>
              {variant.titles.map((option, index) => (
                <button
                  type="button"
                  key={`${option}-${index}`}
                  className={`pf-variant-title ${title === option ? 'selected' : ''}`}
                  onClick={() => setTitle(option)}
                >
                  {option}
                </button>
              ))}
            </div>
            <label className="config-field pf-caption-field"><span>Título elegido</span>
              <span className="pf-field-with-copy">
                <input className="text-input" value={title} maxLength={100} onChange={(event) => setTitle(event.target.value)} />
                <button type="button" className="pf-copy-button" aria-label="Copiar título" onClick={() => void copyField('title', title)}>
                  {copied === 'title' ? <Check size={15} /> : <Copy size={15} />}
                </button>
              </span>
            </label>
            <label className="config-field pf-caption-field"><span>Descripción</span>
              <span className="pf-field-with-copy">
                <textarea className="text-input pf-textarea" value={description} rows={3} maxLength={4000} onChange={(event) => setDescription(event.target.value)} />
                <button type="button" className="pf-copy-button" aria-label="Copiar descripción" onClick={() => void copyField('description', description)}>
                  {copied === 'description' ? <Check size={15} /> : <Copy size={15} />}
                </button>
              </span>
            </label>
            {PLATFORMS.map((platform) => (
              <label className="config-field pf-caption-field" key={platform}><span>Hashtags · {PLATFORM_LABELS[platform]}</span>
                <span className="pf-field-with-copy">
                  <input className="text-input" value={hashtags[platform]} onChange={(event) => setHashtags((current) => ({ ...current, [platform]: event.target.value }))} placeholder="#ejemplo #clip" />
                  <button type="button" className="pf-copy-button" aria-label={`Copiar hashtags de ${PLATFORM_LABELS[platform]}`} onClick={() => void copyField(platform, hashtags[platform])}>
                    {copied === platform ? <Check size={15} /> : <Copy size={15} />}
                  </button>
                </span>
              </label>
            ))}
            <div className="pf-modal-actions">
              <button type="button" className="subtle-button" onClick={() => void generate()} disabled={generating}>
                {generating ? <LoaderCircle size={15} className="spin" /> : <Sparkles size={15} />}
                Regenerar
              </button>
              <button type="button" className="create-button" onClick={() => void apply()} disabled={saving}>
                {saving ? <LoaderCircle size={15} className="spin" /> : null}
                {saving ? 'Guardando…' : 'Aplicar al clip'}
              </button>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
