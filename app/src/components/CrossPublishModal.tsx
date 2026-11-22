import { useEffect, useState } from 'react';
import { LoaderCircle, Plus, X } from 'lucide-react';
import { apiFetch } from '../auth';
import { API_BASE } from '../context/constants';
import type { PublishAccount } from '../types';
import { responseError } from '../utils';

type TargetDraft = {
  key: number;
  platform: string;
  accountId: string;
  scheduledAt: string;
  visibility: string;
};

const PLATFORMS = ['youtube', 'tiktok', 'instagram'] as const;
const PLATFORM_LABELS: Record<string, string> = {
  youtube: 'YouTube',
  tiktok: 'TikTok',
  instagram: 'Instagram',
};

function emptyTarget(key: number, platform = 'youtube'): TargetDraft {
  return {
    key,
    platform,
    accountId: '',
    scheduledAt: '',
    visibility: platform === 'instagram' ? 'public' : 'private',
  };
}

function toIsoWithZone(localValue: string): string | null {
  if (!localValue) return null;
  const date = new Date(localValue);
  if (Number.isNaN(date.getTime())) return null;
  return date.toISOString();
}

/** Publicación cruzada (Fase 3, ítem 10): los mismos clips a varias
 * plataformas y cuentas a la vez, cada destino con su cuenta y su hora. */
export function CrossPublishModal({ projectId, clipIds, onClose, onDone, onNotice }: {
  projectId: string;
  clipIds: string[];
  onClose: () => void;
  onDone: (groupId: string) => void;
  onNotice: (message: string, kind?: 'info' | 'success' | 'error') => void;
}) {
  const [accounts, setAccounts] = useState<Record<string, PublishAccount[]>>({});
  const [loadingAccounts, setLoadingAccounts] = useState(true);
  const [targets, setTargets] = useState<TargetDraft[]>([emptyTarget(0)]);
  const [nextKey, setNextKey] = useState(1);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;
    apiFetch(`${API_BASE}/api/integrations/accounts`)
      .then(async (response) => {
        if (!response.ok) throw new Error(await responseError(response));
        return response.json() as Promise<Record<string, PublishAccount[]>>;
      })
      .then((result) => { if (active) setAccounts(result); })
      .catch((caught: unknown) => {
        if (active) setError(caught instanceof Error ? caught.message : 'No se pudieron leer las cuentas conectadas.');
      })
      .finally(() => { if (active) setLoadingAccounts(false); });
    return () => { active = false; };
  }, []);

  const updateTarget = (key: number, patch: Partial<TargetDraft>) => {
    setTargets((current) => current.map((target) => {
      if (target.key !== key) return target;
      const next = { ...target, ...patch };
      if (patch.platform === 'instagram') next.visibility = 'public';
      if (patch.platform === 'tiktok' && next.visibility === 'unlisted') next.visibility = 'private';
      if (patch.platform) next.accountId = '';
      return next;
    }));
  };

  const submit = async () => {
    setError('');
    const payloadTargets = [];
    for (const target of targets) {
      const scheduledAt = toIsoWithZone(target.scheduledAt);
      if (target.scheduledAt && !scheduledAt) {
        setError('Revisa la fecha de uno de los destinos: no es válida.');
        return;
      }
      payloadTargets.push({
        platform: target.platform,
        account_id: target.accountId || null,
        scheduled_at: scheduledAt,
        visibility: target.visibility,
      });
    }
    setSubmitting(true);
    try {
      const response = await apiFetch(`${API_BASE}/api/projects/${projectId}/cross-publish`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ clip_ids: clipIds, targets: payloadTargets }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      const result = await response.json() as { group_id: string; jobs: unknown[] };
      onNotice(`Se programaron ${result.jobs.length} publicacion${result.jobs.length === 1 ? 'es' : 'es'}.`, 'success');
      onDone(result.group_id);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'No se pudo programar la publicación cruzada.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="pf-modal-overlay" role="dialog" aria-modal="true" aria-label="Publicar en varias plataformas">
      <section className="pf-modal pf-modal-wide">
        <header className="pf-modal-heading">
          <div>
            <h2>Publicar en varias plataformas</h2>
            <p>{clipIds.length} clip{clipIds.length === 1 ? '' : 's'} · un destino por plataforma y cuenta</p>
          </div>
          <button type="button" className="pf-modal-close" onClick={onClose} aria-label="Cerrar"><X size={18} /></button>
        </header>
        {loadingAccounts && <p className="results-loading"><LoaderCircle size={16} className="spin" /> Cargando cuentas…</p>}
        {error && <p className="error-message" role="alert">{error}</p>}
        {!loadingAccounts && (
          <>
            <div className="pf-target-list">
              {targets.map((target) => {
                const platformAccounts = accounts[target.platform] ?? [];
                return (
                  <div className="pf-target-row" key={target.key}>
                    <label className="config-field"><span>Plataforma</span>
                      <span className="select-wrap"><select value={target.platform} onChange={(event) => updateTarget(target.key, { platform: event.target.value })}>
                        {PLATFORMS.map((platform) => <option key={platform} value={platform}>{PLATFORM_LABELS[platform]}</option>)}
                      </select></span>
                    </label>
                    <label className="config-field"><span>Cuenta</span>
                      <span className="select-wrap"><select value={target.accountId} onChange={(event) => updateTarget(target.key, { accountId: event.target.value })}>
                        <option value="">Predeterminada</option>
                        {platformAccounts.map((account) => (
                          <option key={account.account_id} value={account.account_id}>
                            {account.display_name}{account.is_primary ? ' (predeterminada)' : ''}
                          </option>
                        ))}
                      </select></span>
                      {platformAccounts.length === 0 && <small>Sin cuentas de {PLATFORM_LABELS[target.platform]} conectadas.</small>}
                    </label>
                    <label className="config-field"><span>Cuándo (vacío: cuanto antes)</span>
                      <input className="text-input" type="datetime-local" value={target.scheduledAt} onChange={(event) => updateTarget(target.key, { scheduledAt: event.target.value })} />
                    </label>
                    <label className="config-field"><span>Visibilidad</span>
                      <span className="select-wrap"><select value={target.visibility} disabled={target.platform === 'instagram'} onChange={(event) => updateTarget(target.key, { visibility: event.target.value })}>
                        {target.platform !== 'tiktok' && <option value="unlisted">No listada</option>}
                        {target.platform !== 'instagram' && <option value="private">Privada</option>}
                        <option value="public">Pública</option>
                      </select></span>
                    </label>
                    {targets.length > 1 && (
                      <button type="button" className="pf-target-remove" onClick={() => setTargets((current) => current.filter((item) => item.key !== target.key))} aria-label="Quitar destino">
                        <X size={16} />
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
            {targets.length < 9 && (
              <button type="button" className="subtle-button" onClick={() => { setTargets((current) => [...current, emptyTarget(nextKey)]); setNextKey((key) => key + 1); }}>
                <Plus size={15} /> Añadir destino
              </button>
            )}
            <div className="pf-modal-actions">
              <button type="button" className="subtle-button" onClick={onClose}>Cancelar</button>
              <button type="button" className="create-button" onClick={() => void submit()} disabled={submitting || clipIds.length === 0}>
                {submitting ? <LoaderCircle size={15} className="spin" /> : null}
                {submitting ? 'Programando…' : 'Programar publicación'}
              </button>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
