import { useEffect, useState } from 'react';
import { Gauge } from 'lucide-react';
import { apiFetch } from '../auth';
import { API_BASE } from '../context/constants';
import {
  detectGpuAcceleration,
  parseEncoderInfo,
  type EncoderInfo,
  type GpuAcceleration,
  type GpuDiagnostics,
} from '../utils/systemDiagnostics';

// Diagnóstico del sistema (v1.4.2): aceleración gráfica de la
// interfaz (sonda WebGL) y codificador de video activo (GET
// /api/encoder), visibles en Ajustes sin abrir las herramientas de
// desarrollo. Todo es best-effort: si algo no se puede comprobar, la
// fila muestra "No disponible" en vez de romper la página.
const GPU_STATUS_TEXT: Record<GpuAcceleration, string> = {
  hardware: 'Aceleración GPU activa',
  software: 'Render por software',
  unknown: 'No se pudo comprobar',
};

type EncoderState = EncoderInfo | 'unavailable' | null;

export function SystemDiagnosticsSection() {
  // La sonda WebGL es síncrona y barata (un contexto temporal): se
  // ejecuta una vez al montar la sección.
  const [gpu] = useState<GpuDiagnostics>(() => detectGpuAcceleration());
  const [encoder, setEncoder] = useState<EncoderState>(null);

  useEffect(() => {
    let active = true;
    apiFetch(`${API_BASE}/api/encoder`)
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json() as Promise<unknown>;
      })
      .then((payload) => { if (active) setEncoder(parseEncoderInfo(payload) ?? 'unavailable'); })
      .catch(() => { if (active) setEncoder('unavailable'); });
    return () => { active = false; };
  }, []);

  return (
    <section className="config-card">
      <div className="card-heading"><div><h2>Sistema y aceleración</h2><p>Aceleración gráfica de la interfaz y codificador de video</p></div><Gauge size={20} className="integration-icon" /></div>
      <dl className="diagnostics-list">
        <div className="diagnostics-row">
          <dt>Interfaz</dt>
          <dd>
            <span className={`diagnostics-status diagnostics-status--${gpu.acceleration}`}>{GPU_STATUS_TEXT[gpu.acceleration]}</span>
            {gpu.renderer && <span className="diagnostics-detail">{gpu.renderer}</span>}
            {!gpu.webglAvailable && <span className="diagnostics-detail">WebGL no está disponible en este equipo.</span>}
            {gpu.webglAvailable && !gpu.renderer && <span className="diagnostics-detail">El navegador no reveló el nombre del renderizador.</span>}
          </dd>
        </div>
        <div className="diagnostics-row">
          <dt>Codificador de video</dt>
          <dd>
            {encoder === null ? (
              <span className="diagnostics-detail">Comprobando…</span>
            ) : encoder === 'unavailable' ? (
              <span>No disponible</span>
            ) : (
              <>
                <span>{encoder.label}</span>
                <span className="diagnostics-detail">{encoder.encoder}</span>
              </>
            )}
          </dd>
        </div>
      </dl>
      <p className="settings-explainer">
        La interfaz usa la GPU cuando está disponible. Si ves «Render por software», la vista previa del Studio
        puede ir más lenta: actualiza los controladores de la tarjeta gráfica y vuelve a abrir la aplicación.
      </p>
    </section>
  );
}
