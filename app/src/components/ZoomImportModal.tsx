import { AlertCircle, Download, LoaderCircle, X } from 'lucide-react';
import { useApp } from '../context/AppContext';

// Modal de importación desde Zoom (extraído de App.tsx, ítem 23 de la Fase 2).
export function ZoomImportModal() {
  const {
    navigate,
    zoom: {
      zoomOpen,
      setZoomOpen,
      zoomRecordings,
      zoomLoading,
      zoomError,
      zoomModalRef,
      importZoomFile,
    },
  } = useApp();
  if (!zoomOpen) return null;

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget && !zoomLoading) setZoomOpen(false); }}>
      <section className="record-modal zoom-modal" ref={zoomModalRef} role="dialog" aria-modal="true" aria-labelledby="zoom-title">
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
  );
}
