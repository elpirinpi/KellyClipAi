import { AlertCircle, MonitorUp, Pause, Video, X } from 'lucide-react';
import { useApp } from '../context/AppContext';
import { formatDuration } from '../utils';

// Modal de grabación de video (extraído de App.tsx, ítem 23 de la Fase 2).
export function RecordModal() {
  const {
    recorder: {
      recordOpen,
      setRecordOpen,
      recordMode,
      setRecordMode,
      recording,
      recordingTime,
      recordError,
      recordMic,
      setRecordMic,
      recordModalRef,
      startRecording,
      stopRecording,
    },
  } = useApp();
  if (!recordOpen) return null;

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget && !recording) setRecordOpen(false); }}>
      <section className="record-modal" ref={recordModalRef} role="dialog" aria-modal="true" aria-labelledby="record-title">
        <div className="modal-heading"><div><h2 id="record-title">Grabar video</h2><p>La grabación se guardará en este dispositivo.</p></div><button type="button" className="icon-button" aria-label="Cerrar" disabled={recording} onClick={() => setRecordOpen(false)}><X size={19} /></button></div>
        <div className="record-mode-switch"><button type="button" className={recordMode === 'camera' ? 'active' : ''} onClick={() => setRecordMode('camera')} disabled={recording}><Video size={17} /> Cámara</button><button type="button" className={recordMode === 'screen' ? 'active' : ''} onClick={() => setRecordMode('screen')} disabled={recording}><MonitorUp size={17} /> Pantalla</button></div>
        <label className="record-mic"><input type="checkbox" checked={recordMic} onChange={(event) => setRecordMic(event.target.checked)} disabled={recording} /> Incluir micrófono</label>
        {recording && <p className="recording-status"><span className="recording-dot" /> Grabando · {formatDuration(recordingTime)}</p>}
        {recordError && <p className="error-message" role="alert"><AlertCircle size={16} />{recordError}</p>}
        <button type="button" className={`record-start ${recording ? 'stop' : ''}`} onClick={recording ? stopRecording : startRecording}>{recording ? <><Pause size={17} /> Detener y guardar</> : <><Video size={17} /> Comenzar grabación</>}</button>
        <p className="modal-footnote">Tu navegador solicitará permiso para acceder a la cámara, pantalla o micrófono.</p>
      </section>
    </div>
  );
}
