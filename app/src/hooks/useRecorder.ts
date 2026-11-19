import { useEffect, useRef, useState } from 'react';
import { useFocusTrap } from './useFocusTrap';

// Grabación de video (extraída de App.tsx, ítem 23 de la Fase 2): modal
// de grabar con cámara o pantalla, micrófono opcional y guardado de la
// grabación como importación local al detenerla.

type UseRecorderOptions = {
  importFile: (file: File, fromRecorder?: boolean) => Promise<void>;
};

export function useRecorder({ importFile }: UseRecorderOptions) {
  const [recordOpen, setRecordOpen] = useState(false);
  const [recordMode, setRecordMode] = useState<'camera' | 'screen'>('camera');
  const [recording, setRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [recordError, setRecordError] = useState('');
  const [recordMic, setRecordMic] = useState(true);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recordChunksRef = useRef<Blob[]>([]);
  // Fase 2 (ítem 21): trampa de foco en el modal de grabar.
  const recordModalRef = useRef<HTMLElement>(null);
  useFocusTrap(recordModalRef, recordOpen);

  useEffect(() => {
    if (!recording) return;
    const interval = window.setInterval(() => setRecordingTime((value) => value + 1), 1000);
    return () => window.clearInterval(interval);
  }, [recording]);

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

  return {
    recordOpen,
    setRecordOpen,
    recordMode,
    setRecordMode,
    recording,
    recordingTime,
    recordError,
    setRecordError,
    recordMic,
    setRecordMic,
    recordModalRef,
    startRecording,
    stopRecording,
  };
}
