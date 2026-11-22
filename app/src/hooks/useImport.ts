import { useCallback, useEffect, useRef, useState, type ChangeEvent, type DragEvent } from 'react';
import { accessToken, apiFetch } from '../auth';
import type { StagedMedia, UploadInfo } from '../types';
import { responseError } from '../utils';

const ALLOWED_EXTENSIONS = new Set(['mp4', 'mov', '3gp', 'avi', 'mkv', 'webm', 'mp3', 'wav', 'm4a']);
export const ACCEPT = '.mp4,.mov,.3gp,.avi,.mkv,.webm,.mp3,.wav,.m4a,video/mp4,video/quicktime,video/3gpp,video/x-msvideo,video/webm,audio/mpeg,audio/wav,audio/mp4';

// Fase 2 (ítem 22): límite de subida del backend (backend/schemas.py,
// MAX_UPLOAD_BYTES = 4 GiB). Se valida en cliente antes de subir y se
// muestra en la zona de subida para que el usuario lo conozca por
// adelantado, en vez de descubrirlo con un error del servidor.
export const MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024; // 4 GiB
export const MAX_UPLOAD_LABEL = '4 GB';

// Subida por fragmentos con reanudación (Fase 2, ítem 22): el archivo se
// envía en trozos de UPLOAD_CHUNK_BYTES con `Content-Range: bytes
// <inicio>-<fin>/<total>`. El backend conserva el .part ante cualquier
// interrupción y expone bytes_received en el GET, así que un reintento
// reanuda desde el último byte confirmado en vez de empezar de cero.
const UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024; // 8 MiB por fragmento
const MAX_UPLOAD_ATTEMPTS = 3;
const MAX_CHUNK_ATTEMPTS = 3;
const UPLOAD_STALL_TIMEOUT_MS = 60_000;
const UPLOAD_RETRY_DELAYS_MS = [1000, 2000];

class UploadAttemptError extends Error {
  readonly retryable: boolean;

  constructor(message: string, retryable: boolean) {
    super(message);
    this.name = 'UploadAttemptError';
    this.retryable = retryable;
  }
}

type UseImportOptions = {
  apiBase: string;
  authReady: boolean;
  userId: string | undefined;
  onImportStart: () => void;
  onNotice: (message: string) => void;
};

function extensionOf(name: string): string {
  return name.split('.').pop()?.toLowerCase() ?? '';
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

async function fetchStagedMedia(apiBase: string, stagedId: string): Promise<StagedMedia> {
  const response = await apiFetch(`${apiBase}/api/staged-media/${stagedId}`);
  if (!response.ok) throw new Error(await responseError(response));
  return response.json() as Promise<StagedMedia>;
}

async function sendChunk(
  apiBase: string,
  stagedId: string,
  chunk: Blob,
  start: number,
  total: number,
  onProgress: (progress: number) => void,
  onRequest: (request: XMLHttpRequest | null) => void,
): Promise<StagedMedia> {
  const token = await accessToken();
  if (!token) throw new UploadAttemptError('Inicia sesión antes de cargar un video.', false);
  const end = start + chunk.size - 1;
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    let stalled = false;
    let lastProgressAt = Date.now();
    // Detección de conexión colgada: sin un timeout fijo (un fragmento
    // puede tardar legítimamente en una conexión lenta), se aborta solo
    // si no llega ningún evento de progreso durante UPLOAD_STALL_TIMEOUT_MS.
    const stallTimer = window.setInterval(() => {
      if (Date.now() - lastProgressAt > UPLOAD_STALL_TIMEOUT_MS) {
        stalled = true;
        request.abort();
      }
    }, 5000);
    const settle = (finish: () => void) => {
      window.clearInterval(stallTimer);
      onRequest(null);
      finish();
    };
    request.open('POST', `${apiBase}/api/staged-media/${stagedId}/content`);
    request.setRequestHeader('Content-Type', 'application/octet-stream');
    request.setRequestHeader('Content-Range', `bytes ${start}-${end}/${total}`);
    request.setRequestHeader('Authorization', `Bearer ${token}`);
    request.upload.onprogress = (event) => {
      lastProgressAt = Date.now();
      if (event.lengthComputable && event.total > 0) {
        const uploaded = start + Math.round((event.loaded / event.total) * chunk.size);
        onProgress(Math.min(99, Math.floor((uploaded / total) * 100)));
      }
    };
    request.onload = () => {
      settle(() => {
        if (request.status >= 200 && request.status < 300) {
          try {
            resolve(JSON.parse(request.responseText) as StagedMedia);
          } catch {
            reject(new UploadAttemptError('El servidor respondió sin los datos de la carga.', true));
          }
          return;
        }
        let detail = `El servidor respondió con error (${request.status}).`;
        try {
          const body = JSON.parse(request.responseText) as { detail?: string };
          detail = body.detail ?? detail;
        } catch {
          // Se conserva el mensaje genérico con el código de estado.
        }
        // 5xx: transitorio. 409 (offset por delante) y 400 (fragmento
        // interrumpido, .part conservado): se reanuda desde
        // bytes_received. El resto de 4xx no se reintenta.
        const retryable = request.status >= 500 || request.status === 409 || request.status === 400;
        reject(new UploadAttemptError(detail, retryable));
      });
    };
    request.onerror = () => {
      settle(() => reject(new UploadAttemptError(
        'No se pudo conectar con el almacenamiento local.',
        true,
      )));
    };
    request.onabort = () => {
      settle(() => reject(stalled
        ? new UploadAttemptError(
          'La conexión se quedó sin responder durante la subida.',
          true,
        )
        : new UploadAttemptError('Importación cancelada.', false)));
    };
    onRequest(request);
    request.send(chunk);
  });
}

async function uploadStagedFile(
  apiBase: string,
  stagedId: string,
  file: File,
  onProgress: (progress: number) => void,
  onRequest: (request: XMLHttpRequest | null) => void,
  startAt = 0,
): Promise<void> {
  let offset = startAt;
  let chunkAttempts = 0;
  while (offset < file.size) {
    const chunk = file.slice(offset, Math.min(offset + UPLOAD_CHUNK_BYTES, file.size));
    try {
      const staged = await sendChunk(apiBase, stagedId, chunk, offset, file.size, onProgress, onRequest);
      // El servidor es la fuente de verdad: bytes_received manda.
      if (staged.status === 'failed') {
        throw new UploadAttemptError(staged.error ?? 'La carga falló en el servidor.', false);
      }
      if (staged.status === 'ready') {
        onProgress(100);
        return;
      }
      if (staged.bytes_received <= offset) {
        // El fragmento se aceptó pero no avanzó nada: anomalía del
        // servidor; reintentarlo no ayudaría.
        throw new UploadAttemptError('El servidor no confirmó el fragmento enviado.', false);
      }
      offset = staged.bytes_received;
      onProgress(Math.min(99, Math.floor((offset / file.size) * 100)));
      chunkAttempts = 0;
    } catch (error) {
      const retryable = error instanceof UploadAttemptError
        ? error.retryable
        : error instanceof TypeError;
      if (!retryable) throw error;
      chunkAttempts += 1;
      if (chunkAttempts >= MAX_CHUNK_ATTEMPTS) throw error;
      await new Promise((resolve) => window.setTimeout(
        resolve,
        UPLOAD_RETRY_DELAYS_MS[chunkAttempts - 1] ?? 2000,
      ));
      // Reanudar desde lo confirmado por el servidor.
      const staged = await fetchStagedMedia(apiBase, stagedId);
      if (staged.status === 'ready') {
        onProgress(100);
        return;
      }
      if (staged.status === 'failed') {
        throw new UploadAttemptError(staged.error ?? 'La carga falló en el servidor.', false);
      }
      offset = staged.bytes_received;
    }
  }
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

export async function pollStagedMedia(
  apiBase: string,
  stagedId: string,
  onUpdate: (staged: StagedMedia) => void,
  active: () => boolean,
): Promise<StagedMedia> {
  while (active()) {
    const response = await apiFetch(`${apiBase}/api/staged-media/${stagedId}`);
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

export function useImport({ apiBase, authReady, userId, onImportStart, onNotice }: UseImportOptions) {
  const [fileInfo, setFileInfo] = useState<UploadInfo | null>(null);
  const [thumbnail, setThumbnail] = useState('');
  const [progress, setProgress] = useState<number | null>(null);
  const [uploadError, setUploadError] = useState('');
  // Hotfix v1.4.1 (bug 1): id de la importación fallida cuyo aviso se está
  // mostrando, para poder descartarlo borrando su fila staged. Solo se
  // fija cuando el error viene de una fila staged fallida (restoreUpload
  // o el polling); los demás errores no tienen nada que descartar.
  const [failedStagedId, setFailedStagedId] = useState<string | null>(null);
  // Fase 2 (ítem 22): intento actual de subida (null fuera de un reintento).
  const [uploadAttempt, setUploadAttempt] = useState<{ attempt: number; maxAttempts: number } | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const cancelImportRef = useRef(false);
  const importGenerationRef = useRef(0);
  const uploadRequestRef = useRef<XMLHttpRequest | null>(null);

  useEffect(() => {
    if (!authReady || !userId) return;
    let active = true;
    const restoreUpload = async () => {
      try {
        const response = await apiFetch(`${apiBase}/api/staged-media/latest?source=upload`);
        if (!response.ok) throw new Error(await responseError(response));
        const latest = await response.json() as StagedMedia | null;
        if (!active || !latest) return;
        if (latest.status === 'failed') {
          setFailedStagedId(latest.id);
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
            apiBase,
            latest.id,
            (staged) => {
              if (!active) return;
              setProgress(staged.status === 'ready'
                ? 100
                : staged.file_size > 0
                  ? Math.min(99, Math.floor((staged.bytes_received / staged.file_size) * 100))
                  : 0);
              if (staged.status === 'failed') {
                setFailedStagedId(staged.id);
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
  }, [apiBase, authReady, userId]);

  const importFile = useCallback(async (file: File, fromRecorder = false) => {
    setUploadError('');
    setFailedStagedId(null);
    onNotice('');
    const extension = extensionOf(file.name);
    if (!fromRecorder && !ALLOWED_EXTENSIONS.has(extension)) {
      setUploadError('Formato no compatible. Usa MP4, MOV, 3GP, AVI, MP3, WAV o M4A.');
      return;
    }
    if (file.size === 0) {
      setUploadError('El archivo está vacío. Elige otro archivo.');
      return;
    }
    // Fase 2 (ítem 22): validación en cliente del tamaño máximo (4 GiB),
    // antes de crear la carga en el servidor.
    if (file.size > MAX_UPLOAD_BYTES) {
      setUploadError(`El archivo supera el tamaño máximo admitido de ${MAX_UPLOAD_LABEL}. Elige un archivo más pequeño o comprímelo antes de subirlo.`);
      return;
    }

    uploadRequestRef.current?.abort();
    cancelImportRef.current = false;
    const generation = ++importGenerationRef.current;
    onImportStart();
    setFileInfo(null);
    setThumbnail('');
    setUploadAttempt(null);
    setProgress(0);
    const kind = file.type.startsWith('audio/') || ['mp3', 'wav', 'm4a'].includes(extension) ? 'audio' : 'video';
    const wasCancelled = () => cancelImportRef.current || generation !== importGenerationRef.current;
    try {
      const metadata = await metadataFor(file);
      if (wasCancelled()) {
        setProgress(null);
        return;
      }
      // Subida por fragmentos con reanudación: la carga se crea UNA vez;
      // cada reintento reanuda desde los bytes confirmados por el
      // servidor en vez de recrear la carga y empezar de cero.
      let lastError: Error | null = null;
      let stagedId: string | null = null;
      for (let attempt = 1; attempt <= MAX_UPLOAD_ATTEMPTS; attempt += 1) {
        if (wasCancelled()) {
          setProgress(null);
          return;
        }
        setUploadAttempt({ attempt, maxAttempts: MAX_UPLOAD_ATTEMPTS });
        try {
          let resumeFrom = 0;
          if (stagedId === null) {
            const createResponse = await apiFetch(`${apiBase}/api/staged-media/uploads`, {
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
            if (!createResponse.ok) {
              throw new UploadAttemptError(
                await responseError(createResponse),
                createResponse.status >= 500,
              );
            }
            const staged = await createResponse.json() as StagedMedia;
            stagedId = staged.id;
            if (wasCancelled()) {
              const deleteResponse = await apiFetch(`${apiBase}/api/staged-media/${staged.id}`, { method: 'DELETE' });
              if (!deleteResponse.ok) throw new Error(await responseError(deleteResponse));
              setProgress(null);
              return;
            }
            setFileInfo(stagedUploadInfo(staged));
            setThumbnail(metadata.thumbnail);
          } else {
            // Reanudar: preguntar al servidor desde dónde seguir.
            const staged = await fetchStagedMedia(apiBase, stagedId);
            if (staged.status === 'ready') {
              if (generation === importGenerationRef.current) {
                setProgress(100);
                setUploadError('');
                setUploadAttempt(null);
              }
              return;
            }
            if (staged.status === 'failed') {
              throw new UploadAttemptError(staged.error ?? 'La carga falló en el servidor.', false);
            }
            resumeFrom = staged.bytes_received;
            setProgress(Math.min(99, Math.floor((resumeFrom / file.size) * 100)));
          }
          await uploadStagedFile(
            apiBase,
            stagedId,
            file,
            (value) => {
              if (generation === importGenerationRef.current) setProgress(value);
            },
            (request) => {
              if (generation === importGenerationRef.current) uploadRequestRef.current = request;
            },
            resumeFrom,
          );
          if (generation === importGenerationRef.current) {
            setProgress(100);
            setUploadError('');
            setUploadAttempt(null);
          }
          return;
        } catch (attemptError) {
          if (wasCancelled()) {
            setProgress(null);
            return;
          }
          lastError = attemptError instanceof Error
            ? attemptError
            : new Error('No se pudo guardar el archivo en este dispositivo.');
          const retryable = attemptError instanceof UploadAttemptError
            ? attemptError.retryable
            // Un fallo de red de fetch (TypeError) también se reintenta.
            : attemptError instanceof TypeError;
          if (!retryable || attempt === MAX_UPLOAD_ATTEMPTS) throw lastError;
          await new Promise((resolve) => window.setTimeout(
            resolve,
            UPLOAD_RETRY_DELAYS_MS[attempt - 1] ?? 2000,
          ));
        }
      }
      throw lastError ?? new Error('No se pudo guardar el archivo en este dispositivo.');
    } catch (error) {
      if (generation !== importGenerationRef.current) return;
      if (uploadRequestRef.current) uploadRequestRef.current = null;
      setProgress(null);
      setFileInfo(null);
      setUploadAttempt(null);
      setUploadError(error instanceof Error ? error.message : 'No se pudo guardar el archivo en este dispositivo.');
    }
  }, [apiBase, onImportStart, onNotice]);

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
    setUploadAttempt(null);
    setUploadError('Importación cancelada.');
    // Hotfix v1.4.1 (bug 1): borrar la carga en el servidor. Antes la
    // fila quedaba en 'receiving' y en el siguiente arranque se marcaba
    // como fallida con el aviso "La aplicación se cerró antes de
    // completar la importación", aunque el usuario solo había cancelado.
    const stagedId = fileInfo?.id;
    setFileInfo(null);
    if (stagedId) {
      void apiFetch(`${apiBase}/api/staged-media/${stagedId}`, { method: 'DELETE' }).catch(() => {
        // El borrado es limpieza: si falla, la fila caduca sola a las
        // 24 h y el aviso no reaparece tras la siguiente importación.
      });
    }
  };

  // Hotfix v1.4.1 (bug 1): descartar el aviso de una importación fallida
  // borra su fila staged. El aviso ya se mostró; sin este borrado
  // reaparecía en cada arranque hasta que la fila caducara (24 h).
  const dismissUploadError = useCallback(async () => {
    const stagedId = failedStagedId;
    setFailedStagedId(null);
    setUploadError('');
    if (!stagedId) return;
    try {
      const response = await apiFetch(`${apiBase}/api/staged-media/${stagedId}`, { method: 'DELETE' });
      // 404: la fila ya no existe; nada más que limpiar.
      if (!response.ok && response.status !== 404) throw new Error(await responseError(response));
    } catch {
      // La fila caduca sola a las 24 h; el aviso ya no se muestra en esta sesión.
    }
  }, [apiBase, failedStagedId]);

  return {
    fileInfo,
    setFileInfo,
    thumbnail,
    progress,
    setProgress,
    uploadError,
    setUploadError,
    failedStagedId,
    dismissUploadError,
    uploadAttempt,
    dragging,
    setDragging,
    inputRef,
    importFile,
    chooseFile,
    onDrop,
    cancelImport,
  };
}
