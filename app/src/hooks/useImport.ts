import { useCallback, useEffect, useRef, useState, type ChangeEvent, type DragEvent } from 'react';
import { accessToken, apiFetch } from '../auth';
import type { StagedMedia, UploadInfo } from '../types';

const ALLOWED_EXTENSIONS = new Set(['mp4', 'mov', '3gp', 'avi', 'mkv', 'webm', 'mp3', 'wav', 'm4a']);
export const ACCEPT = '.mp4,.mov,.3gp,.avi,.mkv,.webm,.mp3,.wav,.m4a,video/mp4,video/quicktime,video/3gpp,video/x-msvideo,video/webm,audio/mpeg,audio/wav,audio/mp4';

type UseImportOptions = {
  apiBase: string;
  authReady: boolean;
  userId: string | undefined;
  onImportStart: () => void;
  onNotice: (message: string) => void;
};

async function responseError(response: Response): Promise<string> {
  try {
    const body = await response.json() as { detail?: string };
    return body.detail ?? `El servidor respondió con error (${response.status}).`;
  } catch {
    return `El servidor respondió con error (${response.status}).`;
  }
}

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

async function uploadStagedFile(
  apiBase: string,
  stagedId: string,
  file: File,
  onProgress: (progress: number) => void,
  onRequest: (request: XMLHttpRequest | null) => void,
): Promise<void> {
  const token = await accessToken();
  if (!token) throw new Error('Inicia sesión antes de cargar un video.');
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('POST', `${apiBase}/api/staged-media/${stagedId}/content`);
    request.setRequestHeader('Content-Type', 'application/octet-stream');
    request.setRequestHeader('Authorization', `Bearer ${token}`);
    request.upload.onprogress = (event) => {
      if (event.lengthComputable && event.total > 0) {
        onProgress(Math.min(99, Math.round((event.loaded / event.total) * 100)));
      }
    };
    request.onload = () => {
      onRequest(null);
      if (request.status >= 200 && request.status < 300) {
        resolve();
        return;
      }
      try {
        const body = JSON.parse(request.responseText) as { detail?: string };
        reject(new Error(body.detail ?? `El servidor respondió con error (${request.status}).`));
      } catch {
        reject(new Error(`El servidor respondió con error (${request.status}).`));
      }
    };
    request.onerror = () => {
      onRequest(null);
      reject(new Error('No se pudo conectar con el almacenamiento local.'));
    };
    request.onabort = () => {
      onRequest(null);
      reject(new Error('Importación cancelada.'));
    };
    onRequest(request);
    request.send(file);
  });
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

    uploadRequestRef.current?.abort();
    cancelImportRef.current = false;
    const generation = ++importGenerationRef.current;
    onImportStart();
    setFileInfo(null);
    setThumbnail('');
    setProgress(0);
    const kind = file.type.startsWith('audio/') || ['mp3', 'wav', 'm4a'].includes(extension) ? 'audio' : 'video';
    try {
      const metadata = await metadataFor(file);
      if (cancelImportRef.current || generation !== importGenerationRef.current) {
        setProgress(null);
        return;
      }
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
      if (!createResponse.ok) throw new Error(await responseError(createResponse));
      const staged = await createResponse.json() as StagedMedia;
      if (cancelImportRef.current || generation !== importGenerationRef.current) {
        const deleteResponse = await apiFetch(`${apiBase}/api/staged-media/${staged.id}`, { method: 'DELETE' });
        if (!deleteResponse.ok) throw new Error(await responseError(deleteResponse));
        setProgress(null);
        return;
      }
      setFileInfo(stagedUploadInfo(staged));
      setThumbnail(metadata.thumbnail);
      await uploadStagedFile(
        apiBase,
        staged.id,
        file,
        (value) => {
          if (generation === importGenerationRef.current) setProgress(value);
        },
        (request) => {
          if (generation === importGenerationRef.current) uploadRequestRef.current = request;
        },
      );
      if (generation === importGenerationRef.current) {
        setProgress(100);
        setUploadError('');
      }
    } catch (error) {
      if (generation !== importGenerationRef.current) return;
      if (uploadRequestRef.current) uploadRequestRef.current = null;
      setProgress(null);
      setFileInfo(null);
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
    setUploadError('Importación cancelada.');
  };

  return {
    fileInfo,
    setFileInfo,
    thumbnail,
    progress,
    setProgress,
    uploadError,
    setUploadError,
    dragging,
    setDragging,
    inputRef,
    importFile,
    chooseFile,
    onDrop,
    cancelImport,
  };
}
