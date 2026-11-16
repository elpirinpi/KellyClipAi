# kellyClipAi

App local para subir videos, guardar proyectos y crear clips con transcripción,
análisis LLM y renderizado FFmpeg. No genera resultados de ejemplo: cada clip
se obtiene del video y la transcripción del proyecto.

## Requisitos

- Windows 10/11, PowerShell y Node.js.
- Python 3.13 de 64 bits (recomendado para las ruedas de `faster-whisper`).
- FFmpeg y ffprobe en `PATH`.
- Una clave válida para el proveedor LLM elegido. También se puede configurar
  en Ajustes durante la sesión del navegador.

Comprueba los ejecutables:

```powershell
python --version
node --version
npm --version
Get-Command ffmpeg
Get-Command ffprobe
```

Si FFmpeg no está instalado, confirma antes de ejecutar:

```powershell
winget install Gyan.FFmpeg
```

## Instalación en PowerShell

Desde la carpeta del proyecto:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edita `.env` y configura `LLM_API_KEY`, `LLM_PROVIDER` y, si corresponde,
`LLM_MODEL`/`LLM_BASE_URL`. Las claves guardadas en Ajustes se conservan solo
en la sesión del navegador y se envían al backend local al iniciar el análisis;
no se escriben en SQLite.

Instala el frontend con las versiones fijadas en `app/package-lock.json`:

```powershell
Push-Location .\app
npm ci
Pop-Location
```

## Arranque

```powershell
.\dev.ps1
```

El script abre una ventana PowerShell para FastAPI (`127.0.0.1:8000`) y otra
para Vite (`127.0.0.1:5173`). Cierra ambas ventanas al terminar. No se publican
videos ni se dejan servidores ejecutándose en segundo plano.

La primera transcripción descarga el modelo de Whisper seleccionado desde el
repositorio oficial de modelos. Los proyectos, originales, miniaturas y MP4
resultantes quedan en `data/`; la configuración de cada proyecto y el historial
de eventos SSE se conservan en SQLite.

Los enlaces se descargan con yt-dlp únicamente desde plataformas compatibles;
importa solo contenido propio o para el que tengas autorización. La clave del
proveedor LLM es obligatoria. Si falta, el procesamiento se bloquea con un
mensaje visible y no se crean clips ficticios.

## Editor por transcripción

Desde los resultados, abre **Editar transcripción** en un clip para cambiar sus
subtítulos, quitar frases (lo que recorta ese segmento de audio y video al
exportar), o elegir otro ratio y plantilla. **Vista previa** genera un render
temporal; **Exportar cambios** vuelve a renderizar el clip desde el video
original y actualiza el archivo del clip. Los borradores del editor se guardan
con el proyecto local.

## Lotes, compartir y mejoras

Selecciona varios clips en Resultados y pulsa **Descargar** para crear un ZIP
con sus MP4 reales. **Compartir** crea un enlace aleatorio persistido en SQLite
y una página local de reproducción; solo funciona en esta máquina mientras los
servidores locales estén encendidos. El selector 1080p vuelve a renderizar el
clip desde el original. **Mejorar con IA** conserva los cambios del editor y
aplica un filtro real de paso alto, normalización EBU R128 y compresión ligera.

Para guardar un estilo propio, selecciona una plantilla integrada, escribe un
nombre y pulsa **Guardar estilo**. Se conserva en SQLite y aparece en “Mis
plantillas” después de recargar la app. El estilo guardado reutiliza los
parámetros de la plantilla seleccionada.

## Integraciones y publicaciones

Las credenciales de YouTube, TikTok, Instagram, Zoom, Pexels y Pixabay se
configuran únicamente en `.env`; nunca las compartas en el chat ni en capturas.
Después de editarlas, reinicia el backend. Registra en cada consola OAuth las
URLs locales que aparecen comentadas en `.env.example`. En **Ajustes**, la app
indica qué variables faltan y solo inicia OAuth cuando están configuradas.
Las cuentas conectadas y la cola se guardan en SQLite; los tokens permanecen
en la base local `data/kellyclipai.sqlite3`.

**Publicar** y **Programar clips** requieren una cuenta conectada. La publicación
inmediata muestra el resultado confirmado por la plataforma; si no se confirma,
la interfaz muestra el error en vez de indicar que tuvo éxito. Una tarea
programada publica al llegar su fecha usando la cuenta conectada. Si no hay una
cuenta o credenciales válidas, deja el MP4 real en `data/exports/` y marca la
tarea como exportada localmente, sin publicarlo. Si el backend se interrumpe
durante una publicación, la tarea queda fallida para revisión manual y no se
reintenta automáticamente.

Zoom lista grabaciones MP4 completadas de los últimos 30 días y las descarga
al directorio de proyectos local. B-roll busca un video real de Pexels o
Pixabay por clip, lo integra como imagen superpuesta y guarda proveedor,
creador y atribución en los metadatos del clip. Configura `BROLL_PROVIDER` y
la clave correspondiente en `.env`; si la búsqueda o descarga falla, el
procesamiento muestra el error en vez de sustituirlo por contenido de ejemplo.
Revisa los términos y las condiciones de cada proveedor antes de publicar o
utilizar contenido externo.

Endpoints del Hito 5: `GET /api/integrations`,
`GET /api/integrations/{provider}/connect`,
`GET /api/integrations/{provider}/callback`,
`POST /api/integrations/{provider}/disconnect`,
`GET /api/integrations/zoom/recordings`,
`POST /api/integrations/zoom/import`,
`POST /api/projects/{id}/publish`,
`GET /api/projects/{id}/scheduled`,
`GET /api/scheduled-jobs` y
`GET /api/scheduled-jobs/{job_id}/download`.

Endpoints del Hito 4: `POST /api/projects/{id}/clips/archive`,
`POST /api/projects/{id}/clips/{clip_id}/share`, `GET /api/shares/{token}`,
`POST /api/projects/{id}/clips/{clip_id}/quality`,
`POST /api/projects/{id}/clips/{clip_id}/improve` y
`GET/POST /api/templates`.

## Verificación

```powershell
.\.venv\Scripts\python.exe -m pytest
Push-Location .\app
npm run build
Pop-Location
```
