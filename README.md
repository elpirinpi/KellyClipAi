# kellyClipAi

App local para subir videos, guardar proyectos y crear clips con transcripción,
análisis LLM y renderizado FFmpeg. No genera resultados de ejemplo: cada clip
se obtiene del video y la transcripción del proyecto.

## Requisitos

- Windows 10/11, PowerShell y Node.js.
- Python 3.13 de 64 bits (recomendado para las ruedas de `faster-whisper`).
- FFmpeg y ffprobe en `PATH`.
- Un proyecto Supabase para registro e inicio de sesión.

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
.\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements-dev.lock
Copy-Item .env.example .env
```

`requirements.txt` contiene dependencias de ejecución; `requirements-dev.txt`
añade pruebas y herramientas. Los archivos `requirements.lock`,
`requirements-dev.lock` y `requirements-desktop.lock` fijan dependencias
transitivas con hashes SHA-256. Para actualizarlos, activa `.venv` y ejecuta:

```powershell
.\.venv\Scripts\pip-compile.exe --generate-hashes --output-file=requirements.lock requirements.txt
.\.venv\Scripts\pip-compile.exe --generate-hashes --output-file=requirements-dev.lock requirements-dev.txt
.\.venv\Scripts\pip-compile.exe --generate-hashes --output-file=requirements-desktop.lock requirements-desktop.txt
```

El `.env.example` incluido ya configura el proyecto Supabase de kellyClipAi
con su clave publicable. Si usas otro proyecto, reemplaza `SUPABASE_URL` y
`SUPABASE_ANON_KEY` por la URL y clave publicable (o clave `anon` heredada).
**No uses una clave `service_role`.** En Supabase, habilita Email/Password y
permite las redirecciones a `http://127.0.0.1:5173`,
`http://localhost:5173` y `http://127.0.0.1:8000`. El registro requiere
verificar el correo. En la app instalada, edita
`%LOCALAPPDATA%\kellyClipAi\.env` antes de abrirla por primera vez y reinicia
la aplicación después de cambiar su configuración.

La app muestra una pantalla de acceso antes de permitir usar los proyectos.
Los videos y proyectos siguen guardándose localmente, asociados al usuario
autenticado en la base SQLite del dispositivo. En **Ajustes**, la clave LLM se
cifra con Windows DPAPI y se conserva localmente para esa cuenta de Windows;
no se guarda en Supabase ni se devuelve al navegador. No se puede descifrar al
mover la base de datos a otro equipo o a otra cuenta de Windows.
Al actualizar una instalación con datos anteriores, esos proyectos locales se
asignan al primer usuario que inicie sesión en esa instalación.
Los cambios de esquema de SQLite se aplican mediante migraciones numeradas en
`PRAGMA user_version`; las bases creadas por versiones anteriores se actualizan
al iniciar la aplicación.

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

## Instalador de Windows

Descarga `KellyClipAi-Setup-1.0.5.exe` desde **Assets** del release en GitHub e
inicia el instalador. Incluye la aplicación, FFmpeg/ffprobe y el bootstrapper
oficial de Microsoft WebView2. El instalador requiere internet para instalar
WebView2 si hace falta; el modelo Whisper `small` se descarga la primera vez
que procesas un video. No incluye credenciales LLM ni de redes sociales:
configúralas en Ajustes o en `%LOCALAPPDATA%\kellyClipAi\.env`.

Los proyectos, archivos y base SQLite de la versión instalada se guardan en
`%LOCALAPPDATA%\kellyClipAi\data`. Las cargas locales se transmiten al disco de
la aplicación con progreso real y quedan recuperables al reiniciar. Las
descargas por URL también muestran bytes, velocidad y ETA informados por
yt-dlp. La desinstalación no borra esos archivos.
Al generar clips, el recorte puede seguir el rostro principal con YuNet en
CPU; desactívalo en **Opciones de edición** para volver al encuadre centrado.
El modelo se incluye localmente en la aplicación y el análisis no sube
fotogramas ni detecciones a servicios externos.
FFmpeg 9.0.2 Essentials es un ejecutable separado con licencia GPLv3; el
instalador incluye `licenses\GPL-3.0.txt` y `THIRD_PARTY_NOTICES.md`.

## Actualizaciones automáticas

La app instalada comprueba al iniciar sesión y cada seis horas si existe una
versión estable nueva en los releases oficiales de GitHub. Si hay una
actualización, descarga el instalador en segundo plano y verifica su SHA-256
antes de ofrecer **Instalar actualización**. La instalación abre el asistente
oficial de Windows y requiere confirmación; cerrar y volver a abrir la app
aplica la versión nueva. El proceso de actualización no elimina `.env`,
proyectos ni videos, que se guardan fuera de la carpeta de instalación. El
modo de desarrollo no descarga ni instala actualizaciones.

Para generar el instalador desde el código fuente, instala Python 3.13,
Node.js 22 e Inno Setup 6, y ejecuta desde PowerShell:

```powershell
.\build_windows.ps1
```

El script verifica la suma SHA-256 de FFmpeg y la firma del bootstrapper de
WebView2. El instalador resultante aparece en `artifacts\`. Al publicar un tag
`vX.Y.Z` que coincida con `app/package.json`, GitHub Actions construye el
instalador y lo adjunta al release.

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
configuran en `.env`; nunca las compartas en el chat ni en capturas.
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

Endpoints de autenticación y clave LLM: `GET /api/auth/config`,
`POST/DELETE /api/auth/media-cookie`, `GET /api/settings`,
`PUT/DELETE /api/settings/llm-key`. La API comprueba las sesiones con Supabase
y filtra los proyectos por usuario; los enlaces compartidos permanecen
públicos por diseño. Las claves de API de usuario se cifran en el dispositivo
con DPAPI antes de guardarse en SQLite.

## Verificación

```powershell
.\.venv\Scripts\python.exe -m pytest
Push-Location .\app
npm run build
Pop-Location
```
