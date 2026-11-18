param()

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = $PSScriptRoot
$appDir = Join-Path $root 'app'
$buildDir = Join-Path $root 'build'
$vendorDir = Join-Path $buildDir 'vendor'
$python = Join-Path $root '.venv\Scripts\python.exe'
$ffmpegVersion = '9.0.2'
$ffmpegArchive = Join-Path $vendorDir "ffmpeg-$ffmpegVersion-essentials_build.zip"
$ffmpegUrl = "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-$ffmpegVersion-essentials_build.zip"
$ffmpegSha256 = '60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba'
$webViewSetup = Join-Path $vendorDir 'MicrosoftEdgeWebView2Setup.exe'
$webViewUrl = 'https://go.microsoft.com/fwlink/p/?LinkId=2124703'
$gplLicense = Join-Path $vendorDir 'COPYING.GPLv3'
$ffmpegDir = Join-Path $vendorDir "ffmpeg-$ffmpegVersion-essentials_build"
$distDir = Join-Path $buildDir 'bundle'
$version = (Get-Content (Join-Path $appDir 'package.json') -Raw | ConvertFrom-Json).version

if (-not (Test-Path $python)) {
    & py -3.13 -m venv (Join-Path $root '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo crear el entorno virtual de Python 3.13.' }
}
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw 'No se encontró Python 3.13 en .venv.' }
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw 'Instala Node.js antes de generar el instalador.' }

New-Item -ItemType Directory -Force -Path $vendorDir | Out-Null
Push-Location $appDir
try {
    npm ci
    if ($LASTEXITCODE -ne 0) { throw 'npm ci falló.' }
    npm run build
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo compilar el frontend.' }
} finally {
    Pop-Location
}

& $python -m pip install --require-hashes -r (Join-Path $root 'requirements-desktop.lock')
if ($LASTEXITCODE -ne 0) { throw 'No se pudieron instalar las dependencias de escritorio.' }

if (-not (Test-Path $ffmpegArchive)) {
    Invoke-WebRequest -Uri $ffmpegUrl -OutFile $ffmpegArchive
}
$actualHash = (Get-FileHash -LiteralPath $ffmpegArchive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $ffmpegSha256) {
    throw "La suma SHA-256 de FFmpeg no coincide: $actualHash"
}
if (-not (Test-Path (Join-Path $ffmpegDir 'bin\ffmpeg.exe'))) {
    Expand-Archive -LiteralPath $ffmpegArchive -DestinationPath $vendorDir -Force
}
$ffmpegBin = Join-Path $ffmpegDir 'bin'
foreach ($binary in @('ffmpeg.exe', 'ffprobe.exe')) {
    if (-not (Test-Path (Join-Path $ffmpegBin $binary))) {
        throw "El paquete FFmpeg no contiene $binary."
    }
}
& (Join-Path $ffmpegBin 'ffmpeg.exe') -version | Select-Object -First 1
if ($LASTEXITCODE -ne 0) { throw 'El binario ffmpeg.exe no se pudo ejecutar.' }

if (-not (Test-Path $webViewSetup)) {
    Invoke-WebRequest -Uri $webViewUrl -OutFile $webViewSetup
}
$signature = Get-AuthenticodeSignature -FilePath $webViewSetup
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Microsoft Corporation') {
    throw 'No se pudo verificar la firma del instalador de WebView2 de Microsoft.'
}
foreach ($file in @(
    (Join-Path $root 'backend\models\face_detection_yunet_2023mar.onnx'),
    (Join-Path $root 'backend\models\LICENSE')
)) {
    if (-not (Test-Path $file)) {
        throw "Falta un archivo requerido para el seguimiento facial: $file"
    }
}
if (-not (Test-Path $gplLicense)) {
    Invoke-WebRequest `
        -Uri 'https://raw.githubusercontent.com/FFmpeg/FFmpeg/946fcce07b/COPYING.GPLv3' `
        -OutFile $gplLicense
}

& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --windowed `
    --name KellyClipAi `
    --distpath $distDir `
    --workpath (Join-Path $buildDir 'pyinstaller') `
    --specpath $buildDir `
    --collect-all faster_whisper `
    --collect-all ctranslate2 `
    --collect-all av `
    --collect-all yt_dlp `
    --collect-all uvicorn `
    --collect-all webview `
    --collect-all cv2 `
    --hidden-import uvicorn.logging `
    --hidden-import uvicorn.loops.auto `
    --hidden-import uvicorn.protocols.http.auto `
    --hidden-import uvicorn.lifespan.on `
    --add-data "$(Join-Path $appDir 'dist');web" `
    --add-data "$(Join-Path $root '.env.example');config" `
    --add-data "$(Join-Path $root 'backend\models\face_detection_yunet_2023mar.onnx');backend/models" `
    --add-data "$(Join-Path $root 'backend\models\LICENSE');backend/models" `
    --add-binary "$(Join-Path $ffmpegBin 'ffmpeg.exe');bin" `
    --add-binary "$(Join-Path $ffmpegBin 'ffprobe.exe');bin" `
    (Join-Path $root 'desktop.py')
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller no pudo empaquetar la aplicación.' }

$compilerPath = (Get-Command ISCC.exe -ErrorAction SilentlyContinue).Source
if (-not $compilerPath) {
    $compilerPaths = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 6\ISCC.exe'),
        (Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 6\ISCC.exe'),
        (Join-Path $env:ProgramFiles 'Inno Setup 6\ISCC.exe')
    )
    $compilerPath = $compilerPaths | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $compilerPath) { throw 'Instala Inno Setup 6 para generar el instalador.' }
}

& $compilerPath "/DAppVersion=$version" (Join-Path $root 'installer\KellyClipAi.iss')
if ($LASTEXITCODE -ne 0) { throw 'Inno Setup no pudo crear el instalador.' }

$installer = Join-Path $root "artifacts\KellyClipAi-Setup-$version.exe"
if (-not (Test-Path $installer)) { throw "No se generó el instalador esperado: $installer" }
Write-Host "Instalador creado: $installer"
