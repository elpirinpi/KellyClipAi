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

# --- Firma de código opcional (ver README.md, "Firma de código") ---
# Sin credenciales configuradas el build sigue sin firmar, sin fallar.
# Si la firma está configurada pero falla, el build falla: un instalador
# que debería ir firmado no debe publicarse sin firmar en silencio.
# Nunca se guarda ningún secreto en el repositorio: las credenciales
# llegan por variables de entorno (secretos de GitHub en CI, variables
# locales en un build manual).
$signingMode = 'None'
if ($env:KELLYCLIP_SIGN_PFX_BASE64 -or $env:KELLYCLIP_SIGN_PFX_PATH) {
    $signingMode = 'Pfx'
} elseif ($env:KELLYCLIP_AZURE_TRUSTED_SIGNING_ENDPOINT -or $env:KELLYCLIP_AZURE_TRUSTED_SIGNING_ACCOUNT -or $env:KELLYCLIP_AZURE_TRUSTED_SIGNING_CERT_PROFILE) {
    if (-not ($env:KELLYCLIP_AZURE_TRUSTED_SIGNING_ENDPOINT -and $env:KELLYCLIP_AZURE_TRUSTED_SIGNING_ACCOUNT -and $env:KELLYCLIP_AZURE_TRUSTED_SIGNING_CERT_PROFILE)) {
        throw 'La firma con Azure Trusted Signing está incompleta: configura KELLYCLIP_AZURE_TRUSTED_SIGNING_ENDPOINT, KELLYCLIP_AZURE_TRUSTED_SIGNING_ACCOUNT y KELLYCLIP_AZURE_TRUSTED_SIGNING_CERT_PROFILE.'
    }
    $signingMode = 'AzureTrustedSigning'
}
if ($signingMode -eq 'None') {
    Write-Host 'Firma de código no configurada; el instalador se generará sin firmar (ver README.md, "Firma de código").'
}

function Find-SignTool {
    $command = Get-Command signtool.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $sdkBin = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\bin'
    if (Test-Path $sdkBin) {
        $candidate = Get-ChildItem -Path $sdkBin -Directory |
            Sort-Object Name -Descending |
            ForEach-Object { Join-Path $_.FullName 'x64\signtool.exe' } |
            Where-Object { Test-Path $_ } |
            Select-Object -First 1
        if ($candidate) { return $candidate }
    }
    return $null
}

function Invoke-CodeSigning {
    param([Parameter(Mandatory = $true)][string]$FilePath)

    if ($script:signingMode -eq 'None') { return }
    if (-not (Test-Path $FilePath)) { throw "No se encontró el archivo a firmar: $FilePath" }

    if ($script:signingMode -eq 'Pfx') {
        $signtool = Find-SignTool
        if (-not $signtool) { throw 'La firma de código está configurada pero no se encontró signtool.exe (instala el SDK de Windows 10/11).' }
        $pfxPath = $env:KELLYCLIP_SIGN_PFX_PATH
        $tempPfx = $null
        try {
            if ($env:KELLYCLIP_SIGN_PFX_BASE64) {
                $tempPfx = Join-Path ([System.IO.Path]::GetTempPath()) ("kellyclipai-sign-{0}.pfx" -f [System.Guid]::NewGuid())
                [System.IO.File]::WriteAllBytes($tempPfx, [System.Convert]::FromBase64String($env:KELLYCLIP_SIGN_PFX_BASE64))
                $pfxPath = $tempPfx
            }
            if (-not (Test-Path $pfxPath)) { throw "No se encontró el certificado PFX: $pfxPath" }
            $timestampUrl = $env:KELLYCLIP_SIGN_TIMESTAMP_URL
            if (-not $timestampUrl) { $timestampUrl = 'http://timestamp.digicert.com' }
            $arguments = @('sign', '/fd', 'SHA256', '/f', $pfxPath)
            if ($env:KELLYCLIP_SIGN_PFX_PASSWORD) { $arguments += '/p'; $arguments += $env:KELLYCLIP_SIGN_PFX_PASSWORD }
            $arguments += '/tr'; $arguments += $timestampUrl; $arguments += '/td'; $arguments += 'SHA256'; $arguments += '/v'
            $arguments += $FilePath
            & $signtool @arguments
            if ($LASTEXITCODE -ne 0) { throw "signtool no pudo firmar $FilePath." }
            Write-Host "Firmado con certificado PFX: $FilePath"
        } finally {
            if ($tempPfx -and (Test-Path $tempPfx)) { Remove-Item $tempPfx -Force }
        }
        return
    }

    # AzureTrustedSigning: requiere la CLI azcodesign y autenticación de
    # Azure (variables AZURE_CLIENT_ID, AZURE_CLIENT_SECRET y
    # AZURE_TENANT_ID, o `az login`).
    $azcodesign = (Get-Command azcodesign.exe -ErrorAction SilentlyContinue).Source
    if (-not $azcodesign) { throw 'La firma con Azure Trusted Signing está configurada pero no se encontró azcodesign.exe (dotnet tool install --global Azure.CodeSigning.CommandLine).' }
    & $azcodesign sign `
        -Endpoint $env:KELLYCLIP_AZURE_TRUSTED_SIGNING_ENDPOINT `
        -CodeSigningAccountName $env:KELLYCLIP_AZURE_TRUSTED_SIGNING_ACCOUNT `
        -CertificateProfileName $env:KELLYCLIP_AZURE_TRUSTED_SIGNING_CERT_PROFILE `
        -Files $FilePath `
        -FileDigest SHA256 `
        -TimestampRfc3161 'http://timestamp.acs.microsoft.com' `
        -TimestampDigest SHA256
    if ($LASTEXITCODE -ne 0) { throw "Azure Trusted Signing no pudo firmar $FilePath." }
    Write-Host "Firmado con Azure Trusted Signing: $FilePath"
}

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
    (Join-Path $root 'backend\models\face_landmarker.task'),
    (Join-Path $root 'backend\models\LICENSE'),
    (Join-Path $root 'backend\models\MEDIAPIPE-LICENSE.txt')
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

# Tamaño del bundle (revisado en la Fase 2, v1.1.0):
# - Los 8 --collect-all son necesarios: faster_whisper (assets de voz),
#   ctranslate2 (librerías nativas), av (DLL de FFmpeg de PyAV), yt_dlp
#   (extractores cargados dinámicamente), uvicorn (loops y protocolos
#   cargados por importlib), webview (assets JS y backends), cv2
#   (binario y datos) y mediapipe (binario y modelos). Recortar alguno
#   rompería el bundle o ahorraría menos de 1 MB (uvicorn).
# - --exclude-module solo incluye módulos verificados como no usados en
#   runtime: tkinter (nada lo importa; matplotlib usa el backend Agg
#   cuando Tk no está disponible) y los bindings Qt (no instalados; la
#   app usa EdgeChromium vía pywebview en Windows, nunca Qt).
#   Deliberadamente NO excluidos: matplotlib (lo importa en runtime
#   mediapipe.tasks.python.vision -> drawing_utils -> pyplot, necesario
#   para "Seguir al orador") y PIL (matplotlib.image lo importa al
#   arrancar pyplot). Excluirlos rompería el seguimiento de orador.
# - Sin UPX a propósito (--noupx): el instalador ya comprime el paquete
#   con lzma2 sólido de Inno Setup, así que UPX apenas reduciría la
#   descarga; los ejecutables comprimidos con UPX son un disparador
#   habitual de falsos positivos de antivirus (más grave mientras el
#   instalador no está firmado); y --noupx hace el build reproducible
#   en cualquier PC, tenga o no UPX instalado.
& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --windowed `
    --noupx `
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
    --collect-all mediapipe `
    --exclude-module tkinter `
    --exclude-module PyQt5 `
    --exclude-module PyQt6 `
    --exclude-module PySide2 `
    --exclude-module PySide6 `
    --hidden-import uvicorn.logging `
    --hidden-import uvicorn.loops.auto `
    --hidden-import uvicorn.protocols.http.auto `
    --hidden-import uvicorn.lifespan.on `
    --add-data "$(Join-Path $appDir 'dist');web" `
    --add-data "$(Join-Path $root '.env.example');config" `
    --add-data "$(Join-Path $root 'backend\models\face_detection_yunet_2023mar.onnx');backend/models" `
    --add-data "$(Join-Path $root 'backend\models\face_landmarker.task');backend/models" `
    --add-data "$(Join-Path $root 'backend\models\LICENSE');backend/models" `
    --add-data "$(Join-Path $root 'backend\models\MEDIAPIPE-LICENSE.txt');backend/models" `
    --add-binary "$(Join-Path $ffmpegBin 'ffmpeg.exe');bin" `
    --add-binary "$(Join-Path $ffmpegBin 'ffprobe.exe');bin" `
    (Join-Path $root 'desktop.py')
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller no pudo empaquetar la aplicación.' }

Invoke-CodeSigning -FilePath (Join-Path $distDir 'KellyClipAi\KellyClipAi.exe')

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
Invoke-CodeSigning -FilePath $installer
Write-Host "Instalador creado: $installer"
