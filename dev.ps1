$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$envFile = Join-Path $root '.env'
$exampleFile = Join-Path $root '.env.example'
$appDir = Join-Path $root 'app'

if (-not (Test-Path $python)) {
    throw 'No existe .venv. Sigue los pasos de instalación de README.md.'
}
if (-not (Test-Path $envFile)) {
    Copy-Item -LiteralPath $exampleFile -Destination $envFile
    Write-Host 'Se creó .env desde .env.example. Añade tu clave LLM_API_KEY antes de procesar.'
}

$powershell = Join-Path $PSHOME 'powershell.exe'
$backendCommand = "Set-Location -LiteralPath '$root'; & '$python' -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000"
$frontendCommand = "Set-Location -LiteralPath '$appDir'; npm run dev -- --host 127.0.0.1"

Start-Process -FilePath $powershell -ArgumentList @('-NoExit', '-Command', $backendCommand)
Start-Process -FilePath $powershell -ArgumentList @('-NoExit', '-Command', $frontendCommand)
Write-Host 'Backend: http://127.0.0.1:8000  |  Frontend: http://127.0.0.1:5173'
