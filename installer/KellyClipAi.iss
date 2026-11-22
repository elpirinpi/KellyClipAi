#ifndef AppVersion
  #define AppVersion "1.4.0"
#endif

#define AppName "kellyClipAi"
#define AppPublisher "KELLY BENJAMIN DE LA CRUZ"

; La firma de código del ejecutable y del instalador la hace
; build_windows.ps1 (signtool o Azure Trusted Signing) cuando hay
; credenciales configuradas; no se usa la directiva SignTool aquí para
; tener un único punto de configuración (ver README.md, "Firma de código").

[Setup]
AppId={{A53B9435-4242-49D5-B1D5-41DBE0D0E6D0}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Programs\kellyClipAi
DefaultGroupName={#AppName}
UninstallDisplayIcon={app}\KellyClipAi.exe
OutputDir=..\artifacts
OutputBaseFilename=KellyClipAi-Setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64
ArchitecturesAllowed=x64
DisableProgramGroupPage=yes
CloseApplications=yes
RestartApplications=no

[Tasks]
Name: "desktopicon"; Description: "Crear un acceso directo en el escritorio"; GroupDescription: "Accesos directos:"

[Files]
Source: "..\build\bundle\KellyClipAi\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\build\vendor\MicrosoftEdgeWebView2Setup.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall
Source: "..\build\vendor\COPYING.GPLv3"; DestDir: "{app}\licenses"; DestName: "GPL-3.0.txt"; Flags: ignoreversion
Source: "..\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\kellyClipAi"; Filename: "{app}\KellyClipAi.exe"
Name: "{autodesktop}\kellyClipAi"; Filename: "{app}\KellyClipAi.exe"; Tasks: desktopicon

[Run]
Filename: "{tmp}\MicrosoftEdgeWebView2Setup.exe"; Parameters: "/silent /install"; StatusMsg: "Instalando el runtime de Microsoft Edge WebView2..."; Flags: runhidden waituntilterminated; Check: NeedsWebView2
Filename: "{app}\KellyClipAi.exe"; Description: "Iniciar kellyClipAi"; Flags: postinstall nowait skipifsilent

[Code]
var
  WebView2WasMissing: Boolean;

// GUID del runtime de Microsoft Edge WebView2 Evergreen en EdgeUpdate.
// Microsoft documenta la detección por registro: la clave existe y su
// valor "pv" trae la versión instalada cuando el runtime está presente.
// Se consultan la instalación por máquina (HKLM, vista de 32 bits, donde
// registra EdgeUpdate, y vista de 64 bits por si acaso) y la instalación
// por usuario (HKCU).
function WebView2IsInstalled: Boolean;
var
  Version: string;
begin
  Result :=
    (RegQueryStringValue(HKEY_LOCAL_MACHINE, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) and (Version <> '')) or
    (RegQueryStringValue(HKEY_LOCAL_MACHINE, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) and (Version <> '')) or
    (RegQueryStringValue(HKEY_CURRENT_USER, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) and (Version <> ''));
end;

// Check: de la entrada [Run] del bootstrapper: solo se ejecuta si el
// runtime falta, así que las actualizaciones ya no re-ejecutan un
// instalador innecesario. Guarda el hecho para avisar al final si la
// instalación del bootstrapper no dejó el runtime instalado (sin
// internet, por ejemplo): [Run] no comprueba el código de salida, así
// que el fallo se detecta verificando el registro al terminar.
function NeedsWebView2: Boolean;
begin
  Result := not WebView2IsInstalled;
  if Result then
    WebView2WasMissing := True;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssDone) and WebView2WasMissing and (not WebView2IsInstalled) then
    MsgBox(
      'No se pudo instalar el runtime de Microsoft Edge WebView2, que kellyClipAi necesita para mostrar su ventana.' + #13#10 + #13#10 +
      'La aplicación se instaló, pero no arrancará hasta que instales WebView2. Descárgalo gratis desde:' + #13#10 +
      'https://developer.microsoft.com/microsoft-edge/webview2/' + #13#10 + #13#10 +
      'Si tenías internet durante la instalación y el error persiste, ejecuta de nuevo el instalador.',
      mbError, MB_OK);
end;
