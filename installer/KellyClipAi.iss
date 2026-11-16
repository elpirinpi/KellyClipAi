#ifndef AppVersion
  #define AppVersion "1.0.2"
#endif

#define AppName "kellyClipAi"
#define AppPublisher "KELLY BENJAMIN DE LA CRUZ"

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
Filename: "{tmp}\MicrosoftEdgeWebView2Setup.exe"; Parameters: "/silent /install"; StatusMsg: "Instalando el runtime de Microsoft Edge WebView2..."; Flags: runhidden waituntilterminated
Filename: "{app}\KellyClipAi.exe"; Description: "Iniciar kellyClipAi"; Flags: postinstall nowait skipifsilent
