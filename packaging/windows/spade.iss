; Inno Setup script of SPADE's Windows installer (packaging/windows). Built by
; packaging/windows/build_installer.py, which stages the embedded Python, the dependencies and the
; SPADE source tree into STAGE and passes the defines below to ISCC.
;
; Per-user install (no administrator rights) into %LOCALAPPDATA%\SPADE: SPADE writes its workspace,
; datasets and runs next to its sources, which must therefore be writable.

#ifndef STAGE
  #error STAGE (the staged folder with python\ and SPADE\) must be defined
#endif
#ifndef APPVERSION
  #define APPVERSION "0.1.0"
#endif
#ifndef COMMIT
  #define COMMIT "unknown"
#endif
#ifndef OUTDIR
  #define OUTDIR "."
#endif

[Setup]
AppId={{6C0F6A52-3B1E-4C55-9E0B-5B1D2D8A7E11}
AppName=SPADE
AppVersion={#APPVERSION} (commit {#COMMIT})
VersionInfoVersion={#APPVERSION}
AppPublisher=SPADE project
AppPublisherURL=https://github.com/lucianoradrigan/SPADE
DefaultDirName={localappdata}\SPADE
DefaultGroupName=SPADE
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#OUTDIR}
OutputBaseFilename=SPADE-Setup-{#APPVERSION}-{#COMMIT}
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
WizardStyle=modern
LicenseFile={#STAGE}\SPADE\LICENSE
UninstallDisplayName=SPADE {#APPVERSION} (commit {#COMMIT})

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "{#STAGE}\python\*"; DestDir: "{app}\python"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#STAGE}\SPADE\*"; DestDir: "{app}\SPADE"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\SPADE"; Filename: "{app}\python\python.exe"; Parameters: """{app}\SPADE\packaging\windows\launch.py"""; WorkingDir: "{app}\SPADE"; Comment: "Start SPADE (opens in the browser)"
Name: "{group}\SPADE folder"; Filename: "{app}\SPADE"
Name: "{group}\Uninstall SPADE"; Filename: "{uninstallexe}"
Name: "{userdesktop}\SPADE"; Filename: "{app}\python\python.exe"; Parameters: """{app}\SPADE\packaging\windows\launch.py"""; WorkingDir: "{app}\SPADE"; Comment: "Start SPADE (opens in the browser)"; Tasks: desktopicon

[Run]
Filename: "{app}\python\python.exe"; Parameters: """{app}\SPADE\packaging\windows\launch.py"""; WorkingDir: "{app}\SPADE"; Description: "Start SPADE now"; Flags: postinstall nowait skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\SPADE\src"
