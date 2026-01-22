[Setup]
AppName=TradeCopier
AppVersion=2.0.3
AppPublisher=bbstrading
DefaultDirName={pf}\TradeCopier
DefaultGroupName=TradeCopier
OutputDir=.
OutputBaseFilename=TradeCopier
Compression=lzma
SolidCompression=yes
LicenseFile=LICENSE
SetupIconFile=bbstrader\assets\bbstrader.ico

[Files]
Source: "dist\tcopier.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "bbstrader\assets\bbstrader.ico"; DestDir: "{app}"; Flags: ignoreversion
[Icons]
Name: "{group}\TradeCopier"; Filename: "{app}\tcopier.exe"; IconFilename: "{app}\bbstrader.ico"
Name: "{group}\Uninstall tradecopier"; Filename: "{uninstallexe}"
Name: "{commondesktop}\tradecopier"; Filename: "{app}\tcopier.exe"; IconFilename: "{app}\bbstrader.ico"; Tasks: desktopicon



[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"
