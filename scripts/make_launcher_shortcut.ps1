# Creates "Claude Trader.lnk" on the Desktop - the on-demand button that runs
# scripts\launch.ps1 (starts Ollama + IB Gateway + dashboard, then shows a menu).
#
# Idempotent: re-run any time to repoint or refresh the shortcut.
#   .\scripts\make_launcher_shortcut.ps1
#   .\scripts\make_launcher_shortcut.ps1 -AlsoStartMenu   # add a Start Menu copy too

param([switch]$AlsoStartMenu)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..").Path

# Borrow the IB Gateway icon so the button is recognisable on the Desktop;
# fall back to a generic shell icon if the standalone install moved.
$icon = ""
foreach ($cand in @(
    (Join-Path $env:LOCALAPPDATA "Programs\ibgateway\ibgateway.exe"),
    "C:\Jts\ibgateway\1045\ibgateway.exe"
)) {
    if (Test-Path $cand) { $icon = $cand; break }
}
if (-not $icon) { $icon = "$env:SystemRoot\System32\shell32.dll,137" }

function New-LauncherShortcut([string]$Path) {
    $ws  = New-Object -ComObject WScript.Shell
    $lnk = $ws.CreateShortcut($Path)
    $lnk.TargetPath = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
    # -NoExit keeps the window open if the script exits early so errors stay readable.
    $lnk.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$root\scripts\launch.ps1`""
    $lnk.WorkingDirectory = $root
    $lnk.WindowStyle = 1          # normal window - the menu needs to be visible
    $lnk.IconLocation = $icon
    $lnk.Description = "Start Claude Trader on demand (Ollama + IB Gateway + dashboard)"
    $lnk.Save()
    Write-Host "  + $Path" -ForegroundColor Green
}

Write-Host ""
Write-Host "Creating the on-demand button" -ForegroundColor White
New-LauncherShortcut (Join-Path ([Environment]::GetFolderPath('Desktop')) "Claude Trader.lnk")

if ($AlsoStartMenu) {
    $programs = Join-Path ([Environment]::GetFolderPath('StartMenu')) "Programs"
    New-LauncherShortcut (Join-Path $programs "Claude Trader.lnk")
}

Write-Host ""
Write-Host "Done. Double-click 'Claude Trader' on the Desktop to start everything." -ForegroundColor Green
Write-Host "Tip: right-click it -> Pin to taskbar for one-click access." -ForegroundColor DarkGray
Write-Host ""
