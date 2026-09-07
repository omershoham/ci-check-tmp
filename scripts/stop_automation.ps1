# Retires every unattended Claude Trader process, reversibly.
#
# Disables (does not delete) the four scheduled tasks and moves the three
# Startup-folder shortcuts into scripts\disabled_startup\ so nothing launches at
# logon. After this the only way anything starts is the Desktop button ->
# scripts\launch.ps1. Undo with scripts\restore_automation.ps1.
#
#   .\scripts\stop_automation.ps1            # disable + park shortcuts
#   .\scripts\stop_automation.ps1 -KeepRunning   # leave already-running procs alive

param([switch]$KeepRunning)

$ErrorActionPreference = "Stop"
$root    = (Resolve-Path "$PSScriptRoot\..").Path
$parked  = Join-Path $root "scripts\disabled_startup"
$startup = [Environment]::GetFolderPath('Startup')

$tasks = @(
    "ClaudeTrader Deep",
    "ClaudeTrader Light",
    "ClaudeTrader Gateway Watchdog",
    "ClaudeTrader Weekly Digest"
)
$shortcuts = @("ClaudeTrader Dashboard.lnk", "IBC Gateway.lnk", "Ollama.lnk")

New-Item -ItemType Directory -Force -Path $parked | Out-Null

Write-Host ""
Write-Host "Disabling scheduled tasks" -ForegroundColor White
foreach ($t in $tasks) {
    $task = Get-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue
    if (-not $task) { Write-Host "  - $t (not registered)" -ForegroundColor DarkGray; continue }
    # Stop a run in flight first, otherwise Disable leaves it going.
    try { Stop-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue } catch {}
    Disable-ScheduledTask -TaskName $t -ErrorAction Stop | Out-Null
    Write-Host "  x $t -> Disabled" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Parking Startup shortcuts in scripts\disabled_startup" -ForegroundColor White
foreach ($s in $shortcuts) {
    $src = Join-Path $startup $s
    if (-not (Test-Path $src)) { Write-Host "  - $s (not present)" -ForegroundColor DarkGray; continue }
    Move-Item -Path $src -Destination (Join-Path $parked $s) -Force
    Write-Host "  x $s -> parked" -ForegroundColor Yellow
}

if (-not $KeepRunning) {
    Write-Host ""
    Write-Host "Note: anything already running (Gateway, dashboard, Ollama) is untouched." -ForegroundColor DarkGray
    Write-Host "      Use option [5] in the launcher menu to shut the stack down." -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "Automation retired. Nothing will start on a timer or at logon." -ForegroundColor Green
Write-Host "Start on demand with the 'Claude Trader' Desktop shortcut," -ForegroundColor Green
Write-Host "or:  powershell -ExecutionPolicy Bypass -File `"$root\scripts\launch.ps1`"" -ForegroundColor Green
Write-Host "Undo with: .\scripts\restore_automation.ps1" -ForegroundColor DarkGray
Write-Host ""
