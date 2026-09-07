# Undoes scripts\stop_automation.ps1: re-enables the four scheduled tasks and
# puts the Startup-folder shortcuts back, restoring the old always-on behaviour.
#
#   .\scripts\restore_automation.ps1

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

Write-Host ""
Write-Host "Re-enabling scheduled tasks" -ForegroundColor White
foreach ($t in $tasks) {
    $task = Get-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue
    if (-not $task) {
        Write-Host "  - $t (not registered - re-run setup_windows_tasks.ps1)" -ForegroundColor DarkGray
        continue
    }
    Enable-ScheduledTask -TaskName $t -ErrorAction Stop | Out-Null
    Write-Host "  + $t -> Ready" -ForegroundColor Green
}

Write-Host ""
Write-Host "Restoring Startup shortcuts" -ForegroundColor White
if (Test-Path $parked) {
    foreach ($f in (Get-ChildItem -Path $parked -Filter *.lnk -ErrorAction SilentlyContinue)) {
        Move-Item -Path $f.FullName -Destination (Join-Path $startup $f.Name) -Force
        Write-Host "  + $($f.Name) -> Startup" -ForegroundColor Green
    }
} else {
    Write-Host "  - nothing parked in $parked" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "Automation restored. The Desktop button still works alongside it." -ForegroundColor Green
Write-Host ""
