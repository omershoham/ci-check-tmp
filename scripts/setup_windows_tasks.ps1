# NOTE (2026-09-03): the project moved to ON-DEMAND operation - see scripts\launch.ps1
# and the Desktop "Claude Trader" button. The tasks below are currently DISABLED by
# scripts\stop_automation.ps1. Running this script RE-REGISTERS them in the Ready state
# and recreates the dashboard logon shortcut, i.e. it puts the always-on setup back.
# To simply re-enable what already exists, use scripts\restore_automation.ps1 instead.
#
# Registers two Windows Scheduled Tasks for Claude Trader:
#   - "ClaudeTrader Deep"  : evening full pipeline (incl. Claude deep research)
#   - "ClaudeTrader Light" : pre-market quick refresh (no Claude)
#
# Auth: the pipeline loads .env (CLAUDE_CODE_OAUTH_TOKEN) automatically, so no
# machine env var is needed. Make sure ANTHROPIC_API_KEY is NOT set anywhere.
#
# Run this from the project root in an elevated PowerShell:
#   .\scripts\setup_windows_tasks.ps1
#
# Times are in the machine's LOCAL time. Aim for deep ~07:00 ET (US pre-market) and
# light ~16:15 ET (just after the close), converted to your timezone, and pass them
# as -DeepTime / -LightTime. The defaults below are only placeholders.
# Keep these in sync with schedule.* in config.yaml.

param(
    [string]$DeepTime  = "14:00",
    [string]$LightTime = "23:15"
)

$ErrorActionPreference = "Stop"
$root   = (Resolve-Path "$PSScriptRoot\..").Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$logdir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null

if (-not (Test-Path $python)) {
    throw "venv python not found at $python. Create it: python -m venv .venv; pip install -r requirements.txt"
}

function Register-TraderTask([string]$Name, [string]$Mode, [string]$Time) {
    # Run via cmd.exe so stdout/stderr are redirected to a log file — Task Scheduler
    # otherwise discards them and pipeline failures are invisible.
    $log = Join-Path $logdir "$Mode.log"
    # NOTE the doubled outer quotes: cmd /c mangles a line that BEGINS with a quoted
    # path and contains further quotes unless the whole command is wrapped again.
    $inner = "`"$python`" -m scripts.run_daily --mode $Mode >> `"$log`" 2>&1"
    $action = New-ScheduledTaskAction -Execute "cmd.exe" `
        -Argument "/c `"$inner`"" -WorkingDirectory $root
    # Weekdays only (Mon-Fri); the run script additionally skips NYSE holidays.
    $trigger = New-ScheduledTaskTrigger -Weekly `
        -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $Time
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun `
        -DontStopOnIdleEnd -ExecutionTimeLimit (New-TimeSpan -Hours 2)
    # NOTE: registered with an interactive token (your current user), so the task runs
    # when you are logged on. For run-when-logged-out, re-register with:
    #   -Principal (New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited)
    Register-ScheduledTask -TaskName $Name -Action $action -Trigger $trigger `
        -Settings $settings -Description "Claude Trader $Mode run" -Force | Out-Null
    Write-Host "Registered '$Name' -> $Mode @ $Time  (logs: $log)"
}

Register-TraderTask -Name "ClaudeTrader Deep"  -Mode "deep"  -Time $DeepTime
Register-TraderTask -Name "ClaudeTrader Light" -Mode "light" -Time $LightTime

# IB Gateway watchdog: relaunch via IBC when port 4001 is dead (every 30 min).
$wdAction = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$root\scripts\gateway_watchdog.ps1`""
$wdTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) `
    -RepetitionInterval (New-TimeSpan -Minutes 30)
$wdSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5)
Register-ScheduledTask -TaskName "ClaudeTrader Gateway Watchdog" -Action $wdAction `
    -Trigger $wdTrigger -Settings $wdSettings `
    -Description "Relaunch IB Gateway via IBC if port 4001 is dead" -Force | Out-Null
Write-Host "Registered 'ClaudeTrader Gateway Watchdog' (every 30 min)"

# Monday-morning digest.
$dgLog = Join-Path $logdir "digest.log"
$dgInner = "`"$python`" -m scripts.weekly_digest >> `"$dgLog`" 2>&1"
$dgAction = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$dgInner`"" -WorkingDirectory $root
$dgTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At "09:00"
$dgSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15)
Register-ScheduledTask -TaskName "ClaudeTrader Weekly Digest" -Action $dgAction `
    -Trigger $dgTrigger -Settings $dgSettings -Description "Monday morning digest" -Force | Out-Null
Write-Host "Registered 'ClaudeTrader Weekly Digest' (Mon 09:00, logs: $dgLog)"

# Always-on dashboard: a Startup-folder shortcut (no admin needed) so the UI
# launches hidden at every logon and stays running.
$startup = [Environment]::GetFolderPath('Startup')
$lnkPath = Join-Path $startup "ClaudeTrader Dashboard.lnk"
$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut($lnkPath)
$lnk.TargetPath = "powershell.exe"
$lnk.Arguments = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$root\scripts\start_dashboard.ps1`""
$lnk.WorkingDirectory = $root
$lnk.WindowStyle = 7   # minimized
$lnk.Description = "Claude Trader always-on dashboard"
$lnk.Save()
Write-Host "Created Startup shortcut -> http://localhost:8501 (launches at logon): $lnkPath"

Write-Host ""
Write-Host "Done. Inspect with: Get-ScheduledTask -TaskName 'ClaudeTrader*'"
Write-Host "Run now to test:    Start-ScheduledTask -TaskName 'ClaudeTrader Light'"
Write-Host "Start the UI now:    & `"$root\scripts\start_dashboard.ps1`"  (or just log off/on)"
