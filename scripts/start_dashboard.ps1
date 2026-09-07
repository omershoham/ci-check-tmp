# Launches the Streamlit dashboard as a persistent local server on port 8501.
# Idempotent: if something is already serving on 8501, it exits without starting a second copy.
# Called by scripts\launch.ps1 (the on-demand button) in a hidden window; because that
# window is invisible, all output is also appended to logs\dashboard.log so a failed
# start is diagnosable. (Formerly the "ClaudeTrader Dashboard" logon task/shortcut.)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..").Path
$streamlit = Join-Path $root ".venv\Scripts\streamlit.exe"
$port = 8501

$busy = Test-NetConnection -ComputerName 127.0.0.1 -Port $port -WarningAction SilentlyContinue
if ($busy.TcpTestSucceeded) {
    Write-Host "Dashboard already running at http://localhost:$port"
    exit 0
}

$log = Join-Path $root "logs\dashboard.log"
New-Item -ItemType Directory -Force -Path (Join-Path $root "logs") | Out-Null

$env:PYTHONIOENCODING = "utf-8"
Write-Host "Starting dashboard at http://localhost:$port ..."
Add-Content -Path $log -Value ("`n===== dashboard start {0} =====" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
# Streamlit writes progress to stderr; Continue keeps that from tripping the Stop preference.
$ErrorActionPreference = "Continue"
& $streamlit run (Join-Path $root "app\dashboard.py") `
    --server.headless true --server.port $port --server.address 127.0.0.1 `
    --browser.gatherUsageStats false *>> $log
