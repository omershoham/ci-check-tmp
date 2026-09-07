# Claude Trader - on-demand launcher ("the button").
#
# Replaces the old always-on setup (scheduled Deep/Light/Watchdog/Digest tasks +
# Startup-folder shortcuts, all retired by scripts/stop_automation.ps1). Nothing
# runs on a timer any more: press the Desktop shortcut when you want to work.
#
# What it does, in order, skipping anything already up:
#   1. Ollama          (port 11434) - local LLM for the sentiment pillar
#   2. IB Gateway/IBC  (port 4001)  - holdings + quotes; may need a 2FA phone push
#   3. Streamlit UI    (port 8501)  - the dashboard, opened in your browser
# then offers a menu so a research run only happens when you ask for one.
#
# Gateway and dashboard are started DETACHED - closing this window leaves them up.
#
# Usage:
#   .\scripts\launch.ps1                 # start stack, open browser, show menu
#   .\scripts\launch.ps1 -Run deep       # start stack then run the deep pipeline
#   .\scripts\launch.ps1 -NoMenu         # start stack only, then exit
#   .\scripts\launch.ps1 -NoBrowser      # don't open a browser tab

param(
    [ValidateSet("", "deep", "light", "digest")] [string]$Run = "",
    [switch]$NoMenu,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$root       = (Resolve-Path "$PSScriptRoot\..").Path
$python     = Join-Path $root ".venv\Scripts\python.exe"
$logdir     = Join-Path $root "logs"
$ollamaExe  = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama app.exe"
$gatewayBat = "C:\IBC\StartGateway.bat"
$OLLAMA_PORT = 11434
$GW_PORT     = 4001
$UI_PORT     = 8501

New-Item -ItemType Directory -Force -Path $logdir | Out-Null

# ---------------------------------------------------------------- helpers ----

# Fast port probe. Test-NetConnection costs 1-2s per miss; a raw TcpClient is instant.
function Test-Port([int]$Port, [int]$TimeoutMs = 600) {
    $client = New-Object System.Net.Sockets.TcpClient
    try   { $ok = $client.ConnectAsync("127.0.0.1", $Port).Wait($TimeoutMs) }
    catch { $ok = $false }
    finally { $client.Close() }
    return $ok
}

# Poll a port until it answers, printing dots so a slow login doesn't look hung.
function Wait-Port([int]$Port, [int]$TimeoutSec) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-Port $Port) { return $true }
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 2
    }
    return $false
}

function Write-Line([string]$Label, [string]$Value, [string]$Color = "Gray") {
    $dots = "." * [Math]::Max(1, 18 - $Label.Length)
    Write-Host ("  {0} {1} " -f $Label, $dots) -NoNewline
    Write-Host $Value -ForegroundColor $Color
}

# True if an IBC/Gateway java process is alive - it may be mid-login (2FA wait),
# and stacking a second instance on top of it causes a session conflict.
function Test-GatewayProcess {
    foreach ($p in (Get-Process -Name "java", "javaw", "ibgateway" -ErrorAction SilentlyContinue)) {
        try {
            $cmdline = (Get-CimInstance Win32_Process -Filter "ProcessId=$($p.Id)" -ErrorAction SilentlyContinue).CommandLine
            if ($cmdline -match "ibcalpha|IBC\.jar|ibgateway") { return $true }
        } catch {}
    }
    return $false
}

# ------------------------------------------------------------- components ----

function Start-Ollama {
    if (Test-Port $OLLAMA_PORT) { Write-Line "Ollama" "up" "Green"; return $true }
    if (-not (Test-Path $ollamaExe)) {
        Write-Line "Ollama" "not installed at $ollamaExe" "Red"
        return $false
    }
    Write-Host "  Ollama ............ starting" -NoNewline
    Start-Process -FilePath $ollamaExe -WindowStyle Hidden
    $ok = Wait-Port $OLLAMA_PORT 40
    Write-Host ""
    if ($ok) { Write-Line "Ollama" "up" "Green" }
    else     { Write-Line "Ollama" "did NOT come up - sentiment pillar falls back" "Yellow" }
    return $ok
}

function Start-Gateway {
    if (Test-Port $GW_PORT) { Write-Line "IB Gateway" "up (port $GW_PORT)" "Green"; return $true }

    if (-not (Test-Path $gatewayBat)) {
        Write-Line "IB Gateway" "IBC not found at $gatewayBat" "Red"
        return $false
    }

    # The Gateway JVM sometimes dies within a minute of launch, logging nothing -
    # seen right after a second-factor timeout tears down the previous session
    # (2026-09-03: two watchdog relaunches and one manual launch died that way,
    # the next attempt logged in cleanly). So retry rather than give up.
    $maxAttempts = 3
    for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {

        if (Test-GatewayProcess) {
            # A JVM is already alive - never stack a second instance on it, that
            # is what causes IBKR session conflicts. Just wait it out.
            Write-Host "  IB Gateway ........ login in progress" -NoNewline
        } else {
            if ($attempt -eq 1) { Write-Host "  IB Gateway ........ launching IBC" -NoNewline }
            else                { Write-Host "  IB Gateway ........ retry $attempt of $maxAttempts" -NoNewline }
            # Redirect the batch output so a silent death leaves something behind.
            Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $gatewayBat `
                -WorkingDirectory "C:\IBC" -WindowStyle Minimized `
                -RedirectStandardOutput (Join-Path $logdir "gateway.out.log") `
                -RedirectStandardError  (Join-Path $logdir "gateway.err.log")
        }

        # A clean login takes ~40-60s.
        if (Wait-Port $GW_PORT 80) { Write-Host ""; Write-Line "IB Gateway" "up (port $GW_PORT)" "Green"; return $true }

        if (Test-GatewayProcess) {
            # Alive but no port: almost certainly sitting on the IB Key 2FA dialog.
            # Give it a long grace window instead of relaunching underneath it.
            if (Wait-Port $GW_PORT 120) { Write-Host ""; Write-Line "IB Gateway" "up (port $GW_PORT)" "Green"; return $true }
            if (Test-GatewayProcess) { break }   # still waiting on the phone - stop retrying
        }
        Write-Host ""
    }

    Write-Host ""
    Write-Line "IB Gateway" "no API port yet" "Yellow"
    Write-Host "     Check your phone for an IB Key push, or the Gateway window for a dialog." -ForegroundColor Yellow
    Write-Host "     The app still works without it - holdings/quotes just stay blank." -ForegroundColor DarkGray
    Write-Host "     Menu option [6] retries the Gateway on its own." -ForegroundColor DarkGray
    return $false
}

function Start-Dashboard {
    if (Test-Port $UI_PORT) { Write-Line "Dashboard" "http://localhost:$UI_PORT" "Green"; return $true }
    if (-not (Test-Path $python)) {
        Write-Line "Dashboard" "venv missing at $python" "Red"
        return $false
    }
    Write-Host "  Dashboard ......... starting" -NoNewline
    # Detached + hidden so it outlives this window, exactly like the old Startup shortcut.
    Start-Process -FilePath "powershell.exe" `
        -ArgumentList "-NoProfile", "-WindowStyle", "Hidden", "-ExecutionPolicy", "Bypass", `
                      "-File", "$root\scripts\start_dashboard.ps1" `
        -WorkingDirectory $root -WindowStyle Hidden
    $ok = Wait-Port $UI_PORT 90
    Write-Host ""
    if ($ok) { Write-Line "Dashboard" "http://localhost:$UI_PORT" "Green" }
    else     { Write-Line "Dashboard" "did NOT come up" "Red" }
    return $ok
}

# Ask the pipeline's own calendar whether the US market trades today, so an
# on-demand press on a weekend/holiday still runs instead of silently skipping.
function Get-ClosedReason {
    if (-not (Test-Path $python)) { return $null }
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    Push-Location $root
    try {
        $code = "from src import market_calendar as m; print('' if m.is_trading_day() else (m.reason_closed() or 'market closed'))"
        $out = & $python -c $code
        if ($LASTEXITCODE -ne 0) { return $null }
        $reason = ($out | Where-Object { $_ } | Select-Object -Last 1)
        if ($reason) { return $reason.ToString().Trim() }
        return ""
    } catch {
        return $null
    } finally {
        Pop-Location
        $ErrorActionPreference = $prev
    }
}

# Freshness + data-health line for the menu: "#115 light 3h ago - health degraded: broker ...".
# The five-week blind spot happened because nothing on screen said the runs were
# portfolio-blind; this puts it in front of you every time you press the button.
function Get-LastRunLine {
    if (-not (Test-Path $python)) { return $null }
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    Push-Location $root
    try {
        # SINGLE quotes only inside this Python: PowerShell 5.1 strips embedded double
        # quotes when it builds the argv for a native exe (Get-ClosedReason has the
        # same constraint). Keys are pulled into variables to keep f-strings simple.
        $code = @'
from datetime import datetime, timezone
from src import storage, health
runs = storage.list_runs(1)
if not runs:
    print('no runs yet')
else:
    r = runs[0]
    keys = r.keys()
    rid, mode, status = r['id'], r['mode'], r['status']
    try:
        h = (datetime.now(timezone.utc) - datetime.fromisoformat(r['started_at'])).total_seconds() / 3600
        age = '%.0fh ago' % h if h < 48 else '%.0fd ago' % (h / 24)
    except Exception:
        age = '?'
    hv = (r['health'] if 'health' in keys else None) or 'unknown'
    hz = health.RunHealth.from_json(r['health_json'] if 'health_json' in keys else None)
    line = '#%s %s %s status=%s health=%s' % (rid, mode, age, status, hv)
    probs = hz.problems()
    if probs:
        line += ' | ' + '; '.join('%s: %s' % (n, note) for n, _, note in probs)
    print(line[:230])
'@
        $out = & $python -c $code
        if ($LASTEXITCODE -ne 0) { return $null }
        return ($out | Where-Object { $_ } | Select-Object -Last 1)
    } catch { return $null }
    finally { Pop-Location; $ErrorActionPreference = $prev }
}

function Invoke-Run([string]$Mode, [string]$ClosedReason) {
    $log = Join-Path $logdir "$Mode.log"
    $force = if ($ClosedReason) { " --force" } else { "" }   # a deliberate press beats the calendar skip
    # Merge stderr into stdout inside cmd, not PowerShell: a native `2>&1` makes PS
    # wrap every stderr line in a red NativeCommandError block, which turns routine
    # yfinance 404s into alarming noise. NOTE the doubled outer quotes - cmd mangles
    # a command that begins with a quoted path unless the whole thing is requoted.
    $inner = "`"$python`" -m scripts.run_daily --mode $Mode$force 2>&1"

    Write-Host ""
    Write-Host "  Running $Mode pipeline (live output, also appended to logs\$Mode.log)" -ForegroundColor Cyan
    Write-Host "  Ctrl+C aborts. ---------------------------------------------------" -ForegroundColor DarkGray

    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONUNBUFFERED = "1"
    Push-Location $root
    try {
        Add-Content -Path $log -Value ("`n===== manual $Mode run {0} =====" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
        & cmd.exe /c $inner | Tee-Object -FilePath $log -Append
    } finally {
        Pop-Location
        $ErrorActionPreference = $prev
    }

    Write-Host "  ------------------------------------------------------------------" -ForegroundColor DarkGray
    Write-Host "  Done. Refresh the dashboard tab to pick up the new run." -ForegroundColor Green
}

function Invoke-Digest {
    $log = Join-Path $logdir "digest.log"
    Write-Host ""
    Write-Host "  Building weekly digest ..." -ForegroundColor Cyan
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $env:PYTHONIOENCODING = "utf-8"
    Push-Location $root
    try {
        & cmd.exe /c "`"$python`" -m scripts.weekly_digest 2>&1" | Tee-Object -FilePath $log -Append
    } finally {
        Pop-Location
        $ErrorActionPreference = $prev
    }
    Write-Host "  Done." -ForegroundColor Green
}

# Stop what this launcher started. Ollama is left alone - it is a harmless tray
# app that other things may be using.
function Stop-Stack {
    Write-Host ""
    try {
        $owners = Get-NetTCPConnection -LocalPort $UI_PORT -State Listen -ErrorAction SilentlyContinue |
                  Select-Object -ExpandProperty OwningProcess -Unique
        if ($owners) {
            foreach ($owner in $owners) {
                Stop-Process -Id $owner -Force -ErrorAction SilentlyContinue
                Write-Host "  Stopped dashboard (pid $owner)." -ForegroundColor Yellow
            }
        } else {
            Write-Host "  Dashboard was not running." -ForegroundColor DarkGray
        }
    } catch {
        Write-Host "  Could not stop the dashboard: $_" -ForegroundColor Red
    }

    $killed = $false
    foreach ($p in (Get-Process -Name "java", "javaw", "ibgateway" -ErrorAction SilentlyContinue)) {
        try {
            $cmdline = (Get-CimInstance Win32_Process -Filter "ProcessId=$($p.Id)" -ErrorAction SilentlyContinue).CommandLine
            if ($cmdline -match "ibcalpha|IBC\.jar|ibgateway") {
                Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
                Write-Host "  Stopped IB Gateway/IBC (pid $($p.Id))." -ForegroundColor Yellow
                $killed = $true
            }
        } catch {}
    }
    if (-not $killed) { Write-Host "  IB Gateway was not running." -ForegroundColor DarkGray }
    Write-Host "  Ollama left running (tray app)." -ForegroundColor DarkGray
}

# ------------------------------------------------------------------- main ----

try { Clear-Host } catch {}   # fails harmlessly when output is redirected
Write-Host ""
Write-Host "  Claude Trader - starting up" -ForegroundColor White
Write-Host "  ---------------------------------"
Start-Ollama  | Out-Null
Start-Gateway | Out-Null
$uiOk = Start-Dashboard
Write-Host ""

if ($uiOk -and -not $NoBrowser) { Start-Process "http://localhost:$UI_PORT" }

$closed = Get-ClosedReason
if ($closed) {
    Write-Host "  Note: US market closed today ($closed) - runs will use --force." -ForegroundColor DarkYellow
    Write-Host ""
}

if ($Run) {
    if ($Run -eq "digest") { Invoke-Digest } else { Invoke-Run $Run $closed }
    if ($NoMenu) { exit 0 }
}

if ($NoMenu) {
    Write-Host "  Stack is up. This window can close - everything keeps running." -ForegroundColor DarkGray
    exit 0
}

while ($true) {
    $ollamaUp = Test-Port $OLLAMA_PORT
    $gwUp     = Test-Port $GW_PORT
    $uiUp     = Test-Port $UI_PORT

    Write-Host "  Claude Trader - ready" -ForegroundColor White
    Write-Host "  ---------------------------------"
    if ($ollamaUp) { Write-Line "Ollama" "up" "Green" } else { Write-Line "Ollama" "down" "Yellow" }
    if ($gwUp) { Write-Line "IB Gateway" "up (port $GW_PORT)" "Green" } else { Write-Line "IB Gateway" "down" "Yellow" }
    if ($uiUp) { Write-Line "Dashboard" "http://localhost:$UI_PORT" "Green" } else { Write-Line "Dashboard" "down" "Yellow" }
    $last = Get-LastRunLine
    if ($last) {
        $color = if ($last -match "health=ok") { "Green" } elseif ($last -match "health=failed|status=error") { "Red" } else { "Yellow" }
        Write-Line "Last run" $last $color
    }
    Write-Host ""
    Write-Host "  [1] Deep research run (Claude, ~15 min)"
    Write-Host "  [2] Light refresh (no Claude)"
    Write-Host "  [3] Weekly digest"
    Write-Host "  [4] Open dashboard in browser"
    Write-Host "  [5] Shut everything down (Gateway + dashboard)"
    Write-Host "  [6] Retry IB Gateway"
    Write-Host "  [Enter] Just leave it running"
    Write-Host ""
    $choice = Read-Host "  >"

    switch ($choice.Trim()) {
        "1" { Invoke-Run "deep"  $closed }
        "2" { Invoke-Run "light" $closed }
        "3" { Invoke-Digest }
        "4" { Start-Process "http://localhost:$UI_PORT" }
        "5" { Stop-Stack; Write-Host ""; exit 0 }
        "6" { Write-Host ""; Start-Gateway | Out-Null }
        ""  {
            Write-Host ""
            Write-Host "  Left running. Dashboard: http://localhost:$UI_PORT" -ForegroundColor Green
            Write-Host "  Press the shortcut again any time." -ForegroundColor DarkGray
            Write-Host ""
            exit 0
        }
        default { Write-Host "  Unknown option '$choice'." -ForegroundColor Yellow }
    }
    Write-Host ""
}
