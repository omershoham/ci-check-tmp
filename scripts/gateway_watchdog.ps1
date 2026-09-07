# IB Gateway watchdog: relaunches IBC/Gateway if the API port is dead.
# Idempotent — exits silently if the Gateway is already up or currently starting.
# Registered as the "ClaudeTrader Gateway Watchdog" scheduled task (every 30 min).
#
# Why this exists: the Gateway can exit for reasons the Startup shortcut never heals
# (session takeover by a manual TWS/mobile login, weekly 2FA timeout, crash) — on
# 2026-06-29 a session conflict killed it and it stayed down for 3 days unnoticed.

$ErrorActionPreference = "SilentlyContinue"
$port = 4001
$log  = "C:\IBC\Logs\watchdog.log"

function Log([string]$msg) {
    Add-Content -Path $log -Value ("{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg)
}

# 1. API port answering? Then all is well.
$up = Test-NetConnection -ComputerName 127.0.0.1 -Port $port -WarningAction SilentlyContinue
if ($up.TcpTestSucceeded) { exit 0 }

# 2. Port down but a Gateway/IBC java process alive? It may be mid-login (2FA wait) —
#    don't stack a second instance on top of it.
$javas = Get-Process -Name "java","javaw","ibgateway" -ErrorAction SilentlyContinue
foreach ($p in $javas) {
    try {
        $cmdline = (Get-CimInstance Win32_Process -Filter "ProcessId=$($p.Id)").CommandLine
        if ($cmdline -match "ibcalpha|IBC\.jar|ibgateway") {
            Log "port $port down but IBC/Gateway process $($p.Id) alive (login in progress?) - waiting"
            exit 0
        }
    } catch {}
}

# 3. Nothing running: relaunch. Full auth after a long outage may trigger a 2FA push.
Log "port $port down and no Gateway process - relaunching StartGateway.bat"
Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "C:\IBC\StartGateway.bat" `
    -WorkingDirectory "C:\IBC" -WindowStyle Minimized
