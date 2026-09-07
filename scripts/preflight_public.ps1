# Pre-publication leak check. Run this BEFORE every commit/push on a public repo:
#
#   .\scripts\preflight_public.ps1
#
# It answers one question: "if I commit right now, what leaves this machine?"
#   1. resolves the exact file set git would track (honouring .gitignore)
#   2. scans ONLY that set for credentials, emails, IBKR account numbers and
#      absolute user paths
#   3. confirms the known-sensitive artefacts are still ignored
#
# Exits 1 on any finding so it can gate a hook or CI step. Works whether or not
# the project has been `git init`-ed yet (it borrows a temp git dir if not).

param([switch]$Quiet)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..").Path
Push-Location $root

$findings = New-Object System.Collections.ArrayList
function Add-Finding([string]$Severity, [string]$What, [string]$Where) {
    $null = $findings.Add([pscustomobject]@{ Severity = $Severity; What = $What; Where = $Where })
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "git not on PATH - cannot resolve the tracked file set." -ForegroundColor Red
    Pop-Location; exit 1
}

# ---- 1. what would git actually track? ----------------------------------------
$tempGit = $null
if (Test-Path (Join-Path $root ".git")) {
    $gitArgs = @()
} else {
    # No repo yet: use a throwaway git dir so we can still evaluate .gitignore
    # without creating a .git in the project.
    $tempGit = Join-Path ([System.IO.Path]::GetTempPath()) ("preflight_" + [guid]::NewGuid().ToString("N") + ".git")
    & git --git-dir=$tempGit init -q --bare
    & git --git-dir=$tempGit config core.bare false
    $gitArgs = @("--git-dir=$tempGit", "--work-tree=$root")
}

$staged = & git @gitArgs add -An . 2>$null |
          ForEach-Object { if ($_ -match "^add '(.+)'$") { $matches[1] } }
$staged = @($staged | Where-Object { $_ })

if (-not $Quiet) {
    Write-Host ""
    Write-Host "Files git would track: $($staged.Count)" -ForegroundColor White
}

# ---- 2. content scan of that set ----------------------------------------------
# Each rule: a regex plus what it means. Deliberately narrow to stay signal-heavy.
$rules = @(
    @{ Name = "Anthropic/Claude token"; Pattern = 'sk-ant-[A-Za-z0-9_\-]{10,}' },
    @{ Name = "generic secret key";     Pattern = 'sk-[A-Za-z0-9]{32,}' },
    @{ Name = "JWT";                    Pattern = 'eyJ[A-Za-z0-9_\-]{20,}' },
    # Require a secret-shaped value: this must fire on a real token but not on the
    # empty keys or the "# ANTHROPIC_API_KEY=  <-- leave unset" note in .env.example.
    @{ Name = "populated token/key var";Pattern = '(CLAUDE_CODE_OAUTH_TOKEN|ANTHROPIC_API_KEY)\s*=\s*[A-Za-z0-9_\-\.]{12,}' },
    @{ Name = "IBKR credential key";    Pattern = 'IbPassword|IbLoginId' },
    @{ Name = "IBKR account number";    Pattern = '\bU\d{7,9}\b' },
    @{ Name = "email address";          Pattern = '[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}' },
    @{ Name = "absolute user path";     Pattern = 'C:[\\/]Users[\\/][A-Za-z0-9._\-]+' },
    @{ Name = "bearer token";           Pattern = 'Bearer\s+[A-Za-z0-9._\-]{16,}' }
)
# Placeholders and doc examples that are safe by construction.
$allow = 'example\.com|your\.name@|SEC_CONTACT_EMAIL=\s*$|CLAUDE_CODE_OAUTH_TOKEN=\s*$|ANTHROPIC_API_KEY=\s*$|contact unset'

foreach ($f in $staged) {
    # This script states the detection patterns literally, so scanning it reports
    # itself. It is reviewed by hand instead.
    if (($f -replace '\\', '/') -eq "scripts/preflight_public.ps1") { continue }
    $full = Join-Path $root $f
    if (-not (Test-Path $full -PathType Leaf)) { continue }
    # Skip binaries: a NUL byte in the first 4KB.
    $head = [System.IO.File]::ReadAllBytes($full) | Select-Object -First 4096
    if ($head -contains 0) { continue }
    foreach ($rule in $rules) {
        $hits = Select-String -Path $full -Pattern $rule.Pattern -AllMatches -ErrorAction SilentlyContinue
        foreach ($h in $hits) {
            if ($h.Line -match $allow) { continue }
            Add-Finding "LEAK" $rule.Name ("{0}:{1}" -f $f, $h.LineNumber)
        }
    }
}

# ---- 3. are the known-sensitive artefacts still ignored? ----------------------
$mustIgnore = @(
    ".env",
    "data/trader.db", "data/trader.db-wal", "data/trader.db-shm",
    "data/digests", "data/cache",
    "logs/deep.log", "logs/light.log",
    ".claude/settings.local.json",
    "scripts/disabled_startup"
)
foreach ($p in $mustIgnore) {
    $full = Join-Path $root $p
    if (-not (Test-Path $full)) { continue }          # nothing to leak
    # A path is safe when nothing under it appears in the tracked set.
    $rel = $p -replace '\\', '/'
    $tracked = @($staged | Where-Object { ($_ -replace '\\', '/') -eq $rel -or ($_ -replace '\\', '/').StartsWith($rel + "/") })
    if ($tracked.Count -gt 0) {
        Add-Finding "TRACKED" "sensitive path is NOT ignored" ($tracked -join ", ")
    }
}

if ($tempGit -and (Test-Path $tempGit)) { Remove-Item $tempGit -Recurse -Force -ErrorAction SilentlyContinue }
Pop-Location

# ---- report -------------------------------------------------------------------
Write-Host ""
if ($findings.Count -eq 0) {
    Write-Host "PREFLIGHT PASS - no credentials, emails, account numbers or personal" -ForegroundColor Green
    Write-Host "                 paths in the $($staged.Count) files git would track." -ForegroundColor Green
    Write-Host ""
    exit 0
}
Write-Host "PREFLIGHT FAIL - $($findings.Count) finding(s):" -ForegroundColor Red
$findings | Format-Table -AutoSize Severity, What, Where | Out-String -Width 200 | Write-Host
Write-Host "Fix or ignore each one before committing." -ForegroundColor Yellow
Write-Host ""
exit 1
