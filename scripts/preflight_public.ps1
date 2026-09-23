# Pre-publication leak check. Run this BEFORE every commit/push on a public repo:
#
#   .\scripts\preflight_public.ps1
#
# It answers one question: "if I push right now, what leaves this machine?"
#   1. resolves the exact file set git would track (honouring .gitignore)
#   2. scans ONLY that set for credentials, emails, IBKR account numbers and
#      absolute user paths
#   3. confirms the known-sensitive artefacts are still ignored
#   4. HISTORY: every author/committer identity must be a noreply address
#   5. HISTORY: commit messages (session URLs, emails, account numbers, tokens)
#   6. HISTORY: the same content rules over EVERY file version ever committed —
#      a secret deleted in a later commit is still public in the old one
#   7. private denylist: terms in `.preflight_denylist.txt` (git-ignored; one term
#      per line, e.g. your real name, personal email, employer) are searched in the
#      files, every past file version, commit messages and identities. The list
#      lives only on your machine, so the public repo never contains the words it
#      is protecting.
#
# Exits 1 on any finding so it can gate a hook or CI step. Works whether or not
# the project has been `git init`-ed yet (it borrows a temp git dir if not; the
# history checks are skipped then — there is no history).
#
#   .\scripts\preflight_public.ps1 -NoHistory   # files only (fast pre-commit use)

param([switch]$Quiet, [switch]$NoHistory)

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

if ($tempGit) {
    # No repo: a dry-run add of everything = what a first commit would contain.
    $staged = & git @gitArgs add -An . 2>$null |
              ForEach-Object { if ($_ -match "^add '(.+)'$") { $matches[1] } }
} else {
    # Existing repo: EVERY tracked file plus every untracked file .gitignore lets
    # through. (A dry-run `add` here lists only CHANGED files, which silently
    # dropped the already-committed ones from the scan.)
    $staged = & git ls-files --cached --others --exclude-standard 2>$null
    $staged = @($staged | Where-Object { $_ -and (Test-Path (Join-Path $root $_) -PathType Leaf) })
}
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
    "scripts/disabled_startup",
    ".preflight_denylist.txt"
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

# ---- private denylist (never committed) ------------------------------------------
$denyPath = Join-Path $root ".preflight_denylist.txt"
$deny = @()
if (Test-Path $denyPath) {
    $deny = @(Get-Content $denyPath -Encoding UTF8 | ForEach-Object { $_.Trim() } |
              Where-Object { $_ -and -not $_.StartsWith("#") })
}
foreach ($term in $deny) {
    foreach ($f in $staged) {
        if (($f -replace '\\', '/') -eq "scripts/preflight_public.ps1") { continue }
        $full = Join-Path $root $f
        if (-not (Test-Path $full -PathType Leaf)) { continue }
        $hits = Select-String -Path $full -Pattern $term -SimpleMatch -ErrorAction SilentlyContinue
        foreach ($h in $hits) { Add-Finding "DENYLIST" "private term #$([array]::IndexOf($deny, $term) + 1)" ("{0}:{1}" -f $f, $h.LineNumber) }
    }
}

# ---- 4-6. history -------------------------------------------------------------------
$historyChecked = $false
if (-not $tempGit -and -not $NoHistory) {
    $revs = @(& git rev-list --all 2>$null)
    if ($revs.Count -gt 0) {
        $historyChecked = $true
        $noreply = '@users\.noreply\.github\.com$|^noreply@anthropic\.com$'

        # 4. identities
        $ids = @(& git log --all --format='%an|%ae%n%cn|%ce' | Sort-Object -Unique)
        foreach ($id in $ids) {
            $parts = $id -split '\|', 2
            if ($parts.Count -lt 2) { continue }
            if ($parts[1] -notmatch $noreply) {
                $n = @(& git log --all --format='%h' "--author=$($parts[1])").Count
                Add-Finding "HISTORY" "commit identity is not a noreply address ($n commit(s))" $id
            }
            foreach ($term in $deny) {
                if ($id.IndexOf($term, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
                    Add-Finding "HISTORY" "private term #$([array]::IndexOf($deny, $term) + 1) in a commit identity" $id
                }
            }
        }

        # 5. commit messages
        $msgRules = @(
            @{ Name = "Claude session URL in message"; Pattern = 'claude\.ai/code/session_' },
            @{ Name = "IBKR account number in message"; Pattern = '\bU\d{7,9}\b' },
            @{ Name = "token in message";              Pattern = 'sk-ant-[A-Za-z0-9_\-]{10,}|eyJ[A-Za-z0-9_\-]{20,}' },
            @{ Name = "email in message";              Pattern = '[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}' }
        )
        foreach ($rev in $revs) {
            $msg = (& git log -1 --format='%B' $rev) -join "`n"
            $short = $rev.Substring(0, 7)
            foreach ($r in $msgRules) {
                foreach ($m in [regex]::Matches($msg, $r.Pattern)) {
                    if ($r.Name -eq "email in message" -and $m.Value -match $noreply) { continue }
                    Add-Finding "HISTORY" $r.Name "commit $short"
                }
            }
            foreach ($term in $deny) {
                if ($msg.IndexOf($term, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
                    Add-Finding "HISTORY" "private term #$([array]::IndexOf($deny, $term) + 1) in a commit message" "commit $short"
                }
            }
        }

        # 6. every file version ever committed (git grep over all revisions)
        $exclude = ':(exclude)scripts/preflight_public.ps1'
        $seen = @{}
        foreach ($rule in $rules) {
            $out = & git grep -I -n -P -e $rule.Pattern $revs -- . $exclude 2>$null
            foreach ($line in $out) {
                if ($line -notmatch '^([0-9a-f]{40}):([^:]+):(\d+):(.*)$') { continue }
                if ($matches[4] -match $allow) { continue }
                $key = "$($rule.Name)|$($matches[2])"
                if ($seen.ContainsKey($key)) { continue }
                $seen[$key] = $true
                Add-Finding "HISTORY" "$($rule.Name) in a past file version" ("{0} (commit {1})" -f $matches[2], $matches[1].Substring(0, 7))
            }
        }
        foreach ($term in $deny) {
            $out = & git grep -I -i -l -F -e $term $revs -- . $exclude 2>$null
            $paths = @($out | ForEach-Object { ($_ -split ':', 2)[1] } | Sort-Object -Unique)
            foreach ($p in $paths) {
                Add-Finding "HISTORY" "private term #$([array]::IndexOf($deny, $term) + 1) in a past file version" $p
            }
        }
    }
}

if ($tempGit -and (Test-Path $tempGit)) { Remove-Item $tempGit -Recurse -Force -ErrorAction SilentlyContinue }
Pop-Location

# ---- report -------------------------------------------------------------------
Write-Host ""
if ($findings.Count -eq 0) {
    Write-Host "PREFLIGHT PASS - no credentials, emails, account numbers or personal" -ForegroundColor Green
    Write-Host "                 paths in the $($staged.Count) files git would track." -ForegroundColor Green
    if ($historyChecked) {
        Write-Host "                 History clean: identities, messages and every past file version." -ForegroundColor Green
    } elseif (-not $NoHistory -and -not $tempGit) {
        Write-Host "                 (no commits yet - history not checked)" -ForegroundColor DarkGray
    }
    if ($deny.Count -gt 0) {
        Write-Host "                 Private denylist: $($deny.Count) term(s) checked." -ForegroundColor Green
    } else {
        Write-Host "                 Tip: list your real name/emails in .preflight_denylist.txt (git-ignored)." -ForegroundColor DarkGray
    }
    Write-Host ""
    exit 0
}
Write-Host "PREFLIGHT FAIL - $($findings.Count) finding(s):" -ForegroundColor Red
$findings | Format-Table -AutoSize Severity, What, Where | Out-String -Width 200 | Write-Host
Write-Host "Fix or ignore each one before committing." -ForegroundColor Yellow
Write-Host ""
exit 1
