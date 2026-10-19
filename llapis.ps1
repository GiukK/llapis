# ===========================================================================
# llapis -- THE canonical command.
# ===========================================================================
# The single entry point for the product. Everything the user runs goes through here.
#
# Verb map -- the first four forward to cli/launcher.ps1, which resolves the runtime;
# the rest are pure HTTP clients implemented in this file:
#
#   serve   -> cli/launcher.ps1 serve   API + web UI on 127.0.0.1:8077
#   chat    -> cli/launcher.ps1 chat    terminal REPL against that same server
#   setup   -> cli/launcher.ps1 setup   quality|speed + VRAM budget
#   agents  -> cli/launcher.ps1 agents  provider JSON for external harnesses
#   status  -> (here) ask a RUNNING server what it is actually running
#   reset   -> (here) clear the server's cached conversation KV
#   stop    -> (here) shut a running server down cleanly
#
# `status`, `reset` and `stop` are deliberately implemented here and not in the launcher:
# they must work when NO launcher is running (the server outlives the shell that started
# it), so they do no runtime resolution at all. `status` and `reset` ask the live server,
# which is the only answer that cannot be stale; `stop` instead identifies the process
# holding the port and ends it, because the HTTP shutdown route is reserved for the web
# UI's power button and is guarded by an origin + nonce a shell has no business forging.
# That is also why none of them touch llapis-user.json or nvidia-smi.
# ===========================================================================
# NO [CmdletBinding()] and NO [Parameter()] attributes, deliberately. Either one makes this an
# ADVANCED script, and an advanced script owns the common parameters -- so `llapis serve -Debug`
# bound -Debug to THIS file (where it does nothing but set $DebugPreference) instead of forwarding
# it, and the launcher never saw it. Same for -Verbose. A plain param() block leaves them in $args,
# which is what a forwarder needs. (Regression: -Debug worked before `llapis` existed and stopped
# working the moment it did.)
param([string]$Command = '')
$Rest = @($args)

$ErrorActionPreference = 'Stop'
$Root     = $PSScriptRoot
$Launcher = Join-Path $Root 'cli\launcher.ps1'

function Get-LlapisPort {
  # -Port <n> anywhere in the tail wins; otherwise the product default. Parsed here
  # so `llapis status -Port 8081` works without loading the launcher.
  for ($i = 0; $i -lt $Rest.Count - 1; $i++) {
    if ($Rest[$i] -imatch '^-{1,2}Port$' -and $Rest[$i + 1] -match '^\d+$') { return [int]$Rest[$i + 1] }
  }
  return 8077
}

function Get-LlapisHost {
  # Same idea for -Host: `llapis status -Host 192.168.1.5` has to reach the server that was
  # started that way. A LAN-bound server does NOT answer on 127.0.0.1, so assuming loopback
  # here would report "no server" about one that is running fine.
  for ($i = 0; $i -lt $Rest.Count - 1; $i++) {
    if ($Rest[$i] -imatch '^-{1,2}Host$' -and $Rest[$i + 1] -notmatch '^-') { return $Rest[$i + 1] }
  }
  return '127.0.0.1'
}

function Show-LlapisUsage {
  Write-Host ""
  Write-Host "LLapis -- local Gemma 4 26B-A4B, served." -ForegroundColor Cyan
  Write-Host ""
  Write-Host "  llapis serve            Start the server + web UI on http://127.0.0.1:8077"
  Write-Host "  llapis chat             Terminal chat (starts or reuses that same server)"
  Write-Host "  llapis setup            Choose the runtime (quality | speed) and the VRAM budget"
  Write-Host "  llapis agents           Print the provider JSON for an agent harness"
  Write-Host "  llapis status           What is the running server actually running?"
  Write-Host "  llapis reset            Clear the cached conversation (KV) on the server"
  Write-Host "  llapis stop             Shut the running server down"
  Write-Host ""
  Write-Host "Useful flags on serve/chat:" -ForegroundColor DarkGray
  Write-Host "  -Port <n>      serve on another port (default 8077)" -ForegroundColor DarkGray
  Write-Host "  -Ctx <n>       cap the context ceiling (default: grow to the model's own)" -ForegroundColor DarkGray
  Write-Host "  -VramBudget <MiB>  cap total VRAM use for this session (0 = use what is free)" -ForegroundColor DarkGray
  Write-Host "  -DryRun        resolve everything and print the plan without starting" -ForegroundColor DarkGray
  Write-Host "  -Open          open the browser once the server is ready" -ForegroundColor DarkGray
  Write-Host "  -Verbose / -Debug / -Trace   progressively louder internals" -ForegroundColor DarkGray
  Write-Host "  -NoVariableKv  pin the context at -Ctx instead of growing it" -ForegroundColor DarkGray
  Write-Host "  -NoLoaderCut   load expert tensors the stock way (costs ~584 MiB VRAM)" -ForegroundColor DarkGray
  Write-Host ""
  Write-Host "LLapis runs a local server on 127.0.0.1: reachable from this machine, not from the" -ForegroundColor DarkGray
  Write-Host "network. By default it starts no outgoing connections, runs no tools and cannot" -ForegroundColor DarkGray
  Write-Host "reach your files. These two flags grant the WEB UI more than that:" -ForegroundColor DarkGray
  Write-Host "  -Tools all     let the model run tools on this machine (read_file, write_file," -ForegroundColor Yellow
  Write-Host "                 exec_shell_command, web_fetch, web_search). A subset is allowed:" -ForegroundColor Yellow
  Write-Host "                 -Tools read_file,web_fetch" -ForegroundColor Yellow
  Write-Host "  -McpProxy      let the UI reach MCP servers through this server" -ForegroundColor Yellow
  Write-Host "  -Host <addr>   bind somewhere other than 127.0.0.1: the LAN can reach it, there is" -ForegroundColor Yellow
  Write-Host "                 no password, and the UI power button goes away" -ForegroundColor Yellow
  Write-Host "Neither tool flag is needed by an agent harness on the API -- it runs its own tools." -ForegroundColor DarkGray
  Write-Host "Run 'llapis help' for the long version." -ForegroundColor DarkGray
  Write-Host ""
}

function Invoke-LlapisStatus {
  $port = Get-LlapisPort
  $base = "http://$(Get-LlapisHost):$port"
  $models = $null
  try { $models = Invoke-RestMethod -Uri "$base/models" -TimeoutSec 3 -ErrorAction Stop } catch {}
  if (-not $models) {
    Write-Host "No LLapis server on $base." -ForegroundColor Yellow
    Write-Host "Start one with:  llapis serve"
    return 3
  }
  $m  = $models.data | Where-Object { $_.id -eq 'llapis' } | Select-Object -First 1
  $st = if ($m) { "$($m.status.value)" } else { '(no llapis profile)' }
  Write-Host ""
  Write-Host "LLapis @ $base" -ForegroundColor Cyan
  Write-Host ("  model id : llapis   ({0})" -f $st)
  if ($st -ne 'loaded') {
    # An unloaded router is a legitimate, serviceable state -- say so instead of erroring.
    Write-Host "  The server is up but no model is resident. Load it from the web UI, or POST /models/load."
    Write-Host ""
    return 0
  }
  $info = $null; $tel = $null
  try { $info = Invoke-RestMethod -Uri "$base/moe/info"      -TimeoutSec 5 -ErrorAction Stop } catch {}
  try { $tel  = Invoke-RestMethod -Uri "$base/moe/telemetry" -TimeoutSec 5 -ErrorAction Stop } catch {}
  # GiB, formatted invariantly. `{0:n0}` renders 2572 as "2.572" under an it-IT console, which
  # sits next to "4608" formatted the same way and reads as two numbers three orders apart.
  function fmtGiB([double]$mib) { return ([Math]::Round($mib / 1024.0, 1)).ToString([Globalization.CultureInfo]::InvariantCulture) }
  if ($info -and -not $info.state) {
    # LIVE numbers, not the launcher's estimate (K is re-fitted at runtime). The runtime is decided from
    # the FOLDER of the promotion source /moe/info reports (models/high = quality, models/low = speed):
    # llapis-user.json holds the REQUEST, and the launcher downgrades quality -> speed when models/high is
    # absent. A path under no known slot falls back to the saved value. (history: docs/code-comments-archive-2026-10-05.md, "status-mode")
    # value instead of guessing.
    $saved = 'unknown'
    try {
      $cfg = Get-Content (Join-Path $Root 'llapis-user.json') -Raw -ErrorAction Stop | ConvertFrom-Json
      if ($cfg.mode) { $saved = $cfg.mode }
    } catch {}
    $live = $null
    if ($info.model_high) {
      $slot = Split-Path (Split-Path ($info.model_high -replace '/', '\') -Parent) -Leaf
      if     ($slot -eq 'low')  { $live = 'speed' }
      elseif ($slot -eq 'high') { $live = 'quality' }
    }
    # A mismatch has exactly two causes, and they need opposite advice, so name the right one:
    #   live=speed, saved=quality -> the Q4 was absent at startup and the launcher downgraded.
    #                                Restarting changes nothing; the missing file is the fix.
    #   anything else             -> `setup` was re-run after this server started, so the saved
    #                                choice is newer than the process. Restarting IS the fix.
    # A generic "see the ALERT at startup" would have pointed at a message that never printed in
    # the second case -- the same class of false report this block exists to remove.
    if ($live -and $saved -ne 'unknown' -and $live -ne $saved) {
      $why = if ($live -eq 'speed' -and $saved -eq 'quality') {
        "setup asked for quality; the high (Q4) model was missing at startup"
      } else {
        "setup now says $saved; this server was started as $live -- 'llapis stop' then start again to switch"
      }
      Write-Host ("  runtime  : {0}   ({1})" -f $live, $why) -ForegroundColor Yellow
    } else {
      Write-Host ("  runtime  : {0}" -f $(if ($live) { $live } else { $saved }))
    }
    Write-Host ("  model    : {0}" -f [IO.Path]::GetFileNameWithoutExtension($info.model_high))
    Write-Host ("  context  : {0} tokens (grows on demand)" -f $info.ctx)
    Write-Host ("  expert cache : K{0} of {1} experts/layer, live-fitted to VRAM" -f $info.k, $info.n_expert)
    Write-Host ("  prefill  : {0}" -f $info.prefill)
    Write-Host ("  spec decode  : {0}" -f $(if ($info.spec.enabled) { "MTP (verify width $($info.spec.k))" } else { 'off' }))
  }
  if ($tel -and $tel.partition) {
    $v = $tel.partition.vram; $r = $tel.partition.ram
    Write-Host ("  VRAM     : {0} GiB LLapis / {1} free of {2} total" -f (fmtGiB $v.ours), (fmtGiB $v.free), (fmtGiB $v.total))
    # `arena_resident` is what is physically in RAM; `arena_committed` is how big the arena IS.
    # They differ by most of the arena on a 16 GB box, which is why both are printed.
    Write-Host ("  RAM      : {0} GiB LLapis working set / {1} free of {2} total" -f (fmtGiB $r.rss), (fmtGiB $r.avail), (fmtGiB $r.total))
    Write-Host ("  cold arena: {0} GiB reserved, {1} GiB resident" -f (fmtGiB $r.arena_committed), (fmtGiB $r.arena_resident))
    Write-Host ("  hit rate : {0:p0} of expert compute served hot   (cache {1:p0} full)" -f $tel.hit_rate, $tel.occupancy)
  }
  Write-Host ""
  Write-Host ("  API : {0}/v1   UI : {0}" -f $base)
  Write-Host ""
  return 0
}

function Invoke-LlapisReset {
  # Explicit conversation reset. The server keeps a PREFIX cache of the last conversation's KV so
  # a follow-up turn re-uses it (measured: 11227 of 11255 prompt tokens cached, 2.4 s instead of
  # 24 s). That cache is content-addressed, so a different conversation simply misses it and is
  # re-prefilled -- there is no correctness reason to clear it. This exists for the two cases where
  # you WANT it gone: reclaiming the KV, and starting a measurement from a known-cold state.
  # POST /slots/0?action=erase is the stock llama-server slot API; the router proxies it to the
  # loaded child, so it works the same whether you are talking to a router or a direct server.
  $port = Get-LlapisPort
  $base = "http://$(Get-LlapisHost):$port"
  try { Invoke-RestMethod -Uri "$base/models" -TimeoutSec 3 -ErrorAction Stop | Out-Null }
  catch {
    Write-Host "No LLapis server on $base." -ForegroundColor Yellow
    return 3
  }
  try {
    # The body is not optional and `{}` is not enough. The router's proxy_post parses it as JSON
    # (empty body -> `parse_error.101 ... empty input` wrapped in a 500) AND reads `model` out of
    # it to pick the child to forward to (missing -> router_validate_model fails with a 400).
    # Both were hit in that order while wiring this up.
    Invoke-RestMethod -Uri "$base/slots/0?action=erase" -Method Post -Body '{"model":"llapis"}' `
                      -ContentType 'application/json' -TimeoutSec 15 -ErrorAction Stop | Out-Null
    Write-Host "Conversation cache cleared. The next turn re-reads its prompt from scratch." -ForegroundColor Green
    return 0
  } catch {
    # The usual cause is "no model loaded" (nothing to erase), which is not an error worth an
    # exit code -- but a real failure should not be swallowed either, so print what came back.
    Write-Host "Could not clear the cache: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host "(If no model is loaded there is nothing to clear.)"
    return 1
  }
}

function Get-PortOwnerPid([int]$port) {
  # The PID that OWNS the listening socket. Not a name match: a name match would also find
  # a second LLapis serving another port, or any unrelated binary that happens to be called
  # moert-server.exe. The socket is what `llapis stop` is actually about.
  try {
    $c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction Stop | Select-Object -First 1
    if ($c) { return [int]$c.OwningProcess }
  } catch {
    # Get-NetTCPConnection is missing on some Windows editions; netstat always exists.
    # The state word ("LISTENING") is LOCALIZED, so it is not what we match on -- a listening
    # row is the one whose foreign address is the null endpoint, in every language.
    foreach ($line in (netstat -ano -p tcp)) {
      if ($line -match '^\s+TCP\s+(\S+):(\d+)\s+(\S+)\s+\S+\s+(\d+)\s*$') {
        if ([int]$Matches[2] -eq $port -and $Matches[3] -match ':0$') { return [int]$Matches[4] }
      }
    }
  }
  return 0
}

function Get-VerifiedLlapisProcess([int]$procId) {
  # A PID on its own is not an identity: Windows reuses PIDs, and between reading the
  # socket table and sending the kill the owner can have exited and its number been handed
  # to something else. So before anything is signalled the process must still be
  #   - running,
  #   - our moert-server.exe,
  #   - the copy that lives under THIS checkout (not another LLapis install), and
  #   - the same instance, checked by start time at the moment of the kill.
  if ($procId -le 0) { return $null }
  try { $p = Get-Process -Id $procId -ErrorAction Stop } catch { return $null }
  $path = $null
  try { $path = $p.Path } catch { }        # access denied on a foreign process is a refusal, not an error
  if (-not $path) { return $null }
  if ([IO.Path]::GetFileName($path) -ne 'moert-server.exe') { return $null }
  $rootFull = [IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
  if (-not ([IO.Path]::GetFullPath($path)).StartsWith($rootFull, [StringComparison]::OrdinalIgnoreCase)) {
    return $null
  }
  return $p
}

function Stop-VerifiedProcess($p) {
  # Re-verify identity immediately before the kill: the checks above are only as fresh as
  # the moment they ran.
  $again = Get-VerifiedLlapisProcess $p.Id
  if (-not $again) { return $false }
  if ($again.StartTime -ne $p.StartTime) { return $false }   # PID reused between checks
  try { Stop-Process -Id $p.Id -Force -ErrorAction Stop; return $true } catch { return $false }
}

function Invoke-LlapisStop {
  $port = Get-LlapisPort
  $base = "http://$(Get-LlapisHost):$port"

  # Deliberately NOT POST /moe/exit. That route is the WEB UI's power button: it is bound to
  # a loopback origin and a per-process nonce precisely so that only a page this server
  # itself handed out can trigger a shutdown. A shell already has local authority, so it
  # stops the process the way any other local tool would -- by identifying it and ending it.
  # This is also why `stop` is the ONE verb that keeps working in LAN mode, where that route
  # is not registered at all: it never needed it. The socket lookup is address-agnostic.
  $srvPid = Get-PortOwnerPid $port
  $proc   = Get-VerifiedLlapisProcess $srvPid
  if (-not $proc) {
    $listening = $srvPid -gt 0
    if ($listening) {
      Write-Host "Port $port is held by PID $srvPid, which is not an LLapis server from this checkout. Not touching it." -ForegroundColor Yellow
      return 2
    }
    Write-Host "No LLapis server on $base -- nothing to stop." -ForegroundColor Yellow
    return 0
  }

  # `serve` runs a router that spawns one child process for the model. Snapshot the children
  # BEFORE the kill: afterwards the parent link is gone and they would be unattributable.
  $children = @()
  try {
    $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $($proc.Id)" -ErrorAction Stop |
                  ForEach-Object { Get-VerifiedLlapisProcess ([int]$_.ProcessId) } |
                  Where-Object { $_ })
  } catch { }

  if (-not (Stop-VerifiedProcess $proc)) {
    Write-Host "Could not stop LLapis (PID $($proc.Id)); it may have exited already." -ForegroundColor Yellow
    return 1
  }

  # A child whose router is gone exits on its own when the stdin pipe closes. Give it that
  # chance before ending it, so its own cleanup runs.
  foreach ($c in $children) {
    for ($i = 0; $i -lt 20; $i++) {
      if ($c.HasExited) { break }
      Start-Sleep -Milliseconds 250
      $c.Refresh()
    }
    if (-not $c.HasExited) { $null = Stop-VerifiedProcess $c }
  }

  # "Stopped" means the port is free for the next `llapis serve` -- that is the only thing
  # the caller cares about, and it is not the same instant as the process exiting.
  for ($i = 0; $i -lt 40; $i++) {
    if ((Get-PortOwnerPid $port) -eq 0) {
      Write-Host "LLapis stopped (port $port released)." -ForegroundColor Green
      return 0
    }
    Start-Sleep -Milliseconds 250
  }
  Write-Host "LLapis did not release port $port within 10 s." -ForegroundColor Yellow
  return 1
}

if (-not (Test-Path $Launcher)) {
  Write-Error "llapis: runtime not found at $Launcher. Run this from the LLapis checkout."
  exit 1
}

if (-not $Command -or $Command -in @('-h', '--help', '/?')) { Show-LlapisUsage; exit 0 }

switch ($Command.ToLowerInvariant()) {
  'status'  { exit (Invoke-LlapisStatus) }
  'stop'    { exit (Invoke-LlapisStop) }
  'reset'   { exit (Invoke-LlapisReset) }
  # `help` is FORWARDED, not answered here: the long version (model folders, the
  # capability flags, where the logs go) lives in the launcher, and answering it here
  # left that text unreachable for anyone who did not open the file. Bare `llapis`
  # still prints the short usage above.
  { $_ -in @('serve', 'chat', 'setup', 'agents', 'help') } {
    # Forward the tail to the launcher as NAMED parameters.
    #
    # `& $Launcher $verb @Rest` looks like it should work and does not: in PowerShell 5.1, ARRAY
    # splatting binds every element POSITIONALLY, so `-DryRun` comes out as "Impossibile trovare un
    # parametro posizionale" instead of setting the switch. (Verified with a two-line repro; the
    # launcher's own `diag` forwarder carries the same warning about `-Last` binding to `-Path`.)
    # Only HASHTABLE splatting binds by name, so the tail is parsed into one first.
    $named = @{}
    $extra = @()
    for ($i = 0; $i -lt $Rest.Count; $i++) {
      $t = $Rest[$i]
      if ($t -match '^-{1,2}([A-Za-z][A-Za-z0-9]*)$') {
        $name = $Matches[1]
        $next = if ($i + 1 -lt $Rest.Count) { $Rest[$i + 1] } else { $null }
        # The next token is this parameter's VALUE unless it is itself a parameter name. A negative
        # number is a value, so it is tested for first — otherwise `-K -1` would silently become
        # two switches.
        if ($null -ne $next -and ($next -match '^-?\d+(\.\d+)?$' -or $next -notmatch '^-{1,2}[A-Za-z]')) {
          $named[$name] = $next
          $i++
        } else {
          $named[$name] = $true    # switch
        }
      } else {
        $extra += $t
      }
    }
    & $Launcher $_ @named @extra
    exit $LASTEXITCODE
  }
  default {
    Write-Host "llapis: unknown command '$Command'." -ForegroundColor Red
    Show-LlapisUsage
    exit 2
  }
}
