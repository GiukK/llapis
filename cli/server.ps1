# ===========================================================================
# llapis/server.ps1 - serve + server-backed chat launch (the router path).
# ===========================================================================
# DOT-SOURCED by llapis.ps1 AFTER the runtime is resolved (preset.ps1) and
# AFTER chat.ps1 (it calls Invoke-LlapisChatRepl). Shares the launcher scope, so
# it reads $SrvExe/$Rel/$Root/$Port/$Model/$K/$Ctx/$place/$script:mtpCliArgs/
# $script:visionMmproj/$DryRun/... and, for serve|chat, runs the whole block and
# EXITS; for any other mode the guard is false and control returns to the launcher.
# NOTE: the /change re-exec targets llapis.ps1 via $Root, not $PSCommandPath
# (which, inside a dot-sourced file, would resolve to THIS file).
# ===========================================================================

# ---------------------------------------------------------------------------
# serve + server-backed chat: the product path — moert-server (llama-server + moe_dual) with
# the SAME preset env this wrapper builds. OpenAI API on $BindAddress:$Port
# (/v1/chat/completions incl. SSE) + /moe/telemetry + /moe/info + /health.
# `chat` starts (or reuses) the same router and drives it through the REPL above.

# Loopback == "this machine only". Must agree with the runtime's own test
# (server_http_is_loopback_authority): 127.0.0.0/8 as a real dotted quad, localhost, ::1.
# A prefix test on "127." would also accept 127.evil.com, which is the hole that check exists
# to close -- so this one parses the quad too rather than being clever.
function Test-LoopbackAddress([string]$addr) {
  $a = $addr.Trim().Trim('[', ']')
  if ($a -eq 'localhost' -or $a -eq '::1') { return $true }
  $parts = $a -split '\.'
  if ($parts.Count -ne 4) { return $false }
  foreach ($p in $parts) { if ($p -notmatch '^\d{1,3}$' -or [int]$p -gt 255) { return $false } }
  return [int]$parts[0] -eq 127
}
#
# A FUNCTION, not inline dot-sourced code: this block uses `exit` as normal control flow (DryRun,
# ready, /change, errors), and a dot-sourced `exit` does NOT halt the caller (it only returns from
# the dot-sourced file) — but `exit` inside a CALLED function does terminate the whole script. The
# launcher dot-sources this file (to define the function) then calls Invoke-LlapisServe; for the
# other modes the guard returns and the launcher carries on.
function Invoke-LlapisServe {
  if ($Mode -notin @('serve', 'chat')) { return }
  $SrvExe = Join-Path $Rel 'moert-server.exe'
  if (-not (Test-Path $SrvExe)) { Write-Error "moert-server.exe not found: $SrvExe  (build: cmake --build build-cuda --config Release)"; exit 1 }
  if ($PrefillMode -eq 'q4stream' -and -not $script:mtpCliArgs) {
    Write-Error "${Mode}: q4stream server prefill requires the MTP/spec drafter (backend sampling must be off for the layer-major merged prefill graph). Drop -NoMtp, or pass -PrefillMode tbt."
    exit 1
  }
  # C5.5: ONE fixed, flat profile name -- it IS the model id an agent harness sends, so it must not
  # encode K, ctx or the mode (each change used to invalidate the harness's model id). The mode is a
  # saved setup decision (`llapis status` reads it). (history: docs/code-comments-archive-2026-10-05.md, "profile-name")
  $profName = 'llapis'
  $env:MOE_PRESET = $profName

  # ---- ROUTER mode. C1: serve is API-FIRST + LOADED-BY-DEFAULT. We write the
  #      config-derived profile, start the router, POST /models/load, and only print
  #      "Server ready" once the child reports 'loaded'. ----
  $profArgs = [ordered]@{ 'ctx-size' = "$Ctx"; 'gpu-layers' = "$Ngl"; 'jinja' = 'true' }
  # STARTUP: llama's auto-fit is OFF. Nothing it adjusts is unset here (ctx, ngl, K and batch are
  # resolved and VRAM-clamped by the launcher; the C++ preflight refuses a hopeless load), its model
  # prices in the expert tensors the loader cut never allocates, and it cost 2.1 s of startup (a draft-
  # memory probe that fails anyway + a dry-run pass). (history: docs/code-comments-archive-2026-10-05.md, "fit-off")
  $profArgs['fit'] = 'off'
  # Prompt-cache RAM budget (C5.6). Resolved in preset.ps1 against the cold arena's size; see the
  # measured 0.07 tok/s failure documented there. -1 = leave upstream's 8192 MiB default alone
  # (bench/dev paths, or a box where we could not measure RAM).
  if ($null -ne $script:CacheRamResolved -and $script:CacheRamResolved -ge 0) {
    $profArgs['cache-ram'] = "$($script:CacheRamResolved)"
  }
  # C5.6: enable the slot-state endpoints. `POST /slots/{id}?action=erase` is what backs
  # `llapis reset` (an explicit, safe "forget this conversation"), and llama-server refuses the
  # whole /slots action family unless --slot-save-path is set — including erase, which writes
  # nothing. The directory is only ever used by the save/restore actions we do not expose.
  $slotDir = Join-Path $LogsDir 'slots'
  New-Item -ItemType Directory -Force -Path $slotDir | Out-Null
  $profArgs['slot-save-path'] = $slotDir
  if ($PrefillMode -eq 'q4stream') {
    # one streamed decode covers the whole prompt up to batch-size tokens (each decode call
    # streams the Q4 pool once); 4096 bounds the layer-major merged graph's host mask shadows.
    # The batch is VRAM-gated in preset.ps1 (Resolve-Q4sBatch) to cut the per-chunk
    # pool re-streams; default stays 4096. $script:q4sBatch is set there for q4stream chat/serve.
    $b = if ($script:q4sBatch) { "$($script:q4sBatch)" } else { '4096' }
    $profArgs['batch-size'] = $b
  }
  if ($place -contains '--cpu-moe') { $profArgs['cpu-moe'] = 'true' }
  # P4 prefix reuse: every streamed prompt now leaves a SWA context checkpoint (up to ~200 MiB of host
  # RAM each at the 1024-token window). Continuing a conversation restores only the LATEST one, and every
  # extra one is copied into the host prompt cache when the conversation changes (4 = 714 MiB, +0.46 s on
  # each new conversation's first turn, findings 181): keep exactly one.
  if ($env:MOE_DUAL_PREFIX_REUSE -eq '1') { $profArgs['ctx-checkpoints'] = '1' }
  if ($place -contains '--no-mmap') { $profArgs['no-mmap'] = 'true' }
  if ($place -contains '--no-host') { $profArgs['no-host'] = 'true' }
  $ncmoeIx = [Array]::IndexOf($place, '--n-cpu-moe')
  if ($ncmoeIx -ge 0)               { $profArgs['n-cpu-moe'] = "$($place[$ncmoeIx + 1])" }
  if ($script:mtpCliArgs) {
    $profArgs['model-draft']      = $script:mtpCliArgs[1]
    $profArgs['spec-type']        = $script:mtpCliArgs[3]
    $profArgs['spec-draft-n-max'] = $script:mtpCliArgs[5]
  }
  if ($script:visionMmproj) {
    # projector -> child argv (--mmproj). Resident on the GPU, the tower held ~1.1 GiB of VRAM for the whole
    # session (~18 K of the hot cache): under a 5 GiB budget K sat at its floor of 4 for every turn. On the
    # CPU (--no-mmproj-offload) an image took ~40 s more to encode and the 1.1 GiB moved to RAM, where the
    # cold arena needs it (claude dump findings 249/257). So: on the GPU, LAZY -- the weights go to VRAM only
    # while an image is encoded (clip.cpp, LLAPIS_MMPROJ_LAZY). LLAPIS_MMPROJ_LAZY=0 keeps them resident,
    # LLAPIS_MMPROJ_CPU=1 encodes on the CPU.
    $profArgs['mmproj'] = $script:visionMmproj
    if ($env:LLAPIS_MMPROJ_CPU -eq '1') { $profArgs['no-mmproj-offload'] = 'true' }
    elseif (-not $env:LLAPIS_MMPROJ_LAZY) { $env:LLAPIS_MMPROJ_LAZY = '1' }
  }

  # collect every MOE_* var this wrapper just computed for the preset (child env)
  $profEnv = [ordered]@{}
  Get-ChildItem Env: | Where-Object { $_.Name -like 'MOE_*' -and $_.Value } |
    Sort-Object Name | ForEach-Object { $profEnv[$_.Name] = $_.Value }

  # -DryRun: the runtime is fully resolved now (model, K, ctx, vision, MTP, child env). Print
  # exactly what WOULD load and exit before touching the server — the single place that answers
  # "what actually loads?" without a 20 s model load. Every silent override is visible here.
  if ($DryRun) {
    $plan = [ordered]@{
      mode          = $Mode
      profile       = $profName
      bind          = "${BindAddress}:$Port"
      lan_mode      = -not (Test-LoopbackAddress $BindAddress)
      base_model    = $Model
      q4_source     = $Q4Source
      expert_K      = $K
      context       = $Ctx
      variable_kv   = $(if ($script:VkvOn) { [ordered]@{ enabled = $true; boot_ctx = $script:VkvMinCtx; max_ctx = $script:VkvMaxCtx } }
                        else { [ordered]@{ enabled = $false; pinned_ctx = $Ctx; why = $script:VkvOffWhy } })
      gpu_layers    = $Ngl
      cache_label   = $CacheLabel
      vision        = [bool]$script:visionMmproj
      vision_proj   = $(if ($script:visionMmproj) { [IO.Path]::GetFileName($script:visionMmproj) } else { $null })
      mtp           = [bool]$script:mtpCliArgs
      mtp_n_max     = $(if ($script:mtpCliArgs) { $MtpNMax } else { $null })
      prefill_mode  = $PrefillMode
      setup_applied = $SetupApplied
      child_args    = $profArgs
      child_env     = $profEnv
    }
    # The machine-readable copy is written ALWAYS, not behind a flag. It is the artefact
    # to attach to a bug report, and the one a regression check compares -- neither should
    # depend on the user having known to ask for it. The human summary below is what the
    # terminal gets.
    try {
      New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
      ($plan | ConvertTo-Json -Depth 6) | Out-File -Encoding utf8 (Join-Path $LogsDir 'plan.json')
    } catch { }
    Write-Host ""
    Write-Host "LLapis - resolved runtime (DryRun; nothing started)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host ("  profile      : {0}" -f $profName)
    if (Test-LoopbackAddress $BindAddress) {
      Write-Host ("  bind         : {0}:{1} (this machine only)" -f $BindAddress, $Port)
    } else {
      Write-Host ("  bind         : {0}:{1}  LAN MODE - reachable from the network, no password," -f $BindAddress, $Port) -ForegroundColor Yellow
      Write-Host  "                 no shutdown route, no UI power button" -ForegroundColor Yellow
    }
    if ($script:LlapisMode) { Write-Host ("  runtime      : {0}" -f $script:LlapisMode) }
    Write-Host ("  base model   : {0}" -f [IO.Path]::GetFileName($Model))
    Write-Host ("  Q4 source    : {0}" -f [IO.Path]::GetFileName($Q4Source))
    # BOOT K, not the K that serves: it is re-fitted to measured VRAM at the first
    # prompt end and at every context change, so this figure is the floor.
    Write-Host ("  expert cache : K{0} at boot{1}, then re-fitted to measured VRAM" -f $K, $(if ($CacheLabel) { " ($CacheLabel)" } else { "" }))
    Write-Host ("  context      : {0}{1}" -f $Ctx,
                $(if ($script:VkvOn) { " (variable-KV: boot, grows to $($script:VkvMaxCtx))" }
                  else { " (pinned$(if ($script:VkvOffWhy) { " - $($script:VkvOffWhy)" }))" }))
    Write-Host ("  vision       : {0}" -f $(if ($script:visionMmproj) { "ON ($([IO.Path]::GetFileName($script:visionMmproj)))" } else { 'off' }))
    Write-Host ("  MTP spec     : {0}" -f $(if ($script:mtpCliArgs) { "on (n_max=$MtpNMax)" } else { 'off' }))
    Write-Host ("  prefill      : {0}" -f $PrefillMode)
    Write-Host ("  from setup   : {0}" -f $SetupApplied)
    Write-Host ""
    Write-Host ("  full plan    : {0}" -f (Join-Path $LogsDir 'plan.json')) -ForegroundColor DarkGray
    Write-Host "  child env (MOE_*):" -ForegroundColor DarkGray
    foreach ($kv in $profEnv.GetEnumerator()) { Write-Host ("    {0} = {1}" -f $kv.Key, $kv.Value) -ForegroundColor DarkGray }
    Write-Host ""
    exit 0
  }

  # C1.4: API-first serve keeps the model resident until Ctrl+C (no browser-poll watchdog).
  $autoUnload = 0

  # MERGE into the existing profiles file: only our own entry is (re)written; UI-saved
  # profiles survive. (Overwriting the whole file was the M6 fallback bug.)
  # A one-time migration used to purge legacy 'planc-*' / 'gemma4_k=*' profiles here. It was
  # dead by construction in a release: this file lives in the BUILD tree, which a fresh clone
  # creates empty, so there has never been anything of that shape to find.
  $profilesPath = Join-Path $Rel 'moert-profiles.json'
  $existing = $null
  if (Test-Path $profilesPath) {
    try { $existing = Get-Content $profilesPath -Raw | ConvertFrom-Json } catch {
      Write-Host "[llapis] WARNING: $profilesPath is unreadable - rewriting it fresh (saved profiles lost)" -ForegroundColor Yellow
    }
  }
  $profMap = [ordered]@{}
  if ($existing -and $existing.profiles) {
    foreach ($prop in $existing.profiles.PSObject.Properties) {
      if ($prop.Name -eq $profName) { continue }
      $profMap[$prop.Name] = $prop.Value
    }
  }
  Say-Phase ("Saving Runtime parameters (profile '{0}': K={1}, ctx={2}{3}{4})" -f $profName, $K, $Ctx,
             $(if ($script:mtpCliArgs) { ", MTP n_max=$MtpNMax" } else { "" }),
             $(if ($script:visionMmproj) { ", vision" } else { "" }))
  $profMap[$profName] = [ordered]@{ model = $Model; args = $profArgs; env = $profEnv }
  $profiles = [ordered]@{
    version       = 1
    models_dir    = Join-Path $Root 'models'
    auto_unload_s = $autoUnload
    profiles      = $profMap
  }
  $profiles | ConvertTo-Json -Depth 6 | Out-File -Encoding ascii $profilesPath

  # the router must not inherit an active MOE env (children get theirs from the profile);
  # LLAPIS_* stay so the router + its children keep the filter + shared log.
  Get-ChildItem Env: | Where-Object { $_.Name -like 'MOE_*' } | ForEach-Object { Set-Item "Env:$($_.Name)" $null }

  # ---- C1.2 API-FIRST loaded-by-default -------------------------------------------------
  $base   = "http://${BindAddress}:$Port"
  $isChat = ($Mode -eq 'chat')
  Show-LlapisBanner
  Write-Host $(if ($isChat) { "Chat" } else { "Server" }) -ForegroundColor DarkGray
  Write-Host ""
  # C5.5: the header used to hard-code "Mode: quality" and name $Q4Source — under `speed` all
  # three lines described a runtime that was not the one starting (Q4 file, quality, K48).
  # Report what actually loaded. The name to print is the PROMOTION SOURCE: under quality that
  # is the Q4_K_XL file (the base is a derived splice nobody downloaded), and under speed it is
  # the IQ2 file, which is also the base. Both name a file the user actually has.
  # (This used to ask `$effPreset -like 'planc-*'`, which was constant-true — both modes ran
  # that one preset — so the else branch was unreachable and the fallback below could have
  # printed "quality" for a speed run.)
  Write-Host ("Model: {0}" -f [IO.Path]::GetFileNameWithoutExtension($Q4Source))
  Write-Host ("Mode: {0}" -f $(if ($script:LlapisMode) { $script:LlapisMode } else { 'unknown' }))
  Write-Host ("Context: {0}" -f $(if ($script:VkvOn) { "up to $($script:VkvMaxCtx) (starts at $Ctx and grows as the chat does)" } else { "$Ctx (fixed)" }))
  # C5.6 honesty fix: this K is the PRE-LAUNCH estimate, and it is deliberately pessimistic —
  # it has to survive the MTP drafter's load-time peak, which lands on top of slab + NE + KV and
  # is a transient nothing later pays for. The runtime then re-fits K against MEASURED free VRAM
  # at the first prompt end and at every context change, and that number is materially higher:
  # measured on this box, boot 62 -> 71 at ctx 8192 (speed), boot 28 -> 33 (quality). Printing
  # the estimate alone made the product look like it was leaving a third of the expert cache on
  # the table when it was not. Say which number this is, and where to read the real one.
  if ($CacheLabel) {
      Write-Host ("Expert cache: {0}  (K{1} at boot; re-fitted to measured VRAM on the first prompt)" -f $CacheLabel, $K)
      Write-Host "              `llapis status` reports the live value." -ForegroundColor DarkGray
  }
  Write-Host ("Speculative decode: {0}" -f $(if ($script:mtpCliArgs) { "MTP (n_max=$MtpNMax)" } else { 'off (single-token)' }))
  Write-Host ""

  # C2: chat REUSES an already-running LLapis server on this port when its profile is
  # already loaded (both derive from llapis-user.json, so the config is identical). A busy
  # port with a DIFFERENT profile is an error (loading by name from a rewritten profiles
  # file into a live router is not supported -- stale-config risk).
  $router = $null
  $reusedRouter = $false
  # C5.2: BOTH chat and serve probe the port and reuse a running router (single-instance
  # rule — the server side also enforces it with a named mutex + exclusive port bind, so
  # a second full instance is impossible; this probe just makes the reuse path friendly).
  $probe = $null
  try { $probe = Invoke-RestMethod -Uri "$base/models" -TimeoutSec 2 -ErrorAction Stop } catch {}
  if ($probe) {
    $m = $probe.data | Where-Object { $_.id -eq $profName }
    if ($m -and "$($m.status.value)" -eq 'loaded') {
      $reusedRouter = $true
      Write-Host "Reusing the running LLapis server on port $Port."
      # -Verbose/-Debug/-Trace instrument a server THIS invocation starts: every runtime line
      # (phases, firehose) is emitted by the router + its child, and those processes already
      # exist — they were started by another session, with that session's log level, writing to
      # that session's console. So the flag is silently inert here, which reads as "-Debug shows
      # nothing on the terminal" (reported 2026-07-28) when in fact there was nothing left to
      # attach to. Say so, and give the one command that fixes it.
      if ($DebugMode -or $VerboseMode) {
        $lvlName = if ($TraceMode) { '-Trace' } elseif ($DebugMode) { '-Debug' } else { '-Verbose' }
        Write-Host ""
        Write-Host "$lvlName has nothing to attach to: the runtime is already up, and its output belongs to" -ForegroundColor Yellow
        Write-Host "the session that started it (plus that run's log in $LogsDir)." -ForegroundColor Yellow
        Write-Host "To watch it from here, restart the runtime under $lvlName :" -ForegroundColor Yellow
        Write-Host "    llapis stop" -ForegroundColor Yellow
        Write-Host "    llapis serve $lvlName" -ForegroundColor Yellow
      }
    } else {
      Write-Error "Port $Port already serves LLapis but profile '$profName' is not loaded there. Stop that server (llapis stop -Port $Port) or pass -Port <other>."
      exit 1
    }
  }
  if ($reusedRouter -and -not $isChat) {
    Write-Host ""
    Write-Host "Server already running:"
    Write-Host ("  API : {0}/v1   (OpenAI-compatible; POST /v1/chat/completions)" -f $base)
    Write-Host ("  UI  : {0}" -f $base)
    Write-Host ("  Model id: {0}" -f $profName)
    Write-Host "Leaving it as is (it belongs to the session that started it)."
    exit 0
  }

  if (-not $reusedRouter) {
    Say-Phase "Starting LLapis server (router) on http://${BindAddress}:$Port"
    Write-Host $(if ($isChat) { "Starting server (chat backend)..." } else { "Starting server..." })
    # Start the router in the background (inherits LLAPIS_* env); normal mode redirects its output to log
    # files (the wrapper owns the terminal), -Debug keeps the console. --no-models-autoload: with autoload,
    # ANY request naming a model makes the router load it, force-killing an in-progress load (reproduced:
    # a stale profile replaced the correct one mid-boot). We always POST /models/load ourselves and wait
    # for 'loaded', so autoload only exposed that bug class. (history: docs/code-comments-archive-2026-10-05.md, "router-start")
    # than guessing at which specific client was the source.
    $srvArgsRouter = @('--host',"$BindAddress",'--port',"$Port",'--no-models-autoload')

    # ---- LAN mode -------------------------------------------------------------------
    # Anything but a loopback address means other machines can reach this server. The
    # runtime already treats that as a different security posture -- POST /moe/exit and
    # GET /moe/exit-token are not registered at all, so the web UI's power button does not
    # render -- but the person typing the flag deserves to be told, once, in full.
    if (-not (Test-LoopbackAddress $BindAddress)) {
      Write-Host ""
      Write-Host "[llapis] LAN MODE: binding $BindAddress, not 127.0.0.1." -ForegroundColor Yellow
      Write-Host "         Anything that can reach this machine on port $Port can now use the model," -ForegroundColor Yellow
      Write-Host "         read the conversation, and load or unload it. There is NO password: LLapis" -ForegroundColor Yellow
      Write-Host "         does not ask for one, because on 127.0.0.1 it does not need one." -ForegroundColor Yellow
      Write-Host "         The shutdown route is not registered in this mode, so the web UI's power" -ForegroundColor Yellow
      Write-Host "         button is gone -- stop the server from this machine with 'llapis stop'." -ForegroundColor Yellow
      Write-Host "         -Tools and -McpProxy stay off unless you also asked for them; do not." -ForegroundColor Yellow
      Write-Host ""
    }

    # ---- Capability grants: OFF unless the user asked (see the launcher's params) ----
    # These are the only two things LLapis can do that reach past the conversation, so
    # they are opt-in and they say so out loud when enabled. Both serve the WEB UI only:
    # an agent harness on /v1/chat/completions sends its own tool definitions and runs
    # them in its own process, and is unaffected either way.
    if ($Tools) {
      $srvArgsRouter += @('--tools', $Tools)
      Write-Host "[llapis] TOOLS ENABLED ($Tools): the model can now run these on THIS machine" -ForegroundColor Yellow
      Write-Host "         through the web UI. exec_shell_command runs arbitrary commands as you;" -ForegroundColor Yellow
      Write-Host "         write_file overwrites files. The UI asks before each new tool, and" -ForegroundColor Yellow
      Write-Host "         remembers an 'always'. That prompt is in the browser: any program on" -ForegroundColor Yellow
      Write-Host "         this machine can call POST /tools directly and skip it." -ForegroundColor Yellow
    }
    if ($McpProxy) {
      $srvArgsRouter += '--ui-mcp-proxy'
      Write-Host "[llapis] MCP PROXY ENABLED: the web UI can reach MCP servers by fetching them" -ForegroundColor Yellow
      Write-Host "         through this server, so the server now makes OUTGOING requests on the" -ForegroundColor Yellow
      Write-Host "         page's behalf. Without this flag it starts none." -ForegroundColor Yellow
    }
    if ($DebugMode) {
      $router = Start-Process -FilePath $SrvExe -ArgumentList $srvArgsRouter -NoNewWindow -PassThru
    } else {
      $rOut = $LogFile -replace '\.log$','.router.out.log'
      $rErr = $LogFile -replace '\.log$','.router.err.log'
      $router = Start-Process -FilePath $SrvExe -ArgumentList $srvArgsRouter -NoNewWindow -PassThru -RedirectStandardOutput $rOut -RedirectStandardError $rErr
    }
    # before the load request: the model child is spawned later by the router and inherits the job
    [void](Register-LlapisChildTree $router)
  }

  $script:chatReplResult = $null
  try {
    if (-not $reusedRouter) {
      # 1) wait for the router HTTP to accept connections (/models is served even when unloaded)
      # STARTUP: 150 ms, not 500. The router needs ~2.3 s to be reachable (process
      # start + the ggml/CUDA DLL load, cublasLt64 alone is 460 MB), and the old grain rounded
      # that up to the next half-second on every launch for nothing — the probe is a loopback
      # GET against a socket that is either bound or not. Same budget in wall time (30 s).
      $up = $false
      for ($i = 0; $i -lt 200 -and -not $up; $i++) {
        if ($router.HasExited) { throw "router exited during startup (rc=$($router.ExitCode)); see $LogFile" }
        try { Invoke-RestMethod -Uri "$base/models" -TimeoutSec 2 -ErrorAction Stop | Out-Null; $up = $true }
        catch { Start-Sleep -Milliseconds 150 }
      }
      if (-not $up) { throw "router did not come up on $base; see $LogFile" }

      # 2) load the config-derived profile by name. A router-side admission-gate refusal
      #    (C5.2: memory preflight / crash backoff) comes back as an HTTP error with the
      #    reason in the body — surface it instead of a raw exception.
      Say-Phase "Requesting model load ('$profName') — child phases follow"
      $body = "{`"model`":`"$profName`"}"
      try {
        Invoke-RestMethod -Uri "$base/models/load" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 10 -ErrorAction Stop | Out-Null
      } catch {
        throw "model load request refused: $($_.Exception.Message) — see $LogFile"
      }

      # 3) wait until the child reports 'loaded'. C5.2: NO blind re-POST — a child that
      #    dies flips the status back to 'unloaded' WITH its exit code; re-requesting the
      #    load in a loop was the OOM respawn storm (each attempt re-reads the 7.8 GB
      #    arena). One failure = abort with the reason.
      # STARTUP: poll at 250 ms, not 1000. The child reaches 'loaded' at an arbitrary
      # point inside the tick, so a 1 s grain added ~500 ms of pure wait to the AVERAGE launch and
      # up to 1 s to the worst one — on a 19 s startup that is real, and it is invisible in the
      # child's own log (which is why it survived this long). The progress dots now tick every
      # ~1 s regardless of the poll rate, so the terminal looks identical. Same 300 s budget.
      $loaded = $false
      $sawLoading = $false
      $dotAt = 0
      for ($i = 0; $i -lt 1200 -and -not $loaded; $i++) {
        if ($router.HasExited) { throw "router exited while loading (rc=$($router.ExitCode)); see $LogFile" }
        try {
          $m = (Invoke-RestMethod -Uri "$base/models" -TimeoutSec 3 -ErrorAction Stop).data |
                 Where-Object { $_.id -eq $profName }
          $st = if ($m) { "$($m.status.value)" } else { '' }
          if ($st -eq 'loaded')  { $loaded = $true; break }
          if ($st -eq 'loading') { $sawLoading = $true }
          if ($st -eq 'unloaded' -and $sawLoading) {
            $rc = try { [int]$m.status.exit_code } catch { -1 }
            $why = switch ($rc) {
              86 { "the memory preflight refused the load (not enough free VRAM/RAM — close whatever holds it, or lower K/ctx)" }
              87 { "another LLapis instance already holds the model lock (one instance per session)" }
              default { "the model child exited with code $rc (crash — see the log)" }
            }
            throw "model '$profName' failed to load: $why. Full log: $LogFile"
          }
        } catch { if ($_.Exception.Message -like "model '*failed to load*") { throw } }
        Start-Sleep -Milliseconds 250
        if ($i -ge $dotAt -and -not $DebugMode) { Write-Host "." -NoNewline; $dotAt = $i + 4 }   # -Debug: phases stream instead
      }
      Write-Host ""
      if (-not $loaded) { throw "model '$profName' did not reach 'loaded' in time; see $LogFile" }
    }

    if ($isChat) {
      # C2: run the streaming REPL against the loaded server; the finally below stops the
      # router if (and only if) this invocation started it.
      $script:chatReplResult = Invoke-LlapisChatRepl -Base $base -ModelId $profName
    } else {
      # 4) ready — the OpenAI-compatible API is the product surface
      Write-Host ""
      Write-Host "Server ready:"
      Write-Host ("  API : {0}/v1   (OpenAI-compatible; POST /v1/chat/completions)" -f $base)
      Write-Host ("  UI  : {0}   (optional browser client)" -f $base)
      Write-Host ("  Model id: {0}" -f $profName)
      Write-Host ""
      Write-Host "Press Ctrl+C to stop."

      # optional browser (secondary): -Open forces; otherwise ask once (normal interactive)
      $doOpen = [bool]$Open
      if (-not $doOpen -and $Interactive) {
        $ans = Read-Host "Open browser? [y/N]"
        if ($ans -match '^[yY]') { $doOpen = $true }
      }
      if ($doOpen) { Start-Process $base | Out-Null }

      # 5) stay foreground until the router exits or Ctrl+C
      while (-not $router.HasExited) { Start-Sleep -Milliseconds 500 }
      exit $router.ExitCode
    }
  }
  finally {
    if ($router -and -not $router.HasExited) {
      # We started this router, so we hold its handle -- no identification step and no
      # HTTP round trip are needed here (POST /moe/exit is the web UI's button, and it
      # would refuse a request from a shell anyway). The child follows its parent out
      # when the stdin pipe closes; the second later gives it room to do that first.
      Stop-Process -Id $router.Id -Force -ErrorAction SilentlyContinue
      Start-Sleep -Seconds 1
    }
  }

  # chat epilogue (router already stopped/left-alone by the finally above): /change reruns
  # the setup (now a single quality|speed question) and re-execs chat so the chosen runtime
  # takes effect (old exit-42 contract).
  if ($script:chatReplResult -eq 'change') {
    if ($Interactive -and -not $DebugMode) {
      $s = Invoke-LlapisWizard
      if (-not $s) { exit 0 }
      # re-exec the ENTRY POINT, not $PSCommandPath: inside this dot-sourced file that
      # would be cli\server.ps1. $Root is the checkout root (set in cli\launcher.ps1).
      & (Join-Path $Root 'llapis.ps1') 'chat'
      exit $LASTEXITCODE
    }
    Write-Host "[llapis] /change needs an interactive normal-mode shell; run:  llapis setup" -ForegroundColor Yellow
  }
  exit 0
}