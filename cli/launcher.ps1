# ===========================================================================
# launcher.ps1 - resolves the runtime and starts it.
# ===========================================================================
# Dot-sourced by llapis.ps1 for the verbs that need a resolved runtime:
# serve, chat, setup, agents. It turns the saved setup into the environment and
# placement flags moert-server expects, stages the DLLs, and hands over.
#
# TWO runtimes, ONE code path. They are the same dual-precision runtime; the
# only difference is which file each precision tier points at:
#
#   quality  base = the non-expert-Q4 splice (models/cache/base_neq4.gguf),
#            promotions streamed from the Q4_K_XL source in models/high/.
#            Hot experts at Q4 in VRAM over an IQ2 cold tier in RAM.
#
#   speed    base = the IQ2 in models/low/, and the promotion source is THAT
#            SAME FILE. Hot and cold therefore hold identical bytes, so output
#            quality matches stock exactly -- but the dual machinery still runs,
#            which is what keeps the streamed prefill instead of the
#            token-by-token one. That prefill is the whole speed edge.
#
# `setup` chooses between them and stores one field. Everything else is derived:
# K is fitted to measured VRAM and re-fitted whenever the context changes, the
# context grows on demand, the prefill mode follows from the drafter, and the
# cache policy is the measured one. There is nothing here to tune.
#
# `agents` emits a CONSTANT provider JSON with the single model id `llapis`, so
# an agent harness is configured once and never again.
# ===========================================================================
[CmdletBinding()]
param(
  # ---------------------------------------------------------------------------
  # THE public surface: twelve parameters (+ -Verbose/-Debug). Everything else is DERIVED: K is fitted
  # to measured VRAM, the context grows on demand, the prefill follows from the drafter, the cache policy
  # is the measured one. Three groups: where it listens / how big it may get / what it does on start;
  # -Tools / -McpProxy GRANT CAPABILITY (off unless asked); -NoVariableKv / -NoLoaderCut / -NoMtp turn
  # off a default (measured on ONE box: another machine needs a way back without editing the source).
  # ---------------------------------------------------------------------------
  [Parameter(Position = 0, Mandatory = $true)]
  [ValidateSet('serve', 'chat', 'setup', 'agents', 'help')][string]$Mode,

  # The address the server binds. 127.0.0.1 keeps it on this machine; ANY other value is LAN
  # mode -- reachable by other devices, with the shutdown route unregistered and the UI's power
  # button gone. See the warning block in cli/server.ps1.
  #
  # The VARIABLE is $BindAddress, not $Host: $Host is a PowerShell automatic variable (the host
  # UI object) and a param of that name shadows it for the whole script. The alias gives the
  # user the `-Host` they expect without taking that risk.
  [Alias('Host')][string]$BindAddress = '127.0.0.1',

  [int]$Port       = 8077,   # HTTP port on the bound address
  [int]$Ctx        = 4096,   # ceiling for the context; it still grows on demand up to this
  [int]$VramBudget = 0,      # MiB the runtime may occupy; 0 = whatever is free
  [switch]$Vision,           # load the vision projector (needs models/mmproj)
  [switch]$Open,             # open the browser once the server is ready
  [switch]$DryRun,           # resolve everything, print the plan, start nothing
  [switch]$Trace,            # raw runtime logs

  # Let the WEB UI run tools on this machine. Empty = off (the default).
  # `all`, or a comma-separated subset: read_file, write_file, exec_shell_command,
  # web_fetch, web_search. Does NOT affect agent harnesses on the API — those send
  # their own tools and run them themselves.
  [string]$Tools   = '',
  # Let the WEB UI reach MCP servers, by fetching them THROUGH this server.
  [switch]$McpProxy,

  # --- the two escape hatches --------------------------------------------------
  # Pin the context instead of growing it. By default LLapis boots at a small
  # context and rebuilds it larger on demand, re-fitting K to the MEASURED free
  # VRAM at every step — which is what lets a two-turn chat keep a big expert
  # cache instead of paying for a context it never reaches. The rebuild is a
  # live reallocation of the KV cache, and it was measured on one GPU. If a
  # session misbehaves around a context change, this pins it at -Ctx for good.
  [switch]$NoVariableKv,

  # Load the expert tensors the stock way. By default the loader is CUT: llama
  # allocates no expert VRAM and moert owns the cold tier out of its own store,
  # which is worth ~584 MiB. It needs models/cache/low.experts.index — without
  # it the cut disables itself anyway. Use this to force the older borrowed
  # path, which costs that VRAM back but touches none of the ownership
  # machinery.
  [switch]$NoLoaderCut,

  # Decode WITHOUT the MTP drafter: a MEASUREMENT lever (it separates what the expert cache is worth
  # from what speculation is worth, findings 149-150). It also turns the streamed prefill off (q4stream
  # needs backend sampling off): the prompt is read token by token at ~75 tok/s.
  [switch]$NoMtp,

  # Keep the drafter's full vocabulary head (272 MiB) in host RAM instead of VRAM: the GPU reads only the
  # candidate rows of the two-stage head over PCIe. Worth it only when VRAM is tight (claude dump findings 248-249).
  [switch]$DraftHeadInRam
)

# ===========================================================================
# Internal configuration -- NOT user surface.
# ===========================================================================
# The values the product runs with: they were command-line parameters while this was a research
# harness. Only the ones the resolution below still reads are left (elision E4, 2026-10-05).
$Model              = $null   # resolved from the saved setup (models/low, models/cache)
$Q4Source           = $null   # the promotion source: models/high (quality) or the IQ2 file (speed)
$K                  = 48      # boot K; re-fitted to the measured VRAM by the runtime
$Ngl                = 99
$ColdArena          = ''      # the owned cold arena index (models/cache/low.experts.index)
$NoExpertAlloc      = $false  # the loader cut (on when the cold arena exists)
$ColdCpuMax         = 8       # CPU cold sidecar width; scaled to the MTP verify width below
$ColdCpuWorkers     = 10
$MtpNMax            = 3       # MTP drafter n_max (verify width 4)


# Continue (not Stop): the native exe writes progress to stderr; under output redirection Stop would
# turn the first stderr line into a terminating error. The guards below use explicit `exit 1`.
$ErrorActionPreference = 'Continue'
# $PSBoundParameters is the CALLING invocation's automatic variable; a dot-sourced FILE (unlike a
# dot-sourced code block) gets its own empty one, so llapis/preset.ps1's `$PSBoundParameters.
# ContainsKey(...)` checks (explicit-flag-wins-over-tuned-default) were silently always false --
# every CLI override of K/ColdCpuMax on
# chat/serve was discarded since the C5.4 decomposition. Capture it here as a PLAIN variable (those
# DO cross a dot-source file boundary) for preset.ps1 to check instead.
$CallerBound = $PSBoundParameters
[Threading.Thread]::CurrentThread.CurrentCulture = [Globalization.CultureInfo]::InvariantCulture

# $Root is the REPOSITORY root, not this file's directory: the launcher lives in cli/,
# while build-cuda/, models/ and llapis-user.json all sit one level up.
$CliDir = $PSScriptRoot
$Root   = Split-Path $CliDir -Parent
$Rel    = Join-Path $Root 'build-cuda\Release'
$Bin    = Join-Path $Root 'build-cuda\bin\Release'
$SrvExe = Join-Path $Rel 'moert-server.exe'

# ===========================================================================
# C0 log level (normal | verbose | debug), driven by the common -Verbose / -Debug
# switches. Normal is the default; -Verbose summarizes internals; -Debug shows the raw
# ggml/CUDA/moert firehose AND unlocks the advanced presets/flags. The C++ side (
# moert-server) reads LLAPIS_LOG to decide what to MIRROR to the terminal; every raw log
# line is always written to LLAPIS_LOG_FILE regardless of level.
# ===========================================================================
$DebugMode   = $PSBoundParameters.ContainsKey('Debug')   -or ($DebugPreference   -eq 'Continue')
$VerboseMode = $PSBoundParameters.ContainsKey('Verbose') -or ($VerbosePreference -eq 'Continue')
$TraceMode   = [bool]$Trace
# C5: -Debug = PHASE mode (runtime operates normally; one clean line per logical-flow step, the
# raw firehose goes to the log file only). -Trace = the pre-C5 -Debug (full raw firehose).
$LogLevel = if ($TraceMode) { 'trace' } elseif ($DebugMode) { 'debug' } elseif ($VerboseMode) { 'verbose' } else { 'normal' }
if ($TraceMode) { $DebugMode = $true }   # -Trace implies every -Debug unlock (advanced flags, raw bench dumps)

# ps1-side phase lines (C5): the wrapper's own steps in the same voice as the C++ llapis_phase
# lines (which carry a [+  12.3s] process-relative timestamp; the wrapper's steps are instant).
function Say-Phase([string]$m) { if ($DebugMode) { Write-Host ("[phase    ] {0}" -f $m) -ForegroundColor Cyan } }

# One full log file per invocation, next to the script. Append-only + shared-safe: the router,
# its loaded child, and any plan/prepare subprocess all target the SAME file (the C++ filter
# opens it with O_APPEND and writes whole lines). PS creates the path; children INHERIT the env.
$LogsDir = Join-Path $Root 'logs'
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
$LogFile = Join-Path $LogsDir ("{0}_{1}.log" -f (Get-Date -Format yyyy-MM-dd_HHmmss), $Mode)
$env:LLAPIS_LOG      = $LogLevel
$env:LLAPIS_LOG_FILE = $LogFile
$env:LLAPIS_NO_STATUS = $null

# `help` needs no runtime; print usage (short in normal, full in -Debug) and exit before any
# exe/model resolution. Real slash-command / wizard work lands in later C0 phases.
if ($Mode -eq 'help') {
  Write-Host ""
  Write-Host "Usage:"
  Write-Host "  llapis chat      Terminal chat (streams from the local LLapis server; starts it if needed)"
  Write-Host "  llapis serve     Start the LLapis server + web UI (OpenAI-compatible API)"
  Write-Host "  llapis agents    Emit the provider JSON for agent clients"
  Write-Host "  llapis setup     Check the models and choose the runtime (quality | speed)"
  Write-Host "  llapis status    Report what the running server is actually running"
  Write-Host "  llapis reset     Clear the server's cached conversation"
  Write-Host "  llapis stop      Stop the running server"
  Write-Host ""
  Write-Host "Models are read from fixed folders under models\ : high\ (higher precision, quality only),"
  Write-Host "  low\ (the fast model, required), mtp\ (drafter, required), mmproj\ (only for -Vision)."
  Write-Host "Context is adaptive up to the model's 262144; -Ctx <n> lowers that ceiling."
  Write-Host ""
  Write-Host "LLapis listens on 127.0.0.1 -- that is a local server, reachable by anything running"
  Write-Host "on this machine, and not by the network. By default it starts NO outgoing network"
  Write-Host "connections, runs NO tools, and gives the model NO access to your files. Two flags"
  Write-Host "change that, and only for the WEB UI:" -ForegroundColor Gray
  Write-Host ""
  Write-Host "  -Tools all        Let the model run tools here: read_file, write_file," -ForegroundColor Yellow
  Write-Host "                    exec_shell_command, web_fetch, web_search. Name a subset"
  Write-Host "                    instead of 'all' to grant less, e.g. -Tools read_file,web_fetch."
  Write-Host "                    The UI asks before each new tool and remembers an 'always';"
  Write-Host "                    that prompt lives in the browser, so any program on this"
  Write-Host "                    machine can call the endpoint directly and skip it."
  Write-Host "  -McpProxy         Let the UI reach MCP servers by fetching them through this" -ForegroundColor Yellow
  Write-Host "                    server, which makes outbound requests on the page's behalf."
  Write-Host ""
  Write-Host "And one that changes who can reach LLapis at all:"
  Write-Host "  -Host <addr>      Bind somewhere other than 127.0.0.1. Anything that can reach" -ForegroundColor Yellow
  Write-Host "                    this machine on the port can then use the model and read the" -ForegroundColor Yellow
  Write-Host "                    conversation -- there is no password. The shutdown route is not"
  Write-Host "                    registered in that mode, so the UI power button disappears;"
  Write-Host "                    stop the server from this machine with 'llapis stop'."
  Write-Host ""
  Write-Host "Neither affects an agent harness on the API (see 'llapis agents'): a harness sends"
  Write-Host "its own tools and runs them in its own process, with or without these flags."
  Write-Host ""
  Write-Host "Two escape hatches, for when a default measured on one GPU does not suit yours:"
  Write-Host "  -NoVariableKv     Pin the context at -Ctx. Normally LLapis boots small and rebuilds"
  Write-Host "                    the KV cache larger on demand, re-fitting the expert cache to the"
  Write-Host "                    measured free VRAM each time. Pin it if a session misbehaves"
  Write-Host "                    around a context change."
  Write-Host "  -NoLoaderCut      Let llama.cpp own the expert tensors instead of moert. Costs about"
  Write-Host "                    584 MiB of VRAM back. The cut already disables itself when"
  Write-Host "                    models/cache/low.experts.index is missing."
  Write-Host "  -NoMtp            Decode without the MTP speculative drafter. A measurement lever:"
  Write-Host "                    it is the only way to separate the expert cache's contribution"
  Write-Host "                    from speculation's. Also turns the streamed prefill off, so the"
  Write-Host "                    prompt is read token by token (~75 tok/s at any length)."
  Write-Host "                    KNOWN DEFECT (findings 150): needs -NoLoaderCut. With the cut on,"
  Write-Host "                    the ubatch decoupling is gated on the spec width, the cut forces"
  Write-Host "                    ubatch 1, and the first decode step crashes."
  Write-Host ""
  Write-Host "One lever for small GPUs:"
  Write-Host "  -DraftHeadInRam   Keep the MTP drafter's vocabulary head (272 MiB) in system RAM" -ForegroundColor Yellow
  Write-Host "                    instead of VRAM; the GPU reads only the ~2 000 candidate rows it"
  Write-Host "                    needs per draft over PCIe (~0.2 ms each, about +1 ms per step)."
  Write-Host "                    The freed VRAM holds ~3 more experts per layer."
  Write-Host "                    USE IT when the VRAM budget is tight -- around 5-6 GiB for the"
  Write-Host "                    model (measured: +9.7% decode at -VramBudget 5120)."
  Write-Host "                    DO NOT use it with the whole GPU: there the cache is already ~95%"
  Write-Host "                    hit and the PCIe reads cost more than the slots give (measured:"
  Write-Host "                    +3.3% instead of +4.2% at full VRAM). Needs the two-stage drafter"
  Write-Host "                    (models/mtp/*.draft-head-*.gguf, built by 'moert-server prepare'); without it the flag does nothing."
  Write-Host ""
  if ($DebugMode) {
    Write-Host "Logging: -Verbose = summarized internals; -Debug = PHASE trace (one line per logical"
    Write-Host "  flow step, runtime operates normally); -Trace = full raw ggml/CUDA/moert firehose."
  } else {
    Write-Host "Use -Debug to watch the runtime's phases (pinning, prefill, refreshes); -Trace for raw logs."
  }
  Write-Host ""
  Write-Host "Full log of each run: $LogsDir\<timestamp>_<mode>.log"
  exit 0
}

# ===========================================================================
# C0 guided setup (llapis-user.json) + wizard. PowerShell is orchestration only:
# it resolves paths, runs the wizard, reads/writes the config, and calls the C++
# planner (`moert-server plan`) for the K envelope. It never parses runtime logs
# to infer state and never computes cache policy itself.
# ===========================================================================
$ModelsRoot = Join-Path $Root 'models'
$ConfigPath = Join-Path $Root 'llapis-user.json'
# Interactive = a real console with stdin attached (the wizard needs it).
$Interactive = [Environment]::UserInteractive -and (-not [Console]::IsInputRedirected)

# Wrapper info lines that belong to the runtime internals: shown only in -Verbose / -Debug so
# normal mode stays quiet (C0a). Warnings/errors are printed unconditionally elsewhere.
function Say-Info([string]$m) { if ($VerboseMode -or $DebugMode) { Write-Host $m -ForegroundColor DarkCyan } }

# --- VRAM prefetch: the FIRST nvidia-smi of a session costs ~0.75 s (driver wake-up), later ones
# 0.04 s. Fired HERE, before the first dot-source, and collected when the clamp needs it; nothing in
# this process allocates VRAM before then. On any failure the callers fall back to their own query,
# then to the named conservative estimate -- never to a silent 0. (history: docs/code-comments-archive-2026-10-05.md, "vram-prefetch")
$script:VramPrefetchProc = $null
$script:VramPrefetchOut  = Join-Path ([IO.Path]::GetTempPath()) ("llapis-vram-$PID.txt")
$script:VramPrefetch     = $null   # @{ free = <MiB>; total = <MiB> } once collected
# Only for the modes that actually read a VRAM number. `agents`/`help` return before any
# consumer runs, and an uncollected prefetch would leave its temp file behind.
if ($Mode -in @('serve', 'chat', 'setup')) {
  try {
    $script:VramPrefetchProc = Start-Process -FilePath 'nvidia-smi' `
        -ArgumentList '--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits' `
        -NoNewWindow -PassThru -RedirectStandardOutput $script:VramPrefetchOut -ErrorAction Stop
  } catch { $script:VramPrefetchProc = $null }
}

# Collect the prefetch exactly once. Returns $null if it was never started, did not finish in time,
# or produced something we cannot parse — callers treat that as "no measurement".
function Get-VramPrefetch {
  if ($null -ne $script:VramPrefetch) { return $script:VramPrefetch }
  if (-not $script:VramPrefetchProc)  { return $null }
  try {
    if (-not $script:VramPrefetchProc.HasExited) { $script:VramPrefetchProc.WaitForExit(4000) | Out-Null }
    if (-not $script:VramPrefetchProc.HasExited) { return $null }
    $line = (Get-Content $script:VramPrefetchOut -ErrorAction Stop | Select-Object -First 1)
    if ("$line" -match '^\s*(\d+)\s*,\s*(\d+)\s*$') {
      $script:VramPrefetch = @{ free = [int]$Matches[1]; total = [int]$Matches[2] }
    }
  } catch {}
  finally {
    $script:VramPrefetchProc = $null
    try { Remove-Item $script:VramPrefetchOut -Force -ErrorAction SilentlyContinue } catch {}
  }
  return $script:VramPrefetch
}

# Fatal-error stop for the dot-sourced modules: [Environment]::Exit, NOT `exit`. Inside a dot-sourced
# file `exit` only unwinds the dot-source and the launcher went on to start a degraded server (proven:
# a ctx-K FATAL came up without the drafter). Every call site is a pre-launch guard, so a hard process
# exit has nothing to clean up. (history: docs/code-comments-archive-2026-10-05.md, "stop-launcher")
function Stop-Launcher([int]$code = 1) { [Environment]::Exit($code) }

# C5.5: the LM Studio fallback paths are GONE. Model discovery is the hardcoded
# models/{high,low,mtp,mmproj} layout defined in llapis/setup.ps1 — one place, no machine-specific
# absolute paths, and a missing file produces a named alert instead of a silent alternate source.

# --- setup / config / wizard subsystem: extracted to llapis/setup.ps1 and dot-sourced here so
#     its functions share this scope (paths, Say-* helpers) exactly as before (C5.4 decomposition). ---
. (Join-Path $CliDir 'setup.ps1')

# `setup` mode: (re)run the wizard unconditionally, then exit.
if ($Mode -eq 'setup') {
  if (-not $Interactive) { Write-Error "setup needs an interactive terminal."; exit 1 }
  $s = Invoke-LlapisWizard
  if ($s) { exit 0 } else { exit 1 }
}

# `agents` mode: emit a Pi-style provider JSON for the local OpenAI-compatible API, derived from
# llapis-user.json. Default = print; -Out <path> = write a file; -Copy = clipboard.
if ($Mode -eq 'agents') {
  # C5.5: this JSON is a CONSTANT (no config, no runtime parameter), so an agent harness is configured
  # ONCE. One model id: on 8 GB only one runtime is resident, and quality|speed is a `setup` decision.
  # contextWindow and maxTokens are both 65536 (user decision 2026-07-28): servable through variable-KV
  # (K shrinks as a conversation fills the window); a large maxTokens matters because the model emits
  # reasoning_content before content. Template, not ConvertTo-Json (PS5.1 unwraps 1-element arrays).
  # (history: docs/code-comments-archive-2026-10-05.md, "agents-json")
  $modelId = 'llapis'
  $json = @"
{
  "providers": {
    "llapis": {
      "baseUrl": "http://127.0.0.1:$Port/v1",
      "api": "openai-completions",
      "apiKey": "llapis-local",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "supportsUsageInStreaming": false,
        "maxTokensField": "max_tokens"
      },
      "models": [
        {
          "id": "$modelId",
          "name": "LLapis - Gemma 4 26B-A4B (local)",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 65536,
          "maxTokens": 65536,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        }
      ]
    }
  }
}
"@
  Write-Host $json
  exit 0
}

# ===========================================================================
# Setup consumption (chat/serve, and every non-debug run).
# ===========================================================================
# The saved setup carries ONE field -- mode. Model paths come from the hardcoded layout (setup.ps1), K
# is fitted to measured VRAM, the context is adaptive, vision is a launch flag. So this block resolves
# the layout, applies the FAIL MATRIX and picks the mode. A missing setup defaults to quality (speed
# when the high model is absent) and says so. (history: docs/code-comments-archive-2026-10-05.md, "setup-consumption")
$SetupApplied  = $false
$CacheLabel    = $null
$script:LlapisMode     = $null
$script:LlapisModels   = $null
# VRAM ceiling for the whole runtime, in MiB. 0 = MAX (use whatever is free). Set from the saved
# setup below, or from -VramBudget. preset.ps1 turns it into the keep-free figure the boot clamp
# and the runtime refit both consume, so K is maximized INSIDE it rather than against the card.
$script:VramBudgetMib = if ($PSBoundParameters.ContainsKey('VramBudget')) { [int]$VramBudget } else { 0 }
# This used to be guarded by a $ManualConfig test on -Preset/-Model/-Q4Source. None of the three
# is a parameter any more (they are internal variables), so ContainsKey could never be true and
# the guard was constant-false. Removed rather than left as decoration.
if (($Mode -in @('chat', 'serve')) -or -not $DebugMode) {
  $s = Read-LlapisSetup
  $want = 'quality'
  if (Test-LlapisSetup $s) {
    $want = $s.mode
  } elseif ($s) {
    Say-Info "[llapis] llapis-user.json has no usable 'mode' — defaulting to quality. Run 'setup' to choose."
  }

  # Hardcoded layout + fail matrix. Halts on a missing low/mtp model; downgrades
  # quality -> speed (loudly) when the high model is absent.
  $script:LlapisModels = Resolve-LlapisModels
  $script:LlapisMode   = Assert-LlapisModels $script:LlapisModels $want

  if ($script:LlapisMode -eq 'quality') {
    # base = the non-expert-Q4 splice (preset.ps1 resolves it out of models/cache), promotions
    # streamed from the Q4_K_XL source: hot experts at Q4 in VRAM over an IQ2 cold tier in RAM.
    $Q4Source = $script:LlapisModels.High
  } else {
    # speed = the precision-matched IQ2 configuration: both tiers point at the same IQ2 file (base AND
    # promotion source), so output matches stock while the dual runtime keeps the streamed prefill
    # (measured 5/5 vs stock b9932+MTP: TTFT 4287 vs 8013 ms, prefill 1137 vs 593 tok/s).
    # (history: docs/code-comments-archive-2026-10-05.md, "speed-mode")
    $Model    = $script:LlapisModels.Low
    $Q4Source = $script:LlapisModels.Low
  }
  # VRAM budget (setup v3): 0 = MAX. An explicit -VramBudget wins for this session.
  if (-not $PSBoundParameters.ContainsKey('VramBudget') -and
      ($s -and $s.PSObject.Properties.Name -contains 'vram_budget_mib')) {
    $script:VramBudgetMib = [int]$s.vram_budget_mib
  }
  $CacheLabel   = 'Auto'      # K is VRAM-fitted now; the label is a status-line word, not a choice
  $SetupApplied = $true
  # LLAPIS_CACHE_LABEL and LLAPIS_MODE used to be exported here. Nothing read them — not the
  # runtime, not the patch, not another script — and the comment that justified LLAPIS_MODE
  # ("/moe/info reports it") was simply false. `llapis status` reads the mode from
  # llapis-user.json, which is where the decision actually lives.
  Say-Info ("[llapis] runtime = {0} (saved setup); models from {1}" -f $script:LlapisMode, $ModelsRoot)
}

# ===========================================================================
# Variable-KV: DEFAULT ON for chat/serve. Boot at 8192, grow on demand up to the model-native 262144;
# K is re-fitted to the measured VRAM at every step. -Ctx <n> CAPS the ceiling (a cap at or below the
# boot size pins); -NoVariableKv pins at -Ctx (else 8192); no saved setup pins.
# ===========================================================================
# (history: docs/code-comments-archive-2026-10-05.md, "vkv-default")
$script:VkvOn      = $false
$script:VkvMinCtx  = 8192      # = vkv::Params::min_ctx; kept in sync explicitly, not by luck
$script:VkvMaxCtx  = 0
$script:VkvNativeCtx = 262144  # = vkv::Params::max_ctx (the model/train limit)
$script:VkvOffWhy  = $null
if ($Mode -in @('chat', 'serve')) {
  if ($NoVariableKv) {
    $script:VkvOffWhy = '-NoVariableKv'
    if (-not $PSBoundParameters.ContainsKey('Ctx')) { $Ctx = $script:VkvMinCtx }
  }
  elseif (-not $SetupApplied) { $script:VkvOffWhy = 'no saved setup (dev/bench path)' }
  else {
    $ceil = if ($PSBoundParameters.ContainsKey('Ctx')) { $Ctx } else { $script:VkvNativeCtx }
    if ($ceil -le $script:VkvMinCtx) {
      # A cap at or under the boot size leaves no ladder to climb — pin instead of pretending.
      $script:VkvOffWhy = "-Ctx $ceil is at or below the $($script:VkvMinCtx) boot floor"
      $Ctx = $ceil
    } else {
      $script:VkvOn     = $true
      $script:VkvMaxCtx = $ceil
      $Ctx              = $script:VkvMinCtx    # everything downstream (K clamp, profile, child -c) follows the BOOT ctx
    }
  }
}

# --- runtime resolution (preset -> placement + env): extracted to llapis/preset.ps1 and
#     dot-sourced here so it shares this scope exactly as before (C5.4 decomposition). ---
. (Join-Path $CliDir 'preset.ps1')

# --- startup banner (logo + LED wordmark): defines Show-LlapisBanner, called from server.ps1
#     right before the chat/serve header. ---
. (Join-Path $CliDir 'banner.ps1')

# ---------------------------------------------------------------------------
# --- chat REPL: extracted to llapis/chat.ps1 and dot-sourced here (C5.4 decomposition). ---
. (Join-Path $CliDir 'chat.ps1')

# --- OS-specific helpers (child-tree lifetime): llapis/platform.ps1, the one file a port replaces. ---
. (Join-Path $CliDir 'platform.ps1')

# --- serve + server-backed chat launch: extracted to llapis/server.ps1. Dot-source DEFINES the
#     function; the CALL runs it (for serve|chat it exits; for bench it returns and we fall through
#     to the bench path below). It must be a called function, not inline: a dot-sourced `exit` does
#     not halt the launcher. C5.4 decomposition. ---
. (Join-Path $CliDir 'server.ps1')
Invoke-LlapisServe

