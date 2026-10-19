# ===========================================================================
# llapis/preset.ps1 - runtime resolution: mode -> placement + MOE_DUAL_* env.
# ===========================================================================
# DOT-SOURCED by llapis.ps1 after the saved setup is applied and before the server launch, so it shares
# the launcher scope: it reads the params + $K/$Ctx/$Model/$Q4Source/$Vision/... and WRITES the child
# env + $place/$script:mtpCliArgs/$script:visionMmproj. Order is load-bearing (the vision K clamp runs
# before the VRAM clamp; the ColdCpuMax re-stamp runs after MTP sets spec_k).
# The product path: fused dual-tier experts, the boot-slab dynamic cache, the CPU cold sidecar, the MTP
# drafter, the q4stream prefill, vision -- resolved per mode, per VRAM budget and per context.
# ===========================================================================
# ===========================================================================

# --- defaults for model / pack (the known local paths) ---
# `moert-server prepare` derives the runtime artifacts from models/high|low into models/cache/.
$CacheDir = Join-Path $Root 'models\cache'
# Only quality reaches this: speed arrives with $Model already pointing at the IQ2 file, which
# it uses as BOTH tiers. Quality's base is the non-expert-Q4 splice — experts<-IQ2_M, everything
# else<-Q4_K_XL — which only `prepare` produces.
if (-not $Model) { $Model = Join-Path $CacheDir 'base_neq4.gguf' }
# The loader cut is the gate-proven configuration: ON by default when the owned cold arena is
# available, which is what makes it possible. -NoLoaderCut forces the legacy borrowed path.
if (-not $NoLoaderCut) {
  $ix = Join-Path $CacheDir 'low.experts.index'
  if (Test-Path $ix) {
    $NoExpertAlloc = $true
    $ColdArena = $ix   # the cut needs the owned cold-arena source: the prepared store
    Say-Info "[llapis] loader cut ON by default (product config, -584 MiB VRAM); -NoLoaderCut to disable"
  } else {
    Write-Host "[llapis] cold arena not found ($ix) - staying on the no-cut borrowed path" -ForegroundColor Yellow
    Write-Host "            (run 'moert-server prepare' to build the store and get the product config)" -ForegroundColor Yellow
  }
}
# C5.5: fall back to the HARDCODED layout (models/low, models/high), never to a
# machine-specific LM Studio path. The layout is resolved once in llapis/setup.ps1; these two
# lines only matter for the dev/bench paths that skip the setup-consumption block entirely.
if (-not $Model)    { $Model    = (Find-LlapisGguf $script:DirLow  $script:ExpectLow).Path }
if (-not $Q4Source) { $Q4Source = (Find-LlapisGguf $script:DirHigh $script:ExpectHigh).Path }

if (-not $Model)             { Write-Error "no base model resolved: put the low model's GGUF in $($script:DirLow), or pass -Model"; Stop-Launcher 1 }
if (-not (Test-Path $Model)) { Write-Error "model not found: $Model"; Stop-Launcher 1 }

# --- stage the fresh DLLs next to the exe (the build emits them into bin\Release). Skip
#     unchanged files: a running server locks its DLLs and PS 5.1 prints the share-violation
#     even under SilentlyContinue; identical timestamp+size means there is nothing to stage. ---
foreach ($d in 'ggml-base', 'ggml-cpu', 'ggml-cuda', 'ggml', 'llama', 'llama-common', 'llama-server-impl', 'mtmd') {
  $src = Join-Path $Bin "$d.dll"
  $dst = Join-Path $Rel "$d.dll"
  if (-not (Test-Path $src)) { continue }
  $s = Get-Item $src
  $t = $null
  if (Test-Path $dst) { $t = Get-Item $dst }
  if ($t -and $t.LastWriteTimeUtc -eq $s.LastWriteTimeUtc -and $t.Length -eq $s.Length) { continue }
  try { Copy-Item $src $dst -Force -ErrorAction Stop } catch {
    Write-Host "[llapis] note: $d.dll is in use (running server?) -- keeping the already-staged copy" -ForegroundColor DarkGray
  }
}

# --- stage the CUDA runtime next to the exe -------------------------------------------
# These are NOT produced by the build: they ship with the CUDA Toolkit, and they live in
# <CUDA_PATH>\bin\x64 (not \bin), so they are not on PATH by default either. Without them
# the server may still start by resolving some other copy on the system -- which is worse
# than failing, because it runs SLOWER with no error at all. Measured: 42.7 vs 50.8 tok/s
# on the same build, a 12% loss whose only symptom was the number.
$cudaBin = if ($env:CUDA_PATH) { Join-Path $env:CUDA_PATH 'bin\x64' } else { $null }
if ($cudaBin -and (Test-Path $cudaBin)) {
  foreach ($d in 'cudart64_13.dll', 'cublas64_13.dll', 'cublasLt64_13.dll') {
    $src = Join-Path $cudaBin $d
    $dst = Join-Path $Rel $d
    if (-not (Test-Path $src)) { continue }
    if (Test-Path $dst) {
      $s = Get-Item $src; $t = Get-Item $dst
      if ($t.Length -eq $s.Length) { continue }   # already staged, and they never change in place
    }
    try { Copy-Item $src $dst -Force -ErrorAction Stop } catch { }
  }
} elseif (-not (Test-Path (Join-Path $Rel 'cudart64_13.dll'))) {
  Write-Host "[llapis] CUDA_PATH is not set and the CUDA runtime is not staged next to the server." -ForegroundColor Yellow
  Write-Host "         Copy cudart64_13.dll, cublas64_13.dll and cublasLt64_13.dll from" -ForegroundColor Yellow
  Write-Host "         <CUDA Toolkit>\bin\x64 into $Rel -- see the README." -ForegroundColor Yellow
}

# --- clear any inherited MOE_DUAL_* (and our GGML_CUDA_* toggles) so a preset is deterministic.
#     GGML_CUDA_MOE_DUAL_GRAPH leaking between arms in one shell let the graph-disable override
#     survive into a staging-ring run -> mid-capture sync -> CUDA "operation not permitted" crash.
#     An inherited MOE_DUAL_* must never reach the child, EXCEPT the A/B levers below, which a
#     measurement harness exports on purpose (each one is read by the runtime; unset = the default):
#     MOE_DUAL_PREFILL_MIN (the streamed-vs-fused threshold), the policy's prices
#     MOE_DUAL_COST_H / _R / _PER_TOKEN, MOE_DUAL_ASYNC_PROMOTE, MOE_DUAL_ARENA_EXCLUSIVE,
#     MOE_DUAL_GRAPH_SEGMENTS, MOE_DUAL_Q4S_ARENA, MOE_DUAL_PREFIX_REUSE; the bench capture traces
#     MOE_DUAL_ROUTE_TRACE / _CONTENT_TRACE / _CONTENT_SKIP_EMBD; and the speculative-caching arm still under
#     evaluation (claude dump findings 216l): MOE_DUAL_RAM_DIR (batch transport), MOE_DUAL_MTP_NN / _R / _CAP
#     (MTP forecaster), MOE_DUAL_EARLY_READY, MOE_DUAL_DIR_SM (SM copy transport, findings 231). Any other env-only knob read by this file after the wipe would be
#     silently dead, so there are none (elision E4, 2026-10-11). ---
$keepBatchOverrides = @('MOE_DUAL_PREFILL_MIN', 'MOE_DUAL_COST_H', 'MOE_DUAL_COST_R', 'MOE_DUAL_COST_PER_TOKEN', 'MOE_DUAL_ASYNC_PROMOTE', 'MOE_DUAL_ARENA_EXCLUSIVE', 'MOE_DUAL_GRAPH_SEGMENTS', 'MOE_DUAL_Q4S_ARENA', 'MOE_DUAL_PREFIX_REUSE', 'MOE_DUAL_EARLY_READY', 'MOE_DUAL_ROUTE_TRACE', 'MOE_DUAL_DECISION_TRACE', 'MOE_DUAL_CONTENT_TRACE', 'MOE_DUAL_CONTENT_SKIP_EMBD', 'MOE_DUAL_RAM_DIR', 'MOE_DUAL_DIR_SM', 'MOE_DUAL_DIR_SM_THREADS', 'MOE_DUAL_DIR_SM_PIN', 'MOE_DUAL_EARLY_DECIDE', 'MOE_DUAL_DIR_ENGINE', 'MOE_DUAL_DRAFT_GGUF', 'MOE_DUAL_DIR_ENGINE_MIB', 'MOE_DUAL_DIR_SM_PIN_M', 'MOE_DUAL_DIR_SM_RING_MIB', 'MOE_DUAL_MTP_NN', 'MOE_DUAL_MTP_NN_R', 'MOE_DUAL_MTP_NN_CAP', 'MOE_DUAL_LOAN', 'MOE_DUAL_LOAN_FILL', 'MOE_DUAL_ORACLE', 'MOE_DUAL_ORACLE_H', 'MOE_DUAL_ORACLE_R', 'MOE_DUAL_ORACLE_CAP', 'MOE_DUAL_L3PF', 'MOE_DUAL_L3PF_DRY')
Get-ChildItem Env: | Where-Object { $_.Name -like 'MOE_DUAL_*' -and $keepBatchOverrides -notcontains $_.Name } | ForEach-Object {
  # say it: a lever dropped here silently made a whole A/B arm run the default (findings 210)
  Write-Host ("  NOTE: ignoring inherited {0}={1} (not an A/B lever; see `$keepBatchOverrides in cli/preset.ps1)" -f $_.Name, $_.Value) -ForegroundColor Yellow
  Set-Item "Env:$($_.Name)" $null }
$keptLevers = @(Get-ChildItem Env: | Where-Object { $keepBatchOverrides -contains $_.Name } | ForEach-Object { "$($_.Name)=$($_.Value)" })
if ($keptLevers.Count) { Write-Host ("  A/B levers from the environment: " + ($keptLevers -join ' ')) -ForegroundColor Yellow }
$env:GGML_CUDA_MOE_DUAL_GRAPH = $null
$env:GGML_CUDA_DISABLE_GRAPHS = $null

# ===========================================================================
# PREFILL CALIBRATION TABLE -- the box/model-specific constants the resolution reads, all MEASURED on
# this machine (RTX 4070 8 GB, 16 GB RAM, Gemma-4-26B-A4B IQ2_M + Q4_K_XL). Single source of truth: a
# different GPU/SSD/model recalibrates HERE. Write-PrefillPlan prints the resolved values.
# ===========================================================================
# ===========================================================================
$script:PrefillCalib = [ordered]@{
  # ctx / K envelope
  KMin             = $script:FitCalib.KMin   # below this the expert cache is pointless -> refuse ctx
  CtxDrafterCeil   = 16384   # informational only now: ctx above this needs > 1 prefill pool pass (a
                             # prefill-coverage note, NOT the KV/K ceiling -- that is computed below)
  # --- VRAM MODEL for K_max(ctx) ----------------------------------------
  #   VRAM at the drafter-load PEAK(ctx,K) ~= FixedOverhead + DrafterLoadPeak + KvMibPerTok*ctx + SlabMibPerK*K
  # The drafter loads LAST, with the slab + NE + KV all resident -- that peak is what actually gates the
  # load. The three quant-dependent terms -- SlabMibPerK, KvMibPerTok and the non-expert part of
  # FixedOverhead -- are read from the installed GGUFs ($script:ModelTerms, setup.ps1's
  # Get-LlapisModelTerms) and the quant-independent residual lives in setup.ps1's FitCalib. On the
  # reference pair they reproduce the constants this table held until 2026-10-19: 108 / 62 MiB per K
  # (Q4_K_XL / IQ2_M; 61 measured), 0.0195 MiB per token, FixedOverhead 3869 / 3088 MiB.
  DrafterLoadPeakMib = $script:FitCalib.DrafterLoadPeakMib   # transient the post-boot-release baseline does
                             # NOT show: measured peak at ctx32768/K31 was 8174 MiB vs the steady formula's 7835
  SafetyMarginMib  = $script:FitCalib.SafetyMarginMib   # pure safety buffer above the modelled peak
  AssumedFreeVramMib = 7200  # fallback free-VRAM estimate ONLY when nvidia-smi is unavailable (Get-FreeVramMiB
                             # returns $null) -- conservative-idle for an 8 GB card
  # merged-graph / batch envelope
  BatchFloor       = 4096    # historical safe base = split 64 x the 64-merge limit
  MergeSafetyMul   = 64      # proven-safe merged-ubatch count; safeMax = split x this
  # reserve VRAM budget (Resolve-Q4sBatch free-VRAM gate)
  ReserveMibPerTok = 0.18    # merged-graph host mask + compute growth per batch token
  ReserveBudgetMib = 2400    # reserve headroom budget (decoupled prefill: 'auto' reaches the safe ceiling)
  FreeVramMarginMib = 1024   # keep this much VRAM free above the reserve budget
  SplitDefault     = 256     # prefill split: 30 stagings, 326 MiB decode headroom
}

# --- effective-config introspection (R-a) ------------------------------------
# The resolution below is spread across a K-clamp, a VRAM-gated batch function and
# a defaults block, each of which can be overridden by a CLI flag, an env var, the
# calib table or a hard-coded fallback. Add-Plan records each RESOLVED value with
# its PROVENANCE as it is decided; Write-PrefillPlan prints the whole plan once and
# dumps it to JSON so a gate/operator can confirm the child ran the config they
# think it did — instead of inferring it from log side-effects after the fact.
$script:PrefillPlan = [ordered]@{}
$script:kClamped = $false   # set true by the ctx-driven K-clamp below; read by the plan's K provenance
function Add-Plan([string]$name, $value, [string]$source) {
  $script:PrefillPlan[$name] = [pscustomobject]@{ value = $value; source = $source }
}
function Write-PrefillPlan {
  if (-not $script:PrefillPlan -or $script:PrefillPlan.Count -eq 0) { return }
  Write-Host "[llapis] prefill plan (resolved effective config):" -ForegroundColor DarkCyan
  foreach ($k in $script:PrefillPlan.Keys) {
    $e = $script:PrefillPlan[$k]
    Write-Host ("             {0,-20} = {1,-9} [{2}]" -f $k, $e.value, $e.source) -ForegroundColor DarkCyan
  }
  $planDir = Join-Path $Root 'logs'
  try { New-Item -ItemType Directory -Force -Path $planDir | Out-Null } catch {}
  $planPath = Join-Path $planDir 'prefill_plan.json'
  try {
    $obj = [ordered]@{}
    foreach ($k in $script:PrefillPlan.Keys) { $obj[$k] = [ordered]@{ value = "$($script:PrefillPlan[$k].value)"; source = $script:PrefillPlan[$k].source } }
    ($obj | ConvertTo-Json -Compress -Depth 4) | Out-File -Encoding utf8 $planPath
  } catch {}
}

# --- VRAM measurement + K_max(ctx) model (defined before the clamp that consumes them) ---------------
# VRAM budget -> keep-free: the user says how much VRAM LLapis may use; the runtime refits K against how
# much must stay FREE. keep_free = max(default_keep, total_vram - budget): a budget above the card is MAX,
# and 250 MiB (the measured keep_post ladder) is never eroded. (history: docs/code-comments-archive-2026-10-05.md, "keep-free")
# steady state from a real spill).
$script:VkvKeepMib = 250   # = server_main.cpp's g_vkv_keep_post default; kept in sync explicitly
$script:KeepFreeResolved = $null   # memo: two call sites (env stamp + boot clamp), one decision + one message
function Resolve-KeepFreeMiB {
  if ($null -ne $script:KeepFreeResolved) { return $script:KeepFreeResolved }
  $script:KeepFreeResolved = Resolve-KeepFreeMiBOnce
  return $script:KeepFreeResolved
}
function Resolve-KeepFreeMiBOnce {
  $budget = if ($script:VramBudgetMib) { [int]$script:VramBudgetMib } else { 0 }
  if ($budget -le 0) { return $script:VkvKeepMib }
  $total = Get-TotalVramMiB
  if (-not $total) {
    Write-Host "[llapis] -VramBudget given but nvidia-smi is unavailable — cannot convert it to a keep-free target; ignoring the budget." -ForegroundColor Yellow
    return $script:VkvKeepMib
  }
  if ($budget -ge $total) {
    Say-Info ("[llapis] VRAM budget {0} MiB >= card total {1} MiB — treating as MAX." -f $budget, $total)
    return $script:VkvKeepMib
  }
  $keep = $total - $budget
  if ($keep -lt $script:VkvKeepMib) { $keep = $script:VkvKeepMib }
  return [int]$keep
}

function Get-FreeVramMiB {
  # Prefetched (llapis.ps1 fires it before the first dot-source so the driver's first-call wake
  # overlaps the resolution). ONE snapshot serves every pre-launch consumer — the ctx-K clamp and
  # Resolve-Q4sBatch are sizing the same launch and both run before the router exists, so a single
  # consistent reading is if anything more correct than two taken seconds apart.
  $pf = if (Get-Command Get-VramPrefetch -ErrorAction SilentlyContinue) { Get-VramPrefetch } else { $null }
  if ($pf -and $pf.free) { return [int]$pf.free }
  try {
    $out = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -eq 0 -and $out) {
      $v = (($out | Select-Object -First 1) -as [string]).Trim()
      if ($v -match '^\d+$') { return [int]$v }
    }
  } catch {}
  return $null
}

# The runtime's mode, from the resolved promotion source: `speed` points it at the low model itself
# (both tiers at one precision), `quality` at the high one. True also for a hand-rolled
# `-Q4Source <the low file>`, which is how speed was gated.
$script:FitLow = (Find-LlapisGguf $script:DirLow $script:ExpectLow).Path
$script:FitMode = if ($Q4Source -and $script:FitLow -and
                      ([IO.Path]::GetFullPath($Q4Source) -ieq [IO.Path]::GetFullPath($script:FitLow))) { 'speed' } else { 'quality' }
# The installed models' VRAM terms (setup.ps1's Get-LlapisModelTerms: GGUF headers, no load). STRICT: without
# them the K clamp would be guessing for quants it was never measured on, so the launch stops here.
$fitDrafter = if ($script:LlapisModels) { $script:LlapisModels.Mtp } else { (Find-LlapisGguf $script:DirMtp $script:ExpectMtp).Path }
$script:ModelTerms = Get-LlapisModelTerms $script:FitMode $Q4Source $script:FitLow $fitDrafter
if (-not $script:ModelTerms) {
  Write-Host "[llapis] FATAL: cannot read the installed models' sizes: $($script:ModelTermsError)" -ForegroundColor Red
  Stop-Launcher 1
}
Say-Info ("[llapis] model terms ({0}): {1:N1} MiB per hot slot, non-experts {2} MiB, staging {3} MiB, KV {4:N4} MiB/token, cold arena {5} MiB" -f
          $script:FitMode, [double]$script:ModelTerms.slab_mib_per_k, $script:ModelTerms.ne_vram_mib,
          $script:ModelTerms.staging_mib, [double]$script:ModelTerms.kv_mib_per_tok, $script:ModelTerms.cold_arena_mib)
# The C++ admission preflight (moe_dual_init.cpp) demands `slab + headroom` of free VRAM before any device
# allocation; its built-in 3400 is the Q4_K_XL non-experts (2459) + a 941 MiB residual. Left at 3400 with
# another quant it DISAGREES with the clamp below (seen with IQ2: the clamp gave K=62, the preflight refused
# with exit 86), so it is derived from the same terms. LLAPIS_* is the only prefix that survives to the
# child; MOE_DUAL_* would be wiped and silently do nothing.
if (-not $env:LLAPIS_PREFLIGHT_HEADROOM_MIB) {
  $stagingExcess = [Math]::Max(0, [int]$script:ModelTerms.staging_mib - $script:FitCalib.RefStagingMib)
  $env:LLAPIS_PREFLIGHT_HEADROOM_MIB = "$([int]$script:ModelTerms.ne_vram_mib + $script:FitCalib.PreflightResidualMib + $stagingExcess)"
}
function Get-SlabMibPerK { return [double][Math]::Ceiling([double]$script:ModelTerms.slab_mib_per_k) }
function Get-KvMibPerTok { return [double]$script:ModelTerms.kv_mib_per_tok }
# The boot K's upper bound: every expert of a layer. The VRAM model below is what binds in practice; the
# runtime's measured refit (vkv_refit_k) is the authority after the first prompt either way.
function Get-KMaxCap { return [int]$script:ModelTerms.n_expert }
# Modelled VRAM peak at MTP-drafter load (the gating moment: slab + NE + KV all resident) for a given
# ctx/K, in MiB.
function Get-DrafterLoadPeakMiB([int]$ctx, [int]$k) {
  $c = $script:PrefillCalib
  return ((Get-LlapisFixedOverheadMiB $script:ModelTerms) + [double]$c.DrafterLoadPeakMib + ((Get-KvMibPerTok) * $ctx) + ((Get-SlabMibPerK) * $k))
}

# Largest K whose modelled drafter-load peak fits inside (free - safety margin), capped at [0, KMax].
# Returns a hashtable: KMax (>=0), Free (MiB used), Measured (bool: nvidia-smi vs assumed fallback).
function Resolve-KMaxForCtx([int]$ctx, [switch]$Physical) {
  $c = $script:PrefillCalib
  $free = Get-FreeVramMiB
  $measured = ($null -ne $free)
  if (-not $measured) { $free = [int]$c.AssumedFreeVramMib }
  $margin = [double]$c.SafetyMarginMib
  # A VRAM budget raises the effective margin to the keep-free figure it implies, so the BOOT K is
  # already inside the budget instead of being clamped down on the first refit. -Physical asks the
  # other question: what fits on the card at all, ignoring the user's budget.
  $keep = Resolve-KeepFreeMiB
  if (-not $Physical -and $keep -gt $margin) { $margin = [double]$keep }
  $budgetForSlab = $free - $margin - (Get-DrafterLoadPeakMiB $ctx 0)   # peak at K=0 = everything but the slab
  $slabK = Get-SlabMibPerK
  # NOTE (C5.6): this is a BOOT estimate and it is pessimistic on purpose. It must survive the
  # MTP drafter's load peak (DrafterLoadPeakMib, a transient) on top of a FixedOverhead that is a
  # back-solved constant, with SafetyMarginMib on top of that — together ~500 MiB more than the
  # runtime's own steady-state accounting charges. The authority is vkv_refit_k, which measures
  # free VRAM at the first prompt end (staging already released, everything else resident) and
  # rebuilds the slab at the K that actually fits: 62 -> 71 here at ctx 8192. Do NOT "fix" the
  # gap by trimming these constants without a load-peak measurement — they guard the moment the
  # refit cannot see, and exit 86 is what being wrong looks like.
  $kmax = if ($budgetForSlab -le 0) { 0 } else { [int][Math]::Floor($budgetForSlab / $slabK) }
  $kcap = Get-KMaxCap
  if ($kmax -gt $kcap) { $kmax = $kcap }
  if ($kmax -lt 0)       { $kmax = 0 }
  return @{ KMax = $kmax; Free = [int]$free; Measured = $measured }
}

if ($ColdArena)        { $env:MOE_DUAL_COLD_ARENA = $ColdArena }
if ($NoExpertAlloc)    { $env:MOE_DUAL_NO_EXPERT_ALLOC = '1' }
# VRAM budget -> the runtime's keep-free target (C5.5). Stamped here, after this file's MOE_DUAL_*
# wipe, so server.ps1 collects it into the child profile env like every other knob. It drives BOTH
# the boot fit and every post-resize refit (server_main.cpp reads it once into g_vkv_keep_post and
# hands the same value to vkv_set_boot_policy), which is why one number is enough.
# CAVEAT: the refit only runs when variable-KV is armed. With a pinned context (-Ctx / -NoVariableKv)
# the budget still shapes the BOOT K through Resolve-KMaxForCtx above, but nothing re-fits afterwards.
$script:KeepFreeMib = Resolve-KeepFreeMiB
if ($script:KeepFreeMib -ne $script:VkvKeepMib) {
  $env:MOE_DUAL_VKV_KEEP_POST_MIB = "$($script:KeepFreeMib)"
  $tot = Get-TotalVramMiB
  Write-Host ("[llapis] VRAM budget: up to {0} MiB of {1} MiB -> keeping {2} MiB free; the expert cache is sized to what is left and re-fitted at every context change. -VramBudget 0 for MAX." -f $script:VramBudgetMib, $tot, $script:KeepFreeMib) -ForegroundColor Cyan
}
# Variable-KV (DEFAULT ON for chat/serve; resolved in llapis.ps1, see the block after the saved
# setup is applied). MOE_DUAL_VKV is the canonical arming name and is set HERE, after this file's own
# MOE_DUAL_* wipe -- llapis/server.ps1 then collects every MOE_* into the child profile's env, so the
# child inherits it through the router exactly like the other knobs. The MIN/MAX are stamped
# explicitly rather than left to the header's defaults: the launcher already booted the context at
# MIN_CTX, and a silent disagreement between the two would show up as a resize on turn one.
if ($script:VkvOn) {
  $env:MOE_DUAL_VKV         = '1'
  $env:MOE_DUAL_VKV_MIN_CTX = "$($script:VkvMinCtx)"
  $env:MOE_DUAL_VKV_MAX_CTX = "$($script:VkvMaxCtx)"
  Say-Info ("[llapis] variable-KV ON (default): boot ctx {0}, grows on demand to {1}; K re-fits to the measured VRAM at each step. Pass -Ctx <n> to pin a fixed context, or -NoVariableKv." -f $script:VkvMinCtx, $script:VkvMaxCtx)
} elseif ($script:VkvOffWhy -and $Mode -in @('chat','serve')) {
  Say-Info ("[llapis] variable-KV OFF ({0}): context PINNED at {1} for the whole session." -f $script:VkvOffWhy, $Ctx)
}

# --- placement + precision labels + env -------------------------------------------------
# ONE resolution. Both runtimes are the same dual-precision path; what differs is which file
# each tier points at:
#   quality  base = the non-expert-Q4 splice, promotions streamed from the Q4_K_XL source
#   speed    base = the IQ2, and the promotion source is THAT SAME IQ2 file
# so under `speed` hot and cold hold identical bytes -- the dual machinery still runs, which
# is what keeps the streamed q4stream prefill, and quality matches stock exactly.
$place = @('--ngl', "$Ngl")

if (-not (Test-Path $Model)) {
  Write-Error "the base model was not found: $Model`n  Run 'moert-server prepare' to derive it (see the README)."
  Stop-Launcher 1
}
# Start K at the SOURCE's own cap: the ctx clamp below only ever LOWERS it. Starting lower
# would boot small and leave the runtime's refit to climb back, paying a slab rebuild on the
# first prompt for nothing.
$K = Get-KMaxCap

# World C placement: non-experts + attention on GPU, IQ2 expert pool in RAM -> Plan C staging ring.
$place += @('--cpu-moe', '--no-mmap')
# --no-host: the tensors llama.cpp keeps on the CPU (the input layer: token_embd, 748 MiB at Q8_0 in quality,
# 484 at IQ2 in speed, plus the drafter's 272) go to plain RAM instead of a pinned CUDA_Host buffer, which
# WDDM counts as shared GPU memory for the whole session. The lookup reads a few rows per token on the CPU.
# LLAPIS_EMBD_PINNED=1 keeps the pinned buffer (the A arm of the check). -DraftHeadInRam keeps it too: its
# kernels read the drafter's head through the pinned buffer's device alias, and the same bytes in plain RAM,
# registered on the fly, crashed the first decode (CUDA "out of memory" at the slot init, 3.5 GiB free).
if ($env:LLAPIS_EMBD_PINNED -ne '1' -and -not $DraftHeadInRam -and $env:LLAPIS_DRAFT_HEAD_HOST -ne '1') { $place += '--no-host' }
$env:MOE_DUAL_FUSED = '1'
$env:MOE_DUAL_PLANC = '1'
# LIVE dynamic Q4 cache streamed from the promotion source (starts empty: the boot slab, findings 183).
if (-not (Test-Path $Q4Source)) {
  Write-Error "the promotion source was not found: $Q4Source"
  Stop-Launcher 1
}
$env:MOE_DUAL_DYNAMIC = $Q4Source
$env:MOE_DUAL_K       = "$K"
$hotMode = "dynamic K$K (staged; the first prompt's prefill fills it)"

$hotQuant  = Get-LlapisQuantLabel $Q4Source
$coldQuant = Get-LlapisQuantLabel $script:FitLow
$nonexpertPrec = "$hotQuant (GPU, World C)"
$hotPrec = "$hotQuant ($hotMode, VRAM)"; $coldPrec = "$coldQuant (base experts, RAM -> staging ring)"
# C5.4: vision is driven by the REMEMBERED setup preference (llapis-user.json
# `vision`, applied in the setup-consumption block above) + explicit -Vision/-NoVision — NOT a
# disk-probe. The old auto-on ("projector present next to the model => enable vision") silently
# forced the ~1 GiB tower load (+~8 s startup) and the K 32->22 clamp on every text session just
# because mmproj-Q8_0.gguf sits in models/high. Capability-on-disk != user-wants-it-now. When the
# user does want images, `setup` remembers it, or -Vision turns it on for one session.
if ($Vision -and $Mode -in @('chat', 'serve') -and -not $SetupApplied) {
  Say-Info "[llapis] vision ON (explicit -Vision); text sessions default to vision OFF — set it in 'setup' to remember the choice."
}

# C6: the projector is GPU-encoded (~2.9x faster image prefill than CPU-encode, measured 15.95 s vs
# 46-56 s), so K is clamped to 22 to fit the ~1 GiB Q8 tower beside the hot slab on this 8 GB box (K is a
# hit-rate lever, not a speed one). The CPU-encode path and its ArenaShrink RAM relief were removed in
# elision E2 (they were no longer reachable from the launcher).
if ($Vision -and $Mode -in @('chat', 'serve')) {
  $K = 22
  $env:MOE_DUAL_K = "$K"   # re-stamp: the block above already baked the pre-clamp K into the env
  Write-Host "[llapis] K auto-clamped to 22 to fit the GPU-offloaded projector (~1 GiB) in 8 GB VRAM alongside the hot slab." -ForegroundColor Cyan
}

# VRAM-MEASURED ctx-driven K clamp: the MTP drafter loads LAST, on top of slab + NE + ctx-sized KV;
# if that peak exceeds free VRAM the load dies (a CUDA OOM, or a bogus "invalid vector subscript").
# Resolve-KMaxForCtx computes the peak from the validated model against measured free VRAM: a no-op on
# an idle box, a clamp when VRAM is tight, a refusal only if even K=KMin cannot fit. (history: docs/code-comments-archive-2026-10-05.md, "k-clamp")
if ($Mode -in @('chat', 'serve')) {
  $kres  = Resolve-KMaxForCtx $Ctx
  $kCeil = [int]$kres.KMax
  $freeStr = if ($kres.Measured) { "$($kres.Free) MiB free (nvidia-smi)" } else { "~$($kres.Free) MiB free (ESTIMATE -- nvidia-smi unavailable)" }
  # A VRAM budget is a preference ("keep this much free"), not the card's size. When it is the budget
  # -- not the card -- that leaves less than the minimum expert cache, start at K=KMin over budget and
  # say so; refuse only when K=KMin does not fit PHYSICALLY. (findings 184: `quality` at the golden's
  # 5120 MiB budget was refused here while the same message claimed ~141k tokens would fit.)
  if ($kCeil -lt $script:PrefillCalib.KMin) {
    $kphys = Resolve-KMaxForCtx $Ctx -Physical
    if ([int]$kphys.KMax -ge $script:PrefillCalib.KMin) {
      $over = [int][Math]::Ceiling((Get-DrafterLoadPeakMiB $Ctx $script:PrefillCalib.KMin) + $script:PrefillCalib.SafetyMarginMib - ($kres.Free - (Resolve-KeepFreeMiB)))
      Write-Host ("[llapis] WARNING: the VRAM budget leaves room for less than the minimum expert cache in this mode " +
                  "(K ceiling $kCeil < $($script:PrefillCalib.KMin) at ctx $Ctx). Starting at K=$($script:PrefillCalib.KMin), " +
                  "~$over MiB over the budget; raise -VramBudget for a useful cache.") -ForegroundColor Yellow
      $kres  = @{ KMax = $script:PrefillCalib.KMin; Free = $kphys.Free; Measured = $kphys.Measured }
      $kCeil = $script:PrefillCalib.KMin
    }
  }
  if ($kCeil -lt $script:PrefillCalib.KMin) {
    # Largest ctx that WOULD fit at K=KMin, so the message is actionable rather than just "no".
    # (Physical: this branch is reached only when even the card cannot hold K=KMin.)
    $c = $script:PrefillCalib
    $ctxAtKMin = [int][Math]::Floor(($kres.Free - [double]$c.SafetyMarginMib - (Get-DrafterLoadPeakMiB 0 $c.KMin)) / (Get-KvMibPerTok))
    if ($ctxAtKMin -lt 0) { $ctxAtKMin = 0 }
    Write-Host ("[llapis] FATAL: ctx $Ctx does not fit on this GPU right now ($freeStr). Even the minimum " +
                "expert cache (K=$($c.KMin)) plus this ctx's KV cache + the MTP drafter would exceed VRAM at " +
                "load. Largest ctx that fits at K=$($c.KMin) with the current free VRAM: ~$ctxAtKMin tokens. " +
                "Use a smaller -Ctx, or close other GPU apps (a browser holds ~1 GiB). " +
                "Refusing up front rather than launching a degraded (silently MTP-less) server.") -ForegroundColor Red
    Stop-Launcher 1
  }
  if ($K -gt $kCeil) {
    Write-Host ("[llapis] K auto-clamped {0} -> {1} to fit ctx {2} within the measured VRAM budget ({3}); the MTP drafter loads last, on top of slab+NE+KV, and an unclamped K would exhaust VRAM at draft load." -f $K, $kCeil, $Ctx, $freeStr) -ForegroundColor Cyan
    $K = $kCeil
    $env:MOE_DUAL_K = "$K"
    $script:kClamped = $true
  } else {
    Say-Info ("[llapis] K=$K fits ctx $Ctx within the measured VRAM budget ($freeStr; ceiling K=$kCeil) -- no clamp.")
  }
}

$env:GGML_CUDA_DISABLE_GRAPHS = $null

# --- speculative residency (default since 2026-10-03, both modes since P5) --------------------------
# Asynchronous promotion (findings 171) and the cost-aware rule deciding at every token at the price of
# an asynchronous promotion (r 2, H 16, <= 4 promotions/token, 173) -- the only rule since elision E3
# (library defaults; MOE_DUAL_COST_H/_R/_PER_TOKEN override them for an A/B arm). In quality the copies
# come from the promotion FILE (the arena holds IQ2, not the Q4 bytes; 186). The exclusive arena (172)
# is ON by default on the mapped pack since claude dump findings 283 (MOE_DUAL_ARENA_EXCLUSIVE=0 turns it off): it only trims hot experts' pages from RAM
# and keeps the boundary residency and the arena staging (findings 200); on a copied arena it still turns
# both off. The caller's explicit value wins: MOE_DUAL_ASYNC_PROMOTE=0 promotes synchronously.
# (history: docs/code-comments-archive-2026-10-05.md, "spec-residency")
if ($Mode -in @('chat', 'serve')) {
  if (-not (Test-Path 'Env:MOE_DUAL_ASYNC_PROMOTE')) { $env:MOE_DUAL_ASYNC_PROMOTE = '1' }
  $pv = { param($n, $d) $v = [Environment]::GetEnvironmentVariable($n); if ($v) { $v } else { $d } }
  Write-Host ("[llapis] Speculative residency: async promotion={0} exclusive arena={1} | cost rule H={2} r={3} per-token={4}" -f `
              $env:MOE_DUAL_ASYNC_PROMOTE, (& $pv 'MOE_DUAL_ARENA_EXCLUSIVE' '1'),
              (& $pv 'MOE_DUAL_COST_H' '16'), (& $pv 'MOE_DUAL_COST_R' '2'), (& $pv 'MOE_DUAL_COST_PER_TOKEN' '4')) -ForegroundColor Cyan
  # --- the prefill (the only path since elision E3, claude dump findings 179/180/183/186) ---
  # The streamed prefill's staging is a fixed reserve allocated with an EMPTY slab before the first prompt
  # (the boot slab) and K is fitted around it: no slab eviction, no rebuild from the file. Hot experts are
  # staged D2D from the slab, cold ones from the RAM arena (speed) or the SSD (quality: the arena holds IQ2,
  # not the Q4 bytes); each layer's hot set is re-decided in the prefill from the prompt's routing (boundary
  # residency), which is also what fills the slab on the first prompt. MOE_DUAL_Q4S_ARENA=0 reads speed's
  # cold pages from the SSD too (then depth 2 when K allows it).
  if ($script:LlapisMode -eq 'speed' -and -not (Test-Path 'Env:MOE_DUAL_Q4S_ARENA')) { $env:MOE_DUAL_Q4S_ARENA = '1' }
  Write-Host ("[llapis] Prefill: cold pages from the RAM arena={0}" -f $(if ($env:MOE_DUAL_Q4S_ARENA) { $env:MOE_DUAL_Q4S_ARENA } else { '0' })) -ForegroundColor Cyan
  # CUDA graphs over the segments between the MoE ops (Variant P, claude dump findings 176, default since
  # 198): +6.5% decode with MTP in speed, neutral to +2% in quality, golden unchanged. MOE_DUAL_GRAPH_SEGMENTS=0 turns it off.
  if (-not (Test-Path 'Env:MOE_DUAL_GRAPH_SEGMENTS')) { $env:MOE_DUAL_GRAPH_SEGMENTS = '1' }
  # Loaned slots (claude dump findings 247/249, default since 2026-10-13): between streamed prompts the prefill
  # staging reserve (one full expert layer) carries 4 more hot experts per layer; revoked before a streamed prefill,
  # refilled right after it. +7.6% decode at 5 GiB, part of +4.2% at full VRAM; exact. MOE_DUAL_LOAN=0 turns it off.
  if (-not (Test-Path 'Env:MOE_DUAL_LOAN')) { $env:MOE_DUAL_LOAN = '1' }
}

# --- CPU cold sidecar (text and vision, with the loader cut): the measured best live decode config
# (copy stream + sync sidecar: +26% vs off). The async prototype is a NO-GO and is never used.
# (history: docs/code-comments-archive-2026-10-05.md, "sidecar")
# Without the loader cut (-NoLoaderCut) there is no sidecar and the cold path is the Plan C staging ring.
$ColdCpu = $NoExpertAlloc -and ($Mode -in @('chat','serve'))
if ($ColdCpu) {
  $env:MOE_DUAL_COPYSTREAM = '1'                      # routed-only cold copy stream (overlap pinned H2D)
  $env:MOE_DUAL_COLD_CPU = '1'
  $env:MOE_DUAL_COLD_CPU_MAX = "$ColdCpuMax"
  $env:MOE_DUAL_COLD_CPU_WORKERS = "$ColdCpuWorkers"
  Say-Info "[llapis] CPU cold sidecar ON (text profile): copy stream + sidecar, ColdCpuMax $ColdCpuMax"
}
# MTP-assistant speculative drafter: DEFAULT for chat/serve in both modes at n_max=3 (the validated
# fused config). Not optional on the product path: without it the streamed prefill is off (74.5 vs
# 1275 tok/s prompt). The drafter is a standalone GGUF in models/mtp/. (history: docs/code-comments-archive-2026-10-05.md, "mtp-default")
$script:mtpCliArgs = $null
$mtpDefault = ($Mode -in @('serve', 'chat')) -and -not $NoMtp
if ($NoMtp -and $Mode -in @('serve', 'chat')) {
  # THE FUSED WIDTH IS NOT THE DRAFTER. MOE_DUAL_SPEC_K sizes two things that have nothing to do
  # with having a draft model: the dual runtime's fused ceiling (moe_dual.cpp — `fused_ok` requires
  # n_tokens <= spec_kmax, and ANY call above it falls through to the STOCK loader-cut path, which
  # `moe_dual.cpp:696` documents as a live-measured crash, not a fallback), and the server's decode
  # ubatch under the cut (server_main.cpp:2010). Leaving it unset with the cut active is what made
  # the first -NoMtp run die at the first decode step (findings 150: cudaErrorIllegalAddress, then
  # a plain 0xC0000005 under CUDA_LAUNCH_BLOCKING — a host-side touch of a shell tensor with no
  # buffer). So stamp the same width MTP would have used and withhold only the drafter itself:
  # -NoMtp then means "no speculation", not "a differently-shaped runtime".
  $env:MOE_DUAL_SPEC_K        = "$($MtpNMax + 1)"
  $env:MOE_DUAL_SPEC_NODRAFT  = '1'   # width yes, drafter no -- without this the server arms its OWN ngram-simple drafter at spec_k>=2 and "speculation off" measures a different drafter
  Write-Host "[llapis] -NoMtp: speculative decode OFF (no drafter loaded). This is the measurement arm." -ForegroundColor Yellow
  Say-Info "[llapis] -NoMtp: fused width still $($MtpNMax + 1) (MOE_DUAL_SPEC_K) -- it sizes the fused ceiling and the cut's decode ubatch, NOT the drafter; unsetting it crashes the loader cut (findings 150)"
}
if ($mtpDefault) {
    $draftPath = if ($script:LlapisModels) { $script:LlapisModels.Mtp }
                 else                      { (Find-LlapisGguf $script:DirMtp $script:ExpectMtp).Path }
    # Two-stage drafter head (claude dump findings 243/245/249, default since 2026-10-13): when models/mtp holds the
    # derived drafter (<drafter>.draft-head-*.gguf, built by 'moert-server prepare' from assets/mtp, findings 291) it replaces the
    # plain one -- the head reads ~50 MiB per draft instead of 272 (drafts -2.6 ms/step, +3.5% at full VRAM; the verify
    # is the target's, so the text does not depend on it). LLAPIS_DRAFT_HEAD=0 keeps the plain drafter.
    if ($draftPath -and (Test-Path $draftPath) -and $env:LLAPIS_DRAFT_HEAD -ne '0') {
      $dh = Get-ChildItem (Split-Path $draftPath) -Filter (([IO.Path]::GetFileNameWithoutExtension($draftPath)) + '.draft-head-*.gguf') -ErrorAction SilentlyContinue |
            Select-Object -First 1
      if ($dh) { $draftPath = $dh.FullName }
      elseif (Test-Path (Join-Path (Split-Path (Split-Path (Split-Path $draftPath))) 'assets\mtp')) { Write-Host "[llapis] tip: run 'moert-server prepare' once to build the two-stage drafter head (~2.6 ms less per decode step)" -ForegroundColor Yellow }
    }
    # A/B lever (claude dump findings 243): MOE_DUAL_DRAFT_GGUF=<path> loads a given drafter
    if ($env:MOE_DUAL_DRAFT_GGUF -and (Test-Path $env:MOE_DUAL_DRAFT_GGUF)) { $draftPath = $env:MOE_DUAL_DRAFT_GGUF }
    # -DraftHeadInRam (findings 248): the derived drafter's full head stays in host RAM (only with the two-stage head)
    if ($DraftHeadInRam) {
      if ($draftPath -like '*.draft-head-*') { $env:LLAPIS_DRAFT_HEAD_HOST = '1'; Say-Info "[llapis] -DraftHeadInRam: the drafter's vocabulary head stays in system RAM" }
      else { Write-Host "[llapis] -DraftHeadInRam ignored: no two-stage drafter (models/mtp/*.draft-head-*.gguf)" -ForegroundColor Yellow }
    }
    # The product path can no longer reach here without a drafter: Assert-LlapisModels FATALs on a
    # missing models/mtp/ before any of this runs (user decision 2026-07-28 — a silent fallback here
    # costs ~60 s of TTFT via the token-by-token prefill, which is not a "minor degradation").
    # This branch survives for a caller that skipped the setup block entirely.
    if (-not $draftPath -or -not (Test-Path $draftPath)) {
      Write-Host "[llapis] MTP drafter not found (looked in $($script:DirMtp)) -- speculative decode NOT enabled" -ForegroundColor Yellow
    } else {
      $env:MOE_DUAL_SPEC_K = "$($MtpNMax + 1)"   # verify width = n_max+1 (fused Pmax<=4 ceiling: n_max<=3)
      $script:mtpCliArgs = @('--model-draft', $draftPath, '--spec-type', 'draft-mtp', '--spec-draft-n-max', "$MtpNMax")
      Say-Info "[llapis] MTP-assistant speculative decode ENABLED (n_max=$MtpNMax, drafter=$([IO.Path]::GetFileName($draftPath))) -- pairs with the tuned cache policy above for the validated mixed/coding win"
      # C8: the sidecar's ColdCpuMax was stamped to $env:MOE_DUAL_COLD_CPU_MAX
      # already (above, before spec_k was known) at the pre-MTP default of 8 -- re-stamp it now
      # that the real verify-batch width is known (same restamp pattern as the C6 K-clamp fix,
      # setting the PS variable after the env var was already written is a no-op).
      # An MTP verify fire spans up to spec_k tokens, so n_cold (distinct cold experts across the
      # WHOLE fire) can reach spec_k*top_k -- 8 routinely undercounts it (measured 39.2% of fires
      # spilling to the slower GPU staging ring). Scale to the
      # structural ceiling (Pmax = spec_k*8) so it can never spill by construction.
      if ($ColdCpu) {
        $mtpScaledMax = [math]::Min(32, 8 * [int]$env:MOE_DUAL_SPEC_K)
        if ($mtpScaledMax -ne $ColdCpuMax) {
          $ColdCpuMax = $mtpScaledMax
          $env:MOE_DUAL_COLD_CPU_MAX = "$ColdCpuMax"
          Say-Info "[llapis] ColdCpuMax scaled 8 -> $ColdCpuMax for the active MTP verify width (spec_k=$env:MOE_DUAL_SPEC_K) -- measured +47.6% mean decode tok/s raising 8->24, with spill 39.2%->0.7%; 32 = Pmax guarantees zero spill by construction."
        }
      }
    }
}
# q4stream server prefill: DEFAULT for chat/serve with the MTP drafter (backend sampling off is what
# the layer-major merged prefill graph needs); it coexists with vision. Type-generic over the source,
# so speed streams IQ2 and quality Q4 (measured 195 tok/s vs 37.5 fused on 4204 tokens; 929 tok/s
# speed on 11220). (history: docs/code-comments-archive-2026-10-05.md, "q4stream-default")
$PrefillMode = 'tbt'
if ($Mode -in @('chat', 'serve') -and
    $script:mtpCliArgs -and $env:MOE_DUAL_SPEC_K -and [int]$env:MOE_DUAL_SPEC_K -ge 2) {   # the DRAFTER, not the width: since -NoMtp stamps SPEC_K without a drafter (findings 150), spec_k alone no longer implies backend sampling is off, and server.ps1:41 refuses q4stream without it
  $PrefillMode = 'q4stream'
  Say-Info "[llapis] prefill -> q4stream (layer-major streamed prompt, ~5x fused)"
}
elseif ($Mode -in @('chat', 'serve')) {
  # `auto` that could NOT resolve to q4stream. This is the silent-degradation case the streamed
  # prefill exists to avoid: with no spec path (or no dual runtime) the server prefills through the
  # fused per-token path at ubatch = spec_k, i.e. a prompt token costs about what a decode token
  # costs — 74.5 tok/s at ANY prompt length. On a 5k-token prompt that is a full minute of TTFT.
  # It stays legal (it is the correct fallback, not a bug) but it must never be silent: this is
  # exactly the class of "the number that wasn't the number" this file's calibration block warns
  # about, and it cost 63.7 s of unexplained TTFT once already.
  $why = "the MTP/spec drafter is off (q4stream needs backend sampling off for the layer-major merged prefill graph)"
  Write-Host "[llapis] prefill: STREAMING IS OFF -- $why." -ForegroundColor Yellow
  Write-Host "            The prompt will be read token-by-token (~75 tok/s regardless of length)." -ForegroundColor Yellow
  Write-Host "            Drop -NoMtp to get the streamed prefill back." -ForegroundColor Yellow
}
# q4stream server batch-size gate: the server re-streams the expert pool once PER batch-size chunk, so
# a batch up to the prompt length means one pool stream; the cost is the layer-major merged graph's
# reserve, paid once at load.
# The batch targets the session's context ceiling and is clamped twice: by the merged-graph safety
# ceiling (a correctness wall: a layer-major decode past ~128 merged ubatches overflows the host
# mask-shadow ggml context -> GGML_ASSERT at ggml.c:1608) and by the free-VRAM reserve budget.
function Resolve-Q4sBatch {
  param([int]$Ctx, [int]$Ubatch = 64)
  $floor = $script:PrefillCalib.BatchFloor
  $want = $Ctx
  $script:q4sBatchSource = 'auto->ceiling'
  if ($want -lt $floor) { $want = $floor }

  # Safe ceiling. The wall is the merged-ubatch COUNT (= batch / split), NOT the split size:'s
  # "batch > 4096 crashes" was really 128 merges (batch 8192 at split 64). 64 merges is proven safe --
  # the historical 4096/64 default does exactly 64 -- so cap batch at split * 64. That reproduces the
  # old 4096 ceiling exactly at split 64 while correctly allowing a wider batch at wider splits.
  # ctx > 16384 crashes the MTP drafter at LOAD regardless of batch (a separate bug), so the ceiling
  # never exceeds 16384.
  $ctxCeil    = $script:PrefillCalib.CtxDrafterCeil
  $mergeLimit = $Ubatch * $script:PrefillCalib.MergeSafetyMul
  $safeMax = [Math]::Min([Math]::Min($Ctx, $ctxCeil), $mergeLimit)
  if ($want -gt $safeMax) {
    $why = if ($mergeLimit -lt [Math]::Min($Ctx, $ctxCeil)) { "split $Ubatch x the $($script:PrefillCalib.MergeSafetyMul)-merge safety limit = $mergeLimit" } else { "ctx $Ctx caps the single-pass envelope (ctx > $ctxCeil crashes the MTP drafter at load)" }
    # Under variable-KV the target is the 262144 model-native ceiling, so the clamp is structural on
    # every launch, not a misconfiguration: say it quietly.
    if ($script:VkvOn) {
      Say-Info ("[llapis] q4stream batch {0} -> {1} (safe ceiling; {2})" -f $want, $safeMax, $why)
    } else {
      Write-Host ("[llapis] q4stream batch-size {0} EXCEEDS the safe ceiling {1} ({2}). Clamping to {1}." -f $want, $safeMax, $why) -ForegroundColor Yellow
    }
    $want = $safeMax
    $script:q4sBatchSource += '->safe-clamp'
  }

  $mibPerTok = $script:PrefillCalib.ReserveMibPerTok
  $headroom = $script:PrefillCalib.ReserveBudgetMib
  $freeMargin = $script:PrefillCalib.FreeVramMarginMib
  $free = Get-FreeVramMiB

  $budget = $headroom
  if ($null -ne $free) { $budget = [Math]::Min($headroom, $free - $freeMargin) }
  if ($budget -lt 0) { $budget = 0 }

  $target = $want
  while ($want -gt $floor -and (($want - $floor) * $mibPerTok) -gt $budget) {
    $want = [int][Math]::Floor($want / 2)
    if ($want -lt $floor) { $want = $floor }
  }
  $freeStr = if ($null -ne $free) { "${free} MiB" } else { 'n/a (nvidia-smi)' }
  if ($want -lt $target) {
    Write-Host ("[llapis] q4stream batch-size gate: requested {0} -> CLAMPED to {1} (free VRAM {2}, reserve budget {3} MiB @ {4} MiB/tok)" -f $target, $want, $freeStr, $budget, $mibPerTok) -ForegroundColor Yellow
    $script:q4sBatchSource += '->vram-clamp'
  } else {
    Say-Info ("[llapis] q4stream batch-size {0} (ctx {1}, free VRAM {2}); ~1 pool stream for prompts <= {0} tok" -f $want, $Ctx, $freeStr)
  }
  return $want
}

if ($PrefillMode -eq 'q4stream' -and $Mode -in @('chat', 'serve')) {
  $env:MOE_DUAL_PREFILL_Q4SRC = $Q4Source
  # --- prefill configuration resolution ---------------------------------------------------------
  # The DECOUPLED prefill at split 256. Against the previous coupled chunk-64 default it wins on every
  # measured axis at once:
  #   prefill 667.1 vs 93.2 tok/s (7.16x) | 30 vs 60 layer stagings (HALF the SSD read, one pool
  #   pass) | decode VRAM headroom IDENTICAL (326 MiB in every single trial) | decode 23.6 vs 24.9
  #   tok/s and MTP accept 63% vs 61% (overlapping ranges) | 0 crashes.
  # The decouple is what makes the headroom free: n_ubatch (which sizes the PERMANENT graph reserve)
  # stays decode-sized while the prefill splits at 256 on its own one-shot scheduler.
  # Every MOE_DUAL_* env below is set HERE, after this file's own MOE_DUAL_* wipe.
  $PrefillChunk = $script:PrefillCalib.SplitDefault
  $env:MOE_DUAL_PREFILL_DECOUPLE = '1'
  Say-Info "[llapis] prefill decoupled: n_ubatch stays decode-sized (small permanent graph reserve) while prompts split at $PrefillChunk on the one-shot scheduler — 7x prefill at the same decode VRAM headroom."
  if ($Ctx -gt 16384) {
    Write-Host "[llapis] note: ctx $Ctx > 16384 — a single pool pass only covers prompts up to 16384 tokens; longer prompts re-stream per batch-sized chunk. K is auto-clamped for this ctx." -ForegroundColor DarkCyan
  }
  $env:MOE_DUAL_PREFILL_CHUNK = "$PrefillChunk"
  # lever B: resolve the VRAM-gated prompt batch-size (server.ps1 writes it into the child args).
  # Size the single-pass envelope from the session's CEILING, not from the boot context: under
  # variable-KV the context grows, and `--batch-size` is fixed at launch. llama clamps
  # `cparams.n_batch = min(n_ctx, params.n_batch)` at context creation (llama-context.cpp:212), so
  # a ceiling-sized batch is clamped down to the boot context now and re-clamps UP to its full
  # value when the context is rebuilt larger — which is exactly what a grown session needs. Sizing
  # it from the boot context instead would silently cost a whole extra pool re-stream on every
  # prompt longer than 8192 tokens.
  $q4sBatchCtx = if ($script:VkvOn) { $script:VkvMaxCtx } else { $Ctx }
  $script:q4sBatch = Resolve-Q4sBatch -Ctx $q4sBatchCtx -Ubatch $PrefillChunk

  # --- record the resolved plan with provenance (R-a), then print + dump it once ---
  $kSrc = if ($script:kClamped) { 'ctx-clamp(calib)' } else { 'preset/default' }
  Add-Plan 'decoupled'       $true                                      'always'
  Add-Plan 'split'           $PrefillChunk                              'calib:SplitDefault'
  Add-Plan 'batch'           $script:q4sBatch                           $script:q4sBatchSource
  Add-Plan 'ctx'             $Ctx                                       $(if ($CallerBound.ContainsKey('Ctx')) { 'cli:-Ctx' } elseif ($script:VkvOn) { 'vkv:boot(min_ctx)' } else { 'default' })
  Add-Plan 'variableKv'      $(if ($script:VkvOn) { "on->$($script:VkvMaxCtx)" } else { 'off' })  $(if ($script:VkvOn) { 'default(on)' } elseif ($script:VkvOffWhy) { $script:VkvOffWhy } else { 'n/a' })
  Add-Plan 'K'               $K                                         $kSrc
  Add-Plan 'reserveBudgetMib' $script:PrefillCalib.ReserveBudgetMib    'calib:ReserveBudgetMib'
  Write-PrefillPlan
}

# ---- Vision (Gemma4 multimodal): load the mmproj projector so the model can see images -------
# The fork already carries the whole mtmd pipeline (clip gemma4v/gemma4a graphs, image-chunk decode
# that even feeds the MTP drafter); all we add is the projector path + a CPU-encode default and the
# MOE_VISION marker. A loaded projector makes moert-server force the fused per-token prefill (image
# chunks bypass the q4stream phase gate) -- see server_main.cpp has_mmproj guard.
$script:visionMmproj = $null
if ($Vision) {
  if ($Mode -notin @('chat', 'serve')) {
    Write-Host "[llapis] -Vision only applies to 'serve'/'chat'; ignoring for '$Mode'." -ForegroundColor Yellow
  } else {
    # auto-find; prefer the SMALLEST-memory quant (this 8GB box is RAM-starved: the F32 tower is
    # ~2.3 GiB, F16 ~1.15, Q8 ~0.6). C5.5: the projector has its OWN hardcoded folder now
    # (models/mmproj) — it is not a member of the model pair and its absence is never fatal, so
    # keeping it out of models/high leaves that folder to mean exactly one thing. The high folder
    # stays in the search list so an existing install keeps working without moving files.
    $mmDirs = @($script:DirMmproj, $script:DirHigh) | Select-Object -Unique
    $ranked = 'mmproj-Q8_0.gguf','mmproj-Q8.gguf','mmproj-Q6_K.gguf','mmproj-Q5_K.gguf','mmproj-Q4_K.gguf','mmproj-F16.gguf','mmproj-BF16.gguf','mmproj-F32.gguf'
    foreach ($mmDir in $mmDirs) {
      foreach ($cand in $ranked) {
        $p = Join-Path $mmDir $cand
        if (Test-Path $p) { $script:visionMmproj = $p; break }
      }
      if ($script:visionMmproj) { break }
    }
    if (-not $script:visionMmproj) {
      foreach ($mmDir in $mmDirs) {
        $any = Get-ChildItem $mmDir -Filter 'mmproj*.gguf' -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($any) { $script:visionMmproj = $any.FullName; break }
      }
    }
    if (-not $script:visionMmproj) {
      Write-Error "[llapis] -Vision: no mmproj-*.gguf found in $($mmDirs -join ', '). Put a projector (e.g. mmproj-Q8_0.gguf) in models\mmproj."
      Stop-Launcher 1
    }
    $env:MOE_VISION = '1'   # MOE_* => collected into the child profile env; drives /moe/info "vision" + is a guard signal
    $mmMiB = [math]::Round((Get-Item $script:visionMmproj).Length / 1MB)
    $mmWhere = if ($env:LLAPIS_MMPROJ_CPU -eq '1') { 'encoded on the CPU (LLAPIS_MMPROJ_CPU=1)' }
               elseif ($env:LLAPIS_MMPROJ_LAZY -eq '0') { 'resident on the GPU (LLAPIS_MMPROJ_LAZY=0)' }
               else { 'on the GPU only while an image is encoded; its VRAM goes to the expert cache in between' }
    Write-Host "[llapis] VISION ON: projector $([IO.Path]::GetFileName($script:visionMmproj)) (~$mmMiB MiB), $mmWhere" -ForegroundColor Green
    # C4e: q4stream coexists with vision (text segments stream; image chunks decode on the
    # transient path with the q4stream phase closed around them by the vision-room callback).
  }
}

# ---------------------------------------------------------------------------
# C5.6: PROMPT-CACHE RAM budget (llama-server --cache-ram). Upstream's 8192 MiB of serialized prompt
# states competes with the cold arena (~7.9 GiB of this 15.6 GiB box), whose RESIDENCY is the product:
# measured, one 520 MiB saved state pushed available RAM to ~335 MiB and the next short prompt
# prefilled at 0.07 tok/s. So: cache-ram = clamp(total RAM - arena - RamFloorMib, 0, CapMib); 0 (off)
# on a box with no room. (history: docs/code-comments-archive-2026-10-05.md, "cache-ram")

$script:CacheRamCalib = [ordered]@{
  RamFloorMib = 4096   # RAM this machine must keep for the OS, a browser and our own non-arena
                       # RSS. Measured non-arena RSS here is small (~128 MiB); the rest is the
                       # desktop, and 4 GiB is what kept `avail` positive across this session.
  CapMib      = 1024   # never hand the prompt cache more than this even on a roomy box: one
                       # state at ctx 40960 is ~520 MiB, so 1 GiB already caches the cheap cases
                       # and declines the expensive ones -- exactly the shape we want.
}
function Get-TotalRamMiB {
  try {
    $b = (Get-CimInstance Win32_ComputerSystem -ErrorAction Stop).TotalPhysicalMemory
    if ($b) { return [int]($b / 1MB) }
  } catch {}
  return 0
}
function Resolve-CacheRamMib {
  if ($Mode -notin @('chat', 'serve')) { return -1 }            # bench/dev: leave upstream's default
  $ramTotal = Get-TotalRamMiB
  if ($ramTotal -le 0) { return -1 }                            # cannot measure -> do not guess
  # Arena size: the pack next to the index we are about to map. Absent (no cold arena) -> 0,
  # and then the default budget is fine because nothing else is competing for host RAM.
  $arenaMib = 0
  if ($ColdArena -and (Test-Path $ColdArena)) {
    $pack = [IO.Path]::ChangeExtension($ColdArena, $null) -replace '\.experts\.$', '.experts.pack'
    if (-not (Test-Path $pack)) { $pack = ($ColdArena -replace '\.index$', '.pack') }
    if (Test-Path $pack) { $arenaMib = [int]((Get-Item $pack).Length / 1MB) }
  }
  if ($arenaMib -le 0) { return -1 }
  $headroom = $ramTotal - $arenaMib - [int]$script:CacheRamCalib.RamFloorMib
  $val = [Math]::Max(0, [Math]::Min([int]$script:CacheRamCalib.CapMib, $headroom))
  return [int]$val
}
$script:CacheRamResolved = Resolve-CacheRamMib
# Not an Add-Plan entry: Write-PrefillPlan has already printed by the time this file gets here, and
# a row that only ever reaches the JSON would be exactly the kind of half-reported value the plan
# exists to prevent. The resolved figure is visible in `serve -DryRun` under child_args/cache-ram,
# which is the authoritative "what will the child actually run" view anyway.
if ($script:CacheRamResolved -ge 0) {
  if ($script:CacheRamResolved -eq 0) {
    Say-Info "[llapis] prompt cache DISABLED (--cache-ram 0): the cold expert arena needs this box's RAM more than a resumable prompt state does."
  } else {
    Say-Info "[llapis] prompt cache limited to $($script:CacheRamResolved) MiB (--cache-ram; upstream default 8192 would evict the cold arena on this box)."
  }
}
