# ===========================================================================
# llapis/setup.ps1 - model layout + the one remaining setup choice (quality/speed).
# ===========================================================================
# DOT-SOURCED by llapis.ps1 (". llapis\setup.ps1"), NOT run standalone: the
# functions here read the launcher's script-scope paths ($Root, $Rel, $SrvExe,
# $ConfigPath, $ModelsRoot, $Interactive) and the Say-* helpers.
#
# C5.5 — "llapis always exposes the same runtime":
# The old wizard asked FIVE questions (high model, low model, context, VRAM cache
# label, vision) and wrote all five into llapis-user.json. Every one of them then
# leaked outward: into the router profile name (gemma4_k=<K>_ctx=<Ctx>), and from
# there into the `agents` provider JSON — so an agent harness had to be
# reconfigured whenever a runtime parameter changed. That is the bug this file's
# rewrite closes.
#
# What the user still chooses:  quality | speed.  That is all.
# What is now HARDCODED:        the model layout (below).
# What the runtime decides:     K (VRAM-measured clamp), context (variable-KV).
#
# Layout — fixed paths, no discovery, no LM Studio fallback:
#   models/high/    full-precision experts source (Q4_K_XL)   -> quality only
#   models/low/     IQ2 base                                   -> BOTH modes
#   models/mtp/     MTP-assistant speculative drafter          -> REQUIRED
#   models/mmproj/  vision projector                           -> only for -Vision
#   models/cache/   DERIVED artifacts (moert-server prepare)   -> rebuildable
# ===========================================================================

# --- hardcoded model layout -------------------------------------------------
# Set here (not in the launcher) so every consumer — preset.ps1, the wizard, the
# check — reads one definition. $ModelsRoot is the launcher's repo-relative models\.
$script:DirHigh   = Join-Path $ModelsRoot 'high'
$script:DirLow    = Join-Path $ModelsRoot 'low'
$script:DirMtp    = Join-Path $ModelsRoot 'mtp'
$script:DirMmproj = Join-Path $ModelsRoot 'mmproj'

# Expected filenames. These are a PREFERENCE, not a requirement: a folder holding a
# single .gguf resolves to it whatever it is called, so re-quantising or renaming a
# model does not need a code change. Two-or-more .gguf without the expected name is
# the only ambiguous case, and it reports rather than guessing.
# They must be the names the README tells people to download, or the "Expected: ..." line in
# the fail matrix below sends them looking for a file the instructions never mentioned. This one
# said gemma-4-26B-A4B-it-assistant-Q4_K_M.gguf, a research-era name, while the README pins
# mtp-gemma-4-26B-A4B-it-Q8_0.gguf.
$script:ExpectHigh = 'gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf'
$script:ExpectLow  = 'gemma-4-26B-A4B-it-UD-IQ2_M.gguf'
$script:ExpectMtp  = 'mtp-gemma-4-26B-A4B-it-Q8_0.gguf'

# Resolve the one .gguf a layout folder is supposed to hold.
# Returns @{ Path; Why } — Path $null when unresolved, Why always explains.
function Find-LlapisGguf([string]$dir, [string]$expected) {
  if (-not (Test-Path $dir)) { return @{ Path = $null; Why = "folder does not exist: $dir" } }
  $exact = Join-Path $dir $expected
  if (Test-Path $exact) { return @{ Path = (Resolve-Path $exact).Path; Why = 'expected filename' } }
  $all = @(Get-ChildItem -Path $dir -Filter *.gguf -File -ErrorAction SilentlyContinue)
  if ($all.Count -eq 0) { return @{ Path = $null; Why = "no .gguf in $dir (expected $expected)" } }
  if ($all.Count -eq 1) { return @{ Path = $all[0].FullName; Why = "only .gguf present ($($all[0].Name))" } }
  return @{ Path = $null
            Why  = "$($all.Count) .gguf files in $dir and none is named $expected — remove the extras or rename the one you want: $(($all.Name) -join ', ')" }
}

# Total device VRAM in MiB, or $null when nvidia-smi is unavailable. Defined HERE (not in
# preset.ps1 next to Get-FreeVramMiB) because the setup wizard runs before preset.ps1 is
# dot-sourced and needs it to frame the VRAM-budget question in real numbers.
function Get-TotalVramMiB {
  # The launcher fires one nvidia-smi asynchronously before any dot-source (see Get-VramPrefetch in
  # llapis.ps1) precisely so the driver's 0.75 s first-call wake overlaps the resolution instead
  # of blocking it. Use that reading when it arrived; otherwise ask directly, exactly as before.
  $pf = if (Get-Command Get-VramPrefetch -ErrorAction SilentlyContinue) { Get-VramPrefetch } else { $null }
  if ($pf -and $pf.total) { return [int]$pf.total }
  try {
    $out = & nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -eq 0 -and $out) {
      $v = (($out | Select-Object -First 1) -as [string]).Trim()
      if ($v -match '^\d+$') { return [int]$v }
    }
  } catch {}
  return $null
}

# --- VRAM terms of the installed models --------------------------------------
# Which quant sits in models/low and models/high is the user's choice (IQ2 + Q4 here; Q4 + Q8 on a
# bigger card), so nothing below may assume one. The quant-dependent terms -- bytes per hot slot,
# non-expert weights, KV per token, staging layer, cold arena -- come from the GGUF headers through
# `moert-server plan --terms` (no load, no GPU probe), cached per file identity in models/cache.
# What is left is a quant-INDEPENDENT residual, back-solved on the reference box. preset.ps1 (the K
# clamp, the C++ preflight) and the wizard below both read this one table.
$script:FitCalib = @{
  ResidualMib          = 1410  # the boot floor beyond the non-expert weights: drafter, the SWA-KV floor,
                               # the streamed-prefill staging and the compute/graph reserve (reference box:
                               # quality 3869 = 2459 NE + 1410; speed 3088 = 1678 NE + the same 1410)
  RefStagingMib        = 590   # the staging the residual already holds (one Q4_K_XL expert layer). A hot
                               # source with a bigger layer (Q8_0: ~2x) adds the excess; a smaller one is
                               # not credited back, so the reference pair reproduces exactly
  PreflightResidualMib = 941   # moe_dual_init.cpp's admission headroom: 3400 = 2459 NE + 941
  KMin                 = 4     # below this the expert cache is pointless
  DrafterLoadPeakMib   = 340   # transient at the drafter load (see preset.ps1's PrefillCalib)
  SafetyMarginMib      = 300
  MinCtx               = 8192  # the variable-KV floor the runtime always allocates
}

# The terms for one mode, or $null with the reason in $script:ModelTermsError. Cached in
# models/cache/fit-terms-<mode>.json keyed by every input file's path, size and timestamp.
function Get-LlapisModelTerms([string]$mode, [string]$high, [string]$low, [string]$drafter) {
  $script:ModelTermsError = $null
  $files = @($low) + $(if ($mode -eq 'quality') { @($high) } else { @() }) + $(if ($drafter) { @($drafter) } else { @() })
  $key = ($files | ForEach-Object {
    if (-not $_ -or -not (Test-Path $_)) { "missing:$_" }
    else { $i = Get-Item $_; "{0}|{1}|{2}" -f $i.FullName, $i.Length, $i.LastWriteTimeUtc.Ticks }
  }) -join ';'
  $cache = Join-Path $ModelsRoot "cache\fit-terms-$mode.json"
  try {
    if (Test-Path $cache) {
      $c = Get-Content $cache -Raw | ConvertFrom-Json
      if ($c.key -eq $key) { return $c.terms }
    }
  } catch {}
  if (-not (Test-Path $SrvExe)) { $script:ModelTermsError = "moert-server.exe not found: $SrvExe"; return $null }
  $a = @('plan', '--terms', '--mode', $mode, '--low', $low)
  if ($mode -eq 'quality') { $a += @('--high', $high) }
  if ($drafter) { $a += @('--drafter', $drafter) }
  # The exe logs its CUDA device line on stderr. PS 5.1 wraps each stderr line of a native command in an
  # ErrorRecord, which a caller's ErrorActionPreference 'Stop' turns into a terminating error: run it
  # under 'Continue' (function scope) and keep every line as plain text.
  $ErrorActionPreference = 'Continue'
  $out = @(& $SrvExe @a 2>&1 | ForEach-Object { "$_" })
  $json = @($out | Where-Object { $_.StartsWith('{') }) | Select-Object -First 1
  if ($LASTEXITCODE -ne 0 -or -not $json) {
    $script:ModelTermsError = (@($out | Where-Object { $_ -match 'ERROR' }) -join ' ')
    if (-not $script:ModelTermsError) { $script:ModelTermsError = "moert-server plan --terms failed (exit $LASTEXITCODE)" }
    return $null
  }
  $t = $json | ConvertFrom-Json
  try { [ordered]@{ key = $key; terms = $t } | ConvertTo-Json -Depth 4 | Set-Content -Path $cache -Encoding utf8 } catch {}
  return $t
}

# The quant tag of a model file for display (Q4_K_XL, IQ2_M, Q8_0, BF16 ...), or its name when it has none.
function Get-LlapisQuantLabel([string]$path) {
  if (-not $path) { return '?' }
  $n = [IO.Path]::GetFileNameWithoutExtension($path)
  if ($n -match '(?i)(?:^|[-_.])((?:I?Q\d(?:_[A-Z0-9]+)*)|BF16|F16|F32)$') { return $Matches[1].ToUpper() }
  return $n
}

# Everything at boot that does not scale with K or the context, for these models.
function Get-LlapisFixedOverheadMiB($terms) {
  $c = $script:FitCalib
  $stagingExcess = [Math]::Max(0, [double]$terms.staging_mib - $c.RefStagingMib)
  return [double]$terms.ne_vram_mib + $c.ResidualMib + $stagingExcess
}

# The smallest launch: the fixed floor + the drafter's load peak + KV at the minimum context + the
# minimum expert cache + the safety margin. Below it the runtime still starts, at this size, over the
# user's limit; only a card that cannot hold it physically refuses.
function Get-LlapisVramFloorMiB($terms) {
  $c = $script:FitCalib
  return [int][Math]::Ceiling((Get-LlapisFixedOverheadMiB $terms) + $c.DrafterLoadPeakMib +
                              [double]$terms.kv_mib_per_tok * $c.MinCtx +
                              [Math]::Ceiling([double]$terms.slab_mib_per_k) * $c.KMin + $c.SafetyMarginMib)
}

# Resolve the whole layout in one pass. Pure probe: prints nothing, decides nothing.
function Resolve-LlapisModels {
  $h = Find-LlapisGguf $script:DirHigh   $script:ExpectHigh
  $l = Find-LlapisGguf $script:DirLow    $script:ExpectLow
  $m = Find-LlapisGguf $script:DirMtp    $script:ExpectMtp
  # mmproj is not a model pair member and has no canonical single name (mmproj-Q8_0,
  # -F16, ...): take any mmproj*.gguf, smallest-quant preference is preset.ps1's job.
  $vp = $null
  if (Test-Path $script:DirMmproj) {
    $vp = (Get-ChildItem -Path $script:DirMmproj -Filter 'mmproj*.gguf' -File -ErrorAction SilentlyContinue |
           Select-Object -First 1).FullName
  }
  return @{
    High = $h.Path; HighWhy = $h.Why
    Low  = $l.Path; LowWhy  = $l.Why
    Mtp  = $m.Path; MtpWhy  = $m.Why
    Mmproj = $vp
  }
}

# Print the layout check. Returns nothing; the callers decide what a gap means.
function Show-LlapisModelCheck([hashtable]$mm) {
  $fmt = {
    param($label, $path, $why)
    if ($path) {
      Write-Host ("  [OK]      {0,-8} {1}" -f $label, [IO.Path]::GetFileName($path)) -ForegroundColor Green
    } else {
      Write-Host ("  [MISSING] {0,-8} {1}" -f $label, $why) -ForegroundColor Yellow
    }
  }
  Write-Host ""
  Write-Host "Model check" -ForegroundColor Cyan
  Write-Host ("  layout root: {0}" -f $ModelsRoot) -ForegroundColor DarkGray
  Write-Host ""
  & $fmt 'low'    $mm.Low    $mm.LowWhy
  & $fmt 'high'   $mm.High   $mm.HighWhy
  & $fmt 'mtp'    $mm.Mtp    $mm.MtpWhy
  if ($mm.Mmproj) {
    Write-Host ("  [OK]      {0,-8} {1}  (only used with -Vision)" -f 'mmproj', [IO.Path]::GetFileName($mm.Mmproj)) -ForegroundColor DarkGray
  } else {
    Write-Host ("  [ - ]     {0,-8} not present — -Vision unavailable (everything else works)" -f 'mmproj') -ForegroundColor DarkGray
  }
}

# The FAIL MATRIX, in one place (user decision 2026-07-28):
#   low  missing -> FATAL. Both modes need it: speed runs it directly, and quality's
#                   spliced base + cold expert arena are both DERIVED from it.
#   mtp  missing -> FATAL. Not a graceful degradation: without the drafter, quality
#                   cannot use the streamed q4stream prefill (moert-server refuses the
#                   combination) and falls back to token-by-token prefill, measured at
#                   74.5 vs 1275 tok/s on a 4752-token prompt — a ~60 s TTFT regression
#                   that would be invisible to the user. Refuse instead.
#   high missing -> ALERT + forced speed. Quality streams the Q4 source LIVE at runtime,
#                   so a cached base_neq4.gguf is not a substitute.
# Returns the mode that can actually run ('quality'|'speed'), or halts.
function Assert-LlapisModels([hashtable]$mm, [string]$wantMode) {
  if (-not $mm.Low) {
    Write-Host ""
    Write-Host "FATAL: the low model (models\low, the fast one) is required and was not found." -ForegroundColor Red
    Write-Host ("       {0}" -f $mm.LowWhy) -ForegroundColor Red
    Write-Host ("       Put its GGUF here (e.g. IQ2_M on an 8 GB card, Q4 on a bigger one):  {0}" -f $script:DirLow) -ForegroundColor Red
    Write-Host "       Both modes need it: speed runs it directly, quality derives its base and expert arena from it." -ForegroundColor Red
    Write-Host ""
    Stop-Launcher 1
  }
  if (-not $mm.Mtp) {
    Write-Host ""
    Write-Host "FATAL: the MTP drafter is required and was not found." -ForegroundColor Red
    Write-Host ("       {0}" -f $mm.MtpWhy) -ForegroundColor Red
    Write-Host ("       Put the assistant drafter GGUF here:  {0}" -f $script:DirMtp) -ForegroundColor Red
    Write-Host ("       Expected: {0}" -f $script:ExpectMtp) -ForegroundColor Red
    Write-Host "       Without it the prompt is prefilled token-by-token instead of streamed:" -ForegroundColor Red
    Write-Host "       74.5 vs 1275 tok/s measured on a 4752-token prompt (~60 s of extra TTFT)." -ForegroundColor Red
    Write-Host "       That is too large a regression to apply silently, so LLapis refuses instead." -ForegroundColor Red
    Write-Host ""
    Stop-Launcher 1
  }
  if ($wantMode -eq 'quality' -and -not $mm.High) {
    Write-Host ""
    Write-Host "ALERT: the high model (models\high) is missing — falling back to SPEED mode." -ForegroundColor Yellow
    Write-Host ("       {0}" -f $mm.HighWhy) -ForegroundColor Yellow
    Write-Host ("       Put a higher-precision GGUF than the low one here to get quality back (e.g. Q4_K_XL, or Q8_0 with more VRAM):  {0}" -f $script:DirHigh) -ForegroundColor Yellow
    Write-Host "       Quality streams the high model live during decode, so a cached base_neq4.gguf is not a substitute." -ForegroundColor Yellow
    Write-Host ""
    return 'speed'
  }
  return $wantMode
}

# --- saved setup: ONE field ------------------------------------------------
# v2 = { version, mode }. A v1 file (the old five-question setup) is READ, not
# rejected: it already carried mode='quality', and every other field it holds is
# now either hardcoded (paths) or runtime-decided (K, context, vision). So an
# existing install keeps working and simply stops consulting the dead fields.
function Read-LlapisSetup {
  if (-not (Test-Path $ConfigPath)) { return $null }
  try { return (Get-Content $ConfigPath -Raw | ConvertFrom-Json) } catch { return $null }
}

function Test-LlapisSetup([object]$s) {
  if (-not $s) { return $false }
  if (-not ($s.PSObject.Properties.Name -contains 'mode')) { return $false }
  return ($s.mode -in @('quality', 'speed'))
}

function Write-LlapisSetup([string]$mode, [int]$vramBudgetMib = 0) {
  $obj = [ordered]@{
    version          = 3
    mode             = $mode
    # 0 = MAX (use whatever the GPU has free). Otherwise a ceiling in MiB on how much VRAM the
    # runtime may occupy; K is then maximized inside it instead of against the whole card.
    vram_budget_mib  = $vramBudgetMib
    updated_at       = (Get-Date).ToString('o')
  }
  # ascii avoids the PS5.1 UTF-16 default (would break C++/tooling reads)
  $obj | ConvertTo-Json -Depth 3 | Out-File -Encoding ascii $ConfigPath
  return $obj
}

# --- the one question ------------------------------------------------------
function Select-LlapisMode([bool]$highAvailable, [string]$current) {
  while ($true) {
    Write-Host ""
    Write-Host "Choose the runtime:"
    Write-Host ""
    Write-Host "  [1] quality"
    if ($highAvailable) {
      Write-Host "      The high model's experts kept hot in VRAM. Roughly stock speed, better answers."
    } else {
      Write-Host "      UNAVAILABLE — the high model (models\high) is missing." -ForegroundColor Yellow
    }
    Write-Host "  [2] speed"
    Write-Host "      The low model alone. Maximum speed, same quality as stock llama.cpp."
    Write-Host ""
    if ($current) { Write-Host ("  (current: {0})" -f $current) -ForegroundColor DarkGray }
    $ans = Read-Host "> "
    switch -Regex ($ans) {
      '^1$' {
        if (-not $highAvailable) { Write-Host "quality needs models/high — pick speed, or add the high model first." -ForegroundColor Yellow; continue }
        Write-Host "You picked: quality"; return 'quality'
      }
      '^2$' { Write-Host "You picked: speed"; return 'speed' }
      '^$'  {
        $d = if ($highAvailable) { 'quality' } else { 'speed' }
        Write-Host "You picked: $d"; return $d
      }
      default { Write-Host "Please enter 1 or 2." -ForegroundColor Yellow }
    }
  }
}

# VRAM budget: MAX, or a ceiling the whole runtime shrinks to fit.
#
# This replaces "pick a K" as the user-facing memory knob, and it is the right axis: K is a
# CONSEQUENCE of how much VRAM the runtime is allowed to hold, and the runtime already maximizes it
# against a measured budget every time the context changes (`vkv_refit_k`). Asking for the budget
# and deriving K from it is strictly better than asking for K and hoping it fits.
#
# The budget is expressed to the runtime as its complement — "leave at least (total - budget) MiB
# free" — because that is the quantity the refit already takes (`MOE_DUAL_VKV_KEEP_POST_MIB`).
# HONEST SEMANTICS, worth knowing: because it is enforced as free-VRAM-after, another process
# holding VRAM makes the budget bind harder rather than softer (a browser holding ~1 GiB eats into
# the ceiling instead of being ignored). That errs toward not crashing, which is the right way to
# be wrong here, but it does mean the number is "leave room", not "reserve for me".
function Select-LlapisVram([int]$current, [int]$floor = 0) {
  $total = Get-TotalVramMiB
  while ($true) {
    Write-Host ""
    Write-Host "How much VRAM may LLapis use?"
    Write-Host ""
    if ($total) { Write-Host ("  This GPU has {0} MiB ({1:N1} GiB) total." -f $total, ($total / 1024)) -ForegroundColor DarkGray }
    else        { Write-Host "  (nvidia-smi unavailable — a custom value will be taken on trust.)" -ForegroundColor DarkGray }
    if ($floor -gt 0) {
      Write-Host ("  The installed models need ~{0} MiB at the smallest expert cache." -f $floor) -ForegroundColor DarkGray
      if ($total -and $floor -gt $total) {
        Write-Host ("  That is more than this GPU has: the launch will refuse. Use smaller quants in models\low or models\high.") -ForegroundColor Red
      }
    }
    Write-Host ""
    Write-Host "  [1] Max  suggested"
    Write-Host "      Use whatever is free at launch. The expert cache grows to fill it."
    Write-Host "  [2] Set a limit"
    Write-Host "      Leave room for other GPU apps. The whole runtime shrinks to fit;"
    Write-Host "      the expert cache is sized to whatever is left, automatically."
    Write-Host ""
    if ($current -gt 0) { Write-Host ("  (current: {0} MiB)" -f $current) -ForegroundColor DarkGray }
    elseif ($current -eq 0 -and $null -ne $current) { Write-Host "  (current: Max)" -ForegroundColor DarkGray }
    $ans = Read-Host "> "
    switch -Regex ($ans) {
      '^1$' { Write-Host "You picked: Max"; return 0 }
      '^$'  { Write-Host "You picked: Max"; return 0 }
      '^2$' {
        while ($true) {
          $v = Read-Host "Limit in MiB (e.g. 6144 for 6 GiB)"
          if ($v -notmatch '^\d+$') { Write-Host "Please enter a whole number of MiB." -ForegroundColor Yellow; continue }
          $mib = [int]$v
          if ($total -and $mib -ge $total) {
            # The user's rule, applied literally: a limit at or above the card is just Max.
            Write-Host ("{0} MiB is the whole card ({1} MiB) or more — using Max." -f $mib, $total) -ForegroundColor Yellow
            return 0
          }
          # Every limit is accepted. The floor depends on the installed models' quants (Get-LlapisVramFloorMiB);
          # under it the runtime still starts, at the floor, over the limit and saying so. Only a card that
          # cannot hold the floor at all is refused, at launch, with the MiB these models need.
          if ($floor -gt 0 -and $mib -lt $floor) {
            Write-Host ("These models need ~{0} MiB with the smallest expert cache. LLapis will start at that size," -f $floor) -ForegroundColor Yellow
            Write-Host ("over your {0} MiB limit; pick {1} MiB or more to stay inside it." -f $mib, $floor) -ForegroundColor Yellow
          }
          Write-Host "You picked: $mib MiB"
          return $mib
        }
      }
      default { Write-Host "Please enter 1 or 2." -ForegroundColor Yellow }
    }
  }
}

# `setup`: CHECK -> CHOICE -> ALERT/FATAL/GO. No model paths, no context, no K.
function Invoke-LlapisWizard {
  Write-Host ""
  Write-Host "LLapis Setup"
  Write-Host ""
  Write-Host "Models are read from fixed folders — there is nothing to browse for."
  Write-Host "The only choice is which runtime to use."

  $mm = Resolve-LlapisModels
  Show-LlapisModelCheck $mm

  # FATALs first, against the mode the user might pick — checked with 'speed' so the
  # high-model ALERT does not fire here (the menu below states the same thing better).
  [void](Assert-LlapisModels $mm 'speed')

  $prev = $null; $prevVram = 0
  $old = Read-LlapisSetup
  if (Test-LlapisSetup $old) {
    $prev = $old.mode
    if ($old.PSObject.Properties.Name -contains 'vram_budget_mib') { $prevVram = [int]$old.vram_budget_mib }
  }
  $mode = Select-LlapisMode ([bool]$mm.High) $prev
  # the floor for THIS mode's models (the plain drafter: the residual was measured with it)
  $floor = 0
  $terms = Get-LlapisModelTerms $mode $mm.High $mm.Low $mm.Mtp
  if ($terms) { $floor = Get-LlapisVramFloorMiB $terms }
  else { Write-Host ("  (could not read the models' sizes: {0} -- the minimum is checked at launch instead)" -f $script:ModelTermsError) -ForegroundColor DarkGray }
  $vram = Select-LlapisVram $prevVram $floor

  Write-Host ""
  Write-Host "Setup summary"
  Write-Host ""
  Write-Host ("  Runtime:  {0}" -f $mode)
  Write-Host ("  VRAM:     {0}" -f $(if ($vram -gt 0) { "up to $vram MiB (the runtime shrinks to fit)" } else { 'Max (all free VRAM)' }))
  Write-Host ("  Context:  adaptive (grows on demand; -Ctx <n> lowers the ceiling)")
  Write-Host ("  Cache:    automatic (K is maximized inside the VRAM budget)")
  Write-Host ("  MTP:      on ({0})" -f [IO.Path]::GetFileName($mm.Mtp))
  Write-Host ""
  $save = Read-Host "Save this setup? [Y/n]"
  if ($save -match '^[nN]') { Write-Host "Not saved."; return $null }

  $s = Write-LlapisSetup $mode $vram

  # The router profile is a single fixed name now ('llapis'), so the old per-K/ctx
  # profile pileup cannot recur. Sweep any historical entries once: several predate
  # the current defaults and loading one from the web UI would silently run a slower
  # (or VRAM-unsafe) config.
  $profilesPath = Join-Path $Rel 'moert-profiles.json'
  if (Test-Path $profilesPath) {
    $nOld = 0
    try { $nOld = @(((Get-Content $profilesPath -Raw | ConvertFrom-Json).profiles).PSObject.Properties).Count } catch {}
    Remove-Item $profilesPath -Force -ErrorAction SilentlyContinue
    $genIni = $profilesPath -replace '\.json$','.gen.ini'
    if (Test-Path $genIni) { Remove-Item $genIni -Force -ErrorAction SilentlyContinue }
    if ($nOld -gt 1) { Write-Host ("  ({0} old runtime profiles cleared; the next run rebuilds the single 'llapis' profile)" -f $nOld) -ForegroundColor DarkGray }
  }

  Write-Host ""
  Write-Host "Saved."
  Write-Host "You can now run:  llapis chat   |   llapis serve"
  return $s
}
