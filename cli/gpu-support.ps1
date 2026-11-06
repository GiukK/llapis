# ===========================================================================
# cli/gpu-support.ps1 - can THIS package's binaries run on THIS machine's GPU and driver?
# ===========================================================================
# Only the release zip carries bin\llapis-build.json (written by cli/package.ps1): it lists the GPU
# architectures the CUDA code was compiled for. A source build is compiled for the GPU it was built
# on and has no manifest, so there is nothing to check and the answer is 'unknown'.
#
# Without this, a GPU the package has no code for fails at the first kernel launch with
# "no kernel image is available for execution on the device", deep in a log; a driver too old for
# CUDA 13 fails with "CUDA driver version is insufficient". Both are now a sentence that says what
# to do. Dot-sourced by cli/build-check.ps1 and cli/preset.ps1.

function Get-LlapisSmiPath {
  $smi = (Get-Command nvidia-smi -ErrorAction SilentlyContinue).Source
  if (-not $smi -and (Test-Path "$env:SystemRoot\System32\nvidia-smi.exe")) { $smi = "$env:SystemRoot\System32\nvidia-smi.exe" }
  return $smi
}

# Returns @{ Level; Message } where Level is one of:
#   ok      the package holds machine code for this GPU
#   jit     no machine code, but PTX the driver can compile at the first start (slow once, then cached)
#   none    no code this GPU can run              -> refuse to start
#   driver  the driver predates CUDA 13           -> refuse to start
#   unknown no manifest (a source build) or no nvidia-smi: nothing to say
function Test-LlapisGpuSupport([string]$BinDir) {
  $manifest = Join-Path $BinDir 'llapis-build.json'
  if (-not (Test-Path $manifest)) { return @{ Level = 'unknown'; Message = 'no build manifest (a source build is compiled for its own GPU)' } }
  try { $m = Get-Content $manifest -Raw | ConvertFrom-Json } catch { return @{ Level = 'unknown'; Message = "unreadable $manifest" } }
  $smi = Get-LlapisSmiPath
  if (-not $smi) { return @{ Level = 'unknown'; Message = 'nvidia-smi not found' } }

  $hdr = (& $smi 2>$null) -join "`n"
  if ($hdr -match 'CUDA Version:\s*([0-9]+)\.([0-9]+)') {
    $drvMajor = [int]$Matches[1]
    if ($drvMajor -lt [int]$m.cuda_major) {
      $family = if ([int]$m.cuda_major -eq 13) { ' (the R580 family or newer)' } else { '' }
      return @{ Level = 'driver'; Message = ("the NVIDIA driver supports CUDA {0}.{1}; this package needs a driver for CUDA {2}{3}. Update it from https://www.nvidia.com/Download/index.aspx" -f $Matches[1], $Matches[2], $m.cuda_major, $family) }
    }
  }

  $q = & $smi --query-gpu=name,compute_cap --format=csv,noheader 2>$null | Select-Object -First 1
  if (-not $q) { return @{ Level = 'unknown'; Message = 'nvidia-smi reported no GPU' } }
  $f = $q -split ',\s*'
  $parts = $f[1].Trim() -split '\.'
  $major = [int]$parts[0]; $minor = [int]$parts[1]; $cc = $major * 10 + $minor
  $built = (@($m.sass) | ForEach-Object { "sm_$_" }) -join ', '
  # A cubin for sm_XY runs on the same major version with an equal or higher minor (sm_86 on sm_87).
  if (@($m.sass | Where-Object { [Math]::Floor([int]$_ / 10) -eq $major -and [int]$_ -le $cc }).Count) {
    return @{ Level = 'ok'; Message = ("{0} (sm_{1}): this package has machine code for it ({2})" -f $f[0], $cc, $built) }
  }
  # PTX for compute_XY is compiled by the driver for any GPU of compute capability >= XY.
  if (@($m.ptx | Where-Object { [int]$_ -le $cc }).Count) {
    return @{ Level = 'jit'; Message = ("{0} (sm_{1}) is newer than this package's machine code ({2}): the driver compiles the GPU code at the first start, which can take several minutes once, then it is cached" -f $f[0], $cc, $built) }
  }
  $why = if ($cc -lt 75) { 'CUDA 13 does not support GPUs older than the RTX 20 series' } else { 'build LLapis from source for it (README, "Build from source")' }
  return @{ Level = 'none'; Message = ("{0} (sm_{1}): this package has no GPU code it can run (built for {2}): {3}" -f $f[0], $cc, $built, $why) }
}
