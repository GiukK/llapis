# ===========================================================================
# llapis/build-check.ps1 - can this machine build and run LLapis?
# ===========================================================================
# Run through build-check.cmd from the repository root:
#   .\build-check.cmd            check the prerequisites, say what is missing and where to get it
#   .\build-check.cmd -Build     ...then configure, build and test (build-cuda\), and check that the
#                                CUDA kernels were compiled for this machine's GPU
#   .\build-check.cmd -Smoke     ...then start the server with the installed models and ask it one
#                                question (needs the models and a saved `llapis setup`)
#   -NoGpu -Arch 86-real         build on a machine without an NVIDIA GPU (a VM, a CI runner): the GPU
#                                checks are skipped and the CUDA code is compiled for -Arch
#   -Arch <list>                 with a GPU too: compile for these architectures instead of this GPU's
# Every check prints [ OK ], [WARN] or [FAIL]; any FAIL exits 1. Nothing is installed or changed
# except build-cuda\ (with -Build) and logs\ (with -Smoke).
# ===========================================================================
param([switch]$Build, [switch]$Smoke, [switch]$NoGpu, [string]$Arch)

$ErrorActionPreference = 'Continue'
$Root = Split-Path $PSScriptRoot -Parent
$script:fails = 0; $script:warns = 0

function Ok([string]$m)   { Write-Host "[ OK ] $m" -ForegroundColor Green }
function Warn([string]$m, [string]$fix) {
  $script:warns++; Write-Host "[WARN] $m" -ForegroundColor Yellow
  if ($fix) { Write-Host "       $fix" -ForegroundColor Yellow }
}
function Fail([string]$m, [string]$fix) {
  $script:fails++; Write-Host "[FAIL] $m" -ForegroundColor Red
  if ($fix) { Write-Host "       $fix" -ForegroundColor Red }
}
function Section([string]$t) { Write-Host ""; Write-Host "--- $t" -ForegroundColor Cyan }
function VersionOf([string]$s) { try { return [version]$s } catch { return $null } }

Write-Host "LLapis build check  ($Root)"

# --- system --------------------------------------------------------------
Section 'System'
if ([Environment]::Is64BitOperatingSystem) { Ok ("Windows x64 ({0})" -f [Environment]::OSVersion.Version) }
else { Fail 'Windows is not 64-bit' 'LLapis needs 64-bit Windows 10 or 11.' }
Ok ("PowerShell {0}" -f $PSVersionTable.PSVersion)

Add-Type -Namespace LlapisCheck -Name Native -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern bool IsProcessorFeaturePresent(uint feature);
'@ -ErrorAction SilentlyContinue
$cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
# PF_AVX2_INSTRUCTIONS_AVAILABLE = 40 (reported by Windows 10 1809 and later)
if ([LlapisCheck.Native]::IsProcessorFeaturePresent(40)) { Ok "CPU with AVX2: $cpu" }
else { Fail "CPU without AVX2 (or a Windows too old to report it): $cpu" 'The CPU expert kernels need AVX2 (Intel Haswell / AMD Zen or newer).' }

$ramGb = [Math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
if ($ramGb -ge 15) { Ok "RAM: $ramGb GB" }
else { Warn "RAM: $ramGb GB - 16 GB is the floor (the cold expert arena lives in RAM)" 'The server refuses to load with too little RAM; a build still works.' }

$drive = (Get-Item $Root).PSDrive
$freeGb = [Math]::Round($drive.Free / 1GB, 0)
if ($freeGb -ge 45) { Ok "Free disk on ${drive}: $freeGb GB" }
else { Warn "Free disk on ${drive}: $freeGb GB" 'Build ~2 GB; models + derived cache ~22 GB (speed) to ~38 GB (quality).' }

$lp = $null
try { $lp = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -ErrorAction Stop).LongPathsEnabled } catch {}
if ($lp -eq 1) { Ok 'Windows long paths enabled' }
elseif ($Root.Length -le 60) { Ok ("Install path is short enough ({0} characters; long paths are off)" -f $Root.Length) }
else { Warn ("Install path is {0} characters and Windows long paths are off" -f $Root.Length) 'Source paths reach 161 characters: move the folder to a short path such as C:\llapis.' }

# --- GPU and driver ---------------------------------------------------------
Section 'NVIDIA GPU and driver'
$smi = (Get-Command nvidia-smi -ErrorAction SilentlyContinue).Source
if (-not $smi -and (Test-Path "$env:SystemRoot\System32\nvidia-smi.exe")) { $smi = "$env:SystemRoot\System32\nvidia-smi.exe" }
$gpuCc = $null; $driverCuda = $null
if ($NoGpu) {
  if ($Smoke) { Fail '-Smoke needs the GPU: it cannot be combined with -NoGpu' }
  if ($Build -and -not $Arch) { Fail '-NoGpu -Build needs -Arch (there is no GPU to detect)' 'e.g. -Arch 86-real for an RTX 30 card; 75 = RTX 20, 89 = RTX 40, 120 = RTX 50.' }
  Warn 'GPU checks skipped (-NoGpu): this machine can build, not run' $(if ($Arch) { "CUDA code for: $Arch" } else { '' })
} elseif (-not $smi) {
  Fail 'nvidia-smi not found: no NVIDIA driver installed' 'Install the latest NVIDIA driver for your card: https://www.nvidia.com/Download/index.aspx'
} else {
  $q = & $smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader 2>$null | Select-Object -First 1
  if (-not $q) { Fail 'nvidia-smi found no GPU' 'Check that the NVIDIA card is installed and the driver loaded (Device Manager).' }
  else {
    $f = $q -split ',\s*'
    $gpuCc = VersionOf $f[1]
    Ok ("GPU: {0}, compute capability {1}, {2}, driver {3}" -f $f[0], $f[1], $f[3], $f[2])
    if ($gpuCc -and $gpuCc -lt [version]'7.5') {
      Fail "compute capability $($f[1]) is below 7.5" 'CUDA 13 cannot build for Maxwell, Pascal or Volta (GTX 10 series and older): an RTX 20 or newer is needed.'
    }
  }
  $hdr = (& $smi 2>$null) -join "`n"
  if ($hdr -match 'CUDA Version:\s*([0-9]+\.[0-9]+)') { $driverCuda = VersionOf $Matches[1]; Ok "the driver supports CUDA up to $driverCuda" }
  else { Warn 'could not read the CUDA version the driver supports' }
}

# --- CUDA toolkit -----------------------------------------------------------
Section 'CUDA Toolkit'
$cudaVer = $null
if (-not $env:CUDA_PATH) {
  Fail 'CUDA_PATH is not set: the CUDA Toolkit is not installed' 'Install CUDA Toolkit 13.x (13.3 is the tested one) AFTER Visual Studio: https://developer.nvidia.com/cuda-downloads - then open a NEW terminal.'
} else {
  $nvcc = Join-Path $env:CUDA_PATH 'bin\nvcc.exe'
  if (-not (Test-Path $nvcc)) { Fail "nvcc.exe not found under CUDA_PATH ($env:CUDA_PATH)" 'Reinstall the CUDA Toolkit.' }
  else {
    $nv = (& $nvcc --version 2>$null) -join ' '
    if ($nv -match 'release ([0-9]+\.[0-9]+)') { $cudaVer = VersionOf $Matches[1] }
    if (-not $cudaVer) { Fail 'could not read the nvcc version' }
    elseif ($cudaVer.Major -ne 13) { Fail "CUDA Toolkit $cudaVer - LLapis needs 13.x" 'The launcher stages the CUDA 13 runtime (cudart64_13.dll ...). Install CUDA Toolkit 13.x.' }
    else { Ok "CUDA Toolkit $cudaVer ($env:CUDA_PATH)" }
    # CUDA's minor-version compatibility: a 13.x binary runs on any driver of the 13 family (the reference
    # machine builds with 13.3 and runs on a driver that reports 13.1). Only an older MAJOR version fails.
    if ($cudaVer -and $driverCuda) {
      if ($driverCuda.Major -lt $cudaVer.Major) {
        Fail "the driver supports CUDA $driverCuda only; this toolkit builds CUDA $($cudaVer.Major) binaries" 'Update the NVIDIA driver (CUDA 13 needs the R580 driver family or newer).'
      } elseif ($driverCuda -lt $cudaVer) {
        Ok "driver CUDA $driverCuda < toolkit $cudaVer`: fine within CUDA $($cudaVer.Major) (minor-version compatibility)"
      }
    }
    foreach ($d in 'cudart64_13.dll', 'cublas64_13.dll', 'cublasLt64_13.dll') {
      if (-not (Test-Path (Join-Path $env:CUDA_PATH "bin\x64\$d"))) { Warn "$d not found in $env:CUDA_PATH\bin\x64" 'The launcher copies it next to the server from there; reinstall the toolkit if it is missing.' }
    }
  }
}

# --- Visual Studio and its CUDA integration ---------------------------------
Section 'Visual Studio 2022 (C++ tools)'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsPath = $null
if (Test-Path $vswhere) {
  $vsPath = & $vswhere -version '[17.0,18.0)' -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null | Select-Object -First 1
}
if (-not $vsPath) {
  Fail 'Visual Studio 2022 with the C++ x64 tools not found' 'Install "Build Tools for Visual Studio 2022" (free) with the workload "Desktop development with C++": https://visualstudio.microsoft.com/downloads/ (Tools for Visual Studio).'
} else {
  Ok "Visual Studio 2022 C++ tools: $vsPath"
  $bc = Join-Path $vsPath 'MSBuild\Microsoft\VC\v170\BuildCustomizations'
  $props = @(Get-ChildItem $bc -Filter 'CUDA *.props' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name)
  $want = if ($cudaVer) { "CUDA $($cudaVer.Major).$($cudaVer.Minor).props" } else { $null }
  if ($want -and ($props -contains $want)) { Ok "CUDA integration for Visual Studio: $want" }
  elseif ($props.Count) { Fail ("Visual Studio has the CUDA integration for {0}, not for the installed toolkit {1}" -f ($props -join ', '), $cudaVer) 'Run the CUDA installer again: Custom -> select "Visual Studio Integration" only.' }
  else { Fail 'Visual Studio has no CUDA integration (CMake will say "No CUDA toolset found")' 'Usual cause: CUDA was installed before Visual Studio. Run the CUDA installer again: Custom -> "Visual Studio Integration".' }
}

# --- CMake ------------------------------------------------------------------
Section 'CMake'
# Visual Studio's own CMake first: it is the one the README and the reference machine use, and a CMake from
# another toolchain on the PATH (Strawberry Perl, MinGW ...) can misdetect the MSVC compiler.
function CMakeVersion([string]$exe) {
  if (((& $exe --version 2>$null) -join ' ') -match 'version ([0-9]+\.[0-9]+(\.[0-9]+)?)') { return VersionOf $Matches[1] }
  return $null
}
$cmake = $null; $cmVer = $null
if ($vsPath) {
  $vsCmake = Join-Path $vsPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
  if (Test-Path $vsCmake) { $cmake = $vsCmake; $cmVer = CMakeVersion $vsCmake }
}
if (-not $cmVer -or $cmVer -lt [version]'3.24') {
  $pathCmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
  if ($pathCmake) { $v2 = CMakeVersion $pathCmake; if ($v2 -and (-not $cmVer -or $v2 -gt $cmVer)) { $cmake = $pathCmake; $cmVer = $v2 } }
}
if (-not $cmVer) { Fail 'CMake not found' 'It comes with Visual Studio''s "Desktop development with C++" (C++ CMake tools), or from https://cmake.org/download/' }
elseif ($cmVer -lt [version]'3.24') { Fail "CMake $cmVer is older than 3.24" 'Older CMake cannot detect the GPU architecture; use the one inside Visual Studio 2022 or a current one.' }
else { Ok "CMake $cmVer ($cmake)" }

Section 'Git (only to clone the repository)'
if (Get-Command git -ErrorAction SilentlyContinue) { Ok ((git --version) -join ' ') } else { Warn 'git not found' 'Only needed to clone; a zip of the repository works without it.' }

# --- summary of the prerequisites ------------------------------------------------
Write-Host ""
if ($script:fails) {
  Write-Host ("{0} problem(s) to fix, {1} warning(s). Fix the [FAIL] lines and run the check again." -f $script:fails, $script:warns) -ForegroundColor Red
  exit 1
}
Write-Host ("All prerequisites present ({0} warning(s))." -f $script:warns) -ForegroundColor Green
if (-not $Build -and -not $Smoke) { Write-Host 'Next: .\build-check.cmd -Build'; exit 0 }

# --- build and test -------------------------------------------------------
$rel = Join-Path $Root 'build-cuda\Release'
if ($Build) {
  Section 'Configure (build-cuda)'
  $cudaRoot = $env:CUDA_PATH -replace '\\', '/'   # forward slashes: a backslash in a -D value is an escape to CMake
  $log = Join-Path $Root 'build-cuda-check.log'
  $cfg = @('-S', $Root, '-B', (Join-Path $Root 'build-cuda'), '-G', 'Visual Studio 17 2022', '-A', 'x64', '-DGGML_CUDA=ON', "-DCUDAToolkit_ROOT=$cudaRoot")
  if ($Arch) { $cfg += "-DCMAKE_CUDA_ARCHITECTURES=$Arch" }
  & $cmake @cfg 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $log -Encoding utf8
  if ($LASTEXITCODE -ne 0) { Fail "configure failed - see $log"; Get-Content $log -Tail 25; exit 1 }
  $arch = Select-String -Path $log -Pattern 'Using CMAKE_CUDA_ARCHITECTURES=(\S+)' | Select-Object -First 1
  Ok ("configured" + $(if ($arch) { " (CUDA architectures: $($arch.Matches[0].Groups[1].Value))" } else { '' }))

  Section 'Build (Release; ~5-15 minutes)'
  & $cmake --build (Join-Path $Root 'build-cuda') --config Release --parallel 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $log -Encoding utf8 -Append
  if ($LASTEXITCODE -ne 0) {
    Fail "build failed - see $log"
    Select-String -Path $log -Pattern ' error |error C|error MSB|fatal error' | Select-Object -First 15 | ForEach-Object { Write-Host "  $($_.Line.Trim())" }
    exit 1
  }
  if (Test-Path (Join-Path $rel 'moert-server.exe')) { Ok 'moert-server.exe built' } else { Fail 'the build finished but moert-server.exe is missing' ; exit 1 }

  Section 'Unit tests (no GPU, no model)'
  $ctest = Join-Path (Split-Path $cmake) 'ctest.exe'
  Push-Location (Join-Path $Root 'build-cuda')
  $t = & $ctest -C Release 2>&1 | ForEach-Object { "$_" }
  Pop-Location
  $sum = $t | Where-Object { $_ -match 'tests passed' } | Select-Object -First 1
  if ($LASTEXITCODE -eq 0) { Ok $sum } else { Fail "unit tests failed: $sum"; $t | Where-Object { $_ -match 'Failed|\*\*\*' } | ForEach-Object { Write-Host "  $_" } }

  Section 'CUDA kernels compiled for this GPU'
  $cuobj = Join-Path $env:CUDA_PATH 'bin\cuobjdump.exe'
  $dll = Join-Path $Root 'build-cuda\bin\Release\ggml-cuda.dll'
  if ($gpuCc -and (Test-Path $cuobj) -and (Test-Path $dll)) {
    $sms = @(& $cuobj --list-elf $dll 2>$null | ForEach-Object { if ("$_" -match 'sm_([0-9]+)') { [int]$Matches[1] } } | Sort-Object -Unique)
    $need = $gpuCc.Major * 10 + $gpuCc.Minor
    # a cubin for sm_XY runs on the same major version with an equal or higher minor
    $runs = @($sms | Where-Object { [Math]::Floor($_ / 10) -eq $gpuCc.Major -and $_ -le $need })
    if ($runs.Count) { Ok ("ggml-cuda.dll holds code for sm_{0}; this GPU is sm_{1}" -f ($sms -join ', sm_'), $need) }
    else { Fail ("ggml-cuda.dll holds code for sm_{0} only; this GPU is sm_{1}" -f ($sms -join ', sm_'), $need) 'Delete build-cuda\ and run the check with -Build again (the architecture is detected on a fresh configure).' }
  } elseif ($NoGpu -and (Test-Path $cuobj) -and (Test-Path $dll)) {
    $sms = @(& $cuobj --list-elf $dll 2>$null | ForEach-Object { if ("$_" -match 'sm_([0-9]+)') { [int]$Matches[1] } } | Sort-Object -Unique)
    if ($sms.Count) { Ok ("ggml-cuda.dll holds code for sm_{0} (no GPU here to compare with)" -f ($sms -join ', sm_')) }
    else { Fail 'ggml-cuda.dll holds no GPU code' }
  } else { Warn 'could not inspect the compiled CUDA architectures (cuobjdump or ggml-cuda.dll missing)' }
}

# --- end-to-end: the server on this GPU -------------------------------------------
if ($Smoke) {
  Section 'Smoke test: start the server and ask one question'
  if (-not (Test-Path (Join-Path $rel 'moert-server.exe'))) { Fail 'no build: run with -Build first'; exit 1 }
  if (-not (Test-Path (Join-Path $Root 'llapis-user.json'))) { Fail 'no saved setup' 'Run .\llapis.cmd setup once, then this check with -Smoke.'; exit 1 }
  $out = Join-Path $env:TEMP 'llapis-smoke.out'
  $t0 = Get-Date
  $null = Start-Process -FilePath (Join-Path $Root 'llapis.cmd') -ArgumentList 'serve' -WorkingDirectory $Root -RedirectStandardOutput $out -RedirectStandardError "$out.err" -PassThru -WindowStyle Hidden
  $log = $null; $state = 'timeout'
  for ($i = 0; $i -lt 150; $i++) {
    Start-Sleep 2
    $log = Get-ChildItem (Join-Path $Root 'logs') -Filter '*_serve.log' -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -gt $t0 } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($log -and (Select-String -Path $log.FullName -Pattern 'boot baseline' -Quiet)) { $state = 'up'; break }
    if ($log -and (Select-String -Path $log.FullName -Pattern 'PREFLIGHT FAILED|crashed|CUDA error|no kernel image' -Quiet)) { $state = 'failed'; break }
    if ((Test-Path $out) -and (Select-String -Path $out -Pattern 'FATAL' -Quiet)) { $state = 'failed'; break }
  }
  if ($state -ne 'up') {
    Fail "the server did not start ($state)" ("Send these logs: $out" + $(if ($log) { " and $($log.FullName)" } else { '' }))
    Select-String -Path $out -Pattern 'FATAL|WARNING' -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $($_.Line)" }
    if ($log) { Select-String -Path $log.FullName -Pattern 'PREFLIGHT FAILED|CUDA error|no kernel image' | Select-Object -First 5 | ForEach-Object { Write-Host "  $($_.Line)" } }
  } else {
    $k = Select-String -Path $log.FullName -Pattern 'Runtime parameters loaded.*' | Select-Object -First 1
    Ok ("server up: " + $(if ($k) { $k.Matches[0].Value } else { $log.Name }))
    try {
      $body = '{"model":"llapis","messages":[{"role":"user","content":"Name three primary colors."}],"max_tokens":64,"temperature":0}'
      $r = Invoke-RestMethod -Uri 'http://127.0.0.1:8077/v1/chat/completions' -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 300
      Ok ("answered: {0} tokens at {1:N1} tok/s (a first answer is slow: the expert cache starts cold)" -f $r.usage.completion_tokens, $r.timings.predicted_per_second)
    } catch { Fail "the request failed: $_" "Send the log: $($log.FullName)" }
  }
  & (Join-Path $Root 'llapis.cmd') stop | Out-Null
}

Write-Host ""
if ($script:fails) { Write-Host ("{0} problem(s), {1} warning(s)." -f $script:fails, $script:warns) -ForegroundColor Red; exit 1 }
Write-Host ("Everything checked passed ({0} warning(s))." -f $script:warns) -ForegroundColor Green
exit 0
