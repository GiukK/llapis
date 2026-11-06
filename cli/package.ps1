# ===========================================================================
# cli/package.ps1 - build the release zip: LLapis without a compiler.
# ===========================================================================
#   powershell -ExecutionPolicy Bypass -File cli\package.ps1 [-Arch <list>] [-OutDir dist] [-SkipBuild]
#
# A user of the zip needs an NVIDIA driver and nothing else: no Git, no Visual Studio, no CUDA
# Toolkit. So the binaries must run on ANY supported GPU and CPU, not the build machine's:
#   - CUDA: machine code for every consumer generation CUDA 13 supports, plus PTX for the newest
#     so a later GPU can compile it at its first start (see -Arch);
#   - CPU: GGML_NATIVE=OFF with AVX2/FMA/F16C, the floor build-check already requires. A native
#     build on a CPU with AVX-512 makes a binary that crashes on one without it.
# It builds in build-release\ (never touches build-cuda\), runs the unit tests, checks with
# cuobjdump that every architecture is really in ggml-cuda.dll, and writes
#   <OutDir>\llapis-<version>-win-x64\   and   <OutDir>\llapis-<version>-win-x64.zip
# The zip also carries the CUDA runtime DLLs (cudart, cublas, cublasLt) and the Visual C++ runtime
# (msvcp140, vcruntime140, vcruntime140_1, vcomp140), both redistributable, and bin\llapis-build.json
# which cli/gpu-support.ps1 reads to tell a user whether the package can run on their GPU.
# ===========================================================================
param(
  # 75 = RTX 20 / GTX 16, 86 = RTX 30, 89 = RTX 40, 120 = RTX 50 (machine code); '120' without
  # -real also embeds compute_120 PTX for GPUs newer than the list.
  [string]$Arch = '75-real;86-real;89-real;120',
  [string]$OutDir = 'dist',
  [switch]$SkipBuild
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
$B    = Join-Path $Root 'build-release'
if (-not [IO.Path]::IsPathRooted($OutDir)) { $OutDir = Join-Path $Root $OutDir }

function Step([string]$t) { Write-Host ""; Write-Host "--- $t" -ForegroundColor Cyan }
function Die([string]$m) { Write-Host "[FAIL] $m" -ForegroundColor Red; exit 1 }

# --- toolchain (the same lookups as build-check) --------------------------------
if (-not $env:CUDA_PATH) { Die 'CUDA_PATH is not set (CUDA Toolkit 13.x needed)' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -version '[17.0,18.0)' -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
if (-not $vs) { Die 'Visual Studio 2022 with the C++ tools not found' }
$cmake = Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
if (-not (Test-Path $cmake)) { $cmake = (Get-Command cmake).Source }
$ctest = Join-Path (Split-Path $cmake) 'ctest.exe'
$cudaRoot = $env:CUDA_PATH -replace '\\', '/'
$nv = (& (Join-Path $env:CUDA_PATH 'bin\nvcc.exe') --version) -join ' '
if ($nv -notmatch 'release ([0-9]+)\.([0-9]+)') { Die 'cannot read the nvcc version' }
$cudaMajor = [int]$Matches[1]; $cudaVersion = "$($Matches[1]).$($Matches[2])"

$sha = (git -C $Root rev-parse --short HEAD).Trim()
$dirty = [bool](git -C $Root status --porcelain --untracked-files=no)
$version = (Get-Date -Format 'yyyy.MM.dd') + "-$sha" + $(if ($dirty) { '-dirty' } else { '' })
$name = "llapis-$version-win-x64"
Write-Host "LLapis package $name  (CUDA $cudaVersion, architectures $Arch)"
if ($dirty) { Write-Host '[WARN] the working tree has uncommitted changes: the version says -dirty' -ForegroundColor Yellow }

# --- configure + build + test ------------------------------------------------------
if (-not $SkipBuild) {
  Step 'Configure (build-release, portable)'
  $vvar = "CUDA_PATH_V{0}" -f ($cudaVersion -replace '\.', '_')
  if (-not [Environment]::GetEnvironmentVariable($vvar)) { [Environment]::SetEnvironmentVariable($vvar, $env:CUDA_PATH) }
  & $cmake -S $Root -B $B -G 'Visual Studio 17 2022' -A x64 -T "cuda=$cudaRoot" "-DCUDAToolkit_ROOT=$cudaRoot" `
      -DGGML_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=$Arch" `
      -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=OFF
  if ($LASTEXITCODE) { Die 'configure failed' }

  Step 'Build (Release)'
  $t = Measure-Command { & $cmake --build $B --config Release --parallel | Out-Host }
  if ($LASTEXITCODE) { Die 'build failed' }
  Write-Host ("built in {0:N1} min" -f $t.TotalMinutes)

  Step 'Unit tests'
  Push-Location $B; & $ctest -C Release --output-on-failure | Out-Host; $rc = $LASTEXITCODE; Pop-Location
  if ($rc) { Die 'unit tests failed' }
}

$Rel = Join-Path $B 'Release'
$Bin = Join-Path $B 'bin\Release'

# --- every requested architecture must really be in the binary ----------------------
Step 'GPU code in ggml-cuda.dll'
$cuobj = Join-Path $env:CUDA_PATH 'bin\cuobjdump.exe'
$dll = Join-Path $Bin 'ggml-cuda.dll'
$sass = @(& $cuobj --list-elf $dll | ForEach-Object { if ("$_" -match 'sm_([0-9]+)') { [int]$Matches[1] } } | Sort-Object -Unique)
$ptx  = @(& $cuobj --list-ptx $dll | ForEach-Object { if ("$_" -match 'sm_([0-9]+)') { [int]$Matches[1] } } | Sort-Object -Unique)
$want = @($Arch -split ';' | ForEach-Object { [int](($_ -split '-')[0]) } | Sort-Object -Unique)
$lost = @($want | Where-Object { $sass -notcontains $_ })
if ($lost.Count) { Die ("ggml-cuda.dll lacks machine code for sm_{0} (it has sm_{1})" -f ($lost -join ', sm_'), ($sass -join ', sm_')) }
Write-Host ("[ OK ] machine code: sm_{0}; PTX: {1}" -f ($sass -join ', sm_'), $(if ($ptx.Count) { 'compute_' + ($ptx -join ', compute_') } else { 'none' })) -ForegroundColor Green

# --- assemble ---------------------------------------------------------------------
Step "Assemble $name"
$P = Join-Path $OutDir $name
if (Test-Path $P) { Remove-Item -Recurse -Force $P }
$pb = New-Item -ItemType Directory -Force (Join-Path $P 'bin')

Copy-Item (Join-Path $Rel 'moert-server.exe') $pb
foreach ($d in 'ggml-base', 'ggml-cpu', 'ggml-cuda', 'ggml', 'llama', 'llama-common', 'llama-server-impl', 'mtmd') {
  Copy-Item (Join-Path $Bin "$d.dll") $pb
}
foreach ($d in 'cudart64_13.dll', 'cublas64_13.dll', 'cublasLt64_13.dll') {
  $s = Join-Path $env:CUDA_PATH "bin\x64\$d"; if (-not (Test-Path $s)) { $s = Join-Path $env:CUDA_PATH "bin\$d" }
  if (-not (Test-Path $s)) { Die "$d not found in the CUDA Toolkit" }
  Copy-Item $s $pb
}
# Visual C++ runtime, app-local: Microsoft allows redistributing these next to the application.
$redist = Get-ChildItem (Join-Path $vs 'VC\Redist\MSVC') -Directory | Where-Object { $_.Name -match '^\d+\.\d+\.\d+$' } |
          Sort-Object { [version]$_.Name } | Select-Object -Last 1
foreach ($sub in 'Microsoft.VC143.CRT', 'Microsoft.VC143.OpenMP') {
  $dir = Join-Path $redist.FullName "x64\$sub"
  if (-not (Test-Path $dir)) { Die "Visual C++ redistributable not found: $dir" }
  Copy-Item (Join-Path $dir '*.dll') $pb
}
foreach ($d in 'msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'vcomp140.dll') {
  if (-not (Test-Path (Join-Path $pb $d))) { Die "Visual C++ runtime: $d missing" }
}

[ordered]@{
  version    = $version
  commit     = (git -C $Root rev-parse HEAD).Trim()
  cuda       = $cudaVersion
  cuda_major = $cudaMajor
  sass       = $sass
  ptx        = $ptx
  cpu        = 'x86-64 with AVX2, FMA, F16C'
} | ConvertTo-Json | Out-File -Encoding ascii (Join-Path $pb 'llapis-build.json')

# The scripts and data the runtime reads, from git (exactly what is committed, nothing local).
$tmpTar = Join-Path $OutDir "$name.src.tar"
git -C $Root archive --format=tar -o $tmpTar HEAD llapis.cmd llapis.ps1 build-check.cmd cli assets models LICENSE THIRD_PARTY_LICENSES.txt README.md
if ($LASTEXITCODE) { Die 'git archive failed' }
tar -x -f $tmpTar -C $P
Remove-Item $tmpTar
Remove-Item (Join-Path $P 'cli\package.ps1') -ErrorAction SilentlyContinue   # a build tool, meaningless without the sources

@"
LLapis $version -- Windows x64, NVIDIA

This is the ready-to-run package: no compiler, no CUDA Toolkit, no Git needed.
Requirements: Windows 10/11 x64, an NVIDIA RTX 20 series or newer, a current NVIDIA driver
(CUDA 13: the R580 family or newer), 16 GB of RAM, a CPU with AVX2.

  1. .\build-check.cmd                     checks this machine
  2. download the models into models\     (README.md, section "Models")
  3. .\bin\moert-server.exe prepare
  4. .\llapis.cmd setup
  5. .\llapis.cmd serve -Open

GPU code included: machine code for sm_$($sass -join ', sm_')$(if ($ptx.Count) { "; PTX for compute_$($ptx -join ', compute_')" }).

Third-party binaries in bin\: the NVIDIA CUDA runtime (cudart64_13, cublas64_13, cublasLt64_13),
redistributed under the NVIDIA CUDA Toolkit End User License Agreement
(https://docs.nvidia.com/cuda/eula/), and the Microsoft Visual C++ runtime (msvcp140, vcruntime140,
vcruntime140_1, vcomp140), redistributed under the Visual Studio redistribution terms. They are
not covered by LLapis' MIT licence. Everything else: see LICENSE and THIRD_PARTY_LICENSES.txt.
"@ | Out-File -Encoding ascii (Join-Path $P 'PACKAGE.txt')

# --- zip --------------------------------------------------------------------------
Step 'Zip'
$zip = Join-Path $OutDir "$name.zip"
if (Test-Path $zip) { Remove-Item $zip }
Push-Location $OutDir
tar -a -c -f "$name.zip" $name
$rc = $LASTEXITCODE
Pop-Location
if ($rc) { Die 'zip failed' }
$size = (Get-Item $zip).Length / 1MB
Write-Host ("[ OK ] {0}  ({1:N0} MB)" -f $zip, $size) -ForegroundColor Green
$hash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
"$hash  $name.zip" | Out-File -Encoding ascii "$zip.sha256"
Write-Host "       SHA-256 $hash"
