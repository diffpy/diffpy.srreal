# Install static Boost.Serialization and GSL with vcpkg for Windows PyPI wheels.
#
# Usage: tools/install-vcpkg-deps.ps1 PREFIX TRIPLET
#
# Windows counterpart of install-vcpkg-deps.sh: checks out vcpkg at the
# builtin-baseline commit of vcpkg.json, installs into PREFIX/installed and
# copies license texts to PREFIX/licenses.
param(
    [Parameter(Mandatory = $true)] [string] $Prefix,
    [Parameter(Mandatory = $true)] [string] $Triplet
)
$ErrorActionPreference = "Stop"

function Invoke-Checked {
    $command, $arguments = $args
    # Destructuring unwraps a single argument; keep splatting an array.
    $arguments = @($arguments)
    & $command @arguments
    if ($LASTEXITCODE -ne 0) { throw "$command failed with exit code $LASTEXITCODE" }
}

$project = Split-Path -Parent $PSScriptRoot
$baseline = (Get-Content (Join-Path $project "vcpkg.json") -Raw | ConvertFrom-Json)."builtin-baseline"
$vcpkgRoot = Join-Path $Prefix "vcpkg"
$vcpkg = Join-Path $vcpkgRoot "vcpkg.exe"

# Prefer the current compiler's runtime over older DLLs elsewhere on PATH.
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$vsRoot = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if ($LASTEXITCODE -ne 0 -or -not $vsRoot) { throw "Cannot locate the Visual C++ toolset." }
$redistVersion = (Get-Content (Join-Path $vsRoot "VC\Auxiliary\Build\Microsoft.VCRedistVersion.default.txt") -Raw).Trim()
$redistRoot = Join-Path $vsRoot "VC\Redist\MSVC\$redistVersion\x64"
$crt = Get-ChildItem $redistRoot -Directory -Filter "Microsoft.VC*.CRT" | Select-Object -First 1
if (-not $crt) { throw "Cannot locate the x64 Visual C++ runtime in $redistRoot." }
$redist = Join-Path $Prefix "redist"
New-Item -ItemType Directory -Force -Path $redist | Out-Null
Copy-Item (Join-Path $crt.FullName "*.dll") $redist -Force
Write-Host "Staged Visual C++ runtime $redistVersion from $($crt.FullName)"

if (-not (Test-Path $vcpkg)) {
    Invoke-Checked git init -q $vcpkgRoot
    Invoke-Checked git -C $vcpkgRoot fetch -q --depth 1 https://github.com/microsoft/vcpkg.git $baseline
    Invoke-Checked git -C $vcpkgRoot -c advice.detachedHead=false checkout -q FETCH_HEAD
    Invoke-Checked (Join-Path $vcpkgRoot "bootstrap-vcpkg.bat") -disableMetrics
}

Invoke-Checked $vcpkg install `
    "--x-manifest-root=$project" `
    "--x-install-root=$(Join-Path $Prefix 'installed')" `
    "--overlay-triplets=$(Join-Path $project 'tools\vcpkg-triplets')" `
    "--triplet=$Triplet"

$share = Join-Path $Prefix "installed\$Triplet\share"
$licenses = Join-Path $Prefix "licenses"
New-Item -ItemType Directory -Force -Path (Join-Path $licenses "boost"), (Join-Path $licenses "gsl") | Out-Null
Copy-Item (Join-Path $share "boost-serialization\copyright") (Join-Path $licenses "boost\LICENSE_1_0.txt")
Copy-Item (Join-Path $share "gsl\copyright") (Join-Path $licenses "gsl\COPYING")
