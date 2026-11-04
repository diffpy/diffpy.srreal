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
