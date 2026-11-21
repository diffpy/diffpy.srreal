# Install the compiler, static runtime and CUDA build rules for MSBuild.
param([Parameter(Mandatory = $true)] [string] $Prefix)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$url = "https://developer.download.nvidia.com/compute/cuda/13.5.0/network_installers/cuda_13.5.0_windows_x86_64_network.exe"
$sha256 = "a03530c404b881bb2b34d23de3fd6a74e819ed9d4fd5633f428c30cf85f012e3"
$installer = Join-Path $env:TEMP "cuda_13.5.0_windows_x86_64_network.exe"
Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $installer
if ((Get-FileHash $installer -Algorithm SHA256).Hash -ne $sha256) {
    throw "CUDA installer checksum mismatch."
}
$components = @(
    "-s", "nvcc_13.5", "nvvm_13.5", "crt_13.5", "cudart_13.5", "cccl_13.5",
    "visual_studio_integration_13.5"
)
$setup = Start-Process -FilePath $installer -ArgumentList $components -Wait -PassThru
if ($setup.ExitCode -ne 0) { throw "CUDA installer exited with code $($setup.ExitCode)." }

$licenseDir = New-Item -ItemType Directory -Force -Path (Join-Path $Prefix "licenses/cuda")
Copy-Item (Join-Path $env:CUDA_PATH_V13_5 "EULA.txt") $licenseDir
