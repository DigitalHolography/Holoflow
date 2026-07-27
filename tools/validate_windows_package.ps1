param(
    [Parameter(Mandatory = $true)]
    [string]$StageDirectory
)

$ErrorActionPreference = "Stop"
$stage = (Resolve-Path -LiteralPath $StageDirectory).Path
$bin = Join-Path $stage "bin"

$requiredFiles = @(
    "bin\holovibes.exe",
    "bin\qtadvanceddocking-qt6.dll",
    "plugins\platforms\qwindows.dll",
    "share\holovibes\nvrtc\cuda\include\cuda_runtime.h",
    "share\holovibes\nvrtc\cuda\include\cccl\cuda\std\version",
    "share\holovibes\nvrtc\mathdx\include\cusolverdx.hpp",
    "share\holovibes\nvrtc\mathdx\include\cusolverdx_io.hpp",
    "share\holovibes\nvrtc\mathdx\include\commondx\types.hpp",
    "share\holovibes\nvrtc\mathdx\include\cusolverdx\types.hpp",
    "share\holovibes\nvrtc\mathdx\cutlass\include\cutlass\cutlass.h",
    "share\holovibes\nvrtc\mathdx\lib\libcusolverdx.fatbin",
    "share\licenses\NVIDIA-CUDA-EULA.txt",
    "share\licenses\NVIDIA-MathDx-LICENSE.txt"
)

foreach ($relativePath in $requiredFiles) {
    $path = Join-Path $stage $relativePath
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required runtime file is missing: $path"
    }
}

$requiredPatterns = @(
    "cublas64_*.dll",
    "cublasLt64_*.dll",
    "cufft64_*.dll",
    "nvJitLink_*.dll",
    "nvrtc64_*.dll",
    "nvrtc-builtins64_*.dll",
    "msvcp140.dll",
    "vcomp140.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll"
)

foreach ($pattern in $requiredPatterns) {
    $matches = @(Get-ChildItem -LiteralPath $bin -Filter $pattern -File)
    if ($matches.Count -ne 1) {
        throw "Expected exactly one '$pattern' in '$bin', found $($matches.Count)."
    }
}

$dumpbin = (Get-Command dumpbin.exe -ErrorAction Stop).Source
$binaries = @(Get-ChildItem -LiteralPath $bin -Recurse -File |
    Where-Object { $_.Extension -in @(".exe", ".dll") })

$stagedNames = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)
foreach ($binary in $binaries) {
    [void]$stagedNames.Add($binary.Name)
}

$missing = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)

foreach ($binary in $binaries) {
    $output = & $dumpbin /dependents $binary.FullName
    if ($LASTEXITCODE -ne 0) {
        throw "dumpbin failed for '$($binary.FullName)'."
    }

    foreach ($line in $output) {
        if ($line -notmatch "^\s+([A-Za-z0-9_.+-]+\.dll)\s*$") {
            continue
        }

        $dependency = $Matches[1]
        if ($stagedNames.Contains($dependency) -or
            $dependency.StartsWith("api-ms-", [System.StringComparison]::OrdinalIgnoreCase) -or
            $dependency.StartsWith("ext-ms-", [System.StringComparison]::OrdinalIgnoreCase) -or
            (Test-Path -LiteralPath (Join-Path $env:WINDIR "System32\$dependency"))) {
            continue
        }

        [void]$missing.Add("$dependency (required by $($binary.Name))")
    }
}

if ($missing.Count -gt 0) {
    throw "Unresolved staged runtime dependencies:`n$($missing -join "`n")"
}

$version = (Get-Content -LiteralPath (Join-Path $PSScriptRoot "..\VERSION") -Raw).Trim()
$versionInfo = (Get-Item -LiteralPath (Join-Path $bin "holovibes.exe")).VersionInfo
if ($versionInfo.ProductVersion -ne $version -or $versionInfo.FileVersion -ne $version) {
    throw "Executable metadata '$($versionInfo.ProductVersion)'/'$($versionInfo.FileVersion)' does not match VERSION '$version'."
}

Write-Host "Validated $($binaries.Count) staged PE files for Holoflow $version."
