param(
    [string]$Module = "holoflow",
    [string]$CoveragePath = "build/coverage/coverage.cobertura.xml",
    [int]$Limit = 25
)

$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$resolvedCoveragePath = Join-Path $repositoryRoot $CoveragePath
if (-not (Test-Path -LiteralPath $resolvedCoveragePath)) {
    throw "Coverage report was not found: $resolvedCoveragePath"
}

[xml]$coverage = Get-Content -LiteralPath $resolvedCoveragePath
$package = @($coverage.coverage.packages.package) |
    Where-Object { [string]$_.name -eq $Module } |
    Select-Object -First 1
if (-not $package) {
    $available = @($coverage.coverage.packages.package.name) -join ", "
    throw "Coverage module '$Module' was not found. Available modules: $available"
}

$files = @(
    foreach ($class in @($package.classes.class)) {
        $lines = @($class.lines.line)
        $covered = @($lines | Where-Object { [int]$_.hits -gt 0 }).Count
        [pscustomobject]@{
            File = [System.IO.Path]::GetRelativePath(
                $repositoryRoot,
                [string]$class.filename
            ).Replace("\", "/")
            Covered = $covered
            Coverable = $lines.Count
            Missing = $lines.Count - $covered
            Percent = if ($lines.Count -eq 0) { 100.0 } else { 100.0 * $covered / $lines.Count }
        }
    }
)

$coveredTotal = ($files | Measure-Object Covered -Sum).Sum
$coverableTotal = ($files | Measure-Object Coverable -Sum).Sum
$percentTotal = if ($coverableTotal -eq 0) { 100.0 } else { 100.0 * $coveredTotal / $coverableTotal }

Write-Output ("{0}: {1}/{2} lines ({3:N1}%)" -f
    $Module, $coveredTotal, $coverableTotal, $percentTotal)
Write-Output ""
Write-Output "Largest remaining gaps:"
$files |
    Where-Object { $_.Missing -gt 0 } |
    Sort-Object @{ Expression = "Missing"; Descending = $true }, File |
    Select-Object -First $Limit |
    Format-Table File, Covered, Coverable, Missing, @{Label = "Coverage"; Expression = { "{0:N1}%" -f $_.Percent } } -AutoSize
