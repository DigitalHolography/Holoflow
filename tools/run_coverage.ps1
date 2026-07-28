param(
    [string]$TestPreset = "test-Coverage",
    [string]$OutputDirectory = "build/coverage"
)

$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$outputPath = Join-Path $repositoryRoot $OutputDirectory
$settingsPath = Join-Path $repositoryRoot "test/coverage.runsettings"
$coberturaPath = Join-Path $outputPath "coverage.cobertura.xml"
$junitPath = Join-Path $outputPath "test-results.junit.xml"
$htmlPath = Join-Path $outputPath "html"

New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

$collector = Get-Command "Microsoft.CodeCoverage.Console" -ErrorAction SilentlyContinue
if (-not $collector) {
    throw "Microsoft.CodeCoverage.Console was not found. Install the Visual Studio Code Coverage component."
}

$dotnet = Get-Command "dotnet" -ErrorAction SilentlyContinue
if (-not $dotnet) {
    throw "dotnet was not found. Install the .NET SDK used by the pinned ReportGenerator tool."
}

Push-Location $repositoryRoot
try {
    & $collector.Source collect `
        --settings $settingsPath `
        --output $coberturaPath `
        --output-format cobertura `
        -- `
        ctest --preset $TestPreset --output-junit $junitPath
    if ($LASTEXITCODE -ne 0) {
        throw "Coverage test run failed with exit code $LASTEXITCODE."
    }

    if (-not (Test-Path -LiteralPath $coberturaPath) -or
        (Get-Item -LiteralPath $coberturaPath).Length -eq 0) {
        throw "Coverage collection did not produce a non-empty Cobertura report."
    }

    [xml]$coverage = Get-Content -LiteralPath $coberturaPath
    $productionClasses = @(
        $coverage.coverage.packages.package.classes.class |
            Where-Object { $_.filename -match '[\\/]src[\\/](curaii|holofile|holoflow|holoflow_event|holonp|holotask|holovibes)[\\/]' }
    )
    if ($productionClasses.Count -eq 0) {
        throw "Coverage report contains no Holoflow production source files."
    }

    & $dotnet.Source tool restore
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to restore the pinned ReportGenerator tool."
    }

    & $dotnet.Source tool run reportgenerator `
        "-reports:$coberturaPath" `
        "-targetdir:$htmlPath" `
        "-reporttypes:Html;MarkdownSummaryGithub" `
        "-filefilters:-*/test/*;-*/.fc/*"
    if ($LASTEXITCODE -ne 0) {
        throw "ReportGenerator failed with exit code $LASTEXITCODE."
    }

    $summary = Get-ChildItem -LiteralPath $htmlPath -Filter "*.md" | Select-Object -First 1
    if ($env:GITHUB_STEP_SUMMARY -and $summary) {
        Get-Content -LiteralPath $summary.FullName | Add-Content -LiteralPath $env:GITHUB_STEP_SUMMARY
    }

    Write-Output "Cobertura report: $coberturaPath"
    Write-Output "HTML report: $htmlPath"
}
finally {
    Pop-Location
}
