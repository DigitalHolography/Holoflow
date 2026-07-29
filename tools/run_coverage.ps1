param(
    [string]$TestPreset = "test-Coverage",
    [string]$OutputDirectory = "build/coverage"
)

$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$outputPath = Join-Path $repositoryRoot $OutputDirectory
$settingsPath = Join-Path $repositoryRoot "test/coverage.runsettings"
$rawCoberturaPath = Join-Path $outputPath "coverage.raw.cobertura.xml"
$coberturaPath = Join-Path $outputPath "coverage.cobertura.xml"
$junitPath = Join-Path $outputPath "test-results.junit.xml"
$htmlPath = Join-Path $outputPath "html"
$productionModules = @(
    "curaii",
    "holofile",
    "holoflow",
    "holoflow_event",
    "holonp",
    "holotask",
    "holovibes"
)

New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

$collector = Get-Command "Microsoft.CodeCoverage.Console" -ErrorAction SilentlyContinue
if (-not $collector) {
    throw "Microsoft.CodeCoverage.Console was not found. Install the Visual Studio Code Coverage component."
}

$dotnet = Get-Command "dotnet" -ErrorAction SilentlyContinue
if (-not $dotnet) {
    throw "dotnet was not found. Install the .NET SDK used by the pinned ReportGenerator tool."
}

function ConvertTo-ModuleCoverage {
    param(
        [Parameter(Mandatory)]
        [xml]$Coverage,

        [Parameter(Mandatory)]
        [string]$Destination
    )

    $moduleExpression = $productionModules -join "|"
    $sourceExpression = "^src[\\/](?<module>$moduleExpression)[\\/](?<relative>.+)$"
    $sourceRows = @(
        foreach ($class in $Coverage.coverage.packages.package.classes.class) {
            $filename = [System.IO.Path]::GetFullPath([string]$class.filename)
            $relativeToRepository = [System.IO.Path]::GetRelativePath(
                $repositoryRoot,
                $filename
            )

            if ($relativeToRepository -notmatch $sourceExpression) {
                continue
            }

            $module = $Matches.module
            $relativeToModule = $Matches.relative.Replace("\", "/")
            foreach ($line in @($class.lines.line)) {
                [pscustomobject]@{
                    Module = $module
                    Relative = $relativeToModule
                    Filename = $filename
                    Number = [int]$line.number
                    Hits = [int]$line.hits
                }
            }
        }
    )

    $reportedModules = @($sourceRows.Module | Sort-Object -Unique)
    $missingModules = @($productionModules | Where-Object { $_ -notin $reportedModules })
    if ($missingModules.Count -ne 0) {
        throw "Coverage report is missing production modules: $($missingModules -join ', ')."
    }

    $document = [System.Xml.XmlDocument]::new()
    $declaration = $document.CreateXmlDeclaration("1.0", "utf-8", $null)
    $document.AppendChild($declaration) | Out-Null

    $coverageElement = $document.CreateElement("coverage")
    $coverageElement.SetAttribute("branch-rate", "0")
    $coverageElement.SetAttribute("complexity", "0")
    $coverageElement.SetAttribute("version", "holoflow-module-union")
    $coverageElement.SetAttribute("timestamp", [string]$Coverage.coverage.timestamp)
    $document.AppendChild($coverageElement) | Out-Null

    $packagesElement = $document.CreateElement("packages")
    $coverageElement.AppendChild($packagesElement) | Out-Null

    $totalCovered = 0
    $totalValid = 0

    foreach ($moduleGroup in $sourceRows | Group-Object Module | Sort-Object Name) {
        $packageElement = $document.CreateElement("package")
        $packageElement.SetAttribute("name", $moduleGroup.Name)
        $packageElement.SetAttribute("branch-rate", "0")
        $packageElement.SetAttribute("complexity", "0")

        $classesElement = $document.CreateElement("classes")
        $packageCovered = 0
        $packageValid = 0

        foreach ($fileGroup in $moduleGroup.Group | Group-Object Filename | Sort-Object Name) {
            $representative = $fileGroup.Group | Select-Object -First 1
            $mergedLines = @(
                $fileGroup.Group |
                    Group-Object Number |
                    ForEach-Object {
                        [pscustomobject]@{
                            Number = [int]$_.Name
                            Hits = [int](($_.Group | Measure-Object Hits -Maximum).Maximum)
                        }
                    } |
                    Sort-Object Number
            )

            $fileCovered = @($mergedLines | Where-Object { $_.Hits -gt 0 }).Count
            $fileValid = $mergedLines.Count
            $packageCovered += $fileCovered
            $packageValid += $fileValid

            $classElement = $document.CreateElement("class")
            $classElement.SetAttribute("name", $representative.Relative)
            $classElement.SetAttribute("filename", $representative.Filename)
            $classElement.SetAttribute("branch-rate", "0")
            $classElement.SetAttribute("complexity", "0")
            $classElement.SetAttribute(
                "line-rate",
                [string]::Format(
                    [Globalization.CultureInfo]::InvariantCulture,
                    "{0:R}",
                    $(if ($fileValid -eq 0) { 0.0 } else { $fileCovered / $fileValid })
                )
            )

            $classElement.AppendChild($document.CreateElement("methods")) | Out-Null
            $linesElement = $document.CreateElement("lines")
            foreach ($mergedLine in $mergedLines) {
                $lineElement = $document.CreateElement("line")
                $lineElement.SetAttribute("number", [string]$mergedLine.Number)
                $lineElement.SetAttribute("hits", [string]$mergedLine.Hits)
                $lineElement.SetAttribute("branch", "False")
                $linesElement.AppendChild($lineElement) | Out-Null
            }
            $classElement.AppendChild($linesElement) | Out-Null
            $classesElement.AppendChild($classElement) | Out-Null
        }

        $packageElement.SetAttribute(
            "line-rate",
            [string]::Format(
                [Globalization.CultureInfo]::InvariantCulture,
                "{0:R}",
                $(if ($packageValid -eq 0) { 0.0 } else { $packageCovered / $packageValid })
            )
        )
        $packageElement.AppendChild($classesElement) | Out-Null
        $packagesElement.AppendChild($packageElement) | Out-Null

        $totalCovered += $packageCovered
        $totalValid += $packageValid
    }

    $coverageElement.SetAttribute("lines-covered", [string]$totalCovered)
    $coverageElement.SetAttribute("lines-valid", [string]$totalValid)
    $coverageElement.SetAttribute(
        "line-rate",
        [string]::Format(
            [Globalization.CultureInfo]::InvariantCulture,
            "{0:R}",
            $(if ($totalValid -eq 0) { 0.0 } else { $totalCovered / $totalValid })
        )
    )

    $document.Save($Destination)
}

Push-Location $repositoryRoot
try {
    & $collector.Source collect `
        --settings $settingsPath `
        --output $rawCoberturaPath `
        --output-format cobertura `
        -- `
        ctest --preset $TestPreset --output-junit $junitPath
    if ($LASTEXITCODE -ne 0) {
        throw "Coverage test run failed with exit code $LASTEXITCODE."
    }

    if (-not (Test-Path -LiteralPath $rawCoberturaPath) -or
        (Get-Item -LiteralPath $rawCoberturaPath).Length -eq 0) {
        throw "Coverage collection did not produce a non-empty Cobertura report."
    }

    [xml]$rawCoverage = Get-Content -LiteralPath $rawCoberturaPath
    ConvertTo-ModuleCoverage -Coverage $rawCoverage -Destination $coberturaPath

    & $dotnet.Source tool restore
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to restore the pinned ReportGenerator tool."
    }

    & $dotnet.Source tool run reportgenerator `
        "-reports:$coberturaPath" `
        "-targetdir:$htmlPath" `
        "-reporttypes:Html;MarkdownSummaryGithub"
    if ($LASTEXITCODE -ne 0) {
        throw "ReportGenerator failed with exit code $LASTEXITCODE."
    }

    $summary = Get-ChildItem -LiteralPath $htmlPath -Filter "*.md" | Select-Object -First 1
    if ($env:GITHUB_STEP_SUMMARY -and $summary) {
        Get-Content -LiteralPath $summary.FullName | Add-Content -LiteralPath $env:GITHUB_STEP_SUMMARY
    }

    Write-Output "Raw Cobertura report: $rawCoberturaPath"
    Write-Output "Module Cobertura report: $coberturaPath"
    Write-Output "HTML report: $htmlPath"
}
finally {
    Pop-Location
}
