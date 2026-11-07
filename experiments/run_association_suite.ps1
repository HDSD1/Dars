param(
    [string]$Python = 'D:\PythonEnvs\uav_mec_gpu\python.exe',
    [string]$ReferenceManifest = '',
    [string]$Tag = ('association_' + (Get-Date -Format 'yyyyMMdd_HHmmss')),
    [int]$TimingRepeats = 3,
    [int]$SwapEvaluations = 1000,
    [int]$SwapRounds = 20
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $ReferenceManifest) {
    $ReferenceManifest = Join-Path $repoRoot 'results/reviewer_direct_objective_base_v3_seed200000_200099_20261030/manifest.json'
}
$ReferenceManifest = (Resolve-Path -LiteralPath $ReferenceManifest).Path
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
$destination = Join-Path (Join-Path $repoRoot 'results') $Tag
if (Test-Path -LiteralPath $destination) { throw "Output already exists: $destination" }
New-Item -ItemType Directory -Path $destination | Out-Null
$scriptPath = Join-Path $PSScriptRoot 'compare_association.py'
$cases = @(
    @{Name='base'; Episodes=100; Seed=310000},
    @{Name='high_load'; Episodes=50; Seed=320000},
    @{Name='unseen_order'; Episodes=50; Seed=330000}
)
foreach ($case in $cases) {
    $outputPath = Join-Path $destination $case.Name
    $logPath = Join-Path $destination ($case.Name + '.log')
    Write-Host "Starting $($case.Name): $($case.Episodes) episodes; log: $logPath"
    & $Python -u $scriptPath --reference-manifest $ReferenceManifest `
        --scenario $case.Name --episodes $case.Episodes --seed-start $case.Seed `
        --timing-repeats $TimingRepeats --swap-evaluations $SwapEvaluations `
        --swap-rounds $SwapRounds --output-dir $outputPath 2>&1 | Tee-Object -FilePath $logPath
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $($case.Name). See $logPath" }
}
Write-Host "All association evaluations completed: $destination"
