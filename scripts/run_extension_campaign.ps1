[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectRoot,
    [Parameter(Mandatory=$true)]
    [ValidateSet('task26', 'task27', 'task28')]
    [string]$Task,
    [string]$PeriodicKFreeze,
    [switch]$PreviewOnly,
    [switch]$DryRunOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot 'implementation\.envs\coop-pinned\Scripts\python.exe'
$Config = Join-Path $Repo 'configs\sample_fg\extension.yaml'
$Campaign = Join-Path $Repo 'configs\sample_fg\extension_campaign.yaml'
$DataRoot = Join-Path $ProjectRoot 'data'
$ManifestRoot = Join-Path $ProjectRoot 'provenance\task2_data_manifests'
$ClipCache = Join-Path $ProjectRoot 'implementation\.cache\clip'
$OutputRoot = Join-Path $ProjectRoot 'runs\estimator_extension'
$R2OutputRoot = Join-Path $ProjectRoot 'runs\paper_reproduction'

foreach ($RequiredPath in @(
    $Repo, $Python, $Config, $Campaign, $DataRoot, $ManifestRoot, $ClipCache
)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required extension path is missing: $RequiredPath"
    }
}

if ($Task -eq 'task28') {
    if (-not $PeriodicKFreeze) {
        throw 'Task 28 requires -PeriodicKFreeze.'
    }
    $PeriodicKFreeze = (Resolve-Path -LiteralPath $PeriodicKFreeze).Path
    $Freeze = Get-Content -LiteralPath $PeriodicKFreeze -Raw | ConvertFrom-Json
    if ($Freeze.schema_version -ne 'sample_fg.periodic_k_freeze.v1') {
        throw 'Unsupported periodic-K freeze schema.'
    }
    $FrozenK = @($Freeze.selected_k_values | ForEach-Object { [int]$_ })
    if ($FrozenK.Count -lt 1 -or $FrozenK.Count -gt 2) {
        throw 'Task 28 freeze must retain one or two K values.'
    }
}
elseif ($PeriodicKFreeze) {
    throw '-PeriodicKFreeze is accepted only for Task 28.'
}

$Cells = @()
if ($Task -eq 'task26') {
    $Cells += [pscustomobject]@{ Dataset='dtd'; Seed=1; Estimator='ema';      K=$null }
    $Cells += [pscustomobject]@{ Dataset='dtd'; Seed=1; Estimator='exact';    K=$null }
    $Cells += [pscustomobject]@{ Dataset='dtd'; Seed=1; Estimator='periodic'; K=4 }
}
elseif ($Task -eq 'task27') {
    foreach ($K in @(2, 4, 8, 16)) {
        $Cells += [pscustomobject]@{ Dataset='dtd'; Seed=1; Estimator='periodic'; K=$K }
    }
}
else {
    foreach ($Dataset in @('dtd', 'eurosat')) {
        foreach ($Seed in @(1, 2, 3)) {
            $Cells += [pscustomobject]@{ Dataset=$Dataset; Seed=$Seed; Estimator='ema'; K=$null }
            $Cells += [pscustomobject]@{ Dataset=$Dataset; Seed=$Seed; Estimator='exact'; K=$null }
            foreach ($K in $FrozenK) {
                $Cells += [pscustomobject]@{ Dataset=$Dataset; Seed=$Seed; Estimator='periodic'; K=$K }
            }
        }
    }
}

function Get-EstimatorTag {
    param([Parameter(Mandatory=$true)]$Cell)
    if ($Cell.Estimator -eq 'periodic') {
        return "periodic-k$($Cell.K)"
    }
    return $Cell.Estimator
}

function Get-CompletedRuns {
    param([Parameter(Mandatory=$true)]$Cell)
    $EstimatorTag = Get-EstimatorTag -Cell $Cell
    $CellRoot = Join-Path $OutputRoot (
        '{0}\shots_16\sample\{1}\seed_{2}' -f
        $Cell.Dataset, $EstimatorTag, $Cell.Seed
    )
    if (-not (Test-Path -LiteralPath $CellRoot)) {
        return
    }
    foreach ($SummaryFile in Get-ChildItem -LiteralPath $CellRoot -Recurse -Filter 'summary.json' -File) {
        $Summary = Get-Content -LiteralPath $SummaryFile.FullName -Raw | ConvertFrom-Json
        $Identity = $Summary.run_identity
        if (
            $Summary.status -eq 'completed' -and
            $Summary.smoke -eq $false -and
            $Summary.allow_scientific_summary -eq $true -and
            $Identity.experiment_id -eq (@{task26='E0'; task27='E1'; task28='E2'}[$Task]) -and
            $Identity.dataset -eq $Cell.Dataset -and
            [int]$Identity.shots -eq 16 -and
            [int]$Identity.seed -eq [int]$Cell.Seed -and
            $Identity.method_tag -eq 'sample' -and
            $Identity.estimator_tag -eq $EstimatorTag
        ) {
            [pscustomobject]@{
                Path = $SummaryFile.FullName
                RunId = [string]$Identity.run_id
                ConfigHash = [string]$Identity.config_sha256
            }
        }
    }
}

function Get-ReusableR2EMA {
    param([Parameter(Mandatory=$true)]$Cell)
    $CellRoot = Join-Path $R2OutputRoot (
        '{0}\shots_16\sample\ema\seed_{1}' -f $Cell.Dataset, $Cell.Seed
    )
    if (-not (Test-Path -LiteralPath $CellRoot)) {
        return
    }
    foreach ($SummaryFile in Get-ChildItem -LiteralPath $CellRoot -Recurse -Filter 'summary.json' -File) {
        $Summary = Get-Content -LiteralPath $SummaryFile.FullName -Raw | ConvertFrom-Json
        $Identity = $Summary.run_identity
        if (
            $Summary.status -eq 'completed' -and
            $Summary.smoke -eq $false -and
            $Summary.allow_scientific_summary -eq $true -and
            $Identity.experiment_id -eq 'R2' -and
            $Identity.dataset -eq $Cell.Dataset -and
            [int]$Identity.shots -eq 16 -and
            [int]$Identity.seed -eq [int]$Cell.Seed -and
            $Identity.method_tag -eq 'sample' -and
            $Identity.estimator_tag -eq 'ema'
        ) {
            [pscustomobject]@{
                Path = $SummaryFile.FullName
                RunId = [string]$Identity.run_id
                ConfigHash = [string]$Identity.config_sha256
            }
        }
    }
}

$Mutex = [System.Threading.Mutex]::new($false, 'SAMPLE_Extension_Campaign_Launcher')
$LockTaken = $false
try {
    try {
        $LockTaken = $Mutex.WaitOne(0)
    }
    catch [System.Threading.AbandonedMutexException] {
        $LockTaken = $true
    }
    if (-not $LockTaken) {
        throw 'Another extension launcher is already active.'
    }
    $ActiveRuns = @(
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like '*train_sample_fg*' }
    )
    if ($ActiveRuns.Count -gt 0) {
        throw "Training is already active (Python PIDs: $(($ActiveRuns.ProcessId | Sort-Object) -join ', '))."
    }

    Set-Location -LiteralPath $Repo
    foreach ($Cell in $Cells) {
        $Label = '{0} dataset={1} seed={2} estimator={3} K={4}' -f `
            $Task, $Cell.Dataset, $Cell.Seed, $Cell.Estimator, $Cell.K
        if ($Task -eq 'task28' -and $Cell.Estimator -eq 'ema') {
            $Reusable = @(Get-ReusableR2EMA -Cell $Cell)
            if ($Reusable.Count -eq 0) {
                throw "Task 28 requires a completed matching R2 EMA artifact: $Label"
            }
            $Hashes = @($Reusable.ConfigHash | Sort-Object -Unique)
            if ($Hashes.Count -ne 1) {
                throw "Reusable R2 EMA attempts have conflicting configs: $Label"
            }
            Write-Host "REUSE immutable R2 EMA: $Label run_id=$($Reusable[0].RunId)"
            continue
        }
        $Completed = @(Get-CompletedRuns -Cell $Cell)
        if ($Completed.Count -gt 0) {
            $Hashes = @($Completed.ConfigHash | Sort-Object -Unique)
            if ($Hashes.Count -ne 1) {
                throw "Completed attempts have conflicting configs: $Label"
            }
            Write-Host "SKIP completed: $Label"
            continue
        }
        if ($PreviewOnly) {
            Write-Host "WOULD RUN: $Label"
            continue
        }
        $RunArguments = @(
            '--task', $Task,
            '--dataset', $Cell.Dataset,
            '--shots', '16',
            '--seed', [string]$Cell.Seed,
            '--method', 'sample',
            '--estimator', $Cell.Estimator,
            '--data-root', $DataRoot,
            '--manifest-root', $ManifestRoot,
            '--clip-cache', $ClipCache,
            '--output-root', $OutputRoot,
            '--config', $Config,
            '--campaign-config', $Campaign,
            '--recovery-interval-epochs', '10'
        )
        if ($null -ne $Cell.K) {
            $RunArguments += @('--periodic-k', [string]$Cell.K)
        }
        if ($Task -eq 'task28') {
            $RunArguments += @('--periodic-k-freeze', $PeriodicKFreeze)
        }
        if ($DryRunOnly) {
            $RunArguments += '--dry-run'
        }
        Write-Host "STARTING: $Label"
        & $Python '.\train_sample_fg_extension.py' @RunArguments
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
        if (-not $DryRunOnly -and @(Get-CompletedRuns -Cell $Cell).Count -eq 0) {
            throw "Experiment exited without a completed summary: $Label"
        }
    }
}
finally {
    if ($LockTaken) {
        $Mutex.ReleaseMutex()
    }
    $Mutex.Dispose()
}
