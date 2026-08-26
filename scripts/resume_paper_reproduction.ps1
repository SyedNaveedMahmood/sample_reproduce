[CmdletBinding()]
param(
    [string]$ProjectRoot = 'E:\SAMPLE\sample_full_gradient_spec_docs',
    [switch]$PreviewOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Repo = Join-Path $ProjectRoot 'implementation\CoOp'
$Python = Join-Path $ProjectRoot 'implementation\.envs\coop-pinned\Scripts\python.exe'
$Config = Join-Path $Repo 'configs\sample_fg\paper_reproduction.yaml'
$DataRoot = Join-Path $ProjectRoot 'data'
$ManifestRoot = Join-Path $ProjectRoot 'provenance\task2_data_manifests'
$ClipCache = Join-Path $ProjectRoot 'implementation\.cache\clip'
$OutputRoot = Join-Path $ProjectRoot 'runs\paper_reproduction'
$AnalysisRoot = Join-Path $ProjectRoot 'analysis_output\paper_reproduction'

foreach ($RequiredPath in @(
    $Repo,
    $Python,
    $Config,
    $DataRoot,
    $ManifestRoot,
    $ClipCache,
    $OutputRoot
)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required paper-reproduction path is missing: $RequiredPath"
    }
}

$Cells = @(
    [pscustomobject]@{ Dataset='dtd';     Seed=1; Method='coop';   Estimator='none' }
    [pscustomobject]@{ Dataset='dtd';     Seed=1; Method='sam';    Estimator='none' }
    [pscustomobject]@{ Dataset='dtd';     Seed=1; Method='sample'; Estimator='ema'  }
    [pscustomobject]@{ Dataset='dtd';     Seed=2; Method='coop';   Estimator='none' }
    [pscustomobject]@{ Dataset='dtd';     Seed=2; Method='sam';    Estimator='none' }
    [pscustomobject]@{ Dataset='dtd';     Seed=2; Method='sample'; Estimator='ema'  }
    [pscustomobject]@{ Dataset='dtd';     Seed=3; Method='coop';   Estimator='none' }
    [pscustomobject]@{ Dataset='dtd';     Seed=3; Method='sam';    Estimator='none' }
    [pscustomobject]@{ Dataset='dtd';     Seed=3; Method='sample'; Estimator='ema'  }
    [pscustomobject]@{ Dataset='eurosat'; Seed=1; Method='coop';   Estimator='none' }
    [pscustomobject]@{ Dataset='eurosat'; Seed=1; Method='sam';    Estimator='none' }
    [pscustomobject]@{ Dataset='eurosat'; Seed=1; Method='sample'; Estimator='ema'  }
    [pscustomobject]@{ Dataset='eurosat'; Seed=2; Method='coop';   Estimator='none' }
    [pscustomobject]@{ Dataset='eurosat'; Seed=2; Method='sam';    Estimator='none' }
    [pscustomobject]@{ Dataset='eurosat'; Seed=2; Method='sample'; Estimator='ema'  }
    [pscustomobject]@{ Dataset='eurosat'; Seed=3; Method='coop';   Estimator='none' }
    [pscustomobject]@{ Dataset='eurosat'; Seed=3; Method='sam';    Estimator='none' }
    [pscustomobject]@{ Dataset='eurosat'; Seed=3; Method='sample'; Estimator='ema'  }
)

function Get-CompletedRuns {
    param(
        [Parameter(Mandatory=$true)]$Cell
    )

    $CellRoot = Join-Path $OutputRoot (
        '{0}\shots_16\{1}\{2}\seed_{3}' -f
        $Cell.Dataset,
        $Cell.Method,
        $Cell.Estimator,
        $Cell.Seed
    )
    if (-not (Test-Path -LiteralPath $CellRoot)) {
        return
    }

    foreach ($SummaryFile in Get-ChildItem -LiteralPath $CellRoot -Recurse -Filter 'summary.json' -File) {
        try {
            $Summary = Get-Content -LiteralPath $SummaryFile.FullName -Raw | ConvertFrom-Json
        }
        catch {
            throw "Malformed summary file: $($SummaryFile.FullName)"
        }
        $Identity = $Summary.run_identity
        if (
            $Summary.status -eq 'completed' -and
            $Summary.smoke -eq $false -and
            $Summary.allow_scientific_summary -eq $true -and
            $Identity.experiment_id -eq 'R2' -and
            $Identity.dataset -eq $Cell.Dataset -and
            [int]$Identity.shots -eq 16 -and
            [int]$Identity.seed -eq [int]$Cell.Seed -and
            $Identity.method_tag -eq $Cell.Method -and
            $Identity.estimator_tag -eq $Cell.Estimator
        ) {
            [pscustomobject]@{
                Path = $SummaryFile.FullName
                RunId = [string]$Identity.run_id
                ConfigHash = [string]$Identity.config_sha256
            }
        }
    }
}

$Mutex = [System.Threading.Mutex]::new(
    $false,
    'SAMPLE_R2_Paper_Reproduction_Launcher'
)
$LockTaken = $false
try {
    try {
        $LockTaken = $Mutex.WaitOne(0)
    }
    catch [System.Threading.AbandonedMutexException] {
        $LockTaken = $true
    }
    if (-not $LockTaken) {
        throw 'Another paper-reproduction launcher is already active.'
    }

    $ActiveRuns = @(
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like '*train_sample_fg.py*' }
    )
    if ($ActiveRuns.Count -gt 0) {
        $ProcessIds = ($ActiveRuns.ProcessId | Sort-Object) -join ', '
        throw "Training is already active (Python PIDs: $ProcessIds)."
    }

    Set-Location -LiteralPath $Repo
    $PendingCount = 0
    foreach ($Cell in $Cells) {
        $Label = '{0} seed={1} method={2} estimator={3}' -f
            $Cell.Dataset,
            $Cell.Seed,
            $Cell.Method,
            $Cell.Estimator
        $Completed = @(Get-CompletedRuns -Cell $Cell)
        if ($Completed.Count -gt 0) {
            $Hashes = @($Completed.ConfigHash | Sort-Object -Unique)
            if ($Hashes.Count -ne 1) {
                throw "Completed attempts have conflicting configs: $Label"
            }
            Write-Host "SKIP completed: $Label ($($Completed.Count) attempt(s))"
            continue
        }

        $PendingCount += 1
        if ($PreviewOnly) {
            Write-Host "WOULD RUN: $Label"
            continue
        }

        Write-Host "STARTING: $Label"
        $RunArguments = @(
            '--config', $Config,
            '--experiment-id', 'R2',
            '--dataset', $Cell.Dataset,
            '--shots', '16',
            '--seed', [string]$Cell.Seed,
            '--method', $Cell.Method,
            '--estimator', $Cell.Estimator,
            '--data-root', $DataRoot,
            '--manifest-root', $ManifestRoot,
            '--clip-cache', $ClipCache,
            '--output-root', $OutputRoot,
            '--recovery-interval-epochs', '10'
        )
        & $Python '.\train_sample_fg.py' @RunArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Experiment failed: $Label, exit code $LASTEXITCODE"
        }
        $Completed = @(Get-CompletedRuns -Cell $Cell)
        if ($Completed.Count -eq 0) {
            throw "Experiment exited successfully without a completed summary: $Label"
        }
        Write-Host "COMPLETED: $Label"
    }

    if ($PreviewOnly) {
        Write-Host "PREVIEW COMPLETE: $PendingCount of $($Cells.Count) cells remain."
        return
    }

    $AggregateDir = Join-Path $AnalysisRoot 'aggregate'
    $TablesDir = Join-Path $AnalysisRoot 'tables'
    $PlotsDir = Join-Path $AnalysisRoot 'plots'

    & $Python '.\analysis\aggregate_results.py' `
        --input-root $OutputRoot `
        --output-dir $AggregateDir `
        --mode scientific `
        --allow-invalid
    if ($LASTEXITCODE -ne 0) {
        throw "Aggregation failed: exit code $LASTEXITCODE"
    }

    & $Python '.\analysis\make_tables.py' `
        --input-dir $AggregateDir `
        --output-dir $TablesDir `
        --mode scientific
    if ($LASTEXITCODE -ne 0) {
        throw "Table generation failed: exit code $LASTEXITCODE"
    }

    & $Python '.\analysis\plot_diagnostics.py' `
        --input-dir $AggregateDir `
        --output-dir $PlotsDir `
        --mode scientific
    if ($LASTEXITCODE -ne 0) {
        throw "Plot generation failed: exit code $LASTEXITCODE"
    }

    Write-Host 'FULL PAPER REPRODUCTION COMPLETED'
    Write-Host "Runs:   $OutputRoot"
    Write-Host "Tables: $TablesDir"
    Write-Host "Plots:  $PlotsDir"
}
finally {
    if ($LockTaken) {
        $Mutex.ReleaseMutex()
    }
    $Mutex.Dispose()
}
