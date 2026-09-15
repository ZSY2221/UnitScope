param(
    [string]$Python = 'python',
    [Parameter(Mandatory=$true)]
    [string]$OfficialRepo,
    [string]$Output = '',
    [int[]]$Seeds = @(0, 1, 2, 3, 4),
    [int]$Rounds = 50,
    [int]$LocalEpochs = 1,
    [double]$Sigma = 5.0
)

$ErrorActionPreference = 'Stop'
$releaseRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$repo = (Resolve-Path -LiteralPath $OfficialRepo).Path
$out = if ($Output) { $Output } else { Join-Path $releaseRoot 'reproduced\uldp_fl_native_v1' }
$logDir = Join-Path $out 'logs'
New-Item -ItemType Directory -Force -Path $out, $logDir | Out-Null

$methods = @('ULDP-AVG', 'ULDP-AVG-w')
$rows = [System.Collections.Generic.List[object]]::new()
$manifest = [ordered]@{
    paper = 'ULDP-FL, PVLDB 2024'
    repository = 'https://github.com/FumiyukiKato/uldp-fl'
    claimed_commit = 'd820e24f5c3cc7275a2b1e78111a46c0da1e8ebf'
    dataset = 'MNIST'
    native_semantics = 'across-silo user-level DP with complete user IDs'
    seeds = $Seeds
    rounds = $Rounds
    local_epochs = $LocalEpochs
    sigma = $Sigma
    n_users = 100
    n_silos = 5
    sampling_rate_q = 0.5
    methods = $methods
    execution = 'official repository simulator, native MNIST configuration'
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $out 'manifest.json') -Encoding UTF8

Push-Location $repo
foreach ($seed in $Seeds) {
    foreach ($method in $methods) {
        $safeMethod = $method.ToLower().Replace('-', '_')
        $base = "seed${seed}_${safeMethod}"
        $logPath = Join-Path $logDir "$base.log"
        $rawPath = Join-Path $out "$base.json"
        if (Test-Path -LiteralPath $rawPath) {
            Write-Host "SKIP $base"
            continue
        }

        $args = @(
            '.\src\run_simulation.py',
            '--dataset_name=mnist', '--verbose=1', '--gpu_id=0',
            "--agg_strategy=$method", '--n_users=100', '--n_silos=5', '--n_silo_per_round=5',
            '--global_learning_rate=10.0', '--clipping_bound=1.0', "--sigma=$Sigma",
            "--n_total_round=$Rounds", '--local_learning_rate=0.01', "--local_epochs=$LocalEpochs",
            '--user_dist=uniform-iid', '--silo_dist=uniform', '--user_alpha=0.3', '--silo_alpha=1.5',
            "--times=1", "--seed=$seed", "--version=9010", '--n_labels=2', '--sampling_rate_q=0.5'
        )
        Write-Host "RUN $base"
        & $Python @args 2>&1 | Tee-Object -FilePath $logPath
        if ($LASTEXITCODE -ne 0) {
            throw "ULDP-FL failed for $base with exit code $LASTEXITCODE"
        }
        $saved = Select-String -LiteralPath $logPath -Pattern 'Results saved to (.+)$' | Select-Object -Last 1
        if ($null -eq $saved) {
            throw "Could not locate official result path for $base"
        }
        $source = $saved.Matches[0].Groups[1].Value.Trim()
        if (-not (Test-Path -LiteralPath $source)) {
            throw "Official result path does not exist for ${base}: $source"
        }
        Copy-Item -LiteralPath $source -Destination $rawPath -Force
        $rows.Add([ordered]@{
            seed = $seed
            method = $method
            raw_result = $rawPath
            official_result = $source
            log = $logPath
            rounds = $Rounds
            local_epochs = $LocalEpochs
            sigma = $Sigma
        })
        $rows | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $out 'runs.json') -Encoding UTF8
    }
}
Pop-Location
Write-Host "ULDP-FL native reproduction complete: $out"
