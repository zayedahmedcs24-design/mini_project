# PowerShell runner for training and evaluating U-Net and DeepLabV3+ across seeds 0, 1, 2
# Execution continues sequentially overnight; does not halt on single failures.

$ErrorActionPreference = "Continue"

# Resolve directories
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location -Path $ProjectRoot

# Python executable
$PythonExe = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

# Directories
$LogsDir = Join-Path $ProjectRoot "results\logs"
$CheckpointsDir = Join-Path $ProjectRoot "results\checkpoints"
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
New-Item -ItemType Directory -Force -Path $CheckpointsDir | Out-Null

$PipelineLog = Join-Path $LogsDir "pipeline_summary.log"
$Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"============================================================" | Out-File -FilePath $PipelineLog -Append -Encoding utf8
"PIPELINE RUN STARTED: $Timestamp" | Out-File -FilePath $PipelineLog -Append -Encoding utf8
"============================================================" | Out-File -FilePath $PipelineLog -Append -Encoding utf8

$Models = @("unet_resnet34", "deeplabv3p_resnet34")
$Seeds = @(0, 1, 2)
$Splits = @("valid", "test", "bolivia")

Write-Host "Starting sequential multi-seed pipeline across seeds: $($Seeds -join ', ')..." -ForegroundColor Cyan

foreach ($Model in $Models) {
    $ConfigFile = "configs\$Model.yaml"
    if (-not (Test-Path $ConfigFile)) {
        Write-Warning "Config file not found: $ConfigFile. Skipping..."
        "FAILED: Config file not found $ConfigFile" | Out-File -FilePath $PipelineLog -Append
        continue
    }

    foreach ($Seed in $Seeds) {
        $RunName = "${Model}_seed${Seed}"
        $RunLog = Join-Path $LogsDir "$RunName.log"
        $CheckpointPath = Join-Path $CheckpointsDir "${RunName}_best.pth"
        
        $StartTime = Get-Date
        Write-Host "`n[$($StartTime.ToString('HH:mm:ss'))] === STARTING $RunName ===" -ForegroundColor Green
        "=== STARTING RUN: $RunName at $($StartTime.ToString('yyyy-MM-dd HH:mm:ss')) ===" | Out-File -FilePath $RunLog -Encoding utf8
        
        # 1. Train with early stopping on validation flood IoU
        if (Test-Path $CheckpointPath) {
            Write-Host "Checkpoint already exists at $CheckpointPath; skipping training." -ForegroundColor Yellow
            "TRAIN SKIPPED: $RunName (Checkpoint already exists)" | Out-File -FilePath $PipelineLog -Append -Encoding utf8
        } else {
            Write-Host "Training $RunName (logging to $RunLog)..."
            try {
                & $PythonExe train.py --config $ConfigFile --seed $Seed *>> $RunLog
                $TrainExitCode = $LASTEXITCODE
                if ($TrainExitCode -ne 0) {
                    Write-Warning "Training $RunName exited with error code $TrainExitCode. Check $RunLog."
                    "TRAIN FAILED: $RunName (Exit code $TrainExitCode)" | Out-File -FilePath $PipelineLog -Append -Encoding utf8
                } else {
                    Write-Host "Training $RunName completed successfully." -ForegroundColor Green
                    "TRAIN SUCCESS: $RunName" | Out-File -FilePath $PipelineLog -Append -Encoding utf8
                }
            } catch {
                Write-Warning "Exception during training of ${RunName}: $_"
                "TRAIN EXCEPTION: ${RunName} - $_" | Out-File -FilePath $PipelineLog -Append -Encoding utf8
            }
        }

        # 2. Evaluate on valid, test, bolivia
        if (Test-Path $CheckpointPath) {
            foreach ($Split in $Splits) {
                $EvalResultJson = Join-Path $ProjectRoot "results\${RunName}_${Split}.json"
                if (Test-Path $EvalResultJson) {
                    Write-Host "Evaluation results already exist at $EvalResultJson; skipping." -ForegroundColor Yellow
                    continue
                }
                Write-Host "Evaluating $RunName on $Split split..."
                try {
                    & $PythonExe eval.py --model $RunName --config $ConfigFile --checkpoint $CheckpointPath --split $Split *>> $RunLog
                    $EvalExitCode = $LASTEXITCODE
                    if ($EvalExitCode -ne 0) {
                        Write-Warning "Eval $RunName on $Split failed with code $EvalExitCode."
                        "EVAL FAILED: $RunName on $Split (Exit code $EvalExitCode)" | Out-File -FilePath $PipelineLog -Append -Encoding utf8
                    } else {
                        "EVAL SUCCESS: $RunName on $Split" | Out-File -FilePath $PipelineLog -Append -Encoding utf8
                    }
                } catch {
                    Write-Warning "Exception evaluating ${RunName} on ${Split}: $_"
                    "EVAL EXCEPTION: ${RunName} on ${Split} - $_" | Out-File -FilePath $PipelineLog -Append -Encoding utf8
                }
            }
        } else {
            Write-Warning "Checkpoint not found at $CheckpointPath; skipping evaluation."
            "EVAL SKIPPED: Checkpoint missing for $RunName" | Out-File -FilePath $PipelineLog -Append
        }

        $EndTime = Get-Date
        $Duration = $EndTime - $StartTime
        "Finished $RunName in $($Duration.TotalMinutes.ToString('F1')) minutes." | Out-File -FilePath $RunLog -Append -Encoding utf8
        "COMPLETED $RunName in $($Duration.TotalMinutes.ToString('F1')) min" | Out-File -FilePath $PipelineLog -Append -Encoding utf8
        Write-Host "Finished $RunName in $($Duration.TotalMinutes.ToString('F1')) minutes." -ForegroundColor Cyan
    }
}

$EndTimestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Host "`nAll runs finished at $EndTimestamp. Generating final metrics summary table..." -ForegroundColor Green
"============================================================" | Out-File -FilePath $PipelineLog -Append -Encoding utf8
"ALL RUNS COMPLETED: $EndTimestamp" | Out-File -FilePath $PipelineLog -Append -Encoding utf8
"============================================================" | Out-File -FilePath $PipelineLog -Append -Encoding utf8

# 3. Generate summary table now that all runs are finished
try {
    & $PythonExe scripts/make_results_table.py *>> $PipelineLog
    Write-Host "Summary table created at results/metrics_summary_table.csv" -ForegroundColor Green
} catch {
    Write-Warning "Failed to generate summary table: $_"
}
