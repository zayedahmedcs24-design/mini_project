#!/usr/bin/env bash
# Shell runner for training and evaluating U-Net and DeepLabV3+ across seeds 0, 1, 2
# Execution continues sequentially overnight; does not halt on single failures.

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

# Python executable
if [ -f "$PROJECT_ROOT/venv/bin/python" ]; then
    PYTHON_EXE="$PROJECT_ROOT/venv/bin/python"
elif [ -f "$PROJECT_ROOT/venv/Scripts/python.exe" ]; then
    PYTHON_EXE="$PROJECT_ROOT/venv/Scripts/python.exe"
else
    PYTHON_EXE="python"
fi

LOGS_DIR="$PROJECT_ROOT/results/logs"
CHECKPOINTS_DIR="$PROJECT_ROOT/results/checkpoints"
mkdir -p "$LOGS_DIR" "$CHECKPOINTS_DIR"

PIPELINE_LOG="$LOGS_DIR/pipeline_summary.log"
echo "============================================================" >> "$PIPELINE_LOG"
echo "PIPELINE RUN STARTED: $(date '+%Y-%m-%d %H:%M:%S')" >> "$PIPELINE_LOG"
echo "============================================================" >> "$PIPELINE_LOG"

MODELS=("unet_resnet34" "deeplabv3p_resnet34")
SEEDS=(0 1 2)
SPLITS=("valid" "test" "bolivia")

echo "Starting sequential multi-seed pipeline across seeds: ${SEEDS[*]}..."

for MODEL in "${MODELS[@]}"; do
    CONFIG_FILE="configs/${MODEL}.yaml"
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "[WARN] Config file not found: $CONFIG_FILE. Skipping..." | tee -a "$PIPELINE_LOG"
        continue
    fi

    for SEED in "${SEEDS[@]}"; do
        RUN_NAME="${MODEL}_seed${SEED}"
        RUN_LOG="$LOGS_DIR/${RUN_NAME}.log"
        CHECKPOINT_PATH="$CHECKPOINTS_DIR/${RUN_NAME}_best.pth"

        START_TIME=$(date '+%s')
        echo -e "\n=== STARTING $RUN_NAME at $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$RUN_LOG"

        # 1. Train with early stopping on validation flood IoU
        echo "Training $RUN_NAME (logging to $RUN_LOG)..."
        if $PYTHON_EXE train.py --config "$CONFIG_FILE" --seed "$SEED" >> "$RUN_LOG" 2>&1; then
            echo "Training $RUN_NAME completed successfully." | tee -a "$PIPELINE_LOG"
        else
            EXIT_CODE=$?
            echo "[ERROR] Training $RUN_NAME failed with exit code $EXIT_CODE. Check $RUN_LOG." | tee -a "$PIPELINE_LOG"
        fi

        # 2. Evaluate on valid, test, bolivia
        if [ -f "$CHECKPOINT_PATH" ]; then
            for SPLIT in "${SPLITS[@]}"; do
                echo "Evaluating $RUN_NAME on $SPLIT split..."
                if $PYTHON_EXE eval.py --model "$RUN_NAME" --config "$CONFIG_FILE" --checkpoint "$CHECKPOINT_PATH" --split "$SPLIT" >> "$RUN_LOG" 2>&1; then
                    echo "EVAL SUCCESS: $RUN_NAME on $SPLIT" >> "$PIPELINE_LOG"
                else
                    E_CODE=$?
                    echo "[ERROR] Eval $RUN_NAME on $SPLIT failed with code $E_CODE." >> "$PIPELINE_LOG"
                fi
            done
        else
            echo "[WARN] Checkpoint not found at $CHECKPOINT_PATH; skipping evaluation." | tee -a "$PIPELINE_LOG"
        fi

        END_TIME=$(date '+%s')
        ELAPSED_SEC=$(( END_TIME - START_TIME ))
        ELAPSED_MIN=$(awk -v s="$ELAPSED_SEC" 'BEGIN { printf "%.1f", s / 60 }')
        echo "Completed $RUN_NAME in ${ELAPSED_MIN} minutes." | tee -a "$PIPELINE_LOG"
    done
done

echo -e "\nAll runs finished. Generating final metrics summary table..." | tee -a "$PIPELINE_LOG"
$PYTHON_EXE scripts/make_results_table.py >> "$PIPELINE_LOG" 2>&1 || true
echo "Pipeline execution completed."
