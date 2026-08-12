#!/bin/bash
#SBATCH --job-name=pedagogy_ollama
#SBATCH --output=/oscar/data/tserre/schen336/aria_summer_camp/adaptive_pedagogy_benchmark_v4_final/logs/%x_%j.out
#SBATCH --error=/oscar/data/tserre/schen336/aria_summer_camp/adaptive_pedagogy_benchmark_v4_final/logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu-he
#SBATCH --account=carney-tserre-condo2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

# Adaptive Pedagogy Benchmark v4-final -> Ollama on Oscar.
# Starts a private ollama server on this compute node, runs the benchmark for
# every requested model, then scores and aggregates the results.
#
# Submit:   sbatch run_benchmark_ollama.sh
# Override: MODELS="qwen3:14b gpt-oss:20b" REPS=5 sbatch run_benchmark_ollama.sh

set -euo pipefail

PROJ=/oscar/data/tserre/schen336/aria_summer_camp/adaptive_pedagogy_benchmark_v4_final

# ---- experiment configuration (override via environment at submit time) ----
MODELS="${MODELS:-llama3.1:8b llama3.2:3b qwen3:8b gemma3:12b mistral:7b}"
CONDITIONS="${CONDITIONS:-main controls}"
REPS="${REPS:-3}"
TEMPERATURE="${TEMPERATURE:-1.0}"
# 256 truncated every qwen3 answer: its reasoning channel ate the whole budget,
# so 360 calls returned done_reason="length" with empty content. See
# QWEN3_FAILURE_ANALYSIS.md. 4096 is far above the ~20 tokens the other models
# need, and costs them nothing -- generation still stops at the closing brace.
MAX_TOKENS="${MAX_TOKENS:-4096}"
# Must exceed MAX_TOKENS plus the prompt: num_ctx is the TOTAL window in Ollama,
# so leaving this at 4096 would make a 4096-token generation overrun it and
# trigger context shifting.
NUM_CTX="${NUM_CTX:-8192}"
WORKERS="${WORKERS:-1}"
OUTDIR="${OUTDIR:-$PROJ/results}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$PROJ/logs" "$OUTDIR"
cd "$PROJ"

echo "Job ID:      ${SLURM_JOB_ID:-none}"
echo "Node:        ${SLURMD_NODENAME:-$(hostname)}"
echo "Start time:  $(date)"
echo "Models:      $MODELS"
echo "Conditions:  $CONDITIONS"
echo "Reps:        $REPS  Temperature: $TEMPERATURE"
echo "Out dir:     $OUTDIR"
nvidia-smi || true

# ---- ollama server on this node -------------------------------------------
module load ollama/0.21.0-llj6

export OLLAMA_MODELS=/oscar/data/shared/ollama_models
# Unique port per job so two jobs sharing a node do not collide.
PORT=$(( 11434 + ${SLURM_JOB_ID:-0} % 2000 ))
export OLLAMA_HOST="127.0.0.1:${PORT}"
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL="$WORKERS"
export OLLAMA_KEEP_ALIVE=15m

ollama serve > "$PROJ/logs/ollama_${SLURM_JOB_ID:-manual}.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT

# Wait until the server actually answers before sending work.
for i in $(seq 1 60); do
    if ollama list > /dev/null 2>&1; then
        echo "ollama up on $OLLAMA_HOST after $((i * 2))s"
        break
    fi
    sleep 2
done
ollama list

# ---- python environment ----------------------------------------------------
module load miniforge3/25.3.0-3
source /oscar/runtime/software/x86_64_v3/miniforge3-25.3.0-3-a6hhdjzejtacz63sugjqnvgosfqz63ul/etc/profile.d/conda.sh
conda activate vllm_inference

# ---- generate ---------------------------------------------------------------
python "$PROJ/scripts/run_ollama.py" \
    --models $MODELS \
    --conditions $CONDITIONS \
    --reps "$REPS" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    --num-ctx "$NUM_CTX" \
    --workers "$WORKERS" \
    --host "$OLLAMA_HOST" \
    --outdir "$OUTDIR" \
    $EXTRA_ARGS

# ---- score and aggregate ----------------------------------------------------
python "$PROJ/scripts/score_ollama_runs.py" --outdir "$OUTDIR"

echo "End time: $(date)"
