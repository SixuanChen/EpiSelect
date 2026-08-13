#!/bin/bash
#SBATCH --job-name=episelect_l2
#SBATCH --output=/oscar/data/tserre/schen336/aria_summer_camp/EpiSelect_Level2_AdvancedLogic_v3/logs/%x_%j.out
#SBATCH --error=/oscar/data/tserre/schen336/aria_summer_camp/EpiSelect_Level2_AdvancedLogic_v3/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-he
#SBATCH --account=carney-tserre-condo2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

# EpiSelect Level 2 Advanced Logic v3 -> Ollama text models on Oscar.
#
# Same experiment design as the v4 adaptive-pedagogy run in
# results_maxtok4096_temp0: the same five models, 3 repetitions, temperature 0,
# seeds 1001-1003, num_predict 4096, num_ctx 8192, format=json, one resident
# model at a time.
#
# What differs is the protocol, not the settings. Level 2's primary track is
# staged: one role-neutral Stage-1 inference call per context whose exact reply
# is replayed into both the Teacher and Imposter Stage-2 branches. Per model per
# rep that is 120 primary contexts (360 calls) + 120 null calls + 12 no-history
# contexts (36 calls) = 516 calls.
#
# Submit:   sbatch run_benchmark_ollama.sh
# Override: MODELS="qwen3:8b" REPS=1 SUITES="primary" sbatch run_benchmark_ollama.sh
# Smoke:    EXTRA_ARGS="--limit 2" OUTDIR=results_smoke sbatch run_benchmark_ollama.sh

set -euo pipefail

PROJ=/oscar/data/tserre/schen336/aria_summer_camp/EpiSelect_Level2_AdvancedLogic_v3

# ---- experiment configuration (override via environment at submit time) ----
MODELS="${MODELS:-llama3.1:8b llama3.2:3b qwen3:8b gemma3:12b mistral:7b}"
SUITES="${SUITES:-main}"          # main = primary + null + nohistory
REPS="${REPS:-3}"
TEMPERATURE="${TEMPERATURE:-0.0}"
# 4096 is far above the ~20 tokens a non-reasoning model needs for
# {"rule":...} / {"choice":...}, and costs them nothing -- generation still
# stops at the closing brace. It is what keeps qwen3's thinking channel from
# eating the whole budget before the answer starts (see the v4 run report).
MAX_TOKENS="${MAX_TOKENS:-4096}"
# num_ctx is the TOTAL window in Ollama -- prompt AND generation -- so it must
# exceed MAX_TOKENS plus the prompt. Level 2 Stage-2 prompts carry the Stage-1
# prompt, the replayed assistant answer and the action block, so they are
# longer than v4's; 8192 still leaves the full 4096 budget free.
NUM_CTX="${NUM_CTX:-8192}"
SEED="${SEED:-1000}"
OUTDIR="${OUTDIR:-$PROJ/results_maxtok4096_temp0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$PROJ/logs" "$OUTDIR"
cd "$PROJ"

echo "Job ID:      ${SLURM_JOB_ID:-none}"
echo "Node:        ${SLURMD_NODENAME:-$(hostname)}"
echo "Start time:  $(date)"
echo "Models:      $MODELS"
echo "Suites:      $SUITES"
echo "Reps:        $REPS  Temperature: $TEMPERATURE  Max tokens: $MAX_TOKENS  num_ctx: $NUM_CTX"
echo "Out dir:     $OUTDIR"
nvidia-smi || true

# ---- ollama server on this node -------------------------------------------
module load ollama/0.21.0-llj6

export OLLAMA_MODELS=/oscar/data/shared/ollama_models
# Unique port per job so two jobs sharing a node do not collide.
PORT=$(( 11434 + ${SLURM_JOB_ID:-0} % 2000 ))
export OLLAMA_HOST="127.0.0.1:${PORT}"
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
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

# ---- fail before burning GPU time on a broken build ------------------------
python "$PROJ/self_test.py"

# ---- generate ---------------------------------------------------------------
python "$PROJ/scripts/run_ollama.py" \
    --models $MODELS \
    --suites $SUITES \
    --reps "$REPS" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    --num-ctx "$NUM_CTX" \
    --seed "$SEED" \
    --host "$OLLAMA_HOST" \
    --outdir "$OUTDIR" \
    $EXTRA_ARGS

# ---- score and aggregate ----------------------------------------------------
python "$PROJ/scripts/score_ollama_runs.py" --outdir "$OUTDIR"

# ---- figures ----------------------------------------------------------------
# plot_results.py dispatches on the summary shapes it finds (primary, null,
# nohistory) and exits 0 when there is nothing to plot, so this is safe to call
# unconditionally under `set -e`.
python "$PROJ/scripts/plot_results.py" --results "$OUTDIR"

echo "End time: $(date)"
