#!/bin/bash
#SBATCH --job-name=pedagogy_vlm
#SBATCH --output=/oscar/data/tserre/schen336/aria_summer_camp/adaptive_pedagogy_benchmark_v4_final/logs/%x_%j.out
#SBATCH --error=/oscar/data/tserre/schen336/aria_summer_camp/adaptive_pedagogy_benchmark_v4_final/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-he
#SBATCH --account=carney-tserre-condo2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

# Vision track of the Adaptive Pedagogy Benchmark -> Ollama VLMs on Oscar.
# Same 120 items as the text run, same gold file, same scorer: only the modality
# changes. Each item sends one 224x224 PNG plus the rewritten prompt.
#
# Submit:   sbatch run_vision_benchmark_ollama.sh
# Override: MODELS="qwen2.5vl:32b" REPS=1 sbatch run_vision_benchmark_ollama.sh

set -euo pipefail

PROJ=/oscar/data/tserre/schen336/aria_summer_camp/adaptive_pedagogy_benchmark_v4_final

# ---- experiment configuration (override via environment at submit time) ----
MODELS="${MODELS:-qwen2.5vl:7b llama3.2-vision:11b gemma3:12b}"
REPS="${REPS:-3}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
# num_ctx is the TOTAL window: prompt + image + generation. For MAX_TOKENS to be
# a real generation budget the window must exceed it plus the input, and images
# are not free -- a 224x224 stimulus costs a few hundred tokens on top of the
# ~550-token prompt, more for llama3.2-vision. 8192 keeps 4096 fully available.
NUM_CTX="${NUM_CTX:-8192}"
WORKERS="${WORKERS:-1}"
REQUESTS="${REQUESTS:-$PROJ/vision/all_120_vlm_requests.jsonl}"
CONDITION="${CONDITION:-all120}"
OUTDIR="${OUTDIR:-$PROJ/results_vision}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$PROJ/logs" "$OUTDIR"
cd "$PROJ"

echo "Job ID:      ${SLURM_JOB_ID:-none}"
echo "Node:        ${SLURMD_NODENAME:-$(hostname)}"
echo "Start time:  $(date)"
echo "Models:      $MODELS"
echo "Requests:    $REQUESTS"
echo "Reps:        $REPS  Temperature: $TEMPERATURE  Max tokens: $MAX_TOKENS"
echo "Out dir:     $OUTDIR"
nvidia-smi || true

# ---- stimuli must exist (vision/ is gitignored, so a fresh clone lacks them) --
if [ ! -d "$PROJ/vision/images_224" ]; then
    echo "ERROR: $PROJ/vision/images_224 is missing." >&2
    echo "Run 'python scripts/render_vision.py' on a login node first." >&2
    exit 1
fi
echo "Stimuli:     $(ls "$PROJ/vision/images_224"/*.png | wc -l) PNGs"

# ---- ollama server on this node -------------------------------------------
module load ollama/0.21.0-llj6

export OLLAMA_MODELS=/oscar/data/shared/ollama_models
PORT=$(( 11434 + ${SLURM_JOB_ID:-0} % 2000 ))
export OLLAMA_HOST="127.0.0.1:${PORT}"
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL="$WORKERS"
export OLLAMA_KEEP_ALIVE=15m

ollama serve > "$PROJ/logs/ollama_vlm_${SLURM_JOB_ID:-manual}.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT

for i in $(seq 1 60); do
    if ollama list > /dev/null 2>&1; then
        echo "ollama up on $OLLAMA_HOST after $((i * 2))s"
        break
    fi
    sleep 2
done

# ---- python environment ----------------------------------------------------
module load miniforge3/25.3.0-3
source /oscar/runtime/software/x86_64_v3/miniforge3-25.3.0-3-a6hhdjzejtacz63sugjqnvgosfqz63ul/etc/profile.d/conda.sh
conda activate vllm_inference

# ---- generate ---------------------------------------------------------------
python "$PROJ/scripts/run_vlm.py" \
    --backend ollama \
    --models $MODELS \
    --requests "$REQUESTS" \
    --condition "$CONDITION" \
    --reps "$REPS" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    --num-ctx "$NUM_CTX" \
    --workers "$WORKERS" \
    --host "$OLLAMA_HOST" \
    --outdir "$OUTDIR" \
    $EXTRA_ARGS

# ---- score and aggregate ----------------------------------------------------
# Only the task condition has a gold file; the perception probe is scored by
# scripts/score_vision_perception.py instead.
if [ "$CONDITION" != "perception" ]; then
    python "$PROJ/scripts/score_ollama_runs.py" --outdir "$OUTDIR"
fi

# ---- figures ----------------------------------------------------------------
# plot_results.py dispatches on what the directory holds (task scores, perception
# metrics, or assistant_framing profiles) and exits 0 when there is nothing to
# plot, so this is safe to call unconditionally under `set -e`.
python "$PROJ/scripts/plot_results.py" --results "$OUTDIR"

echo "End time: $(date)"
