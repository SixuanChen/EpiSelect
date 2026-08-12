#!/bin/bash
#SBATCH --job-name=pedagogy_perception
#SBATCH --output=/oscar/data/tserre/schen336/aria_summer_camp/adaptive_pedagogy_benchmark_v4_final/logs/%x_%j.out
#SBATCH --error=/oscar/data/tserre/schen336/aria_summer_camp/adaptive_pedagogy_benchmark_v4_final/logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-he
#SBATCH --account=carney-tserre-condo2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

# Perception probe: the same 120 stimuli with NO task, role or rule -- just
# "list every object you can see". Deliberately a separate job from the task
# run, so asking the model to describe the image cannot scaffold its answer and
# the task prompts stay byte-comparable with the text run.
#
# Scored against vision/objects.csv, the renderer's own per-object ground truth.
# Read it next to the task scores: a model that cannot resolve which object
# carries the black border cannot infer the rule either, and would otherwise be
# recorded as a pedagogy failure rather than a perception one.
#
# Submit:   sbatch run_perception_probe_ollama.sh
# Override: MODELS="qwen2.5vl:32b" REPS=1 sbatch run_perception_probe_ollama.sh

set -euo pipefail

PROJ=/oscar/data/tserre/schen336/aria_summer_camp/adaptive_pedagogy_benchmark_v4_final

MODELS="${MODELS:-qwen2.5vl:7b llama3.2-vision:11b gemma3:12b}"
REPS="${REPS:-3}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
NUM_CTX="${NUM_CTX:-8192}"
WORKERS="${WORKERS:-1}"
REQUESTS="${REQUESTS:-$PROJ/vision/all_120_perception_requests.jsonl}"
OUTDIR="${OUTDIR:-$PROJ/results_perception}"
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

ollama serve > "$PROJ/logs/ollama_perception_${SLURM_JOB_ID:-manual}.log" 2>&1 &
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
# condition=perception keeps these files out of the way of the task run and
# tells score_ollama_runs.py to skip them: there is no gold answer file here,
# the ground truth is per-object in vision/objects.csv.
python "$PROJ/scripts/run_vlm.py" \
    --backend ollama \
    --models $MODELS \
    --requests "$REQUESTS" \
    --condition perception \
    --reps "$REPS" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    --num-ctx "$NUM_CTX" \
    --workers "$WORKERS" \
    --host "$OLLAMA_HOST" \
    --outdir "$OUTDIR" \
    $EXTRA_ARGS

# ---- score ------------------------------------------------------------------
python "$PROJ/scripts/score_vision_perception.py" --outdir "$OUTDIR"

echo "End time: $(date)"
