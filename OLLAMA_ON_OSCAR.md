# Running the benchmark on Oscar with Ollama

Three files:

| file | role |
| --- | --- |
| `run_benchmark_ollama.sh` | SLURM job: starts `ollama serve` on the compute node, runs generation, then scoring |
| `scripts/run_ollama.py` | generation — models × conditions × repetitions, one independent call per item |
| `scripts/score_ollama_runs.py` | scoring — calls `score_predictions.py` / `score_ablations.py`, aggregates to CSV |

Both Python scripts are standard-library only (they speak Ollama's HTTP API directly),
so no `pip install` is needed.

## Submit

```bash
cd /oscar/data/tserre/schen336/aria_summer_camp/adaptive_pedagogy_benchmark_v4_final
sbatch run_benchmark_ollama.sh
```

Override the experiment without editing the file:

```bash
MODELS="llama3.1:8b qwen3:14b gpt-oss:20b" CONDITIONS="main controls ablations" \
REPS=5 TEMPERATURE=1.0 sbatch run_benchmark_ollama.sh
```

| variable | default | meaning |
| --- | --- | --- |
| `MODELS` | `llama3.1:8b llama3.2:3b qwen3:8b gemma3:12b mistral:7b` | Ollama tags, space separated |
| `CONDITIONS` | `main controls` | see table below |
| `REPS` | `3` | repetitions per model × condition (rep *k* uses seed `1000+k`) |
| `TEMPERATURE` | `1.0` | sampled, so reps measure real model variability; use `TEMPERATURE=0.0 REPS=1` for a greedy deterministic pass |
| `MAX_TOKENS` | `256` | `options.num_predict` — raise to ~1024 for reasoning models (`deepseek-r1`, `gpt-oss`) |
| `NUM_CTX` | `4096` | context window |
| `WORKERS` | `1` | concurrent requests (also sets `OLLAMA_NUM_PARALLEL`) |
| `OUTDIR` | `results` | output root |
| `EXTRA_ARGS` | — | passed straight to `run_ollama.py`, e.g. `--limit 10` for a smoke test |

## Instruct-tuned chat models in the shared store

Verified present as of 2026-08-11. Sizes are what fits on one 48 GB `gpu-he` card.

| family | tags | fits on 1 GPU |
| --- | --- | --- |
| `llama3.1` | `8b`, `70b`, `405b` | `8b` |
| `llama3.2` | `1b`, `3b` | both |
| `llama3.3` | `latest` (70b) | tight |
| `qwen3` | `4b`, `8b`, `14b`, `30b`, `32b` | up to `32b` |
| `qwen3.5` | `4b`, `9b`, `27b`, `35b`, `122b` | up to `35b` |
| `gemma3` | `1b`, `4b`, `12b`, `27b` | up to `27b` |
| `gemma2` | `2b`, `9b`, `27b` | up to `27b` |
| `mistral` / `mistral-small` / `mistral-nemo` | `7b` / `22b`, `24b` / `latest` | yes |
| `phi3` | `latest`, `medium` | both |
| `deepseek-r1` | `7b`, `8b`, `14b`, `32b`, `70b` | up to `32b` — reasoning, raise `MAX_TOKENS` |
| `gpt-oss` | `20b`, `120b` | `20b` — reasoning, raise `MAX_TOKENS` |

Skip the embedding (`bge-m3`, `nomic-embed-text`), vision (`llava*`, `*-vision`,
`qwen2.5vl`), and code (`codellama`, `qwen2.5-coder`, `starcoder*`) entries — they
are not suited to this benchmark.

The default `MODELS` list is `llama3.1:8b llama3.2:3b qwen3:8b gemma3:12b mistral:7b`;
all five were checked to have complete manifests *and* blobs on disk.

## Conditions

| name | requests | scored against |
| --- | --- | --- |
| `main` | `main_100_joint_requests.jsonl` | `main_100_gold.jsonl` |
| `all120` | `all_120_requests.jsonl` | `all_120_gold.jsonl` (main + both controls in one file) |
| `controls` | expands to `controls_diagnosis` + `controls_choice` | respective control gold files |
| `ablations` | expands to the three `optional_ablation_*` sets | `main_100_gold.jsonl` via `score_ablations.py` |

## Output layout

```
results/
├── raw/
│   └── <condition>__<model_slug>__rep01.jsonl   predictions + run metadata
├── scores/
│   ├── <condition>__<model_slug>__rep01_summary.json
│   └── <condition>__<model_slug>__rep01_details.jsonl
├── run_manifest.json      every sweep's config, counts, timings (prior sweeps kept under "history")
├── per_run_metrics.csv    one row per (condition, model, rep)
├── aggregate_metrics.csv  mean + sd across reps, per model × condition
└── aggregate_metrics.json same, with the raw per-rep values
```

Each row in `raw/*.jsonl` carries the fields `score_predictions.py` needs
(`item_id`, `response`) plus `raw_response`, `json_extraction`, `wall_seconds`,
`eval_count`, `prompt_eval_count`, `seed`, `temperature`, `attempts`. So the raw
file is simultaneously the prediction file and the audit trail — nothing else
needs to be kept.

## Notes

- **Resumable.** Re-running the same command skips `item_id`s already recorded
  successfully in the output file, so a job that hits the wall clock can just be
  resubmitted. Rows that errored are retried.
- **JSON handling.** Requests use Ollama's `format: "json"` constrained decoding.
  If a model still wraps the answer in prose or `<think>` blocks, the runner
  extracts the first valid JSON object and stores it in `response` while keeping
  the untouched text in `raw_response`; `json_extraction` records which happened
  (`exact` / `extracted` / `failed`). Pass `--no-json-format` or `--no-extract`
  to turn either behaviour off.
- **Model availability.** The job reads the shared read-only store at
  `/oscar/data/shared/ollama_models`. Compute nodes have no outbound network, so
  anything not already in the store cannot be pulled inside the job — the runner
  therefore **exits with code 2 before any GPU time is spent** if a requested tag
  is missing, printing the full list of what is available. To see that list
  yourself, run from a login node:

  ```bash
  ls /oscar/data/shared/ollama_models/manifests/registry.ollama.ai/library         # names
  ls /oscar/data/shared/ollama_models/manifests/registry.ollama.ai/library/qwen3   # tags for one
  ```

  Always write the tag explicitly (`qwen3:8b`, not `qwen3`) — most families in the
  store have no `latest` tag, so a bare name will not resolve.
- **GPU sizing.** The default `gpu-he` partition gives 48 GB cards. 8B–14B models
  are comfortable; for `gpt-oss:120b` or `llama3.1:70b` request more than one GPU
  and raise `--mem`.
- **Scoring separately.** Scoring never needs a GPU, so you can rerun it any time
  on a login node: `python3 scripts/score_ollama_runs.py --outdir results`.
