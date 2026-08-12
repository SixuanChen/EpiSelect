#!/usr/bin/env python3
"""Score every Ollama run produced by scripts/run_ollama.py and aggregate.

Walks results/raw/<condition>__<model>__rep<NN>.jsonl, calls the repo's own
scorers (score_predictions.py for main/all120/controls, score_ablations.py for
the ablations), and writes:

  results/scores/<condition>__<model>__rep<NN>_summary.json   per-run summary
  results/scores/<condition>__<model>__rep<NN>_details.jsonl  per-item detail
  results/per_run_metrics.csv       one row per (condition, model, rep)
  results/aggregate_metrics.csv     mean / sd / n across reps per model
  results/aggregate_metrics.json    same, nested

Usage:
  python scripts/score_ollama_runs.py --outdir results
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# condition -> (gold file, scorer, ablation mode or None)
CONDITION_SCORING: dict[str, tuple[str, str, str | None]] = {
    "main": ("benchmark/main_100_gold.jsonl", "predictions", None),
    "all120": ("benchmark/all_120_gold.jsonl", "predictions", None),
    "controls_diagnosis": ("benchmark/controls_no_history_diagnosis_10_gold.jsonl", "predictions", None),
    "controls_choice": ("benchmark/controls_no_history_choice_10_gold.jsonl", "predictions", None),
    "ablation_choice_only": ("benchmark/main_100_gold.jsonl", "ablations", "choice_only"),
    "ablation_diagnosis_only": ("benchmark/main_100_gold.jsonl", "ablations", "diagnosis_only"),
    "ablation_action_given_rule": ("benchmark/main_100_gold.jsonl", "ablations", "action_given_rule"),
}

# Conditions that are deliberately NOT scored for accuracy. assistant_framing has
# no correct answer: the model is never told to teach, so a teacher-like choice is
# not "right" and an imposter-like choice is not "wrong". What matters is which
# profile the answer matches, which scripts/analyze_assistant_framing.py reports.
RECORD_ONLY = {"assistant_framing"}

# Scalar metrics carried into the CSVs, in report order.
MAIN_METRICS = [
    "main_rule_inference_accuracy",
    "main_action_success",
    "main_joint_success",
    "teacher_rule_inference_accuracy",
    "teacher_pedagogical_selection_accuracy",
    "teacher_joint_accuracy",
    "imposter_rule_inference_accuracy",
    "imposter_blend_in_success",
    "imposter_specific_blend_in_rate",
    "imposter_conservative_blend_in_rate",
    "imposter_joint_success",
    "selection_success_given_correct_inference",
    "selection_success_given_wrong_inference",
    "no_history_diagnosis_unknown_accuracy",
    "no_history_choice_insufficient_accuracy",
    "quartet_rule_inference_consistency",
    "quartet_action_consistency",
    "quartet_joint_consistency",
    "teacher_learner_rule_reversal_consistency",
    "imposter_specific_rule_reversal_consistency",
    "specific_goal_reversal_consistency",
]
ABLATION_METRICS = ["accuracy", "teacher_accuracy", "imposter_accuracy"]

RUN_RE = re.compile(r"^(?P<condition>.+)__(?P<model>.+)__rep(?P<rep>\d+)\.jsonl$")


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def run_health(pred_path: Path) -> dict:
    """Delivery-level stats the benchmark scorers do not track."""
    rows = read_jsonl(pred_path)
    n = len(rows)
    errors = sum(1 for r in rows if "error" in r)
    extraction = defaultdict(int)
    for r in rows:
        extraction[r.get("json_extraction", "n/a")] += 1
    walls = [r["wall_seconds"] for r in rows if isinstance(r.get("wall_seconds"), (int, float))]
    evals = [r["eval_count"] for r in rows if isinstance(r.get("eval_count"), int)]
    return {
        "n_rows": n,
        "n_errors": errors,
        "n_json_exact": extraction.get("exact", 0),
        "n_json_extracted": extraction.get("extracted", 0),
        "n_json_failed": extraction.get("failed", 0),
        "mean_wall_seconds": round(statistics.fmean(walls), 3) if walls else None,
        "mean_eval_tokens": round(statistics.fmean(evals), 1) if evals else None,
    }


def score_run(repo_root: Path, condition: str, pred_path: Path,
              scores_dir: Path) -> dict | None:
    gold_rel, scorer, mode = CONDITION_SCORING[condition]
    gold = repo_root / gold_rel
    stem = pred_path.stem
    summary_out = scores_dir / f"{stem}_summary.json"

    if scorer == "predictions":
        cmd = [
            sys.executable, str(repo_root / "score_predictions.py"),
            "--gold", str(gold), "--pred", str(pred_path),
            "--summary-out", str(summary_out),
            "--details-out", str(scores_dir / f"{stem}_details.jsonl"),
        ]
    else:
        cmd = [
            sys.executable, str(repo_root / "score_ablations.py"),
            "--mode", mode, "--gold", str(gold), "--pred", str(pred_path),
            "--out", str(summary_out),
        ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  !! scoring failed for {pred_path.name}:\n{proc.stderr}", file=sys.stderr)
        return None
    return json.loads(summary_out.read_text(encoding="utf-8"))


def fmt(x) -> str:
    return "" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="only score these conditions (default: everything found)")
    args = ap.parse_args()

    raw_dir = args.outdir / "raw"
    if not raw_dir.is_dir():
        raise SystemExit(f"no raw predictions directory at {raw_dir}")
    scores_dir = args.outdir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    per_run: list[dict] = []
    for pred_path in sorted(raw_dir.glob("*.jsonl")):
        m = RUN_RE.match(pred_path.name)
        if not m:
            print(f"  skipping unrecognised file name {pred_path.name}", file=sys.stderr)
            continue
        condition, model, rep = m["condition"], m["model"], int(m["rep"])
        if condition in RECORD_ONLY:
            print(f"  skipping {condition!r}: record-only, no accuracy to report "
                  f"(see scripts/analyze_assistant_framing.py)", file=sys.stderr)
            continue
        if condition not in CONDITION_SCORING:
            print(f"  skipping unknown condition {condition!r}", file=sys.stderr)
            continue
        if args.conditions and condition not in args.conditions:
            continue

        print(f"scoring {pred_path.name} ...", flush=True)
        summary = score_run(args.repo_root, condition, pred_path, scores_dir)
        if summary is None:
            continue
        # Keyed off the scorer that actually ran, not the condition's name: a
        # condition scored by score_ablations.py reports the ablation metrics
        # whether or not it happens to be called "ablation_*".
        metrics = (ABLATION_METRICS if CONDITION_SCORING[condition][1] == "ablations"
                   else MAIN_METRICS)
        row = {"condition": condition, "model": model, "rep": rep}
        row.update(run_health(pred_path))
        row.update({k: summary.get(k) for k in metrics})
        # Only score_predictions.py reports these coverage fields.
        row["num_missing"] = summary.get("num_missing")
        per_run.append(row)

    if not per_run:
        # Not an error: a run directory may hold only record-only conditions.
        print("nothing to score in this directory (record-only conditions are "
              "analysed separately)", file=sys.stderr)
        return 0

    all_metric_cols = [c for c in MAIN_METRICS + ABLATION_METRICS]
    health_cols = ["n_rows", "n_errors", "num_missing", "n_json_exact",
                   "n_json_extracted", "n_json_failed",
                   "mean_wall_seconds", "mean_eval_tokens"]
    per_run_cols = ["condition", "model", "rep"] + health_cols + all_metric_cols

    per_run_csv = args.outdir / "per_run_metrics.csv"
    with per_run_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(per_run_cols)
        for r in sorted(per_run, key=lambda x: (x["condition"], x["model"], x["rep"])):
            w.writerow([fmt(r.get(c)) for c in per_run_cols])

    # Aggregate across repetitions.
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in per_run:
        grouped[(r["condition"], r["model"])].append(r)

    agg_rows: list[dict] = []
    agg_json: dict[str, dict] = {}
    for (condition, model), runs in sorted(grouped.items()):
        row = {"condition": condition, "model": model, "n_reps": len(runs)}
        nested: dict[str, object] = {"n_reps": len(runs)}
        for col in health_cols:
            vals = [r[col] for r in runs if isinstance(r.get(col), (int, float))]
            row[col] = round(statistics.fmean(vals), 3) if vals else None
        for col in all_metric_cols:
            vals = [r[col] for r in runs if isinstance(r.get(col), (int, float))]
            if not vals:
                row[f"{col}_mean"] = row[f"{col}_sd"] = None
                continue
            mean = statistics.fmean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            row[f"{col}_mean"], row[f"{col}_sd"] = mean, sd
            nested[col] = {"mean": mean, "sd": sd, "n": len(vals), "values": vals}
        agg_rows.append(row)
        agg_json[f"{condition}|{model}"] = nested

    agg_cols = (["condition", "model", "n_reps"] + health_cols
                + [f"{c}{suf}" for c in all_metric_cols for suf in ("_mean", "_sd")])
    agg_csv = args.outdir / "aggregate_metrics.csv"
    with agg_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(agg_cols)
        for r in agg_rows:
            w.writerow([fmt(r.get(c)) for c in agg_cols])
    (args.outdir / "aggregate_metrics.json").write_text(
        json.dumps(agg_json, indent=2), encoding="utf-8")

    # Compact console leaderboard for the main condition.
    print(f"\nwrote {per_run_csv}\nwrote {agg_csv}\n")
    main_rows = [r for r in agg_rows if r["condition"] in ("main", "all120")]
    if main_rows:
        print(f"{'model':28s} {'cond':8s} {'reps':>4s} {'rule_acc':>10s} "
              f"{'action':>10s} {'joint':>10s} {'json_fail':>10s}")
        for r in sorted(main_rows, key=lambda x: -(x.get("main_joint_success_mean") or 0)):
            print(f"{r['model'][:28]:28s} {r['condition'][:8]:8s} {r['n_reps']:4d} "
                  f"{fmt(r.get('main_rule_inference_accuracy_mean')):>10s} "
                  f"{fmt(r.get('main_action_success_mean')):>10s} "
                  f"{fmt(r.get('main_joint_success_mean')):>10s} "
                  f"{fmt(r.get('n_json_failed')):>10s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
