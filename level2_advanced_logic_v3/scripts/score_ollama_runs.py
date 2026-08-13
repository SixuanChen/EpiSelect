#!/usr/bin/env python3
"""Score every raw run in an Ollama results directory and aggregate to CSV.

Mirrors scripts/score_ollama_runs.py from the v4 adaptive-pedagogy benchmark:
walk <outdir>/raw/<suite>__<model>__rep<NN>.jsonl, hand each file to the scorer
that suite needs, then roll the summaries up into per-run and per-model CSVs.

Scorer per suite:
  primary, abl_*_staged/flat with teacher/imposter gold -> score_predictions.py
                                                           (primary) or
                                                           score_ablations.py
  null, abl_naturalistic_staged, abl_all_truthful_null  -> score_naturalistic.py
  nohistory                                             -> scored here, because
      its gold is keyed by context_id with expected_rule=UNKNOWN /
      expected_action=E and neither shipped scorer indexes it that way.

Usage:
  python scripts/score_ollama_runs.py --outdir results_maxtok4096_temp0
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_RE = re.compile(r"^(?P<suite>.+?)__(?P<model>.+)__rep(?P<rep>\d+)\.jsonl$")

# suite -> (gold file, scorer)
SUITE_SCORING: dict[str, tuple[str, str]] = {
    "primary":                 ("benchmark/primary_240_gold.jsonl", "predictions"),
    "null":                    ("benchmark/naturalistic_null_120_gold.jsonl", "naturalistic"),
    "nohistory":               ("benchmark/nohistory_12_gold.jsonl", "nohistory"),
    "abl_action_given_gold":   ("benchmark/ablations/action_given_gold_gold.jsonl", "ablations"),
    "abl_joint_single_call":   ("benchmark/ablations/joint_single_call_gold.jsonl", "ablations"),
    "abl_choice_only":         ("benchmark/ablations/choice_only_gold.jsonl", "ablations"),
    "abl_closed_hypothesis":   ("benchmark/ablations/closed_hypothesis_staged_gold.jsonl", "ablations"),
    "abl_neutral_goals":       ("benchmark/ablations/neutral_goals_staged_gold.jsonl", "ablations"),
    "abl_underdetermined":     ("benchmark/ablations/underdetermined_staged_gold.jsonl", "ablations"),
    "abl_all_truthful_staged": ("benchmark/ablations/all_truthful_staged_gold.jsonl", "ablations"),
    "abl_naturalistic_staged": ("benchmark/ablations/naturalistic_staged_gold.jsonl", "naturalistic"),
    "abl_all_truthful_null":   ("benchmark/ablations/all_truthful_null_gold.jsonl", "naturalistic"),
}


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def json_obj(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            v = json.loads(x.strip())
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}
    return {}


def run_script(script: str, argv: list[str]) -> None:
    cmd = [sys.executable, str(REPO_ROOT / script)] + argv
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"{script} failed:\n{proc.stdout}\n{proc.stderr}")


def score_nohistory(gold_path: Path, pred_path: Path, summary_out: Path,
                    details_out: Path) -> None:
    """UNKNOWN / E calibration control.

    Gold is one row per context (expected_rule UNKNOWN, expected_action E), while
    predictions are one row per role branch, so inference is scored once per
    context and abstention once per branch.
    """
    gold = {g["context_id"]: g for g in read_jsonl(gold_path)}
    rows = [r for r in read_jsonl(pred_path) if "item_id" in r]
    details, by_context = [], {}
    choice_counts, role_abstain = Counter(), defaultdict(list)

    for r in rows:
        g = gold.get(r.get("context_id"))
        if g is None:
            continue
        rule = str(json_obj(r.get("raw_inference_response")).get("rule", "")).strip().upper() or None
        choice = str(json_obj(r.get("raw_action_response")).get("choice", "")).strip().upper() or None
        rule_unknown = rule == g["expected_rule"]
        abstained = choice == g["expected_action"]
        by_context[r["context_id"]] = rule_unknown
        choice_counts[choice or "unparseable"] += 1
        role_abstain[r.get("role")].append(abstained)
        details.append({"item_id": r["item_id"], "context_id": r["context_id"],
                        "role": r.get("role"), "feature_pair": g["feature_pair"],
                        "reported_rule": rule, "rule_is_unknown": rule_unknown,
                        "reported_choice": choice, "abstained": abstained,
                        "both_correct": rule_unknown and abstained})

    def mean(xs):
        xs = list(xs)
        return None if not xs else sum(xs) / len(xs)

    summary = {
        "num_gold_contexts": len(gold),
        "num_scored_rows": len(details),
        "unknown_rate": mean(by_context.values()),
        "abstention_rate": mean(d["abstained"] for d in details),
        "unknown_and_abstain_rate": mean(d["both_correct"] for d in details),
        "choice_counts": dict(choice_counts),
    }
    for role, vals in role_abstain.items():
        summary[f"{role}_abstention_rate"] = mean(vals)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    details_out.write_text("".join(json.dumps(d) + "\n" for d in details), encoding="utf-8")


def generation_health(pred_path: Path) -> dict:
    """Run-level facts about generation itself, independent of correctness."""
    rows = read_jsonl(pred_path)
    n = len(rows) or 1
    staged = any("raw_inference_response" in r for r in rows)
    walls = [w for r in rows
             for w in ([r.get("inference_wall_seconds"), r.get("action_wall_seconds")]
                       if staged else [r.get("wall_seconds")])
             if isinstance(w, (int, float))]
    if staged:
        failed = sum(1 for r in rows if r.get("inference_extraction") == "failed"
                     or r.get("action_extraction") == "failed")
        exact = sum(1 for r in rows if r.get("inference_extraction") == "exact"
                    and r.get("action_extraction") == "exact")
    else:
        failed = sum(1 for r in rows if r.get("json_extraction") == "failed")
        exact = sum(1 for r in rows if r.get("json_extraction") == "exact")
    return {
        "n_rows": len(rows),
        "n_errors": sum(1 for r in rows if "error" in r),
        "json_exact_rate": exact / n,
        "json_failed_rate": failed / n,
        "mean_wall_seconds_per_call": round(statistics.mean(walls), 3) if walls else None,
    }


def flatten(summary: dict, prefix: str = "") -> dict:
    """Keep scalar metrics only; nested breakdowns stay in the JSON summaries."""
    out = {}
    for k, v in summary.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[prefix + k] = v
        elif isinstance(v, bool):
            out[prefix + k] = int(v)
    return out


def write_csv(path: Path, rows: list[dict], lead: list[str]) -> None:
    if not rows:
        return
    cols = lead + sorted({k for r in rows for k in r} - set(lead))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, required=True,
                    help="results directory holding raw/ (absolute or repo-relative)")
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = ap.parse_args()

    outdir = args.outdir if args.outdir.is_absolute() else args.repo_root / args.outdir
    raw_dir, scores_dir = outdir / "raw", outdir / "scores"
    if not raw_dir.is_dir():
        raise SystemExit(f"no raw/ directory under {outdir}")
    scores_dir.mkdir(parents=True, exist_ok=True)

    per_run: list[dict] = []
    for pred in sorted(raw_dir.glob("*.jsonl")):
        m = RUN_RE.match(pred.name)
        if not m:
            print(f"skip (unrecognised name): {pred.name}", file=sys.stderr)
            continue
        suite, model, rep = m["suite"], m["model"], int(m["rep"])
        if suite not in SUITE_SCORING:
            print(f"skip (no scorer for suite {suite!r}): {pred.name}", file=sys.stderr)
            continue
        gold_rel, scorer = SUITE_SCORING[suite]
        gold = args.repo_root / gold_rel
        stem = pred.stem
        summary_out = scores_dir / f"{stem}_summary.json"
        details_out = scores_dir / f"{stem}_details.jsonl"

        if scorer == "predictions":
            run_script("score_predictions.py",
                       ["--gold", str(gold), "--pred", str(pred),
                        "--summary-out", str(summary_out),
                        "--details-out", str(details_out)])
        elif scorer == "naturalistic":
            run_script("score_naturalistic.py",
                       ["--gold", str(gold), "--pred", str(pred), "--out", str(summary_out)])
            side = Path(str(summary_out) + ".details.jsonl")
            if side.exists():
                side.replace(details_out)
        elif scorer == "ablations":
            run_script("score_ablations.py",
                       ["--gold", str(gold), "--pred", str(pred), "--out", str(summary_out)])
        else:
            score_nohistory(gold, pred, summary_out, details_out)

        summary = json.loads(summary_out.read_text(encoding="utf-8"))
        row = {"suite": suite, "model": model, "rep": rep}
        row.update(generation_health(pred))
        row.update(flatten(summary))
        per_run.append(row)
        print(f"scored {pred.name} -> {summary_out.name}", flush=True)

    if not per_run:
        print("nothing scored", file=sys.stderr)
        return 1

    lead = ["suite", "model", "rep", "n_rows", "n_errors"]
    write_csv(outdir / "per_run_metrics.csv", per_run, lead)

    # Aggregate across repetitions: mean and SD per (suite, model).
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in per_run:
        grouped[(r["suite"], r["model"])].append(r)
    agg_rows, agg_json = [], {}
    for (suite, model), rows in sorted(grouped.items()):
        out = {"suite": suite, "model": model, "n_reps": len(rows)}
        keys = {k for r in rows for k, v in r.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
                and k not in ("rep",)}
        for k in sorted(keys):
            vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
            if not vals:
                continue
            out[f"{k}_mean"] = round(statistics.mean(vals), 4)
            out[f"{k}_sd"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
        agg_rows.append(out)
        agg_json[f"{suite}|{model}"] = out
    write_csv(outdir / "aggregate_metrics.csv", agg_rows, ["suite", "model", "n_reps"])
    (outdir / "aggregate_metrics.json").write_text(json.dumps(agg_json, indent=2),
                                                   encoding="utf-8")

    print(f"\nwrote {outdir/'per_run_metrics.csv'}\n"
          f"      {outdir/'aggregate_metrics.csv'}\n"
          f"      {outdir/'aggregate_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
