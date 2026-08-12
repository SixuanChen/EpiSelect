#!/usr/bin/env python3
"""End-to-end sanity test: regenerate, validate, create perfect predictions, score."""
from __future__ import annotations
import json, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def main():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        bench = td / "benchmark"
        subprocess.run([sys.executable, str(ROOT/"generate_benchmark.py"), "--outdir", str(bench)], check=True, capture_output=True)
        subprocess.run([sys.executable, str(ROOT/"validate_benchmark.py"), str(bench/"main_100_full.jsonl")], check=True, capture_output=True)
        gold = read_jsonl(bench/"all_120_gold.jsonl")
        pred = td/"perfect.jsonl"
        with pred.open("w") as f:
            for g in gold:
                t = g["trial_type"]
                if t == "main_joint":
                    payload = {"inferred_rule": g["gold_other_rule"], "answer": g["preferred_answer"]}
                elif t == "control_no_history_diagnosis":
                    payload = {"inferred_rule": "UNKNOWN"}
                else:
                    payload = {"answer": "INSUFFICIENT"}
                f.write(json.dumps({"item_id": g["item_id"], **payload}) + "\n")
        out = td/"summary.json"
        subprocess.run([sys.executable, str(ROOT/"score_predictions.py"), "--gold", str(bench/"all_120_gold.jsonl"), "--pred", str(pred), "--summary-out", str(out)], check=True, capture_output=True)
        s = json.loads(out.read_text())
        required = [
            "main_rule_inference_accuracy", "main_action_success", "main_joint_success",
            "teacher_pedagogical_selection_accuracy", "imposter_blend_in_success",
            "imposter_specific_blend_in_rate", "no_history_diagnosis_unknown_accuracy",
            "no_history_choice_insufficient_accuracy", "quartet_joint_consistency",
        ]
        bad = {k: s.get(k) for k in required if s.get(k) != 1.0}
        if bad:
            raise AssertionError(f"self-test failed: {bad}")
        print("SELF-TEST PASSED: generation, validation, parsing, scoring, controls, and matched metrics all return 1.0 for perfect predictions.")

if __name__ == "__main__":
    main()
