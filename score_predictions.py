#!/usr/bin/env python3
"""Automatically score Adaptive Pedagogy Benchmark v4-final.

Input predictions JSONL can use direct fields, e.g.
  {"item_id":"...","inferred_rule":"COLOR_RULE","answer":"B"}
or a raw model string:
  {"item_id":"...","response":"{\"inferred_rule\":\"COLOR_RULE\",\"answer\":\"B\"}"}

The scorer separately reports:
* main rule-inference accuracy
* main action/evidence-selection success
* Teacher unique pedagogical accuracy
* Imposter blend-in success and specific-vs-conservative strategy
* joint diagnosis+action success
* matched reversal/quartet metrics
* no-history control accuracy
* semantic failure-mode distributions

No LLM judge is used.
"""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

MAIN_RULES = {"COLOR_RULE", "SHAPE_RULE"}
MAIN_ANSWERS = {"A", "B", "C", "D"}


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def mean(xs: Iterable[bool | None]):
    vals = [x for x in xs if x is not None]
    return None if not vals else sum(bool(x) for x in vals) / len(vals)


def parse_payload(row: Mapping[str, object]) -> tuple[dict, bool]:
    # Direct structured fields are accepted as strict.
    direct = {k: row[k] for k in ("inferred_rule", "answer") if k in row}
    if direct:
        return direct, True
    response = row.get("response")
    if isinstance(response, dict):
        return dict(response), True
    if isinstance(response, str):
        s = response.strip()
        try:
            p = json.loads(s)
            if isinstance(p, dict):
                return p, True
        except json.JSONDecodeError:
            pass
    return {}, False


def norm_rule(x, allow_unknown=False):
    if x is None:
        return None
    s = str(x).strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {"COLOR": "COLOR_RULE", "SHAPE": "SHAPE_RULE", "COLOUR_RULE": "COLOR_RULE", "COLOUR": "COLOR_RULE"}
    s = aliases.get(s, s)
    valid = set(MAIN_RULES)
    if allow_unknown:
        valid.add("UNKNOWN")
    return s if s in valid else None


def norm_answer(x, allow_insufficient=False):
    if x is None:
        return None
    s = str(x).strip().upper()
    valid = set(MAIN_ANSWERS)
    if allow_insufficient:
        valid.add("INSUFFICIENT")
    return s if s in valid else None


def selected_option(g, answer):
    if answer not in MAIN_ANSWERS:
        return None
    return next((o for o in g.get("target_options", []) if o["label"] == answer), None)


def outcome_for(role, selected):
    if selected is None:
        return None
    key = "teacher_outcome_class_relative_to_gold" if role == "teacher" else "imposter_outcome_class_relative_to_gold"
    return selected.get(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, required=True, help="Usually benchmark/all_120_gold.jsonl")
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, default=None)
    ap.add_argument("--details-out", type=Path, default=None)
    args = ap.parse_args()

    gold_rows = read_jsonl(args.gold)
    pred_rows = read_jsonl(args.pred)
    gold = {x["item_id"]: x for x in gold_rows}
    preds = {x["item_id"]: x for x in pred_rows if "item_id" in x}

    details = []
    failure_counts = Counter()
    evidence_counts = Counter()
    outcome_counts = Counter()
    format_counts = Counter()
    missing = []

    for item_id, g in gold.items():
        t = g["trial_type"]
        row = preds.get(item_id)
        if row is None:
            missing.append(item_id)
            failure_counts[f"{t}:missing"] += 1
            continue
        payload, strict = parse_payload(row)
        format_counts[f"{t}:strict"] += int(strict)
        format_counts[f"{t}:total"] += 1

        d = {"item_id": item_id, "trial_type": t, "role": g.get("role"), "base_id": g.get("base_id"), "strict_json": strict}
        if t == "main_joint":
            inferred = norm_rule(payload.get("inferred_rule"))
            answer = norm_answer(payload.get("answer"))
            inference_correct = inferred == g["gold_other_rule"] if inferred is not None else False
            action_success = answer in set(g["acceptable_answers"]) if answer is not None else False
            preferred = answer == g["preferred_answer"] if answer is not None else False
            selected = selected_option(g, answer)
            evidence = None if selected is None else selected["evidence_class_relative_to_gold"]
            outcome = outcome_for(g["role"], selected)
            if evidence: evidence_counts[evidence] += 1
            if outcome: outcome_counts[outcome] += 1

            # Does the selected action cohere with the model's own reported rule?
            action_consistent_with_reported_rule = None
            if inferred in MAIN_RULES and selected is not None:
                action_consistent_with_reported_rule = bool(selected["utility_by_role_and_inferred_rule"][g["role"]][inferred])

            if inference_correct and action_success and preferred:
                fm = "fully_correct_preferred"
            elif inference_correct and action_success:
                fm = "correct_inference_acceptable_conservative"
            elif not inference_correct and action_consistent_with_reported_rule:
                fm = "wrong_inference_action_consistent_with_report"
            elif inference_correct and not action_success:
                fm = f"correct_inference_wrong_action:{outcome or 'invalid_format'}"
            elif inferred is None or answer is None:
                fm = "invalid_or_missing_field"
            else:
                fm = f"wrong_inference_and_action:{outcome or 'invalid_format'}"
            failure_counts[fm] += 1
            d.update({
                "gold_inferred_rule": g["gold_other_rule"],
                "reported_inferred_rule": inferred,
                "rule_inference_correct": inference_correct,
                "reported_answer": answer,
                "acceptable_answers": g["acceptable_answers"],
                "preferred_answer": g["preferred_answer"],
                "action_success": action_success,
                "preferred_strategy": preferred,
                "joint_success": inference_correct and action_success,
                "selected_branch_type": None if selected is None else selected["branch_type"],
                "selected_evidence_class": evidence,
                "selected_outcome": outcome,
                "action_consistent_with_reported_rule": action_consistent_with_reported_rule,
                "failure_mode": fm,
            })
        elif t == "control_no_history_diagnosis":
            inferred = norm_rule(payload.get("inferred_rule"), allow_unknown=True)
            ok = inferred == "UNKNOWN"
            fm = "control_unknown_correct" if ok else "control_spurious_rule_inference"
            failure_counts[fm] += 1
            d.update({"reported_inferred_rule": inferred, "control_success": ok, "failure_mode": fm})
        elif t == "control_no_history_choice":
            answer = norm_answer(payload.get("answer"), allow_insufficient=True)
            ok = answer == "INSUFFICIENT"
            fm = "control_insufficient_correct" if ok else "control_spurious_targeted_choice"
            failure_counts[fm] += 1
            selected = selected_option(g, answer)
            d.update({
                "reported_answer": answer,
                "control_success": ok,
                "selected_branch_type": None if selected is None else selected.get("branch_type"),
                "failure_mode": fm,
            })
        details.append(d)

    main_rows = [x for x in details if x["trial_type"] == "main_joint"]
    teachers = [x for x in main_rows if x["role"] == "teacher"]
    imposters = [x for x in main_rows if x["role"] == "imposter"]
    diag_controls = [x for x in details if x["trial_type"] == "control_no_history_diagnosis"]
    choice_controls = [x for x in details if x["trial_type"] == "control_no_history_choice"]

    summary: Dict[str, object] = {
        "num_gold_total": len(gold),
        "num_predictions_scored": len(details),
        "num_missing": len(missing),
        "main_num_trials": len(main_rows),
        "main_rule_inference_accuracy": mean(x.get("rule_inference_correct") for x in main_rows),
        "main_action_success": mean(x.get("action_success") for x in main_rows),
        "main_joint_success": mean(x.get("joint_success") for x in main_rows),
        "teacher_rule_inference_accuracy": mean(x.get("rule_inference_correct") for x in teachers),
        "teacher_pedagogical_selection_accuracy": mean(x.get("action_success") for x in teachers),
        "teacher_joint_accuracy": mean(x.get("joint_success") for x in teachers),
        "imposter_rule_inference_accuracy": mean(x.get("rule_inference_correct") for x in imposters),
        "imposter_blend_in_success": mean(x.get("action_success") for x in imposters),
        "imposter_specific_blend_in_rate": mean(x.get("preferred_strategy") for x in imposters),
        "imposter_conservative_blend_in_rate": mean(x.get("selected_evidence_class") == "conservative_intersection" for x in imposters),
        "imposter_joint_success": mean(x.get("joint_success") for x in imposters),
        "selection_success_given_correct_inference": mean(x.get("action_success") for x in main_rows if x.get("rule_inference_correct") is True),
        "selection_success_given_wrong_inference": mean(x.get("action_success") for x in main_rows if x.get("rule_inference_correct") is False),
        "no_history_diagnosis_unknown_accuracy": mean(x.get("control_success") for x in diag_controls),
        "no_history_choice_insufficient_accuracy": mean(x.get("control_success") for x in choice_controls),
    }

    # Matched counterfactual metrics at the 25 base-rule level.
    by_base = defaultdict(list)
    for x in main_rows:
        by_base[x["base_id"]].append(x)
    quartet_rule = []
    quartet_action = []
    quartet_joint = []
    teacher_rule_reversal = []
    imposter_specific_rule_reversal = []
    goal_reversal_specific = []
    for vals in by_base.values():
        by = {(v["gold_inferred_rule"], v["role"]): v for v in vals}
        expected = {(r, role) for r in MAIN_RULES for role in ("teacher", "imposter")}
        if set(by) != expected:
            continue
        quartet_rule.append(all(by[c]["rule_inference_correct"] for c in expected))
        quartet_action.append(all(by[c]["action_success"] for c in expected))
        quartet_joint.append(all(by[c]["joint_success"] for c in expected))
        tc, ts = by[("COLOR_RULE", "teacher")], by[("SHAPE_RULE", "teacher")]
        teacher_rule_reversal.append(tc["action_success"] and ts["action_success"] and tc["reported_answer"] != ts["reported_answer"])
        ic, is_ = by[("COLOR_RULE", "imposter")], by[("SHAPE_RULE", "imposter")]
        imposter_specific_rule_reversal.append(ic["preferred_strategy"] and is_["preferred_strategy"] and ic["reported_answer"] != is_["reported_answer"])
        for r in MAIN_RULES:
            trow, irow = by[(r, "teacher")], by[(r, "imposter")]
            goal_reversal_specific.append(trow["preferred_strategy"] and irow["preferred_strategy"] and trow["reported_answer"] != irow["reported_answer"])

    summary.update({
        "num_complete_base_quartets": len(quartet_rule),
        "quartet_rule_inference_consistency": mean(quartet_rule),
        "quartet_action_consistency": mean(quartet_action),
        "quartet_joint_consistency": mean(quartet_joint),
        "teacher_learner_rule_reversal_consistency": mean(teacher_rule_reversal),
        "imposter_specific_rule_reversal_consistency": mean(imposter_specific_rule_reversal),
        "specific_goal_reversal_consistency": mean(goal_reversal_specific),
        "selected_evidence_class_counts": dict(evidence_counts),
        "selected_outcome_counts": dict(outcome_counts),
        "failure_mode_counts": dict(failure_counts),
        "strict_format_counts": dict(format_counts),
        "missing_item_ids": missing[:20],
    })

    print(json.dumps(summary, indent=2))
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.details_out:
        args.details_out.parent.mkdir(parents=True, exist_ok=True)
        with args.details_out.open("w", encoding="utf-8") as f:
            for x in details:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
