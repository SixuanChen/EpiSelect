#!/usr/bin/env python3
"""Score EpiSelect Level 2 v3 staged Teacher/Imposter predictions.

Recommended row (one row per role-specific action item):
{
  "item_id": "...__role-teacher",
  "context_id": "...",
  "raw_inference_response": "{\"rule\":\"IF(A,B)\"}",
  "raw_action_response": "{\"choice\":\"2\"}"
}

Open rule inference is scored semantically: formulas are parsed and evaluated over
all four A/B states, so logically equivalent expressions receive identical credit.
Action choices use 1-4 plus E.  Primary semantic roles are Informative,
Compatible-positive, Compatible-negative, and Invalid-control.  The two Compatible
polarities have identical Imposter success scoring but are reported separately.
No LLM judge is used.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List

from logic_core import CANONICAL_FUNCTIONS, RuleParseError, parse_rule_expression

CHOICES = {"1", "2", "3", "4", "E"}


def read_jsonl(path: Path) -> List[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def json_obj(x):
    if isinstance(x, dict):
        return x
    if not isinstance(x, str):
        return {}
    try:
        v = json.loads(x.strip())
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def extract_rule(row: dict):
    for k in ("reported_rule", "rule"):
        if row.get(k) is not None:
            return str(row[k]).strip()
    for k in ("raw_inference_response", "inference_response", "response_stage1"):
        p = json_obj(row.get(k))
        if p.get("rule") is not None:
            return str(p["rule"]).strip()
    p = json_obj(row.get("response", row.get("raw_response")))
    if p.get("rule") is not None:
        return str(p["rule"]).strip()
    return None


def extract_choice(row: dict):
    for k in ("reported_choice", "choice", "reported_answer", "answer"):
        if row.get(k) is not None:
            return str(row[k]).strip().upper()
    for k in ("raw_action_response", "action_response", "response_stage2", "response", "raw_response"):
        p = json_obj(row.get(k))
        val = p.get("choice", p.get("answer"))
        if val is not None:
            return str(val).strip().upper()
    return None


def mean(xs: Iterable[bool]):
    vals = list(xs)
    return None if not vals else sum(vals) / len(vals)


def truth_table_agreement(a, b):
    if a is None:
        return None
    return sum(int(x == y) for x, y in zip(a, b)) / 4.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path)
    ap.add_argument("--details-out", type=Path)
    args = ap.parse_args()

    gold_rows = read_jsonl(args.gold)
    pred_rows = read_jsonl(args.pred)
    pred = {r["item_id"]: r for r in pred_rows if "item_id" in r}

    details = []
    missing, invalid_formula, invalid_choice = [], [], []
    semantic_counts = defaultdict(Counter)
    confusion = Counter()

    for g in gold_rows:
        iid = g["item_id"]
        r = pred.get(iid)
        if r is None:
            missing.append(iid)
            continue

        rule = extract_rule(r)
        choice = extract_choice(r)
        parsed_bits = None
        inferred_name = None
        rule_parse_success = False
        if rule and rule.upper() == "UNKNOWN":
            rule_parse_success = True
            inferred_name = "UNKNOWN"
        elif rule:
            try:
                parsed_bits = parse_rule_expression(rule)
                rule_parse_success = True
                inferred_name = next(
                    (n for n, b in CANONICAL_FUNCTIONS.items() if tuple(b) == tuple(parsed_bits)),
                    "SEMANTIC_OTHER",
                )
            except RuleParseError:
                invalid_formula.append(iid)
        else:
            invalid_formula.append(iid)

        choice_parse_success = choice in CHOICES
        if not choice_parse_success:
            invalid_choice.append(iid)

        gold_bits = tuple(g["gold_user_rule_truth_table"])
        inference_correct = bool(parsed_bits is not None and tuple(parsed_bits) == gold_bits)
        rule_state_accuracy = truth_table_agreement(parsed_bits, gold_bits)
        if inferred_name is not None:
            confusion[(g["gold_user_rule"], inferred_name)] += 1

        option_by_label = {o["label"]: o for o in g["options"]}
        selected = option_by_label.get(choice) if choice_parse_success and choice != "E" else None
        selected_role = "abstain" if choice == "E" else (selected["semantic_role"] if selected else "invalid_format")
        selected_policy = "abstain" if choice == "E" else (selected.get("policy_class") if selected else "invalid_format")
        selected_truthful = None if selected is None else bool(selected["label_is_truthful"])

        target_answers = g.get("target_answers", g.get("preferred_answers", []))
        action_success = bool(choice in target_answers) if choice_parse_success else False
        joint_success = inference_correct and action_success

        semantic_counts[g["role"]][selected_role] += 1
        details.append({
            "item_id": iid,
            "context_id": g["context_id"],
            "base_id": g["base_id"],
            "role": g["role"],
            "rule_family": g["rule_family"],
            "rule_group": g["rule_group"],
            "feature_pair": g["feature_pair"],
            "pdd": g["pdd"],
            "gold_user_rule": g["gold_user_rule"],
            "gold_user_rule_expression": g["gold_user_rule_expression"],
            "reported_rule": rule,
            "reported_rule_canonical": inferred_name,
            "rule_parse_success": rule_parse_success,
            "rule_state_accuracy": rule_state_accuracy,
            "inference_correct": inference_correct,
            "reported_choice": choice,
            "choice_parse_success": choice_parse_success,
            "selected_semantic_role": selected_role,
            "selected_policy_class": selected_policy,
            "selected_label_truthful": selected_truthful,
            "action_success": action_success,
            "joint_success": joint_success,
            # Backward-compatible aliases for existing plotting code.
            "preferred_action_correct": action_success,
            "broad_action_success": action_success,
            "joint_preferred_success": joint_success,
            "raw_inference_response": r.get("raw_inference_response", r.get("inference_response")),
            "raw_action_response": r.get("raw_action_response", r.get("action_response", r.get("response"))),
        })

    result = {
        "num_gold": len(gold_rows),
        "num_scored": len(details),
        "num_missing": len(missing),
        "rule_parse_rate": mean(d["rule_parse_success"] for d in details),
        "choice_parse_rate": mean(d["choice_parse_success"] for d in details),
        "rule_inference_accuracy": mean(d["inference_correct"] for d in details),
        "mean_rule_truth_table_agreement": (
            sum(d["rule_state_accuracy"] for d in details if d["rule_state_accuracy"] is not None)
            / max(1, sum(d["rule_state_accuracy"] is not None for d in details))
        ),
        "action_success": mean(d["action_success"] for d in details),
        "joint_inference_action_success": mean(d["joint_success"] for d in details),
        "invalid_control_selection_rate": mean(d["selected_semantic_role"] == "control_invalid" for d in details),
        "abstention_rate": mean(d["selected_semantic_role"] == "abstain" for d in details),
        "truthful_choice_rate_given_nonabstention": mean(
            bool(d["selected_label_truthful"]) for d in details if d["selected_label_truthful"] is not None
        ),
        "reference_rates": {
            "teacher_uniform_1to4_target_rate": 0.25,
            "imposter_uniform_1to4_compatible_rate": 0.50,
            "teacher_uniform_among_three_truthful_options": 1/3,
            "imposter_uniform_among_three_truthful_options": 2/3,
            "note": "Teacher and Imposter raw action-success percentages have different target-set sizes and should not be directly compared as equivalent accuracies.",
        },
    }

    for role in ("teacher", "imposter"):
        vals = [d for d in details if d["role"] == role]
        nonabs = [d for d in vals if d["selected_semantic_role"] != "abstain"]
        result[f"{role}_rule_inference_accuracy"] = mean(d["inference_correct"] for d in vals)
        result[f"{role}_action_success"] = mean(d["action_success"] for d in vals)
        result[f"{role}_action_success_given_nonabstention"] = mean(d["action_success"] for d in nonabs)
        result[f"{role}_action_success_given_correct_inference"] = mean(
            d["action_success"] for d in vals if d["inference_correct"]
        )
        result[f"{role}_joint_inference_action_success"] = mean(d["joint_success"] for d in vals)
        result[f"{role}_informative_rate"] = mean(d["selected_semantic_role"] == "informative" for d in vals)
        result[f"{role}_compatible_rate"] = mean(d["selected_policy_class"] == "compatible" for d in vals)
        result[f"{role}_compatible_positive_rate"] = mean(d["selected_semantic_role"] == "compatible_positive" for d in vals)
        result[f"{role}_compatible_negative_rate"] = mean(d["selected_semantic_role"] == "compatible_negative" for d in vals)
        result[f"{role}_invalid_control_selection_rate"] = mean(d["selected_semantic_role"] == "control_invalid" for d in vals)
        result[f"{role}_abstention_rate"] = mean(d["selected_semantic_role"] == "abstain" for d in vals)

    # Unique Stage-1 contexts and strict goal-sensitive role contrast.
    by_context = defaultdict(list)
    for d in details:
        by_context[d["context_id"]].append(d)
    unique_inf, role_rule_agree, strict_role_contrast = [], [], []
    for vals in by_context.values():
        if vals:
            unique_inf.append(vals[0]["inference_correct"])
        if len(vals) == 2:
            role_rule_agree.append(vals[0]["reported_rule"] == vals[1]["reported_rule"])
            byrole = {v["role"]: v for v in vals}
            if set(byrole) == {"teacher", "imposter"}:
                strict_role_contrast.append(
                    byrole["teacher"]["inference_correct"]
                    and byrole["imposter"]["inference_correct"]
                    and byrole["teacher"]["action_success"]
                    and byrole["imposter"]["action_success"]
                    and byrole["teacher"]["reported_choice"] != byrole["imposter"]["reported_choice"]
                )
    result["unique_context_rule_inference_accuracy"] = mean(unique_inf)
    result["teacher_imposter_stage1_exact_agreement"] = mean(role_rule_agree)
    result["strict_goal_sensitive_role_contrast"] = mean(strict_role_contrast)
    result["strict_role_reversal"] = result["strict_goal_sensitive_role_contrast"]  # compatibility alias

    # Four-cell base quartet: 2 hidden-user rules x 2 explicit goals.
    by_base = defaultdict(list)
    for d in details:
        by_base[d["base_id"]].append(d)
    quart = []
    for vals in by_base.values():
        if len(vals) == 4:
            quart.append(all(v["joint_success"] for v in vals))
    result["num_complete_base_quartets"] = len(quart)
    result["strict_four_cell_quartet"] = mean(quart)

    for field in ("rule_family", "rule_group", "feature_pair", "pdd"):
        groups = defaultdict(list)
        for d in details:
            groups[str(d[field])].append(d)
        result[f"by_{field}"] = {
            k: {
                "n": len(v),
                "inference_accuracy": mean(x["inference_correct"] for x in v),
                "action_success": mean(x["action_success"] for x in v),
                "joint_success": mean(x["joint_success"] for x in v),
                "informative_rate": mean(x["selected_semantic_role"] == "informative" for x in v),
                "compatible_rate": mean(x["selected_policy_class"] == "compatible" for x in v),
                "compatible_positive_rate": mean(x["selected_semantic_role"] == "compatible_positive" for x in v),
                "compatible_negative_rate": mean(x["selected_semantic_role"] == "compatible_negative" for x in v),
                "invalid_control_rate": mean(x["selected_semantic_role"] == "control_invalid" for x in v),
            }
            for k, v in sorted(groups.items())
        }

    result["semantic_choice_counts_by_role"] = {r: dict(c) for r, c in semantic_counts.items()}
    result["rule_confusion"] = {f"gold={g}|pred={p}": n for (g, p), n in confusion.items()}
    result["missing_item_ids"] = missing[:20]
    result["invalid_formula_item_ids"] = invalid_formula[:20]
    result["invalid_choice_item_ids"] = invalid_choice[:20]

    txt = json.dumps(result, indent=2)
    print(txt)
    if args.summary_out:
        args.summary_out.write_text(txt, encoding="utf-8")
    if args.details_out:
        args.details_out.write_text("".join(json.dumps(x) + "\n" for x in details), encoding="utf-8")


if __name__ == "__main__":
    main()
