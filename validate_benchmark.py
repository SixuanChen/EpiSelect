#!/usr/bin/env python3
"""Structural validator for Adaptive Pedagogy Benchmark v4-final."""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(p: Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, nargs="?", default=Path("benchmark/main_100_full.jsonl"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = read_jsonl(args.path)
    errors = []
    if len(rows) != 100:
        errors.append(f"expected 100 main rows, got {len(rows)}")
    ids = [r["item_id"] for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate item_ids")

    groups = defaultdict(list)
    preferred_letters = Counter()
    teacher_letters = Counter()
    history_selected_side = Counter()
    target_branch_positions = defaultdict(Counter)

    for r in rows:
        groups[r["base_id"]].append(r)
        hist_texts = []
        for h in r["history"]:
            opts = h["options"]
            hist_texts += [o["text"] for o in opts]
            history_selected_side[h["selected_side"]] += 1
            if len(opts) != 2 or opts[0]["text"] == opts[1]["text"]:
                errors.append(f"bad history pair {r['item_id']}")
        target_texts = [o["text"] for o in r["target_options"]]
        if len(target_texts) != 4 or len(set(target_texts)) != 4:
            errors.append(f"bad target uniqueness {r['item_id']}")
        if set(hist_texts) & set(target_texts):
            errors.append(f"history-target object repetition {r['item_id']}: {set(hist_texts)&set(target_texts)}")
        branches = [o["branch_type"] for o in r["target_options"]]
        if set(branches) != {"intersection_positive", "color_only_positive", "shape_only_positive", "negative"}:
            errors.append(f"wrong semantic target set {r['item_id']}: {branches}")
        for o in r["target_options"]:
            target_branch_positions[o["branch_type"]][o["label"]] += 1
        preferred_letters[r["preferred_answer"]] += 1
        if r["role"] == "teacher":
            teacher_letters[r["preferred_answer"]] += 1
            if len(r["acceptable_answers"]) != 1:
                errors.append(f"teacher not unique {r['item_id']}")
        else:
            if len(r["acceptable_answers"]) != 2:
                errors.append(f"imposter expected 2 accepted answers {r['item_id']}")
            classes = {next(o["evidence_class_relative_to_gold"] for o in r["target_options"] if o["label"] == a) for a in r["acceptable_answers"]}
            if classes != {"current_rule_specific", "conservative_intersection"}:
                errors.append(f"wrong imposter accepted classes {r['item_id']}: {classes}")

    if len(groups) != 25:
        errors.append(f"expected 25 base groups, got {len(groups)}")
    for base, vals in groups.items():
        if len(vals) != 4:
            errors.append(f"{base}: expected quartet of 4, got {len(vals)}")
            continue
        cells = {(v["gold_other_rule"], v["role"]) for v in vals}
        expected = {(r, role) for r in ("COLOR_RULE", "SHAPE_RULE") for role in ("teacher", "imposter")}
        if cells != expected:
            errors.append(f"{base}: bad quartet cells {cells}")
        targets = [[(o["label"], o["text"], o["branch_type"]) for o in v["target_options"]] for v in vals]
        if any(t != targets[0] for t in targets[1:]):
            errors.append(f"{base}: targets/order not identical across quartet")
        # Counterfactual histories must show same option pair order, only chosen item differs.
        color_hist = next(v["history"] for v in vals if v["gold_other_rule"] == "COLOR_RULE")
        shape_hist = next(v["history"] for v in vals if v["gold_other_rule"] == "SHAPE_RULE")
        for hc, hs in zip(color_hist, shape_hist):
            if [o["text"] for o in hc["options"]] != [o["text"] for o in hs["options"]]:
                errors.append(f"{base}: history option order mismatch across counterfactual")
            if hc["selected_text"] == hs["selected_text"]:
                errors.append(f"{base}: counterfactual selected same history item")
        # Each branch learner should choose left once/right once over two trials.
        for rule_label in ("COLOR_RULE", "SHAPE_RULE"):
            h = next(v["history"] for v in vals if v["gold_other_rule"] == rule_label)
            if Counter(x["selected_side"] for x in h) != Counter({"left":1, "right":1}):
                errors.append(f"{base}: {rule_label} history side not balanced")

    summary = {
        "valid": not errors,
        "num_main_trials": len(rows),
        "num_base_rules": len(groups),
        "preferred_answer_letter_counts": dict(preferred_letters),
        "teacher_gold_letter_counts": dict(teacher_letters),
        "history_selected_side_counts": dict(history_selected_side),
        "target_branch_position_counts": {k: dict(v) for k,v in target_branch_positions.items()},
        "errors": errors[:100],
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    raise SystemExit(0 if not errors else 1)

if __name__ == "__main__":
    main()
