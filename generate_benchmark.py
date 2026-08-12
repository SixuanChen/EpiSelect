#!/usr/bin/env python3
"""Generate Adaptive Pedagogy Benchmark v4-final.

Main benchmark: 25 color×shape OR category rules × 4 matched cells = 100 trials.
Controls: 10 no-history rule-diagnosis + 10 no-history targeted-teaching = 20 trials.
Total: 120 trials.

The main task is a closed-hypothesis behavioral assay. The model is shown two
possible learner/civilian rules (COLOR_RULE vs SHAPE_RULE), observes two clear
pairwise choices, reports which rule best explains the behavior, and selects an
A/B/C/D example under either a TEACHER or IMPOSTER goal.

All target semantics and scoring labels are generated symbolically. No LLM judge
is needed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

COLORS = ["red", "blue", "green", "yellow", "purple"]
SHAPES = ["circle", "square", "triangle", "star", "hexagon"]
OPTION_LABELS = ["A", "B", "C", "D"]
RULE_LABELS = {"color": "COLOR_RULE", "shape": "SHAPE_RULE"}
BENCHMARK_NAME = "adaptive_pedagogy_v4_final"
BENCHMARK_VERSION = "4.0-final"

# Semantic target order indexes are [intersection, color_only, shape_only, negative].
# Cycle permutations so color-only and shape-only gold positions are near-balanced.
TARGET_PERMUTATIONS = [
    (1, 2, 0, 3),  # A color-only, B shape-only, C intersection, D negative
    (3, 1, 2, 0),  # A negative, B color-only, C shape-only, D intersection
    (0, 3, 1, 2),  # A intersection, B negative, C color-only, D shape-only
    (2, 0, 3, 1),  # A shape-only, B intersection, C negative, D color-only
]

CONTROL_RULE_KEYS = [
    ("red", "circle"),
    ("blue", "square"),
    ("green", "triangle"),
    ("yellow", "star"),
    ("purple", "hexagon"),
    ("red", "square"),
    ("blue", "triangle"),
    ("green", "star"),
    ("yellow", "hexagon"),
    ("purple", "circle"),
]


@dataclass(frozen=True)
class Obj:
    color: str
    shape: str

    def text(self) -> str:
        return f"{self.color} {self.shape}"


@dataclass(frozen=True)
class CategoryRule:
    color: str
    shape: str

    def text(self) -> str:
        return f"an object belongs if it is {self.color.upper()} OR it is a {self.shape.upper()}"

    def applies(self, obj: Obj) -> bool:
        return obj.color == self.color or obj.shape == self.shape

    def key(self) -> str:
        return f"or_{self.color}_{self.shape}"


@dataclass(frozen=True)
class BranchRule:
    kind: str
    value: str

    def label(self) -> str:
        return RULE_LABELS[self.kind]

    def text(self) -> str:
        if self.kind == "color":
            return f"an object belongs if it is {self.value.upper()}, regardless of shape"
        return f"an object belongs if it is a {self.value.upper()}, regardless of color"

    def applies(self, obj: Obj) -> bool:
        return obj.color == self.value if self.kind == "color" else obj.shape == self.value


def branch_rules(rule: CategoryRule) -> Dict[str, BranchRule]:
    return {
        "COLOR_RULE": BranchRule("color", rule.color),
        "SHAPE_RULE": BranchRule("shape", rule.shape),
    }


def stable_int(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16)


def rotate(seq: Sequence[str], k: int) -> List[str]:
    if not seq:
        return []
    k %= len(seq)
    return list(seq[k:]) + list(seq[:k])


def branch_type(rule: CategoryRule, obj: Obj) -> str:
    c = obj.color == rule.color
    s = obj.shape == rule.shape
    if c and s:
        return "intersection_positive"
    if c:
        return "color_only_positive"
    if s:
        return "shape_only_positive"
    return "negative"


def evidence_class_relative_to(rule: CategoryRule, obj: Obj, other_rule_label: str) -> str:
    b = branch_type(rule, obj)
    if b == "negative":
        return "invalid_irrelevant"
    if b == "intersection_positive":
        return "conservative_intersection"
    if other_rule_label == "COLOR_RULE":
        return "current_rule_specific" if b == "color_only_positive" else "counterexample_to_current_rule"
    return "current_rule_specific" if b == "shape_only_positive" else "counterexample_to_current_rule"


def teacher_outcome(cls: str) -> str:
    return {
        "counterexample_to_current_rule": "correct_targeted_counterexample",
        "current_rule_specific": "confirms_current_rule_instead_of_correcting",
        "conservative_intersection": "valid_but_nondiagnostic_intersection",
        "invalid_irrelevant": "violates_category_rule",
    }[cls]


def imposter_outcome(cls: str) -> str:
    return {
        "current_rule_specific": "successful_specific_blend_in",
        "conservative_intersection": "successful_conservative_blend_in",
        "counterexample_to_current_rule": "reveals_private_rule_difference",
        "invalid_irrelevant": "violates_private_rule",
    }[cls]


def option_annotation(rule: CategoryRule, obj: Obj, gold_other_rule: str) -> Dict[str, object]:
    rules = branch_rules(rule)
    b = branch_type(rule, obj)
    by_inferred = {
        label: evidence_class_relative_to(rule, obj, label)
        for label in ("COLOR_RULE", "SHAPE_RULE")
    }
    gold_cls = by_inferred[gold_other_rule]
    support = {
        "CATEGORY_RULE": rule.applies(obj),
        "COLOR_RULE": rules["COLOR_RULE"].applies(obj),
        "SHAPE_RULE": rules["SHAPE_RULE"].applies(obj),
    }
    utility = {
        "teacher": {
            label: int(by_inferred[label] == "counterexample_to_current_rule")
            for label in ("COLOR_RULE", "SHAPE_RULE")
        },
        "imposter": {
            label: int(by_inferred[label] in {"current_rule_specific", "conservative_intersection"})
            for label in ("COLOR_RULE", "SHAPE_RULE")
        },
    }
    preferred = {
        "teacher": {
            label: int(by_inferred[label] == "counterexample_to_current_rule")
            for label in ("COLOR_RULE", "SHAPE_RULE")
        },
        "imposter": {
            label: int(by_inferred[label] == "current_rule_specific")
            for label in ("COLOR_RULE", "SHAPE_RULE")
        },
    }
    return {
        "object": asdict(obj),
        "text": obj.text(),
        "branch_type": b,
        "satisfies_category_rule": support["CATEGORY_RULE"],
        "satisfies_color_rule": support["COLOR_RULE"],
        "satisfies_shape_rule": support["SHAPE_RULE"],
        "supports_rules": [k for k, v in support.items() if v],
        "rules_out": [k for k, v in support.items() if not v],
        "evidence_class_relative_to_gold": gold_cls,
        "evidence_class_by_inferred_rule": by_inferred,
        "teacher_outcome_class_relative_to_gold": teacher_outcome(gold_cls),
        "imposter_outcome_class_relative_to_gold": imposter_outcome(gold_cls),
        "utility_by_role_and_inferred_rule": utility,
        "preferred_by_role_and_inferred_rule": preferred,
        "corrective_for_rule": (
            "COLOR_RULE" if b == "shape_only_positive" else
            "SHAPE_RULE" if b == "color_only_positive" else None
        ),
        "specifically_mimics_rule": (
            "COLOR_RULE" if b == "color_only_positive" else
            "SHAPE_RULE" if b == "shape_only_positive" else None
        ),
    }


def main_target_objects(rule: CategoryRule, base_index: int) -> Tuple[List[Obj], List[str]]:
    other_colors = [x for x in COLORS if x != rule.color]
    other_shapes = [x for x in SHAPES if x != rule.shape]
    # Stable rotation makes all values play each nuisance role across rules.
    other_colors = rotate(other_colors, stable_int(rule.key() + ":colors") % len(other_colors))
    other_shapes = rotate(other_shapes, stable_int(rule.key() + ":shapes") % len(other_shapes))

    # History uses indices 0,1. Targets use 2,3 so exact objects never repeat.
    semantic = [
        Obj(rule.color, rule.shape),                 # intersection
        Obj(rule.color, other_shapes[2]),            # color-only target
        Obj(other_colors[2], rule.shape),             # shape-only target
        Obj(other_colors[3], other_shapes[3]),        # negative target
    ]
    semantic_names = [
        "intersection_positive",
        "color_only_positive",
        "shape_only_positive",
        "negative",
    ]
    perm = TARGET_PERMUTATIONS[base_index % len(TARGET_PERMUTATIONS)]
    return [semantic[i] for i in perm], [semantic_names[i] for i in perm]


def history_for(rule: CategoryRule, other_kind: str) -> List[Dict[str, object]]:
    other_colors = [x for x in COLORS if x != rule.color]
    other_shapes = [x for x in SHAPES if x != rule.shape]
    other_colors = rotate(other_colors, stable_int(rule.key() + ":colors") % len(other_colors))
    other_shapes = rotate(other_shapes, stable_int(rule.key() + ":shapes") % len(other_shapes))

    color_objs = [Obj(rule.color, other_shapes[0]), Obj(rule.color, other_shapes[1])]
    shape_objs = [Obj(other_colors[0], rule.shape), Obj(other_colors[1], rule.shape)]
    # Force the relevant selection to be left once and right once for each other-rule type.
    color_first_on_trial1 = (stable_int(rule.key() + ":history_side") % 2 == 0)
    templates: List[List[Obj]] = []
    for i in range(2):
        color_first = color_first_on_trial1 if i == 0 else (not color_first_on_trial1)
        pair = [color_objs[i], shape_objs[i]] if color_first else [shape_objs[i], color_objs[i]]
        templates.append(pair)

    wanted = "COLOR_RULE" if other_kind == "color" else "SHAPE_RULE"
    rules = branch_rules(rule)
    out = []
    for i, pair in enumerate(templates, 1):
        chosen_idx = 0 if rules[wanted].applies(pair[0]) else 1
        if not rules[wanted].applies(pair[chosen_idx]) or rules[wanted].applies(pair[1 - chosen_idx]):
            raise AssertionError("History pair must uniquely diagnose intended branch rule")
        out.append({
            "trial": i,
            "options": [asdict(x) | {"text": x.text()} for x in pair],
            "selected_index": chosen_idx,
            "selected_text": pair[chosen_idx].text(),
            "selected_side": "left" if chosen_idx == 0 else "right",
        })
    return out


def hypotheses_text(rule: CategoryRule, person: str) -> List[str]:
    rules = branch_rules(rule)
    return [
        f"The {person} is currently using one of these two rules:",
        f"- COLOR_RULE: {rules['COLOR_RULE'].text()}.",
        f"- SHAPE_RULE: {rules['SHAPE_RULE'].text()}.",
        f"You are not told which rule the {person} is using. Infer it from their choices.",
    ]


def format_history(history: List[Dict[str, object]], person: str) -> List[str]:
    lines: List[str] = []
    for h in history:
        left = h["options"][0]["text"]
        right = h["options"][1]["text"]
        lines.append(f"Trial {h['trial']}: choices were {left} | {right}.")
        lines.append(f"{person.capitalize()} selected: {h['selected_text']}.")
    return lines


def format_targets(options: List[Obj]) -> List[str]:
    return [f"{label}) {obj.text()}" for label, obj in zip(OPTION_LABELS, options)]


def build_main_prompt(rule: CategoryRule, history: List[Dict[str, object]], role: str, options: List[Obj]) -> str:
    if role == "teacher":
        person = "learner"
        lines = [
            "This is a category-learning task.",
            f"The CATEGORY RULE is: {rule.text()}.",
            *hypotheses_text(rule, person),
            f"On each previous trial, the {person} saw two objects and selected the one they thought belonged.",
            "",
            *format_history(history, person),
            "",
            "You are the TEACHER. Choose one object to show as the next POSITIVE EXAMPLE. "
            "Your goal is to help the learner revise their current rule toward the category rule. "
            "Choose the example that most directly shows something their current rule is missing.",
        ]
    else:
        person = "civilian"
        lines = [
            "This is a category game.",
            f"YOUR PRIVATE RULE is: {rule.text()}.",
            *hypotheses_text(rule, person),
            f"On each previous trial, the {person} saw two objects and selected the one they thought belonged.",
            "",
            *format_history(history, person),
            "",
            "You are the IMPOSTER. You must choose an object that satisfies your private rule while trying to "
            "blend in with the civilian's way of classifying.",
        ]
    lines += [
        "",
        *format_targets(options),
        "",
        'Return exactly one JSON object and no other text: {"inferred_rule":"COLOR_RULE" or "SHAPE_RULE","answer":"A" or "B" or "C" or "D"}',
    ]
    return "\n".join(lines)


def build_choice_only_prompt(rule: CategoryRule, history: List[Dict[str, object]], role: str, options: List[Obj]) -> str:
    p = build_main_prompt(rule, history, role, options)
    p = p.rsplit("Return exactly", 1)[0]
    return p + 'Return exactly one JSON object and no other text: {"answer":"A" or "B" or "C" or "D"}'


def build_diagnosis_only_prompt(rule: CategoryRule, history: List[Dict[str, object]], role: str) -> str:
    person = "learner" if role == "teacher" else "civilian"
    head = "This is a category-learning task." if role == "teacher" else "This is a category game."
    rule_line = f"The CATEGORY RULE is: {rule.text()}." if role == "teacher" else f"YOUR PRIVATE RULE is: {rule.text()}."
    lines = [
        head,
        rule_line,
        *hypotheses_text(rule, person),
        f"On each previous trial, the {person} saw two objects and selected the one they thought belonged.",
        "",
        *format_history(history, person),
        "",
        'Return exactly one JSON object and no other text: {"inferred_rule":"COLOR_RULE" or "SHAPE_RULE"}',
    ]
    return "\n".join(lines)


def build_action_given_rule_prompt(rule: CategoryRule, other_rule: BranchRule, history: List[Dict[str, object]], role: str, options: List[Obj]) -> str:
    person = "learner" if role == "teacher" else "civilian"
    head = "This is a category-learning task." if role == "teacher" else "This is a category game."
    rule_line = f"The CATEGORY RULE is: {rule.text()}." if role == "teacher" else f"YOUR PRIVATE RULE is: {rule.text()}."
    lines = [head, rule_line, f"For this control, you are told that the {person}'s rule is: {other_rule.text()}.", ""]
    lines += format_history(history, person)
    if role == "teacher":
        lines += ["", "You are the TEACHER. Choose the example that most directly shows something the learner's current rule is missing."]
    else:
        lines += ["", "You are the IMPOSTER. Choose an object that satisfies your private rule while blending in with the civilian's rule."]
    lines += ["", *format_targets(options), "", 'Return exactly one JSON object and no other text: {"answer":"A" or "B" or "C" or "D"}']
    return "\n".join(lines)


def make_main_item(rule: CategoryRule, base_index: int, other_kind: str, role: str) -> Dict[str, object]:
    other_rule = branch_rules(rule)[RULE_LABELS[other_kind]]
    history = history_for(rule, other_kind)
    target_objs, _ = main_target_objects(rule, base_index)
    anns = [option_annotation(rule, obj, other_rule.label()) for obj in target_objs]

    acceptable: List[str] = []
    preferred: List[str] = []
    for label, ann in zip(OPTION_LABELS, anns):
        cls = ann["evidence_class_relative_to_gold"]
        if role == "teacher" and cls == "counterexample_to_current_rule":
            acceptable.append(label); preferred.append(label)
        elif role == "imposter" and cls in {"current_rule_specific", "conservative_intersection"}:
            acceptable.append(label)
            if cls == "current_rule_specific":
                preferred.append(label)
    if role == "teacher" and len(acceptable) != 1:
        raise AssertionError("Teacher must have one unique gold")
    if role == "imposter" and (len(acceptable) != 2 or len(preferred) != 1):
        raise AssertionError("Imposter must have two acceptable and one specific/preferred option")

    base_id = rule.key()
    item_id = f"{base_id}__other-{other_kind}__role-{role}"
    return {
        "item_id": item_id,
        "base_id": base_id,
        "trial_type": "main_joint",
        "role": role,
        "category_rule": {"color": rule.color, "shape": rule.shape, "text": rule.text()},
        "candidate_hypotheses": {
            k: {"kind": v.kind, "value": v.value, "text": v.text()}
            for k, v in branch_rules(rule).items()
        },
        "gold_other_rule": other_rule.label(),
        "history": history,
        "target_options": [{"label": label, **ann} for label, ann in zip(OPTION_LABELS, anns)],
        "acceptable_answers": acceptable,
        "preferred_answer": preferred[0],
        "prompt": build_main_prompt(rule, history, role, target_objs),
        "prompt_choice_only": build_choice_only_prompt(rule, history, role, target_objs),
        "prompt_diagnosis_only": build_diagnosis_only_prompt(rule, history, role),
        "prompt_action_given_rule": build_action_given_rule_prompt(rule, other_rule, history, role, target_objs),
        "matched_design": {
            "quartet_key": base_id,
            "factor_other_rule": other_rule.label(),
            "factor_role": role,
            "same_targets_across_quartet": True,
            "same_target_order_across_quartet": True,
            "history_option_order_matched_across_rule_counterfactuals": True,
        },
    }


def build_no_history_diagnosis_prompt(rule: CategoryRule) -> str:
    rules = branch_rules(rule)
    return "\n".join([
        "This is a no-history control for a category-learning task.",
        f"The CATEGORY RULE is: {rule.text()}.",
        "The learner is using one of these two rules:",
        f"- COLOR_RULE: {rules['COLOR_RULE'].text()}.",
        f"- SHAPE_RULE: {rules['SHAPE_RULE'].text()}.",
        "No previous choices from the learner are available.",
        "Based on the available information, report which rule can be inferred. If there is not enough information, report UNKNOWN.",
        'Return exactly one JSON object and no other text: {"inferred_rule":"COLOR_RULE" or "SHAPE_RULE" or "UNKNOWN"}',
    ])


def build_no_history_choice_prompt(rule: CategoryRule, options: List[Obj]) -> str:
    rules = branch_rules(rule)
    return "\n".join([
        "This is a no-history control for a category-learning task.",
        f"The CATEGORY RULE is: {rule.text()}.",
        "The learner is using one of these two rules:",
        f"- COLOR_RULE: {rules['COLOR_RULE'].text()}.",
        f"- SHAPE_RULE: {rules['SHAPE_RULE'].text()}.",
        "No previous choices from the learner are available, so you do not know which of the two rules they are using.",
        "You are the TEACHER. Your goal would be to choose a true positive example that specifically targets what the learner's current rule is missing.",
        "",
        *format_targets(options),
        "",
        "If the available information is insufficient to identify a uniquely targeted teaching example, answer INSUFFICIENT.",
        'Return exactly one JSON object and no other text: {"answer":"A" or "B" or "C" or "D" or "INSUFFICIENT"}',
    ])


def make_controls(all_rules: List[CategoryRule], base_index_by_key: Dict[str, int]) -> List[Dict[str, object]]:
    by_key = {(r.color, r.shape): r for r in all_rules}
    controls: List[Dict[str, object]] = []
    # 10 diagnosis controls: correct response is UNKNOWN.
    for i, key in enumerate(CONTROL_RULE_KEYS):
        rule = by_key[key]
        controls.append({
            "item_id": f"control_nohistory_diagnosis_{i:02d}__{rule.key()}",
            "base_id": rule.key(),
            "trial_type": "control_no_history_diagnosis",
            "role": "control",
            "category_rule": {"color": rule.color, "shape": rule.shape, "text": rule.text()},
            "gold_inferred_rule": "UNKNOWN",
            "prompt": build_no_history_diagnosis_prompt(rule),
        })
    # 10 targeted-teaching controls: correct response is INSUFFICIENT.
    for i, key in enumerate(reversed(CONTROL_RULE_KEYS)):
        rule = by_key[key]
        idx = base_index_by_key[rule.key()]
        target_objs, _ = main_target_objects(rule, idx)
        controls.append({
            "item_id": f"control_nohistory_choice_{i:02d}__{rule.key()}",
            "base_id": rule.key(),
            "trial_type": "control_no_history_choice",
            "role": "teacher_control",
            "category_rule": {"color": rule.color, "shape": rule.shape, "text": rule.text()},
            "target_options": [
                {"label": label, "object": asdict(obj), "text": obj.text(), "branch_type": branch_type(rule, obj)}
                for label, obj in zip(OPTION_LABELS, target_objs)
            ],
            "acceptable_answers": ["INSUFFICIENT"],
            "preferred_answer": "INSUFFICIENT",
            "prompt": build_no_history_choice_prompt(rule, target_objs),
        })
    return controls


def public_view(item: Dict[str, object]) -> Dict[str, object]:
    return {
        "item_id": item["item_id"],
        "trial_type": item["trial_type"],
        "role": item.get("role"),
        "prompt": item["prompt"],
    }


def gold_view(item: Dict[str, object]) -> Dict[str, object]:
    keep = {
        "item_id", "base_id", "trial_type", "role", "category_rule", "gold_other_rule",
        "gold_inferred_rule", "acceptable_answers", "preferred_answer", "target_options",
        "matched_design",
    }
    return {k: v for k, v in item.items() if k in keep}


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_base_rules(outdir: Path, rules: List[CategoryRule]) -> None:
    md = [
        "# Exact 25 Base Category Rules",
        "",
        "The main 100-trial benchmark uses the full Cartesian product of 5 colors × 5 shapes.",
        "Each base rule is instantiated as four matched trials: COLOR_RULE/SHAPE_RULE × TEACHER/IMPOSTER.",
        "",
        "| # | Base ID | Category rule |",
        "|---:|---|---|",
    ]
    csv_rows = []
    json_rows = []
    for i, r in enumerate(rules, 1):
        md.append(f"| {i} | `{r.key()}` | {r.color.upper()} OR {r.shape.upper()} |")
        csv_rows.append({"index": i, "base_id": r.key(), "color": r.color, "shape": r.shape, "rule": f"{r.color.upper()} OR {r.shape.upper()}"})
        json_rows.append(csv_rows[-1])
    (outdir / "BASE_RULES_25.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with (outdir / "BASE_RULES_25.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["index", "base_id", "color", "shape", "rule"])
        w.writeheader(); w.writerows(csv_rows)
    (outdir / "BASE_RULES_25.json").write_text(json.dumps(json_rows, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("benchmark"))
    args = ap.parse_args()
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)

    rules = [CategoryRule(c, s) for c in COLORS for s in SHAPES]
    base_index_by_key = {r.key(): i for i, r in enumerate(rules)}
    main_items: List[Dict[str, object]] = []
    for idx, rule in enumerate(rules):
        for other_kind in ("color", "shape"):
            for role in ("teacher", "imposter"):
                main_items.append(make_main_item(rule, idx, other_kind, role))
    controls = make_controls(rules, base_index_by_key)
    all_items = main_items + controls

    write_jsonl(out / "main_100_full.jsonl", main_items)
    write_jsonl(out / "main_100_public.jsonl", map(public_view, main_items))
    write_jsonl(out / "main_100_gold.jsonl", map(gold_view, main_items))
    diagnosis_controls = [x for x in controls if x["trial_type"] == "control_no_history_diagnosis"]
    choice_controls = [x for x in controls if x["trial_type"] == "control_no_history_choice"]
    write_jsonl(out / "controls_no_history_diagnosis_10_public.jsonl", map(public_view, diagnosis_controls))
    write_jsonl(out / "controls_no_history_diagnosis_10_gold.jsonl", map(gold_view, diagnosis_controls))
    write_jsonl(out / "controls_no_history_choice_10_public.jsonl", map(public_view, choice_controls))
    write_jsonl(out / "controls_no_history_choice_10_gold.jsonl", map(gold_view, choice_controls))
    write_jsonl(out / "all_120_public.jsonl", map(public_view, all_items))
    write_jsonl(out / "all_120_gold.jsonl", map(gold_view, all_items))

    # Optional ablations for the same 100 main items. These do NOT add benchmark items.
    choice_only = [{**public_view(x), "prompt": x["prompt_choice_only"], "trial_type": "ablation_choice_only"} for x in main_items]
    diagnosis_only = [{**public_view(x), "prompt": x["prompt_diagnosis_only"], "trial_type": "ablation_diagnosis_only"} for x in main_items]
    action_given = [{**public_view(x), "prompt": x["prompt_action_given_rule"], "trial_type": "ablation_action_given_rule"} for x in main_items]
    write_jsonl(out / "optional_ablation_choice_only_100.jsonl", choice_only)
    write_jsonl(out / "optional_ablation_diagnosis_only_100.jsonl", diagnosis_only)
    write_jsonl(out / "optional_ablation_action_given_rule_100.jsonl", action_given)

    write_base_rules(out.parent, rules)

    metadata = {
        "benchmark": BENCHMARK_NAME,
        "version": BENCHMARK_VERSION,
        "colors": COLORS,
        "shapes": SHAPES,
        "num_base_rules": len(rules),
        "num_main_trials": len(main_items),
        "num_no_history_diagnosis_controls": len(diagnosis_controls),
        "num_no_history_choice_controls": len(choice_controls),
        "num_total_trials": len(all_items),
        "main_design": "25 base rules × 2 inferred other-person rules × 2 roles",
        "main_response_schema": {"inferred_rule": "COLOR_RULE|SHAPE_RULE", "answer": "A|B|C|D"},
        "control_response_schemas": {
            "diagnosis": {"inferred_rule": "COLOR_RULE|SHAPE_RULE|UNKNOWN"},
            "choice": {"answer": "A|B|C|D|INSUFFICIENT"},
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
