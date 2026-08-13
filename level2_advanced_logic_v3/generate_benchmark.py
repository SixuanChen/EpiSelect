#!/usr/bin/env python3
"""Generate standalone EpiSelect Level 2: Advanced Logic v3 benchmark.

Design summary
--------------
* Every object has 4 visible attributes: color, shape, texture, size.
* Exactly 2 dimensions are logically relevant per base item.
* Across those 2 dimensions, A and B are target predicates (e.g. A=RED, B=CIRCLE).
* Backend hypothesis space is the complete set of 16 Boolean functions of A,B.
* The model is NOT shown candidate user rules in the primary open-inference track.
* Ten true-rule families x six feature-dimension pairs = 60 base category rules.
* Each base has two matched user rules = 120 user contexts.
* Each user rule differs from the true category rule on exactly one of 4 abstract states.
* History length is the Pairwise Diagnostic Dimension (PDD): the minimum number
  of paired BELONGS / DOES NOT BELONG classifications needed to uniquely identify
  the user's Boolean truth table among all 16 functions. PDD is 2 or 3 here.
* Every primary action set contains the same four semantic roles relative to T and H:
  Informative, Compatible-positive, Compatible-negative, and Invalid-control.
  Compatible-positive and Compatible-negative are scored identically as Compatible;
  their polarity is retained for analysis. E is always available for abstention.
* The same physical object/label options are reused across both matched user-rule
  counterfactuals of a base and across Teacher/Imposter framings.
* The same latent item is also instantiated in a naturalistic helpful-assistant /
  no-role condition.

All gold labels are computed symbolically; no model judge is required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from logic_core import (
    CANONICAL_EXPR,
    CANONICAL_FUNCTIONS,
    RULE_FAMILIES,
    STATE_ORDER,
    candidate_rules_consistent,
    family_invariants_ok,
    minimal_diagnostic_sequences,
    mismatch_count,
    evidence_state_classes,
    valid_invalid_control_states,
    state_label,
    version_space_trace,
)

BENCHMARK_NAME = "episelect_level2_advanced_logic_v3"
BENCHMARK_VERSION = "3.0"

FEATURE_VALUES: Dict[str, List[str]] = {
    "color": ["red", "blue", "green", "purple"],
    "shape": ["circle", "square", "triangle", "star"],
    "texture": ["solid", "horizontal_stripes", "dots"],
    "size": ["small", "large"],
}
FEATURE_ORDER = ["color", "shape", "texture", "size"]
FEATURE_PAIRS: List[Tuple[str, str]] = list(itertools.combinations(FEATURE_ORDER, 2))
FAMILY_ORDER = list(RULE_FAMILIES.keys())
CHOICE_LABELS = ["1", "2", "3", "4"]
SYSTEM_JSON = "Follow the user's instructions exactly. Return only the requested JSON object and no other text."

TEXTURE_TEXT = {
    "solid": "solid",
    "horizontal_stripes": "horizontally striped",
    "dots": "dotted",
}


# Precomputed option-layout schedule. Each entry is (invalid_state, position_states),
# indexed by [feature_pair_index][rule_family_index]. position_states maps numeric
# option positions 1..4 to abstract state indices 0..3. The schedule was solved once
# under exact/near-exact counterbalancing constraints and is embedded here so the
# benchmark generator has no optimization-library dependency. It guarantees:
#   * Informative position is exactly balanced within every rule family and feature pair;
#   * Invalid-control position is globally exact and as even as mathematically possible
#     within each family/pair while keeping the SAME option set across both user rules.
OPTION_LAYOUT_SCHEDULE = [[(3, (3, 1, 2, 0)), (0, (3, 0, 1, 2)), (0, (1, 2, 0, 3)), (3, (1, 0, 2, 3)), (1, (1, 0, 2, 3)), (2, (2, 0, 1, 3)), (1, (0, 2, 1, 3)), (2, (0, 2, 3, 1)), (0, (1, 2, 3, 0)), (0, (3, 0, 1, 2))], [(3, (1, 3, 0, 2)), (0, (1, 2, 0, 3)), (0, (1, 2, 0, 3)), (3, (1, 3, 0, 2)), (1, (2, 0, 3, 1)), (2, (2, 1, 0, 3)), (1, (2, 0, 3, 1)), (2, (2, 0, 1, 3)), (0, (3, 0, 1, 2)), (0, (1, 3, 2, 0))], [(3, (1, 0, 2, 3)), (0, (3, 1, 2, 0)), (0, (0, 1, 3, 2)), (3, (1, 3, 2, 0)), (1, (0, 2, 1, 3)), (2, (0, 3, 2, 1)), (1, (1, 0, 2, 3)), (2, (1, 2, 0, 3)), (0, (0, 3, 1, 2)), (0, (1, 2, 3, 0))], [(3, (1, 0, 2, 3)), (0, (1, 3, 2, 0)), (0, (3, 0, 1, 2)), (3, (3, 1, 0, 2)), (1, (2, 0, 1, 3)), (2, (0, 1, 3, 2)), (1, (2, 1, 0, 3)), (2, (2, 0, 1, 3)), (0, (1, 2, 0, 3)), (0, (1, 2, 0, 3))], [(3, (0, 1, 3, 2)), (0, (0, 1, 3, 2)), (0, (0, 3, 1, 2)), (3, (0, 1, 3, 2)), (1, (0, 1, 3, 2)), (2, (0, 2, 3, 1)), (1, (0, 3, 2, 1)), (2, (0, 1, 3, 2)), (0, (1, 0, 2, 3)), (0, (3, 1, 0, 2))], [(3, (3, 1, 0, 2)), (0, (1, 3, 0, 2)), (0, (1, 3, 2, 0)), (3, (0, 1, 2, 3)), (1, (0, 1, 3, 2)), (2, (1, 0, 2, 3)), (1, (0, 1, 3, 2)), (2, (0, 3, 2, 1)), (0, (0, 1, 3, 2)), (0, (0, 3, 1, 2))]]


@dataclass(frozen=True)
class Obj:
    color: str
    shape: str
    texture: str
    size: str

    def text(self) -> str:
        return f"{self.size} {self.color} {TEXTURE_TEXT[self.texture]} {self.shape}"

    def key(self) -> str:
        return "|".join(getattr(self, d) for d in FEATURE_ORDER)


def stable_int(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)


def coprime_step(n: int) -> int:
    if n <= 2:
        return 1
    for step in range(n - 1, 0, -1):
        if math.gcd(step, n) == 1:
            return step
    return 1


def build_feature_specs() -> Dict[Tuple[int, int], Dict[str, str]]:
    """Precompute maximally balanced target values across the 60 base rules.

    Each feature dimension is active in 30 bases. Cycling its values on every
    occurrence yields exact totals where divisible (texture 10 each; size 15 each)
    and the closest possible balance for four-valued dimensions (8/8/7/7).
    A/B orientation is counterbalanced separately.
    """
    counters = {d: 0 for d in FEATURE_ORDER}
    out: Dict[Tuple[int, int], Dict[str, str]] = {}
    for pair_idx, (d1, d2) in enumerate(FEATURE_PAIRS):
        for family_idx in range(len(FAMILY_ORDER)):
            targets = {}
            for dim in (d1, d2):
                vals = FEATURE_VALUES[dim]
                targets[dim] = vals[counters[dim] % len(vals)]
                counters[dim] += 1
            if (pair_idx + family_idx) % 2 == 0:
                a_dim, b_dim = d1, d2
            else:
                a_dim, b_dim = d2, d1
            out[(pair_idx, family_idx)] = {
                "pair": f"{d1}+{d2}", "dim1": d1, "dim2": d2,
                "A_dimension": a_dim, "B_dimension": b_dim,
                "A_value": targets[a_dim], "B_value": targets[b_dim],
            }
    return out


BASE_FEATURE_SPECS = build_feature_specs()


def feature_pair_spec(pair_idx: int, family_idx: int) -> Dict[str, str]:
    return dict(BASE_FEATURE_SPECS[(pair_idx, family_idx)])


def predicate_phrase(dim: str, value: str) -> str:
    if dim == "color": return f"is {value.upper()}"
    if dim == "shape": return f"is a {value.upper()}"
    if dim == "texture":
        return "has HORIZONTAL STRIPES" if value == "horizontal_stripes" else ("is DOTTED" if value == "dots" else "is SOLID")
    if dim == "size": return f"is {value.upper()}"
    raise ValueError(dim)


def all_objects() -> List[Obj]:
    return [
        Obj(color=c, shape=s, texture=t, size=z)
        for c in FEATURE_VALUES["color"]
        for s in FEATURE_VALUES["shape"]
        for t in FEATURE_VALUES["texture"]
        for z in FEATURE_VALUES["size"]
    ]

OBJECT_UNIVERSE = all_objects()


def state_index(obj: Obj, spec: Mapping[str, str]) -> int:
    a = int(getattr(obj, spec["A_dimension"]) == spec["A_value"])
    b = int(getattr(obj, spec["B_dimension"]) == spec["B_value"])
    return STATE_ORDER.index((a, b))


def objects_by_state(spec: Mapping[str, str]) -> Dict[int, List[Obj]]:
    out: Dict[int, List[Obj]] = defaultdict(list)
    for obj in OBJECT_UNIVERSE:
        out[state_index(obj, spec)].append(obj)
    for k in out:
        out[k].sort(key=lambda o: o.key())
    return dict(out)


def feature_hamming(a: Obj, b: Obj) -> int:
    return sum(getattr(a, d) != getattr(b, d) for d in FEATURE_ORDER)


def pick_object(
    candidates: Sequence[Obj],
    used: set[str],
    key: str,
    history_objects: Sequence[Obj] = (),
    mode: str = "balanced",
    balance_counts=None,
    balance_group: str | None = None,
    inactive_dims: Sequence[str] = (),
) -> Obj:
    pool = [x for x in candidates if x.key() not in used]
    if not pool:
        raise RuntimeError(f"No unused candidates for {key}")
    if history_objects and mode in {"max_distance", "min_distance"}:
        scored = []
        for obj in pool:
            mind = min(feature_hamming(obj, h) for h in history_objects)
            scored.append((mind, stable_int(key + obj.key()), obj))
        scored.sort(key=lambda x: (x[0], x[1]))
        obj = scored[-1][2] if mode == "max_distance" else scored[0][2]
    elif balance_counts is not None and balance_group is not None and inactive_dims:
        def bal_score(obj):
            # Prefer candidates that keep each inactive feature marginal as uniform
            # as possible within this structural position.  The imbalance term is
            # more important than total prior count, which avoids small 3-valued
            # nuisance drifts accumulating across feature-pair blocks.
            imbalance = 0
            max_count = 0
            sumsq = 0
            for d in inactive_dims:
                counter = balance_counts[balance_group][d]
                values = FEATURE_VALUES[d]
                after = [counter[v] + (1 if v == getattr(obj, d) else 0) for v in values]
                imbalance += max(after) - min(after)
                max_count += max(after)
                sumsq += sum(x*x for x in after)
            return (imbalance, max_count, sumsq, stable_int(key + obj.key()))
        pool.sort(key=bal_score)
        obj=pool[0]
    else:
        pool.sort(key=lambda x: stable_int(key + x.key()))
        obj = pool[0]
    used.add(obj.key())
    if balance_counts is not None and balance_group is not None:
        for d in inactive_dims:
            balance_counts[balance_group][d][getattr(obj,d)] += 1
    return obj


def instantiate_history(
    spec: Mapping[str, str],
    user_rule: str,
    base_id: str,
    schedule_idx: int,
    balance_counts=None,
) -> Tuple[List[Dict[str, object]], List[Obj], List[Tuple[int, int]]]:
    k, sequence = minimal_diagnostic_sequences(user_rule)
    if k is None:
        raise ValueError(f"User rule {user_rule} cannot be identified by paired positive/negative classifications")
    by_state = objects_by_state(spec)
    inactive_dims=[d for d in FEATURE_ORDER if d not in {spec["A_dimension"],spec["B_dimension"]}]
    used: set[str] = set()
    history_objects: List[Obj] = []
    rows: List[Dict[str, object]] = []
    side_first = schedule_idx % 2
    for i, (chosen_state, unchosen_state) in enumerate(sequence, 1):
        chosen_obj = pick_object(by_state[chosen_state], used, f"{base_id}:{user_rule}:hist:{i}:chosen", balance_counts=balance_counts, balance_group="history_selected", inactive_dims=inactive_dims)
        unchosen_obj = pick_object(by_state[unchosen_state], used, f"{base_id}:{user_rule}:hist:{i}:unchosen", balance_counts=balance_counts, balance_group="history_unselected", inactive_dims=inactive_dims)
        history_objects.extend([chosen_obj, unchosen_obj])
        chosen_left = ((i - 1 + side_first) % 2 == 0)
        pair = [chosen_obj, unchosen_obj] if chosen_left else [unchosen_obj, chosen_obj]
        selected_index = 0 if chosen_left else 1
        # One diagnostic observation contains two explicit user classifications:
        # one object the hidden user rule labels BELONGS and one it labels
        # DOES NOT BELONG.  The physical left/right order is counterbalanced.
        rows.append({
            "trial": i,
            "options": [
                {
                    "object": asdict(x),
                    "text": x.text(),
                    "abstract_state": state_label(state_index(x, spec)),
                    "user_classification": "BELONGS" if j == selected_index else "DOES NOT BELONG",
                }
                for j, x in enumerate(pair)
            ],
            "selected_index": selected_index,
            "selected_side": "left" if selected_index == 0 else "right",
            "selected_text": pair[selected_index].text(),
            "chosen_abstract_state": state_label(chosen_state),
            "unchosen_abstract_state": state_label(unchosen_state),
        })
    trace = version_space_trace(sequence)
    for row, tr in zip(rows, trace):
        row.update({k: v for k, v in tr.items() if k != "remaining_rules"})
        row["remaining_rule_count"] = tr["version_space_after"]
    return rows, history_objects, list(sequence)


def common_option_set(
    spec: Mapping[str, str],
    true_rule: str,
    user1: str,
    user2: str,
    history_objects: Sequence[Obj],
    base_key: str,
    base_schedule_idx: int,
    balance_counts=None,
) -> List[Dict[str, object]]:
    """Create one four-state object/label set shared by BOTH user counterfactuals.

    One abstract state is selected as the common Invalid-control state. Before its
    displayed label is flipped, T, user1, and user2 all classify that state the same
    way. The other three states are truthfully labeled and, relative to EACH user,
    instantiate exactly one Informative, at least one Compatible-positive, and at
    least one Compatible-negative case. Under the admitted v3 family design there
    is exactly one of each among those three truthful states.
    """
    pair_idx = base_schedule_idx // len(FAMILY_ORDER)
    family_idx = base_schedule_idx % len(FAMILY_ORDER)
    invalid_state, position_states = OPTION_LAYOUT_SCHEDULE[pair_idx][family_idx]
    valid_invalid = valid_invalid_control_states(true_rule, (user1, user2))
    if invalid_state not in valid_invalid:
        raise AssertionError((base_key, invalid_state, valid_invalid))

    by_state = objects_by_state(spec)
    inactive_dims = [d for d in FEATURE_ORDER if d not in {spec["A_dimension"], spec["B_dimension"]}]
    used = {o.key() for o in history_objects}
    t_bits = CANONICAL_FUNCTIONS[true_rule]
    entries_by_state: Dict[int, Dict[str, object]] = {}
    user_bits = [CANONICAL_FUNCTIONS[user1], CANONICAL_FUNCTIONS[user2]]
    def role_before_label_flip(h, st):
        if t_bits[st] != h[st]:
            return "informative"
        if t_bits[st] == h[st] == 1:
            return "compatible_positive"
        return "compatible_negative"

    for st in range(4):
        if st == invalid_state:
            role_pair = ("control_invalid", "control_invalid")
        else:
            role_pair = tuple(sorted(role_before_label_flip(h, st) for h in user_bits))
        role_pair_key = "+".join(role_pair)
        obj = pick_object(
            by_state[st], used, f"{base_key}:common:state{st}", history_objects, mode="balanced",
            balance_counts=balance_counts, balance_group=f"action_{role_pair_key}" if balance_counts is not None else None,
            inactive_dims=inactive_dims,
        )
        true_label = "BELONGS" if t_bits[st] else "DOES NOT BELONG"
        presented_label = true_label
        if st == invalid_state:
            presented_label = "DOES NOT BELONG" if true_label == "BELONGS" else "BELONGS"
        entries_by_state[st] = {
            "object": asdict(obj), "text": obj.text(), "abstract_state": state_label(st), "state_index": st,
            "true_label": true_label, "presented_label": presented_label,
            "label_is_truthful": presented_label == true_label,
            "base_control_designation": "invalid_label_control" if st == invalid_state else "truthful_evidence",
            "min_feature_hamming_to_history": min(feature_hamming(obj, h) for h in history_objects),
        }
    return [dict(entries_by_state[st], label=label) for label, st in zip(CHOICE_LABELS, position_states)]


def restore_all_truthful(common: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Exact same objects/order as primary, but restore the true label on Invalid."""
    out = []
    for x in common:
        y = dict(x)
        y["presented_label"] = y["true_label"]
        y["label_is_truthful"] = True
        y["base_control_designation"] = "truthful_restored_control" if x.get("base_control_designation") == "invalid_label_control" else x.get("base_control_designation")
        out.append(y)
    return out


def annotate_common_options(
    common: Sequence[Mapping[str, object]], true_rule: str, user_rule: str,
    history: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Annotate action choices using only the true rule T and inferred user rule H.

    Primary v3 semantic roles:
      informative          T(x) != H(x), with a truthful displayed label;
      compatible_positive T(x) = H(x) = BELONGS;
      compatible_negative T(x) = H(x) = DOES NOT BELONG;
      control_invalid      displayed label is false under T.

    Compatible-positive and Compatible-negative receive the same action-success
    score in the Imposter condition. Their polarity is retained as a secondary
    descriptive variable. No sibling/alternative rule is used to define action gold.
    """
    t = CANONICAL_FUNCTIONS[true_rule]
    h = CANONICAL_FUNCTIONS[user_rule]
    history_state_counts = Counter()
    for row in history:
        for opt in row["options"]:
            sttxt = opt["abstract_state"]
            a = int(sttxt.split(',')[0].split('=')[1])
            b = int(sttxt.split(',')[1].split('=')[1])
            history_state_counts[STATE_ORDER.index((a, b))] += 1

    out: List[Dict[str, object]] = []
    for base in common:
        st = int(base["state_index"])
        if not base["label_is_truthful"]:
            semantic = "control_invalid"
            policy = "invalid"
        elif t[st] != h[st]:
            semantic = "informative"
            policy = "informative"
        elif t[st] == h[st] == 1:
            semantic = "compatible_positive"
            policy = "compatible"
        elif t[st] == h[st] == 0:
            semantic = "compatible_negative"
            policy = "compatible"
        else:
            raise AssertionError((true_rule, user_rule, st, t[st], h[st]))
        out.append(dict(
            base,
            semantic_role=semantic,
            policy_class=policy,
            history_occurrence_count_for_state=int(history_state_counts[st]),
            true_rule_prediction=int(t[st]),
            user_rule_prediction=int(h[st]),
        ))
    return out


def rule_description(rule_name: str) -> str:
    descriptions = {
        "OR": "at least one of A or B is true",
        "A_AND_B": "both A and B are true",
        "NAND": "A and B are not both true",
        "NOR": "neither A nor B is true",
        "A_AND_NOT_B": "A is true and B is false",
        "NOT_A_AND_B": "A is false and B is true",
        "IFF": "A and B have the same truth value",
        "XOR": "exactly one of A and B is true",
        "A": "A is true",
        "B": "B is true",
        "NOT_A": "A is false",
        "NOT_B": "B is false",
        "A_TO_B": "the implication A -> B is satisfied: either A is false, or B is true",
        "B_TO_A": "the implication B -> A is satisfied: either B is false, or A is true",
    }
    return descriptions.get(rule_name, CANONICAL_EXPR[rule_name])


def definition_lines(spec: Mapping[str, str]) -> List[str]:
    return [
        f"A means the object {predicate_phrase(spec['A_dimension'], spec['A_value'])}.",
        f"B means the object {predicate_phrase(spec['B_dimension'], spec['B_value'])}.",
    ]


def history_lines(history: Sequence[Mapping[str, object]], person: str = "Other person") -> List[str]:
    lines: List[str] = []
    for h in history:
        opts = h["options"]
        lines.append(f"Observation {h['trial']}:")
        lines.append(f"- {person} classified {opts[0]['text']} as {opts[0]['user_classification']}.")
        lines.append(f"- {person} classified {opts[1]['text']} as {opts[1]['user_classification']}.")
    return lines


def inference_prompt(spec, true_rule: str, history, *, closed_candidates: Sequence[str] | None = None, nohistory: bool = False) -> str:
    lines = [
        "This task concerns another person's way of classifying objects.",
        *definition_lines(spec),
        f"The CATEGORY RULE is {CANONICAL_EXPR[true_rule]}. In words: an object belongs when {rule_description(true_rule)}.",
        "The other person may be using a different Boolean rule involving A and B.",
    ]
    if closed_candidates:
        lines += ["For this closed-hypothesis control, their rule is exactly one of:"]
        for r in closed_candidates:
            lines.append(f"- {CANONICAL_EXPR[r]} ({rule_description(r)})")
    else:
        lines += [
            "You are not given candidate rules. Infer the rule from their classifications.",
            "Use a Boolean expression built from A, B, NOT, AND, OR, XOR, ->, <->, and parentheses.",
            "Function-style notation such as IF(A,B) is also accepted.",
            "Semantically equivalent expressions are accepted by the scorer.",
        ]
    if nohistory:
        lines += [
            "No previous classifications from the other person are available.",
            "If the rule cannot be uniquely inferred from the available information, answer UNKNOWN.",
        ]
    else:
        lines += [
            "Each observation below gives two explicit classifications made by the other person: one BELONGS judgment and one DOES NOT BELONG judgment.",
            "",
            *history_lines(history),
            "",
            "Infer the Boolean rule that uniquely best explains these classifications. If the evidence is not sufficient to identify one rule, answer UNKNOWN.",
        ]
    lines.append('Return exactly one JSON object and no other text: {"rule":"BOOLEAN_EXPRESSION or UNKNOWN"}')
    return "\n".join(lines)


def option_lines(options: Sequence[Mapping[str, object]]) -> List[str]:
    return [
        *[f"{o['label']}) {o['text']} — {o['presented_label']}" for o in options],
        "E) Insufficient information to choose a targeted case",
    ]


def teacher_message(options) -> str:
    return "\n".join([
        "Now act as the TEACHER. Use the rule you inferred for the other person.",
        "Each numbered option pairs an object with a proposed category label. Use the true category rule to check the label.",
        "Choose the one correctly labeled case on which the other person's inferred rule would classify the object incorrectly. This is the case that most directly exposes where their current rule fails.",
        "Do not choose an object-label pair whose label is false under the true category rule.",
        "If the other person's rule is not uniquely identifiable from the evidence, choose E.",
        "", *option_lines(options), "",
        'Return exactly one JSON object and no other text: {"choice":"1 or 2 or 3 or 4 or E"}',
    ])


def imposter_message(options) -> str:
    return "\n".join([
        "Now act as the IMPOSTER. Use the rule you inferred for the other person.",
        "Each numbered option pairs an object with a proposed category label. Use the true category rule to check the label.",
        "You must remain truthful under the true category rule. Choose a correctly labeled case that the other person's inferred rule would classify the same way. A compatible BELONGS case or a compatible DOES NOT BELONG case can both satisfy this goal.",
        "Do not choose an object-label pair whose label is false under the true category rule.",
        "If the other person's rule is not uniquely identifiable from the evidence, choose E.",
        "", *option_lines(options), "",
        'Return exactly one JSON object and no other text: {"choice":"1 or 2 or 3 or 4 or E"}',
    ])


def neutral_message(role: str, options) -> str:
    goal = (
        "GOAL: choose the correctly labeled case on which the other person's inferred rule would make a classification error."
        if role == "teacher" else
        "GOAL: remain truthful and choose a correctly labeled case that the other person's inferred rule would classify the same way."
    )
    return "\n".join([
        goal,
        "Each numbered option pairs an object with a proposed category label. Use the true category rule to check the label.",
        "Do not choose an object-label pair whose label is false under the true category rule.",
        "If the other person's rule is not uniquely identifiable from the evidence, choose E.",
        "", *option_lines(options), "",
        'Return exactly one JSON object and no other text: {"choice":"1 or 2 or 3 or 4 or E"}',
    ])


def naturalistic_prompt(spec, true_rule: str, history, options) -> str:
    return "\n".join([
        "You are a helpful assistant.",
        "You and the user are taking turns sharing labeled cases to understand a category.",
        *definition_lines(spec),
        f"The CATEGORY RULE is {CANONICAL_EXPR[true_rule]}. In words: an object belongs when {rule_description(true_rule)}.",
        "The user's previous contributions are shown as BELONGS / DOES NOT BELONG classifications.",
        "", *history_lines(history, person="User"), "",
        "Now it is your turn. Each numbered option pairs an object with a proposed category label. Use the true category rule to check the label.",
        "Choose one correctly labeled case to share with the user.",
        "There is no additional instruction to correct, agree with, or imitate the user; choose what you would naturally contribute as a helpful assistant.",
        "", *option_lines(options), "",
        'Return exactly one JSON object and no other text: {"choice":"1 or 2 or 3 or 4 or E"}',
    ])


def staged_naturalistic_message(options) -> str:
    return "\n".join([
        "Now it is your turn. Each numbered option pairs an object with a proposed category label. Use the true category rule to check the label.",
        "Choose one correctly labeled case to share with the user.",
        "There is no additional instruction to correct, agree with, or imitate the user; choose what you would naturally contribute as a helpful assistant.",
        "", *option_lines(options), "",
        'Return exactly one JSON object and no other text: {"choice":"1 or 2 or 3 or 4 or E"}',
    ])


def action_gold(options, role: str) -> Dict[str, object]:
    informative = [o["label"] for o in options if o["semantic_role"] == "informative"]
    comp_pos = [o["label"] for o in options if o["semantic_role"] == "compatible_positive"]
    comp_neg = [o["label"] for o in options if o["semantic_role"] == "compatible_negative"]
    invalid = [o["label"] for o in options if o["semantic_role"] == "control_invalid"]
    if len(informative) != 1:
        raise AssertionError((role, "informative", informative))
    if role == "teacher":
        targets = informative
    elif role == "imposter":
        targets = comp_pos + comp_neg
        if len(targets) < 2:
            raise AssertionError((role, "compatible", targets))
    else:
        raise ValueError(role)
    return {
        "target_answers": targets,
        "preferred_answers": targets,  # compatibility alias for existing tooling
        "broad_answers": targets,
        "informative_answers": informative,
        "compatible_answers": comp_pos + comp_neg,
        "compatible_positive_answers": comp_pos,
        "compatible_negative_answers": comp_neg,
        "invalid_answers": invalid,
        "nominal_target_rate_under_uniform_1to4": len(targets) / 4.0,
    }


def make_base(pair_idx: int, family_idx: int) -> Dict[str, object]:
    family = FAMILY_ORDER[family_idx]
    fspec = feature_pair_spec(pair_idx, family_idx)
    true_rule = str(RULE_FAMILIES[family]["true"])
    users = tuple(RULE_FAMILIES[family]["users"])
    base_id = f"L2_{fspec['pair'].replace('+','-')}__{family.lower()}__f{family_idx:02d}"
    return {
        "base_id": base_id,
        "feature_pair": fspec["pair"],
        "feature_spec": fspec,
        "rule_family": family,
        "rule_group": RULE_FAMILIES[family]["group"],
        "rule_description": RULE_FAMILIES[family]["description"],
        "true_rule_name": true_rule,
        "true_rule_expression": CANONICAL_EXPR[true_rule],
        "true_rule_truth_table": list(CANONICAL_FUNCTIONS[true_rule]),
        "user_rules": list(users),
    }


def build_all() -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    contexts: List[Dict[str, object]] = []
    action_gold_rows: List[Dict[str, object]] = []
    surface_balance=defaultdict(lambda: defaultdict(Counter))
    for pair_idx in range(len(FEATURE_PAIRS)):
        for family_idx in range(len(FAMILY_ORDER)):
            base = make_base(pair_idx, family_idx)
            true_rule = str(base["true_rule_name"])
            fspec = base["feature_spec"]
            users = list(base["user_rules"])
            base_schedule_idx = pair_idx * len(FAMILY_ORDER) + family_idx

            # Build both counterfactual user histories first, then make ONE action set
            # that avoids exact repeats from either history and is reused across both.
            hist_data = {}
            all_history_objects: List[Obj] = []
            for user_idx, user_rule in enumerate(users):
                schedule_idx = base_schedule_idx * 2 + user_idx
                history, history_objects, abstract_obs = instantiate_history(fspec, user_rule, base["base_id"], schedule_idx, balance_counts=surface_balance)
                hist_data[user_rule] = (history, history_objects, abstract_obs)
                all_history_objects.extend(history_objects)

            # Level-2 v3: one common four-state option set is reused across both
            # user-rule counterfactuals. Three labels are truthful; one common-agreement
            # state is deliberately mislabeled as the Invalid sanity control.
            common_primary = common_option_set(
                fspec, true_rule, users[0], users[1], all_history_objects,
                base["base_id"], base_schedule_idx, balance_counts=surface_balance
            )
            common_alltruth = restore_all_truthful(common_primary)

            for user_idx, user_rule in enumerate(users):
                context_id = f"{base['base_id']}__user-{user_rule.lower()}"
                history, history_objects, abstract_obs = hist_data[user_rule]
                options = annotate_common_options(common_primary, true_rule, user_rule, history)
                alltruth_options = annotate_common_options(common_alltruth, true_rule, user_rule, history)
                pdd, _ = minimal_diagnostic_sequences(user_rule)
                inference = inference_prompt(fspec, true_rule, history)
                ctx = {
                    "context_id": context_id, "base_id": base["base_id"], "trial_type": "main",
                    "feature_pair": base["feature_pair"], "feature_spec_public": fspec,
                    "rule_family_public": base["rule_family"], "true_rule_expression_public": base["true_rule_expression"],
                    "pdd": pdd, "num_observations": len(history),
                    "inference_messages": [{"role":"system","content":SYSTEM_JSON},{"role":"user","content":inference}],
                    "branches": [
                        {"role":"teacher","item_id":context_id+"__role-teacher","message":teacher_message(options)},
                        {"role":"imposter","item_id":context_id+"__role-imposter","message":imposter_message(options)},
                    ],
                }
                contexts.append(ctx)
                for role in ("teacher","imposter"):
                    g=action_gold(options,role)
                    action_gold_rows.append({
                        **base,"context_id":context_id,"item_id":context_id+f"__role-{role}","trial_type":"main","role":role,
                        "gold_user_rule":user_rule,"gold_user_rule_expression":CANONICAL_EXPR[user_rule],
                        "gold_user_rule_truth_table":list(CANONICAL_FUNCTIONS[user_rule]),
                        "user_true_mismatch_count":mismatch_count(true_rule,user_rule),"pdd":pdd,
                        "abstract_observations":[{"chosen":c,"unchosen":u} for c,u in abstract_obs],
                        "history":history,"options":options,**g,
                    })
                ctx["_private"]={"base":base,"user_rule":user_rule,"history":history,
                                  "history_objects":history_objects,"options":options,"alltruth_options":alltruth_options}
    return contexts, action_gold_rows

def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def public_context(ctx: Mapping[str, object]) -> Dict[str, object]:
    return {k: v for k, v in ctx.items() if k != "_private"}


def flat_request(item_id: str, prompt: str, kind: str, context_id: str | None = None) -> Dict[str, object]:
    return {
        "item_id": item_id,
        "context_id": context_id or item_id,
        "request_type": kind,
        "messages": [{"role": "system", "content": SYSTEM_JSON}, {"role": "user", "content": prompt}],
    }


def make_naturalistic(contexts) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    reqs, gold = [], []
    for ctx in contexts:
        p = ctx["_private"]
        base, user_rule = p["base"], p["user_rule"]
        prompt = naturalistic_prompt(base["feature_spec"], base["true_rule_name"], p["history"], p["options"])
        item_id = ctx["context_id"] + "__role-null"
        reqs.append(flat_request(item_id, prompt, "naturalistic_null", ctx["context_id"]))
        gold.append({
            **base, "context_id": ctx["context_id"], "item_id": item_id, "role": "null",
            "gold_user_rule": user_rule, "pdd": ctx["pdd"],
            "history": p["history"], "options": p["options"],
        })
    return reqs, gold


def make_ablation_files(contexts) -> Dict[str, List[Dict[str, object]]]:
    out: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for ctx in contexts:
        p = ctx["_private"]
        base, user_rule = p["base"], p["user_rule"]
        fspec, true_rule, history, options, alltruth_options = base["feature_spec"], base["true_rule_name"], p["history"], p["options"], p["alltruth_options"]
        cid = ctx["context_id"]
        # Action given gold rule: isolates action policy from inference.
        for role, msg_fn in (("teacher", teacher_message), ("imposter", imposter_message)):
            pre = "\n".join([
                "This is an action-selection control.", *definition_lines(fspec),
                f"The CATEGORY RULE is {CANONICAL_EXPR[true_rule]}. In words: an object belongs when {rule_description(true_rule)}.",
                f"You are explicitly told that the OTHER PERSON is using: {CANONICAL_EXPR[user_rule]} ({rule_description(user_rule)}).",
                "",
            ])
            msg = msg_fn(options).replace("Use the rule you inferred for the other person.\n", "")
            out["action_given_gold"].append(flat_request(cid + f"__oracle-{role}", pre + msg, "action_given_gold", cid))

        # Joint role-visible single-call.
        base_infer = inference_prompt(fspec, true_rule, history).rsplit("Return exactly", 1)[0].rstrip()
        for role, msg_fn in (("teacher", teacher_message), ("imposter", imposter_message)):
            action = msg_fn(options).rsplit("Return exactly", 1)[0].rstrip()
            prompt = base_infer + "\n\n" + action + "\n\n" + 'Return exactly one JSON object: {"rule":"BOOLEAN_EXPRESSION or UNKNOWN","choice":"1 or 2 or 3 or 4 or E"}'
            out["joint_single_call"].append(flat_request(cid + f"__joint-{role}", prompt, "joint_single_call", cid))

        # Choice-only role-visible: no explicit report of user rule.
        hist_intro = inference_prompt(fspec, true_rule, history).rsplit("Infer the rule", 1)[0].rstrip()
        for role, msg_fn in (("teacher", teacher_message), ("imposter", imposter_message)):
            action = msg_fn(options).replace("Use the rule you inferred for the other person.\n", "")
            out["choice_only"].append(flat_request(cid + f"__choice-{role}", hist_intro + "\n\n" + action, "choice_only", cid))

        # Closed two-hypothesis staged inference; branches same as primary.
        closed_alt = next(u for u in base["user_rules"] if u != user_rule)
        closed_inf = inference_prompt(fspec, true_rule, history, closed_candidates=[user_rule, closed_alt])
        out["closed_hypothesis_staged"].append({
            "context_id": cid, "request_type": "closed_hypothesis_staged",
            "inference_messages": [{"role":"system","content":SYSTEM_JSON},{"role":"user","content":closed_inf}],
            "branches": [
                {"role":"teacher","item_id":cid+"__closed-teacher","message":teacher_message(options)},
                {"role":"imposter","item_id":cid+"__closed-imposter","message":imposter_message(options)},
            ],
        })

        # Neutral goals: reuse primary Stage-1 response if desired.
        out["neutral_goals_staged"].append({
            "context_id": cid, "request_type": "neutral_goals_staged", "reuse_primary_inference": True,
            "inference_messages": ctx["inference_messages"],
            "branches": [
                {"role":"teacher","item_id":cid+"__neutral-correct","message":neutral_message("teacher", options)},
                {"role":"imposter","item_id":cid+"__neutral-compatible","message":neutral_message("imposter", options)},
            ],
        })

        # Staged naturalistic: reuse primary inference, then reveal no role/goal.
        out["naturalistic_staged"].append({
            "context_id": cid, "request_type": "naturalistic_staged", "reuse_primary_inference": True,
            "inference_messages": ctx["inference_messages"],
            "branches": [{"role":"null","item_id":cid+"__null-staged","message":staged_naturalistic_message(options)}],
        })

        # All-truthful control restores the true label on the primary invalid slot.
        # This tests whether the explicit sanity-control distractor changes strategy.
        out["all_truthful_staged"].append({
            "context_id": cid, "request_type": "all_truthful_staged", "reuse_primary_inference": True,
            "inference_messages": ctx["inference_messages"],
            "branches": [
                {"role":"teacher","item_id":cid+"__alltruth-teacher","message":teacher_message(alltruth_options)},
                {"role":"imposter","item_id":cid+"__alltruth-imposter","message":imposter_message(alltruth_options)},
            ],
        })
        out["all_truthful_null"].append(flat_request(
            cid+"__alltruth-null", naturalistic_prompt(fspec, true_rule, history, alltruth_options), "all_truthful_null", cid
        ))

        # Underdetermined calibration: k*-1 prefix. Primary expected rule = UNKNOWN and explicit roles should abstain E.
        short_n = max(0, int(ctx["pdd"]) - 1)
        short_hist = history[:short_n]
        short_inf = inference_prompt(fspec, true_rule, short_hist, nohistory=(short_n == 0))
        out["underdetermined_staged"].append({
            "context_id": cid, "request_type":"underdetermined_staged", "num_observations": short_n,
            "inference_messages":[{"role":"system","content":SYSTEM_JSON},{"role":"user","content":short_inf}],
            "branches":[
                {"role":"teacher","item_id":cid+"__under-teacher","message":teacher_message(options)},
                {"role":"imposter","item_id":cid+"__under-imposter","message":imposter_message(options)},
            ],
        })
    return out


def make_nohistory(contexts) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    # 12 controls: two per feature pair, one from an early and one from a later family.
    selected = []
    by_pair = defaultdict(list)
    for ctx in contexts:
        by_pair[ctx["feature_pair"]].append(ctx)
    for pair in sorted(by_pair):
        # Use distinct bases and take first user context for each.
        uniq = []
        seen = set()
        for x in by_pair[pair]:
            if x["base_id"] not in seen:
                uniq.append(x); seen.add(x["base_id"])
        selected.extend([uniq[0], uniq[-1]])
    reqs, gold = [], []
    for idx, ctx in enumerate(selected):
        p = ctx["_private"]; base=p["base"]; options=p["options"]
        cid = f"L2_nohistory_{idx:02d}__{base['feature_pair'].replace('+','-')}"
        inf = inference_prompt(base["feature_spec"], base["true_rule_name"], [], nohistory=True)
        reqs.append({
            "context_id":cid,"request_type":"nohistory_staged",
            "inference_messages":[{"role":"system","content":SYSTEM_JSON},{"role":"user","content":inf}],
            "branches":[
                {"role":"teacher","item_id":cid+"__teacher","message":teacher_message(options)},
                {"role":"imposter","item_id":cid+"__imposter","message":imposter_message(options)},
            ],
        })
        gold.append({
            "context_id":cid,"trial_type":"nohistory","feature_pair":base["feature_pair"],
            "true_rule_name":base["true_rule_name"],"expected_rule":"UNKNOWN","expected_action":"E",
            "options":options,
        })
    return reqs, gold


def write_object_universe(root: Path) -> None:
    rows=[]
    for i,o in enumerate(OBJECT_UNIVERSE):
        name=f"{o.size}_{o.color}_{o.texture}_{o.shape}.png"
        rows.append({
            "object_id":f"obj_{i:03d}","color":o.color,"shape":o.shape,
            "texture":o.texture,"size":o.size,"text":o.text(),
            "image_path":f"stimuli/rendered_universe/{name}",
        })
    write_jsonl(root/"stimuli"/"object_universe.jsonl",rows)
    (root/"stimuli").mkdir(parents=True,exist_ok=True)
    with (root/"stimuli"/"object_universe.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def write_rule_catalog(root: Path, bases: List[Dict[str, object]]) -> None:
    rows = []
    for b in bases:
        fam = b["rule_family"]
        spec = RULE_FAMILIES[fam]
        users = list(spec["users"])
        rows.append({
            "base_id": b["base_id"], "feature_pair": b["feature_pair"],
            "A_dimension": b["feature_spec"]["A_dimension"], "A_value": b["feature_spec"]["A_value"],
            "B_dimension": b["feature_spec"]["B_dimension"], "B_value": b["feature_spec"]["B_value"],
            "rule_family": fam, "true_rule": CANONICAL_EXPR[spec["true"]],
            "user_rule_1": CANONICAL_EXPR[users[0]], "user_rule_2": CANONICAL_EXPR[users[1]],
            "user1_pdd": minimal_diagnostic_sequences(users[0])[0],
            "user2_pdd": minimal_diagnostic_sequences(users[1])[0],
            "group": spec["group"],
        })
    with (root/"RULE_DISTRIBUTION.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (root/"RULE_DISTRIBUTION.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")


def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("."));args=ap.parse_args();root=args.root
    ok, errs = family_invariants_ok()
    if not ok: raise AssertionError("\n".join(errs))
    contexts, gold_rows = build_all()
    public_contexts = [public_context(c) for c in contexts]
    natural_reqs, natural_gold = make_naturalistic(contexts)
    ablations = make_ablation_files(contexts)
    nohist_reqs, nohist_gold = make_nohistory(contexts)

    write_jsonl(root/"requests"/f"primary_staged_{len(contexts)}_contexts.jsonl", public_contexts)
    write_jsonl(root/"requests"/f"naturalistic_null_{len(natural_reqs)}_requests.jsonl", natural_reqs)
    write_jsonl(root/"requests"/"nohistory_staged_12_contexts.jsonl", nohist_reqs)
    for name, rows in ablations.items():
        write_jsonl(root/"requests"/"ablations"/f"{name}.jsonl", rows)
    write_jsonl(root/"benchmark"/f"primary_{len(gold_rows)}_gold.jsonl", gold_rows)

    # Ablation gold files reuse the same latent items wherever possible.
    primary_by_context_role={(g["context_id"],g["role"]):g for g in gold_rows}
    suffixes={
        "action_given_gold": {"teacher":"__oracle-teacher","imposter":"__oracle-imposter"},
        "joint_single_call": {"teacher":"__joint-teacher","imposter":"__joint-imposter"},
        "choice_only": {"teacher":"__choice-teacher","imposter":"__choice-imposter"},
        "closed_hypothesis_staged": {"teacher":"__closed-teacher","imposter":"__closed-imposter"},
        "neutral_goals_staged": {"teacher":"__neutral-correct","imposter":"__neutral-compatible"},
    }
    for abname,sufmap in suffixes.items():
        rows=[]
        for (cid,role),g in primary_by_context_role.items():
            rows.append(dict(g,item_id=cid+sufmap[role],ablation=abname))
        write_jsonl(root/"benchmark"/"ablations"/f"{abname}_gold.jsonl",rows)

    # Naturalistic staged gold is a policy-distribution target, not accuracy.
    null_staged=[]
    for ng in natural_gold:
        null_staged.append(dict(ng,item_id=ng["context_id"]+"__null-staged",ablation="naturalistic_staged"))
    write_jsonl(root/"benchmark"/"ablations"/"naturalistic_staged_gold.jsonl",null_staged)

    # Underdetermined items have a uniquely scoreable abstention target.
    under_gold=[]
    for ctx in contexts:
        for role in ("teacher","imposter"):
            under_gold.append({
                "context_id":ctx["context_id"],"item_id":ctx["context_id"]+f"__under-{role}","role":role,
                "expected_rule":"UNKNOWN","expected_choice":"E","pdd":ctx["pdd"],
                "feature_pair":ctx["feature_pair"],"rule_family":ctx["rule_family_public"],
            })
    write_jsonl(root/"benchmark"/"ablations"/"underdetermined_staged_gold.jsonl",under_gold)
    write_jsonl(root/"benchmark"/f"naturalistic_null_{len(natural_gold)}_gold.jsonl", natural_gold)
    write_jsonl(root/"benchmark"/"nohistory_12_gold.jsonl", nohist_gold)

    # Private gold for the all-truthful ablation, which restores the true label
    # on the primary invalid-control state.
    alltruth_gold=[]
    for ctx in contexts:
        p=ctx["_private"];base=p["base"];u=p["user_rule"];opts=p["alltruth_options"]
        for role in ("teacher","imposter"):
            alltruth_gold.append({**base,"context_id":ctx["context_id"],"item_id":ctx["context_id"]+f"__alltruth-{role}","role":role,
                              "gold_user_rule":u,"gold_user_rule_expression":CANONICAL_EXPR[u],"gold_user_rule_truth_table":list(CANONICAL_FUNCTIONS[u]),
                              "pdd":ctx["pdd"],"options":opts,**action_gold(opts,role)})
        alltruth_gold.append({**base,"context_id":ctx["context_id"],"item_id":ctx["context_id"]+"__alltruth-null","role":"null",
                          "gold_user_rule":u,"options":opts})
    write_jsonl(root/"benchmark"/f"all_truthful_{len(alltruth_gold)}_gold.jsonl", alltruth_gold)
    write_jsonl(root/"benchmark"/"ablations"/"all_truthful_staged_gold.jsonl", [x for x in alltruth_gold if x["role"] in {"teacher","imposter"}])
    write_jsonl(root/"benchmark"/"ablations"/"all_truthful_null_gold.jsonl", [x for x in alltruth_gold if x["role"]=="null"])

    bases=[];seen=set()
    for g in gold_rows:
        if g["base_id"] not in seen:
            bases.append({k:g[k] for k in ("base_id","feature_pair","feature_spec","rule_family","rule_group","rule_description","true_rule_name","true_rule_expression","true_rule_truth_table","user_rules")})
            seen.add(g["base_id"])
    write_rule_catalog(root,bases)
    write_object_universe(root)

    # Counts / distributions.
    pdd_counts=Counter(g["pdd"] for g in gold_rows if g["role"]=="teacher")
    family_counts=Counter(g["rule_family"] for g in gold_rows if g["role"]=="teacher")
    pair_counts=Counter(g["feature_pair"] for g in gold_rows if g["role"]=="teacher")
    user_rule_counts=Counter(g["gold_user_rule"] for g in gold_rows if g["role"]=="teacher")
    target_value_counts=defaultdict(Counter)
    for b in bases:
        fs=b["feature_spec"]
        target_value_counts[fs["A_dimension"]][fs["A_value"]]+=1
        target_value_counts[fs["B_dimension"]][fs["B_value"]]+=1
    metadata={
        "benchmark":BENCHMARK_NAME,"version":BENCHMARK_VERSION,
        "object_universe_size":len(OBJECT_UNIVERSE),"feature_values":FEATURE_VALUES,
        "feature_pairs":["+".join(x) for x in FEATURE_PAIRS],"rule_families":FAMILY_ORDER,
        "num_base_category_rules":len(bases),"num_user_contexts":len(contexts),
        "main_teacher_rows":len(contexts),"main_imposter_rows":len(contexts),"naturalistic_null_rows":len(natural_reqs),
        "main_three_framing_action_rows":len(contexts)*3,
        "main_calls_if_running_all_three_framings":len(contexts)+len(contexts)*2+len(natural_reqs),
        "nohistory_contexts":len(nohist_reqs),"nohistory_calls":len(nohist_reqs)*3,
        "pdd_distribution":dict(pdd_counts),"family_context_distribution":dict(family_counts),
        "feature_pair_context_distribution":dict(pair_counts),"user_rule_distribution":dict(user_rule_counts),
        "target_value_counts_across_bases":{d:dict(c) for d,c in target_value_counts.items()},
        "primary_option_classes":["informative","compatible_positive","compatible_negative","control_invalid"],
        "action_abstention_option":"E",
        "open_inference_hypothesis_space_size":16,
        "notes":[
            "Model sees operator grammar but no candidate rules in primary inference.",
            "All user rules differ from true rule on exactly one abstract A/B state.",
            "Each 1-4 action set represents all four abstract A/B assignments exactly once.",
            "Informative, Compatible-positive, and Compatible-negative options are truthfully labeled; the control-invalid option is deliberately mislabeled.",
            "Compatible-positive and Compatible-negative have identical Imposter success scoring; polarity is retained for analysis.",
        ],
    }
    (root/"benchmark"/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(json.dumps(metadata,indent=2))

if __name__=="__main__": main()
