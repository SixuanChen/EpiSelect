#!/usr/bin/env python3
"""Build the `assistant_framing` condition: the teacher task with the teaching
framing removed.

Same 50 teacher items, same histories, same options, same gold answers. What
changes is only how the task is described:

  main_joint          "You are the TEACHER. Choose one object to show as the next
                       POSITIVE EXAMPLE ... help the learner revise their current
                       rule ... shows something their current rule is missing."
  assistant_framing   "You are a helpful assistant. You and the user are taking
                       turns choosing examples that fit a category rule ... Now it
                       is your turn. Choose one object to share with the user."

No role label, no instruction to teach, no request to infer the rule. The
question it answers: does a model select the informative example *spontaneously*,
or only when told that teaching is the goal? Scored against the same gold, so the
difference in accuracy is attributable to framing alone.

Only the 50 teacher items are used. Imposter items carry a private rule that
conflicts with the shared-goal framing, which has no natural analogue here.

The framing asserts that both examples on each user turn were valid. That is true
of every history round in the benchmark, and is checked at build time rather than
assumed.

Usage:
  python scripts/make_assistant_condition.py
  python scripts/make_assistant_condition.py --verbatim   # no JSON format line
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SYSTEM = ("Follow the user's instructions exactly. Return only the requested "
          "JSON object and no other text.")
ANSWER_FORMAT = ('Return exactly one JSON object and no other text: '
                 '{"answer":"A" or "B" or "C" or "D"}')


def satisfies(obj: dict, rule: dict) -> bool:
    return obj["color"] == rule["color"] or obj["shape"] == rule["shape"]


def build_prompt(item: dict, verbatim: bool) -> str:
    cr = item["category_rule"]
    lines = [
        "You are a helpful assistant.",
        "You and the user are taking turns choosing examples that fit a category rule.",
        f"The CATEGORY RULE is: {cr['text']}.",
        "On each user turn, two valid examples were available and the user chose "
        "one to share.",
    ]
    for h in item["history"]:
        left, right = h["options"]
        lines.append(f"Turn {h['trial']}: available examples were {left['text']} "
                     f"| {right['text']}.")
        lines.append(f"User chose: {h['selected_text']}.")
    lines.append("Now it is your turn. Choose one object to share with the user:")
    for opt in item["target_options"]:
        lines.append(f"{opt['label']}) {opt['text']}")
    prompt = "\n".join(lines)
    if not verbatim:
        # Every other condition ends with this line, and the scorer reads a JSON
        # `answer` key. Without it the framing AND the response format would both
        # differ, and the comparison would no longer isolate the framing.
        prompt += "\n\n" + ANSWER_FORMAT
    return prompt


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", type=Path,
                    default=REPO_ROOT / "benchmark" / "main_100_full.jsonl")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "requests" / "assistant_framing_50_requests.jsonl")
    ap.add_argument("--public", type=Path,
                    default=REPO_ROOT / "benchmark" / "assistant_framing_50_public.jsonl")
    ap.add_argument("--verbatim", action="store_true",
                    help="omit the JSON format line, leaving the prompt exactly as "
                         "written; responses then need free-text answer extraction")
    args = ap.parse_args()

    items = [json.loads(l) for l in args.full.read_text(encoding="utf-8").splitlines() if l.strip()]
    teacher = [r for r in items if r.get("role") == "teacher"]
    if not teacher:
        raise SystemExit(f"no teacher-role items found in {args.full}")

    reqs, public = [], []
    for item in teacher:
        cr = item["category_rule"]
        # The prompt claims both options each turn were valid; verify, do not assume.
        for h in item["history"]:
            for o in h["options"]:
                if not satisfies(o, cr):
                    raise SystemExit(
                        f"{item['item_id']} turn {h['trial']}: {o['text']} does not "
                        f"satisfy the category rule, so 'two valid examples' would "
                        f"be false")
        prompt = build_prompt(item, args.verbatim)
        reqs.append({
            "item_id": item["item_id"],
            "trial_type": "assistant_framing",
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}],
        })
        public.append({
            "item_id": item["item_id"],
            "trial_type": "assistant_framing",
            "base_id": item["base_id"],
            "role": item["role"],
            "gold_other_rule": item["gold_other_rule"],
            "prompt": prompt,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.public.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in reqs) + "\n", encoding="utf-8")
    args.public.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in public) + "\n", encoding="utf-8")

    print(f"{len(reqs)} requests -> {args.out}")
    print(f"{len(public)} public   -> {args.public}")
    print(f"format line: {'omitted (--verbatim)' if args.verbatim else 'appended'}")
    print(f"\nscore against benchmark/main_100_gold.jsonl with "
          f"score_ablations.py --mode choice_only\n")
    print("--- example ---")
    print(reqs[0]["messages"][1]["content"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
