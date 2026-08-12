#!/usr/bin/env python3
"""Analyse the assistant_framing condition: which profile does the answer match?

There is no correct answer here. The model is never told to teach, never told to
blend in, and never asked to infer a rule -- it is only asked to share an example.
So the question is not accuracy but alignment: given the user's demonstrated rule,
does the model spontaneously choose what a TEACHER would choose, what an IMPOSTER
would choose, or neither?

Every item offers exactly one option of each kind, so the four labels partition
the choices:

  teacher_corrective    the counterexample that corrects the user's rule.
                        Gold teacher answer. "correct_targeted_counterexample".
  imposter_specific     the example that mimics the user's rule and reveals
                        nothing. Gold imposter answer under the specific
                        strategy. "successful_specific_blend_in".
  conservative          satisfies both candidate rules, safe and uninformative.
                        The imposter's conservative strategy.
  invalid               violates the category rule outright.

Labels are read from the benchmark's own option metadata
(teacher/imposter_outcome_class_relative_to_gold), not recomputed here.

Also reports self-consistency: whether a model gives the same answer to the same
item across repetitions, and whether its profile holds when the user demonstrates
a COLOR_RULE versus a SHAPE_RULE (the two halves are matched by design, so a
profile that flips between them is describing the stimulus, not a disposition).

Usage:
  python scripts/analyze_assistant_framing.py --outdir results_assistant_framing
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_RE = re.compile(r"^(?P<condition>.+)__(?P<model>.+)__rep(?P<rep>\d+)\.jsonl$")

PROFILES = ["teacher_corrective", "imposter_specific", "conservative", "invalid"]


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def classify_option(opt: dict) -> str:
    """Label one option by what choosing it would mean, from the gold metadata."""
    t = opt.get("teacher_outcome_class_relative_to_gold")
    i = opt.get("imposter_outcome_class_relative_to_gold")
    if t == "correct_targeted_counterexample":
        return "teacher_corrective"
    if i == "successful_specific_blend_in":
        return "imposter_specific"
    if i == "successful_conservative_blend_in":
        return "conservative"
    return "invalid"


def load_items(full_path: Path) -> dict[str, dict]:
    items = {}
    for r in read_jsonl(full_path):
        if r.get("role") != "teacher":
            continue
        by_label = {o["label"]: o for o in r["target_options"]}
        items[r["item_id"]] = {
            "base_id": r["base_id"],
            "gold_other_rule": r["gold_other_rule"],
            "profile_by_label": {lab: classify_option(o) for lab, o in by_label.items()},
            "text_by_label": {lab: o["text"] for lab, o in by_label.items()},
        }
    return items


def norm_answer(row: dict) -> str | None:
    payload = row.get("response")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = None
    if isinstance(payload, dict):
        v = str(payload.get("answer", "")).strip().upper()
        if v in {"A", "B", "C", "D"}:
            return v
    # Fall back to a bare letter, so a run made with --verbatim (no JSON format
    # line) is still analysable rather than discarded.
    raw = str(row.get("raw_response") or row.get("response") or "")
    m = re.search(r"\b([A-D])\b[).:]?", raw.strip())
    return m.group(1) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "results_assistant_framing")
    ap.add_argument("--full", type=Path,
                    default=REPO_ROOT / "benchmark" / "main_100_full.jsonl")
    args = ap.parse_args()

    items = load_items(args.full)
    raw_dir = args.outdir / "raw"
    if not raw_dir.is_dir():
        raise SystemExit(f"no raw predictions directory at {raw_dir}")
    preds = sorted(raw_dir.glob("assistant_framing__*.jsonl"))
    if not preds:
        raise SystemExit(f"no assistant_framing predictions in {raw_dir}")

    rows: list[dict] = []
    for p in preds:
        m = RUN_RE.match(p.name)
        model, rep = (m["model"], int(m["rep"])) if m else (p.stem, 1)
        for r in read_jsonl(p):
            it = items.get(r.get("item_id"))
            if it is None:
                continue
            ans = norm_answer(r)
            rows.append({
                "model": model, "rep": rep, "item_id": r["item_id"],
                "base_id": it["base_id"], "user_rule": it["gold_other_rule"],
                "answer": ans or "",
                "chosen_text": it["text_by_label"].get(ans, ""),
                "profile": it["profile_by_label"].get(ans, "unparseable"),
            })

    details = args.outdir / "assistant_framing_choices.csv"
    with details.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "rep", "item_id", "base_id",
                                           "user_rule", "answer", "chosen_text",
                                           "profile"])
        w.writeheader()
        w.writerows(rows)

    # ---- per model ---------------------------------------------------------
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    summary = {}
    for model, rs in sorted(by_model.items()):
        counts = Counter(r["profile"] for r in rs)
        n = len(rs)
        prof = {k: counts.get(k, 0) / n for k in PROFILES}
        prof["unparseable"] = counts.get("unparseable", 0) / n

        # Same profile under both user rules? The halves are matched by design.
        split = {}
        for rule in ("COLOR_RULE", "SHAPE_RULE"):
            sub = [r for r in rs if r["user_rule"] == rule]
            split[rule] = ({k: sum(1 for r in sub if r["profile"] == k) / len(sub)
                            for k in PROFILES} if sub else {})

        # Self-consistency across reps: same item, same answer every time.
        by_item = defaultdict(list)
        for r in rs:
            by_item[r["item_id"]].append(r["answer"])
        repeated = [v for v in by_item.values() if len(v) > 1]
        consistency = (sum(1 for v in repeated if len(set(v)) == 1) / len(repeated)
                       if repeated else None)

        summary[model] = {"n": n, "profile_rates": prof, "by_user_rule": split,
                          "answer_consistency_across_reps": consistency,
                          "answer_distribution": dict(Counter(r["answer"] for r in rs))}

    (args.outdir / "assistant_framing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = args.outdir / "assistant_framing_profiles.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "n"] + PROFILES + ["unparseable", "consistency"])
        for model, s in summary.items():
            w.writerow([model, s["n"]]
                       + [f"{s['profile_rates'][k]:.4f}" for k in PROFILES]
                       + [f"{s['profile_rates']['unparseable']:.4f}",
                          "" if s["answer_consistency_across_reps"] is None
                          else f"{s['answer_consistency_across_reps']:.4f}"])

    print(f"wrote {details}")
    print(f"wrote {csv_path}\n")
    print(f"{'model':16s} {'n':>5s} {'teacher':>9s} {'imposter':>9s} "
          f"{'conserv':>8s} {'invalid':>8s} {'consist':>8s}")
    for model, s in summary.items():
        p = s["profile_rates"]
        c = s["answer_consistency_across_reps"]
        print(f"{model[:16]:16s} {s['n']:5d} {p['teacher_corrective']:9.3f} "
              f"{p['imposter_specific']:9.3f} {p['conservative']:8.3f} "
              f"{p['invalid']:8.3f} {'' if c is None else f'{c:8.3f}'}")

    print("\nprofile rates by the rule the user demonstrated "
          "(matched halves; a flip between them is stimulus-driven, not a disposition):")
    for model, s in summary.items():
        for rule, d in s["by_user_rule"].items():
            if d:
                print(f"  {model[:16]:16s} {rule:10s} "
                      + "  ".join(f"{k.split('_')[0]}={d[k]:.3f}" for k in PROFILES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
