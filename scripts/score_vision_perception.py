#!/usr/bin/env python3
"""Score the perception probe: what a VLM reports seeing, against the renderer's
own per-object ground truth in vision/objects.csv.

This measures seeing, not reasoning. Run it alongside the task scores to tell
"cannot see the stimulus" apart from "can see it but reasons badly" -- a model
that never resolves which object carries the black border cannot possibly infer
the rule, and would otherwise look like a pedagogy failure.

Metrics, per (model, rep):
  parse_rate          responses yielding a JSON object with an `objects` list
  count_match         images where the number of objects reported is correct
  color_acc           position-matched objects with the right colour
  shape_acc           position-matched objects with the right shape
  binding_acc         right colour AND right shape on the same object
  selection_prec/rec  detection of the thick black selection border (history only)
  section_acc         history vs option assigned correctly

Matching is by `position` ("1L", "2R", "A".."D"); when positions are missing or
duplicated it falls back to reading order, which is the order the prompt asks
for. Falling back is recorded in the details file so it is never silent.

Usage:
  python scripts/score_vision_perception.py --outdir results_perception
  python scripts/score_vision_perception.py --pred results_perception/raw/one.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_RE = re.compile(r"^(?P<condition>.+)__(?P<model>.+)__rep(?P<rep>\d+)\.jsonl$")

COLORS = {"red", "blue", "green", "yellow", "purple"}
SHAPES = {"circle", "square", "triangle", "star", "hexagon"}


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def norm_pos(v) -> str:
    """'t1_L' / 'row 1 left' / '1-L' -> '1L'; option letters -> 'A'..'D'."""
    s = str(v or "").strip().upper().replace("_", "").replace("-", "").replace(" ", "")
    if s.startswith("T"):
        s = s[1:]
    if s in {"A", "B", "C", "D"}:
        return s
    m = re.match(r"^(\d+)(L|R|LEFT|RIGHT)$", s)
    if m:
        return f"{m.group(1)}{m.group(2)[0]}"
    m = re.match(r"^(?:ROW)?(\d+)$", s)
    if m:
        return m.group(1)
    return s


def norm_word(v, allowed: set[str]) -> str | None:
    s = str(v or "").strip().lower()
    s = {"colour": "color", "grey": "gray"}.get(s, s)
    return s if s in allowed else None


def as_bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return True if s in {"true", "yes", "1"} else False if s in {"false", "no", "0"} else None


def load_truth(objects_csv: Path) -> dict[str, list[dict]]:
    """image_id (without .png) -> objects in the reading order the prompt asks for."""
    truth: dict[str, list[dict]] = defaultdict(list)
    for r in csv.DictReader(objects_csv.open()):
        section = "history" if r["region"] == "history" else "option"
        pos = norm_pos(r["slot"]) if section == "history" else r["label"]
        truth[Path(r["image_id"]).stem].append({
            "section": section, "position": pos, "color": r["color"],
            "shape": r["shape"], "selected": r["is_selected"] == "True"})
    return truth


def extract_objects(row: dict):
    """The `objects` list out of a prediction row, or None if unusable."""
    payload = row.get("response")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    objs = payload.get("objects")
    return objs if isinstance(objs, list) else None


def score_file(pred_path: Path, truth: dict[str, list[dict]]) -> tuple[dict, list[dict]]:
    rows = read_jsonl(pred_path)
    details: list[dict] = []
    n_parsed = n_count_match = 0
    color_ok = shape_ok = bind_ok = section_ok = matched = 0
    sel_tp = sel_fp = sel_fn = 0
    fallbacks = 0
    color_conf, shape_conf = Counter(), Counter()

    for row in rows:
        iid = row.get("item_id")
        gold = truth.get(iid)
        if gold is None:
            continue
        objs = extract_objects(row)
        d = {"item_id": iid, "model": row.get("model_name"),
             "n_truth": len(gold), "parsed": objs is not None}
        if objs is None:
            details.append({**d, "n_reported": 0, "matched": 0, "binding_ok": 0,
                            "matched_by": "", "error": row.get("error", "")})
            continue
        n_parsed += 1
        d["n_reported"] = len(objs)
        if len(objs) == len(gold):
            n_count_match += 1

        # Position-keyed where possible; reading order otherwise.
        pred_by_pos = {}
        dupes = False
        for o in objs:
            if not isinstance(o, dict):
                continue
            p = norm_pos(o.get("position"))
            if p in pred_by_pos:
                dupes = True
            pred_by_pos[p] = o
        use_pos = (not dupes
                   and sum(1 for g in gold if g["position"] in pred_by_pos) == len(gold))
        if not use_pos:
            fallbacks += 1
        d["matched_by"] = "position" if use_pos else "reading_order"

        item_matched = item_bind = 0
        for i, g in enumerate(gold):
            if use_pos:
                p = pred_by_pos.get(g["position"])
            else:
                p = objs[i] if i < len(objs) and isinstance(objs[i], dict) else None
            if p is None:
                if g["selected"]:
                    sel_fn += 1
                continue
            matched += 1
            item_matched += 1
            c = norm_word(p.get("color"), COLORS)
            s = norm_word(p.get("shape"), SHAPES)
            if c == g["color"]:
                color_ok += 1
            else:
                color_conf[(g["color"], c or "unparseable")] += 1
            if s == g["shape"]:
                shape_ok += 1
            else:
                shape_conf[(g["shape"], s or "unparseable")] += 1
            if c == g["color"] and s == g["shape"]:
                bind_ok += 1
                item_bind += 1
            if str(p.get("section", "")).strip().lower() == g["section"]:
                section_ok += 1
            # Selection is only meaningful on history objects; options never
            # carry the black border, so counting them would inflate recall.
            if g["section"] == "history":
                pred_sel = as_bool(p.get("selected"))
                if g["selected"] and pred_sel:
                    sel_tp += 1
                elif g["selected"] and not pred_sel:
                    sel_fn += 1
                elif not g["selected"] and pred_sel:
                    sel_fp += 1
        details.append({**d, "matched": item_matched, "binding_ok": item_bind,
                        "error": row.get("error", "")})

    n = len([r for r in rows if r.get("item_id") in truth])
    div = lambda a, b: (a / b) if b else None  # noqa: E731
    summary = {
        "predictions": str(pred_path),
        "n_rows": n,
        "parse_rate": div(n_parsed, n),
        "count_match": div(n_count_match, n),
        "n_objects_matched": matched,
        "color_acc": div(color_ok, matched),
        "shape_acc": div(shape_ok, matched),
        "binding_acc": div(bind_ok, matched),
        "section_acc": div(section_ok, matched),
        "selection_precision": div(sel_tp, sel_tp + sel_fp),
        "selection_recall": div(sel_tp, sel_tp + sel_fn),
        "n_reading_order_fallback": fallbacks,
        "top_color_confusions": [
            {"truth": t, "reported": p, "n": c} for (t, p), c in color_conf.most_common(5)],
        "top_shape_confusions": [
            {"truth": t, "reported": p, "n": c} for (t, p), c in shape_conf.most_common(5)],
    }
    return summary, details


def fmt(x) -> str:
    return "" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "results_perception",
                    help="run directory holding raw/*.jsonl")
    ap.add_argument("--pred", type=Path, default=None,
                    help="score a single predictions file instead of a whole outdir")
    ap.add_argument("--objects", type=Path, default=REPO_ROOT / "vision" / "objects.csv")
    args = ap.parse_args()

    if not args.objects.exists():
        raise SystemExit(
            f"no ground truth at {args.objects}. vision/ is gitignored; regenerate "
            f"it with `python scripts/render_vision.py`.")
    truth = load_truth(args.objects)

    if args.pred:
        preds = [args.pred]
    else:
        raw = args.outdir / "raw"
        if not raw.is_dir():
            raise SystemExit(f"no raw predictions directory at {raw}")
        preds = sorted(raw.glob("*.jsonl"))
    if not preds:
        raise SystemExit("nothing to score")

    scores_dir = args.outdir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    per_run = []
    for p in preds:
        m = RUN_RE.match(p.name)
        print(f"scoring {p.name} ...", flush=True)
        summary, details = score_file(p, truth)
        summary["model"] = m["model"] if m else p.stem
        summary["rep"] = int(m["rep"]) if m else 1
        (scores_dir / f"{p.stem}_perception.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        with (scores_dir / f"{p.stem}_perception_details.csv").open(
                "w", newline="", encoding="utf-8") as fh:
            cols = ["item_id", "model", "n_truth", "n_reported", "parsed",
                    "matched", "binding_ok", "matched_by", "error"]
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(details)
        per_run.append(summary)

    cols = ["model", "rep", "n_rows", "parse_rate", "count_match", "binding_acc",
            "color_acc", "shape_acc", "section_acc", "selection_precision",
            "selection_recall", "n_reading_order_fallback"]
    csv_path = args.outdir / "perception_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in sorted(per_run, key=lambda x: (x["model"], x["rep"])):
            w.writerow([fmt(r.get(c)) for c in cols])

    # Aggregate across reps.
    grouped = defaultdict(list)
    for r in per_run:
        grouped[r["model"]].append(r)
    print(f"\nwrote {csv_path}\n")
    print(f"{'model':24s} {'reps':>4s} {'parse':>7s} {'binding':>8s} {'color':>7s} "
          f"{'shape':>7s} {'sel_rec':>8s}")
    for model, runs in sorted(grouped.items()):
        agg = {}
        for c in ("parse_rate", "binding_acc", "color_acc", "shape_acc", "selection_recall"):
            vals = [r[c] for r in runs if isinstance(r.get(c), float)]
            agg[c] = statistics.fmean(vals) if vals else None
        print(f"{model[:24]:24s} {len(runs):4d} {fmt(agg['parse_rate']):>7s} "
              f"{fmt(agg['binding_acc']):>8s} {fmt(agg['color_acc']):>7s} "
              f"{fmt(agg['shape_acc']):>7s} {fmt(agg['selection_recall']):>8s}")

    worst = min((r for r in per_run if isinstance(r.get("binding_acc"), float)),
                key=lambda r: r["binding_acc"], default=None)
    if worst:
        print(f"\nlowest binding accuracy: {worst['model']} rep{worst['rep']} "
              f"{worst['binding_acc']:.3f}")
        if worst["top_shape_confusions"]:
            c = worst["top_shape_confusions"][0]
            print(f"  most common shape error: {c['truth']} reported as "
                  f"{c['reported']} ({c['n']}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
