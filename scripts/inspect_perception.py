#!/usr/bin/env python3
"""Write a readable report on what the VLMs actually reported seeing.

The aggregate metrics in perception_metrics.csv say how often a model was right.
This says *how* it was wrong, which is the part that changes what you do next:

  - object accuracy split into colour, shape and the two together (binding)
  - selection-border detection, and whether errors are positional
  - stereotypy: how many distinct object lists a model produced across all
    images. A model answering from a template rather than the image gives itself
    away here more clearly than in any accuracy number.
  - a sample of individual trials with ground truth beside the raw response

Output is markdown at the repo root, so it renders on GitHub next to the code.

Usage:
  python scripts/inspect_perception.py
  python scripts/inspect_perception.py --n-examples 5 --seed 7
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import random
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_truth(objects_csv: Path):
    truth = collections.defaultdict(list)
    for r in csv.DictReader(objects_csv.open()):
        section = "history" if r["region"] == "history" else "option"
        pos = r["slot"].replace("t", "").replace("_", "") if section == "history" else r["label"]
        truth[Path(r["image_id"]).stem].append(
            {"section": section, "position": pos, "color": r["color"],
             "shape": r["shape"], "selected": r["is_selected"] == "True"})
    return truth


def objects_of(row):
    payload = row.get("response")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    o = payload.get("objects")
    return o if isinstance(o, list) else None


def key(o):
    return str(o.get("position", "")).upper().replace("_", "").lstrip("T")


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "results_perception")
    ap.add_argument("--objects", type=Path, default=REPO_ROOT / "vision" / "objects.csv")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "VISION_PERCEPTION_ANALYSIS.md")
    ap.add_argument("--n-examples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7, help="fixed so the sample is reproducible")
    ap.add_argument("--task-outdir", type=Path, default=REPO_ROOT / "results_vision",
                    help="task run, to show what the model answered as well as what it saw")
    ap.add_argument("--examples-dir", type=Path,
                    default=REPO_ROOT / "docs" / "perception_examples",
                    help="sampled stimuli are copied here so the report renders "
                         "anywhere; vision/ itself is gitignored")
    args = ap.parse_args()

    truth = load_truth(args.objects)
    files = sorted(glob.glob(str(args.outdir / "raw" / "perception__*.jsonl")))
    if not files:
        raise SystemExit(f"no perception predictions under {args.outdir}/raw/")

    stats = collections.defaultdict(collections.Counter)
    stereo = collections.defaultdict(collections.Counter)
    side = collections.defaultdict(collections.Counter)
    raws = collections.defaultdict(dict)

    for f in files:
        model = os.path.basename(f).split("__")[1]
        rep = os.path.basename(f).split("__")[2]
        for line in open(f):
            row = json.loads(line)
            gold = truth.get(row.get("item_id"))
            if not gold:
                continue
            objs = objects_of(row)
            if objs is None:
                stats[model]["unparsed"] += 1
                continue
            if rep.startswith("rep01"):
                raws[row["item_id"]][model] = (row.get("raw_response") or "").strip()
            stereo[model][tuple((str(o.get("color")), str(o.get("shape")))
                                for o in objs if isinstance(o, dict))] += 1
            pred = {key(o): o for o in objs if isinstance(o, dict)}
            all_obj = all_sel = True
            for t in gold:
                p = pred.get(t["position"])
                stats[model]["objects"] += 1
                if p is None:
                    all_obj = False
                    continue
                c = str(p.get("color", "")).lower() == t["color"]
                s = str(p.get("shape", "")).lower() == t["shape"]
                stats[model]["color_ok"] += c
                stats[model]["shape_ok"] += s
                stats[model]["bind_ok"] += (c and s)
                if not (c and s):
                    all_obj = False
                if t["section"] == "history":
                    stats[model]["hist"] += 1
                    if bool(p.get("selected")) == t["selected"]:
                        stats[model]["sel_ok"] += 1
                    else:
                        all_sel = False
                    sfx = t["position"][-1]
                    side[model][f"n_{sfx}"] += 1
                    side[model][f"truth_{sfx}"] += t["selected"]
                    side[model][f"pred_{sfx}"] += bool(p.get("selected"))
            stats[model]["images"] += 1
            stats[model]["img_objects"] += all_obj
            stats[model]["img_selections"] += all_sel
            stats[model]["img_perfect"] += (all_obj and all_sel)

    models = sorted(stats)
    L = ["# What the VLMs report seeing", "",
         "Generated by `scripts/inspect_perception.py` -- **do not edit by hand**.",
         "",
         "Ground truth is `vision/objects.csv`, written by the renderer, so it is exact.",
         "Every image holds 8 objects: 4 history (2 rounds x left/right) and 4 options.",
         "", "## Object identification", "",
         "| model | images | colour | shape | binding | selection flags | all 8 objects | both selections | perfect |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in models:
        s = stats[m]
        n, h, i = max(s["objects"], 1), max(s["hist"], 1), max(s["images"], 1)
        L.append(f"| `{m}` | {s['images']} | {s['color_ok']/n:.3f} | {s['shape_ok']/n:.3f} | "
                 f"**{s['bind_ok']/n:.3f}** | {s['sel_ok']/h:.3f} | {s['img_objects']/i:.3f} | "
                 f"{s['img_selections']/i:.3f} | {s['img_perfect']/i:.3f} |")
    L += ["",
          "`binding` = colour *and* shape correct on the same object; chance is 1/25 = 0.04.",
          "Colour or shape alone is 1/5 = 0.20. `perfect` = all 8 objects and both",
          "selections correct, which is what the task actually needs.", ""]

    L += ["## Is the model reading the image?", "",
          "Distinct object lists produced across all images. The stimuli are all",
          "different, so a low count means the model is answering from a template.", "",
          "| model | distinct lists | images | most frequent list | share |", "|---|---:|---:|---|---:|"]
    for m in models:
        c = stereo[m]
        if not c:
            continue
        top, n = c.most_common(1)[0]
        desc = " / ".join(f"{a} {b}" for a, b in top[:4]) + (" ..." if len(top) > 4 else "")
        L.append(f"| `{m}` | {len(c)} | {sum(c.values())} | {desc} | "
                 f"{'**' if n > sum(c.values())*0.3 else ''}{n/sum(c.values()):.0%}"
                 f"{'**' if n > sum(c.values())*0.3 else ''} |")

    L += ["", "## Where the selection errors come from", "",
          "Ground truth is 50/50 left/right by design. A model marking mostly one",
          "side is using position as a prior instead of reading the black border.", "",
          "| model | left: truth | left: predicted | right: truth | right: predicted |",
          "|---|---:|---:|---:|---:|"]
    for m in models:
        c = side[m]
        nl, nr = max(c["n_L"], 1), max(c["n_R"], 1)
        L.append(f"| `{m}` | {c['truth_L']/nl:.3f} | **{c['pred_L']/nl:.3f}** | "
                 f"{c['truth_R']/nr:.3f} | **{c['pred_R']/nr:.3f}** |")

    # ---- task answers on the same items -------------------------------------
    gold = {}
    gold_path = REPO_ROOT / "benchmark" / "all_120_gold.jsonl"
    if gold_path.exists():
        for line in gold_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                g = json.loads(line)
                gold[g["item_id"]] = g
    answers = collections.defaultdict(dict)
    for f in sorted(glob.glob(str(args.task_outdir / "raw" / "*__rep01.jsonl"))):
        model = os.path.basename(f).split("__")[1]
        for line in open(f):
            row = json.loads(line)
            payload = row.get("response")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    payload = {}
            if isinstance(payload, dict):
                answers[row.get("item_id")][model] = payload

    # ---- sampled trials -----------------------------------------------------
    rng = random.Random(args.seed)
    picks = rng.sample(sorted(truth), min(args.n_examples, len(truth)))
    L += ["", f"## {len(picks)} sampled trials (`random.seed({args.seed})`, not hand-picked)", ""]
    for iid in picks:
        src = REPO_ROOT / "vision" / "images_224" / f"{iid}.png"
        rel = f"vision/images_224/{iid}.png"
        if src.exists():
            args.examples_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, args.examples_dir / f"{iid}.png")
            rel = os.path.relpath(args.examples_dir / f"{iid}.png", REPO_ROOT)
        L += [f"### `{iid}`", "",
              f"![{iid}]({rel})", "",
              "| slot | ground truth | " + " | ".join(f"`{m}`" for m in sorted(raws[iid])) + " |",
              "|---|---|" + "---|" * len(raws[iid])]
        preds = {m: {key(o): o for o in (objects_of({"response": raw}) or [])}
                 for m, raw in raws[iid].items()}
        for t in truth[iid]:
            cells = []
            for m in sorted(raws[iid]):
                p = preds[m].get(t["position"])
                if p is None:
                    cells.append("—")
                    continue
                got = f"{p.get('color')} {p.get('shape')}"
                ok = (str(p.get("color", "")).lower() == t["color"]
                      and str(p.get("shape", "")).lower() == t["shape"])
                sel = " **SEL**" if p.get("selected") else ""
                cells.append(("✓ " if ok else "✗ ") + got + sel)
            gt = f"{t['color']} {t['shape']}" + (" **SELECTED**" if t["selected"] else "")
            L.append(f"| {t['position']} | {gt} | " + " | ".join(cells) + " |")
        L.append("")

        # What each model answered on the task, for the same image. The point of
        # putting them together: an answer is only meaningful given what the model
        # managed to see.
        g = gold.get(iid)
        if g and answers.get(iid):
            L += ["**Task answers on this image**", "",
                  "| model | inferred rule | answer | chose | correct |",
                  "|---|---|---|---|---|"]
            opts = {o["label"]: o for o in g.get("target_options", [])}
            for m in sorted(answers[iid]):
                a = answers[iid][m]
                ans = str(a.get("answer", "")).upper()
                rule = str(a.get("inferred_rule", "")) or "—"
                chose = opts.get(ans, {}).get("text", "—")
                ok = "✓" if ans in set(g.get("acceptable_answers", [])) else "✗"
                rule_ok = "" if not g.get("gold_other_rule") else (
                    " ✓" if rule == g["gold_other_rule"] else " ✗")
                L.append(f"| `{m}` | {rule}{rule_ok} | {ans} | {chose} | {ok} |")
            L += ["",
                  f"Gold: rule **{g.get('gold_other_rule','—')}**, "
                  f"answer **{'/'.join(g.get('acceptable_answers', []))}** "
                  f"({opts.get(g.get('preferred_answer',''), {}).get('text','—')})", ""]

        for m in sorted(raws[iid]):
            L += [f"<details><summary>raw perception response — <code>{m}</code></summary>", "",
                  "```json", raws[iid][m][:4000], "```", "", "</details>", ""]

    args.out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"  {len(models)} models, {len(picks)} sampled trials, seed {args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
