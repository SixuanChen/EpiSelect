#!/usr/bin/env python3
"""Verification for the rendered vision stimuli (plan steps 1, 3, 4)."""
import json, csv, re, sys, collections
from pathlib import Path
from PIL import Image

BENCH = Path(__file__).resolve().parent.parent
V = BENCH / "vision"
fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)
    return cond


# ---- load sources ---------------------------------------------------------
items = {}
for n in ("benchmark/main_100_full.jsonl",
          "benchmark/controls_no_history_choice_10_gold.jsonl",
          "benchmark/controls_no_history_diagnosis_10_gold.jsonl"):
    for l in open(BENCH / n):
        r = json.loads(l); items[r["item_id"]] = r
gold_ids = {json.loads(l)["item_id"] for l in open(BENCH / "benchmark/all_120_gold.jsonl")}
reqs = [json.loads(l) for l in open(V / "all_120_vlm_requests.jsonl")]
req_ids = [r["item_id"] for r in reqs]
objs = collections.defaultdict(list)
for row in csv.DictReader(open(V / "objects.csv")):
    objs[row["image_id"]].append(row)

# ---- 1. per-object fidelity ----------------------------------------------
for iid, item in items.items():
    rows = objs.get(f"{iid}.png", [])
    hist = [o for o in rows if o["region"] == "history"]
    opts = [o for o in rows if o["region"] == "option"]

    src_h = []
    for t in (item.get("history") or []):
        for i, o in enumerate(t["options"]):
            src_h.append((o["color"], o["shape"], i == t["selected_index"]))
    got_h = [(o["color"], o["shape"], o["is_selected"] == "True") for o in hist]
    check(src_h == got_h, f"{iid}: history mismatch {src_h} != {got_h}")

    src_o = [(o["label"], o["object"]["color"], o["object"]["shape"])
             for o in (item.get("target_options") or [])]
    got_o = [(o["label"], o["color"], o["shape"]) for o in opts]
    check(src_o == got_o, f"{iid}: options mismatch {src_o} != {got_o}")
    check([o[0] for o in got_o] in ([], list("ABCD")), f"{iid}: bad A-D order")

# ---- 2. image files -------------------------------------------------------
imgs = sorted((V / "images_224").glob("*.png"))
check(len(imgs) == 120, f"expected 120 images, got {len(imgs)}")
for p in imgs:
    w, h = Image.open(p).size
    check((w, h) == (224, 224), f"{p.name}: size {w}x{h} != 224x224")
check({p.stem for p in imgs} == gold_ids, "image basenames != gold item_ids")

# ---- 3. scorer compatibility ---------------------------------------------
check(set(req_ids) == gold_ids, "request item_ids != gold item_ids")
check(len(req_ids) == len(set(req_ids)) == 120, "duplicate/missing request ids")
text_order = [json.loads(l)["item_id"] for l in
              open(BENCH / "requests/all_120_requests.jsonl")]
check(req_ids == text_order, "request order differs from the text run")

# ---- 4. leakage: no object descriptions left in prompts -------------------
COLORS = ["red", "blue", "green", "yellow", "purple"]
SHAPES = ["circle", "square", "triangle", "star", "hexagon"]
PAIR = re.compile(r"\b(" + "|".join(COLORS) + r")\s+(" + "|".join(SHAPES) + r")\b", re.I)
for r in reqs:
    txt = r["messages"][1]["content"]
    check("Trial 1: choices were" not in txt, f"{r['item_id']}: history text left in prompt")
    check(not re.search(r"^[A-D]\) ", txt, re.M), f"{r['item_id']}: option list left in prompt")
    # a "<color> <shape>" pair may only appear inside the rule/hypothesis lines
    for m in PAIR.finditer(txt):
        line = txt[txt.rfind("\n", 0, m.start()) + 1: txt.find("\n", m.end())]
        check(("CATEGORY RULE" in line or "PRIVATE RULE" in line
               or line.startswith("- COLOR_RULE") or line.startswith("- SHAPE_RULE")),
              f"{r['item_id']}: object leak {m.group(0)!r} in: {line[:70]!r}")

# ---- report ---------------------------------------------------------------
n_obj = sum(len(v) for v in objs.values())
print(f"items={len(items)} images={len(imgs)} objects={n_obj} requests={len(reqs)}")
if fails:
    print(f"\nFAIL ({len(fails)}):")
    for f in fails[:25]:
        print("  -", f)
    sys.exit(1)
print("\nALL CHECKS PASSED")
