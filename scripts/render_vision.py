#!/usr/bin/env python3
"""
Render the 120 adaptive-pedagogy trials as VLM stimuli.

Matched vision set: same 120 item_ids, same gold file, same scorer -- only the
modality changes. Rendering is fully deterministic (no RNG, no seed): every object
is pinned by the benchmark JSON.

Uses the drawing stack in scripts/shape_bank.py (ALL_COLORS, SHAPE_TEMPLATES,
draw_shape), vendored from binding_probe_repo/datasets/utils.py so this repo
re-renders standalone -- clone it, run this, and the 120 stimuli come back
identical, no external checkout needed.

hexagon is absent from the vendored bank, so a matched template is built at
runtime and appended to an in-memory copy; assets/shape_templates.npz is never
touched.

Usage:
    python scripts/render_vision.py                 # 224x224
    python scripts/render_vision.py --size 448      # pilot
    python scripts/render_vision.py --contact-only  # legibility gate
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFilter, ImageFont

BENCH = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shape_bank  # noqa: E402
from shape_bank import ALL_COLORS, SHAPE_TEMPLATES, SHAPE_INDICES, draw_shape  # noqa: E402

# --------------------------------------------------------------------------
# Shape templates
# --------------------------------------------------------------------------

BENCH_SHAPES = ["circle", "square", "triangle", "star", "hexagon"]
BENCH_COLORS = ["red", "blue", "green", "yellow", "purple"]


def build_hexagon_template(target_dark_frac: float = 0.50) -> np.ndarray:
    """A 32x32 hexagon matching imgs.npy convention: white bg (255), dark shape.

    Radius is calibrated so the dark-pixel fraction lands near the existing
    templates (circle 0.505, square 0.517) -- otherwise the hexagon would be the
    biggest blob on screen and that alone would be a shortcut cue.
    """
    hi = 256
    best = None
    for r in range(hi // 6, hi // 2):
        img = Image.new("L", (hi, hi), 255)
        d = ImageDraw.Draw(img)
        d.regular_polygon((hi // 2, hi // 2, r), n_sides=6, rotation=0, fill=0)
        small = img.resize((32, 32), Image.Resampling.LANCZOS)
        frac = (np.array(small) < 128).mean()
        err = abs(frac - target_dark_frac)
        if best is None or err < best[0]:
            best = (err, np.array(small).astype(np.float64), frac)
    err, tmpl, frac = best
    assert err < 0.02, f"hexagon calibration failed: dark_frac={frac:.3f}"
    return tmpl


def make_template_bank() -> tuple[np.ndarray, dict]:
    """In-memory SHAPE_TEMPLATES + hexagon appended. Disk asset untouched.

    draw_shape() resolves SHAPE_TEMPLATES / SHAPE_INDICES as module globals, so the
    extended bank is rebound on the module once here rather than per call.
    """
    hexa = build_hexagon_template()
    bank = np.concatenate([SHAPE_TEMPLATES, hexa[None, ...]], axis=0)
    idx = {s: SHAPE_INDICES[s] for s in ("circle", "square", "triangle", "star")}
    idx["hexagon"] = bank.shape[0] - 1

    shape_bank.SHAPE_TEMPLATES = bank
    shape_bank.SHAPE_INDICES = idx
    return bank, idx


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

WHITE = (255, 255, 255)
INK = (17, 17, 17)
MUTED = (120, 120, 120)
HAIRLINE = (196, 196, 196)
OUTLINE = (70, 70, 70)


def _font(px: int) -> ImageFont.FreeTypeFont:
    import matplotlib
    ttf = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf"
    return ImageFont.truetype(str(ttf), px)


def _font_bold(px: int) -> ImageFont.FreeTypeFont:
    import matplotlib
    ttf = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
    return ImageFont.truetype(str(ttf), px)


def paste_object(canvas: Image.Image, box, color_name: str, shape: str,
                 bank, idx, shape_px: int, outline: bool = True) -> None:
    """Draw one object centered in `box` using the binding_probe pipeline.

    draw_shape() pastes an OPAQUE white block, which would punch a white square
    through any cell background. So render into a scratch tile with the identical
    call, then composite only the non-white pixels. The tile is strictly 2-color
    (draw_shape thresholds at <128 with no anti-aliasing), so the mask is exact.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    tile = Image.new("RGB", (w, h), WHITE)
    td = ImageDraw.Draw(tile)
    draw_shape(tile, td, (0, 0, w, h), shape, ALL_COLORS[color_name],
               shape_size=shape_px)

    arr = np.array(tile)
    mask_arr = (arr != 255).any(axis=2)
    mask = Image.fromarray((mask_arr * 255).astype(np.uint8), mode="L")

    if outline:
        # Uniform 1px rim on every shape, every color. Without it `yellow`
        # (255,255,0) is nearly invisible on white; applied identically so no
        # color or shape is privileged.
        grown = mask.filter(ImageFilter.MaxFilter(3))
        rim = Image.fromarray(
            ((np.array(grown) > 0) & ~mask_arr).astype(np.uint8) * 255, mode="L")
        canvas.paste(Image.new("RGB", (w, h), OUTLINE), (x1, y1), rim)

    canvas.paste(tile, (x1, y1), mask)


def cell_border(draw: ImageDraw.ImageDraw, box, selected: bool, k: float) -> None:
    """Selection marked by border weight only -- never color, since the task IS
    about color and a colored highlight would contaminate it."""
    if selected:
        draw.rectangle(box, outline=INK, width=max(2, int(round(2 * k))))
    else:
        x1, y1, x2, y2 = box
        step = max(3, int(round(3 * k)))
        for x in range(x1, x2, step * 2):
            draw.line([(x, y1), (min(x + step, x2), y1)], fill=HAIRLINE, width=1)
            draw.line([(x, y2), (min(x + step, x2), y2)], fill=HAIRLINE, width=1)
        for y in range(y1, y2, step * 2):
            draw.line([(x1, y), (x1, min(y + step, y2))], fill=HAIRLINE, width=1)
            draw.line([(x2, y), (x2, min(y + step, y2))], fill=HAIRLINE, width=1)


def render_trial(item: dict, bank, idx, size: int = 224) -> tuple[Image.Image, list[dict]]:
    """One trial -> one square image + per-object ground-truth rows."""
    k = size / 224.0
    S = size

    def s(v):  # scale a 224-space constant
        return int(round(v * k))

    cell = s(42)
    shape_px = s(32)
    canvas = Image.new("RGB", (S, S), WHITE)
    d = ImageDraw.Draw(canvas)
    f_hdr = _font_bold(max(7, s(8)))
    f_lbl = _font_bold(max(8, s(10)))
    f_num = _font(max(7, s(9)))

    objs = []
    history = item.get("history") or []
    options = item.get("target_options") or []

    # ---- observations -----------------------------------------------------
    d.text((s(8), s(5)), "OBSERVED CHOICES", font=f_hdr, fill=MUTED)
    pair_w = 2 * cell + s(14)
    x0 = (S - pair_w) // 2
    y = s(18)
    boxes = []
    if history:
        for hi_, htrial in enumerate(history):
            d.text((x0 - s(16), y + cell // 2 - s(5)), str(htrial["trial"]),
                   font=f_num, fill=MUTED)
            for oi, opt in enumerate(htrial["options"]):
                bx = x0 + oi * (cell + s(14))
                box = (bx, y, bx + cell, y + cell)
                sel = (oi == htrial["selected_index"])
                paste_object(canvas, box, opt["color"], opt["shape"],
                             bank, idx, shape_px)
                boxes.append((box, sel))
                objs.append(dict(
                    region="history", slot=f"t{htrial['trial']}_{'LR'[oi]}",
                    color=opt["color"], shape=opt["shape"],
                    is_selected=sel, is_option=False, label="",
                    branch_type="", center_x=(box[0] + box[2]) / 2,
                    center_y=(box[1] + box[3]) / 2))
            y += cell + s(4)
    else:
        d.text((s(8), s(30)), "(no observations available)", font=f_num, fill=MUTED)
        y = s(18) + 2 * (cell + s(4))

    for box, sel in boxes:
        cell_border(d, box, sel, k)

    # ---- divider ----------------------------------------------------------
    ydiv = s(112)
    d.line([(s(8), ydiv), (S - s(8), ydiv)], fill=HAIRLINE, width=1)

    # ---- options ----------------------------------------------------------
    if options:
        d.text((s(8), s(118)), "OPTIONS", font=f_hdr, fill=MUTED)
        yopt = s(130)
        total = 4 * cell + 3 * s(10)
        ox = (S - total) // 2
        for i, opt in enumerate(options):
            bx = ox + i * (cell + s(10))
            box = (bx, yopt, bx + cell, yopt + cell)
            o = opt["object"]
            paste_object(canvas, box, o["color"], o["shape"], bank, idx, shape_px)
            d.rectangle(box, outline=HAIRLINE, width=1)
            tw = d.textlength(opt["label"], font=f_lbl)
            d.text((bx + cell / 2 - tw / 2, yopt + cell + s(3)), opt["label"],
                   font=f_lbl, fill=INK)
            objs.append(dict(
                region="option", slot=opt["label"], color=o["color"],
                shape=o["shape"], is_selected=False, is_option=True,
                label=opt["label"], branch_type=opt.get("branch_type", ""),
                center_x=(box[0] + box[2]) / 2, center_y=(box[1] + box[3]) / 2))
    else:
        d.text((s(8), s(122)), "(no options -- report the rule only)",
               font=f_num, fill=MUTED)

    return canvas, objs


def render_contact_sheet(bank, idx, size: int = 224) -> Image.Image:
    """5 shapes x 5 colors at the exact cell scale used in the trials.

    This is the legibility gate: if hexagon vs circle is not separable here, the
    benchmark measures perception rather than pedagogy.
    """
    k = size / 224.0
    cell = int(round(42 * k))
    shape_px = int(round(32 * k))
    pad = int(round(16 * k))
    W = pad * 2 + 5 * cell + 4 * int(round(8 * k))
    H = pad * 2 + 5 * cell + 4 * int(round(8 * k)) + int(round(18 * k))
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)
    f = _font(max(7, int(round(9 * k))))
    gap = int(round(8 * k))
    for r, shape in enumerate(BENCH_SHAPES):
        for c, color in enumerate(BENCH_COLORS):
            x = pad + c * (cell + gap)
            y = pad + r * (cell + gap) + int(round(14 * k))
            paste_object(img, (x, y, x + cell, y + cell), color, shape,
                         bank, idx, shape_px)
            d.rectangle((x, y, x + cell, y + cell), outline=HAIRLINE, width=1)
    for c, color in enumerate(BENCH_COLORS):
        d.text((pad + c * (cell + gap), pad), color, font=f, fill=MUTED)
    return img


# --------------------------------------------------------------------------
# Prompt rewriting
# --------------------------------------------------------------------------

RE_HISTORY = re.compile(
    r"^Trial \d+: choices were .*\n(?:Learner|Civilian) selected: .*"
    r"(?:\nTrial \d+: choices were .*\n(?:Learner|Civilian) selected: .*)*$",
    re.MULTILINE)
RE_OPTIONS = re.compile(r"^A\) .*\nB\) .*\nC\) .*\nD\) .*$", re.MULTILINE)
RE_RETURN = re.compile(r"^Return exactly one JSON object and no other text: \{",
                       re.MULTILINE)

IMG_HISTORY_TMPL = (
    "You are given one image of this task. The top section of the image, headed "
    "OBSERVED CHOICES, shows the {who}'s previous rounds: one row per round, "
    "numbered on the left. Each row contains the two objects the {who} was shown, "
    "side by side. The object the {who} selected is outlined with a thick solid "
    "black border; the object they did not select is outlined with a thin dashed "
    "grey border.")
IMG_HISTORY = IMG_HISTORY_TMPL.format(who="learner")
IMG_HISTORY_CIV = IMG_HISTORY_TMPL.format(who="civilian")
IMG_OPTIONS = ("The bottom section of the image, headed OPTIONS, shows the four "
               "objects you may choose from, labelled A, B, C and D from left to "
               "right.")
IMG_NONE = ("You are given one image of this task. Its OBSERVED CHOICES section is "
            "empty and reads \"(no observations available)\", because no previous "
            "rounds were recorded.")

# Vocabulary is pinned so the reported objects can be scored against objects.csv
# without free-text normalisation.
PERCEPTION_INSTRUCTION = (
    "First, report what you can actually see. List every object in the image in "
    "reading order: the OBSERVED CHOICES rows from top to bottom (left object "
    "before right object within a row), then the OPTIONS from left to right. "
    "Describe each object with these five fields:\n"
    "- \"section\": \"history\" for an object in OBSERVED CHOICES, \"option\" for an "
    "object in OPTIONS.\n"
    "- \"position\": for a history object, the row number followed by L or R, e.g. "
    "\"1L\", \"1R\", \"2L\", \"2R\"; for an option, its letter \"A\", \"B\", \"C\" "
    "or \"D\".\n"
    "- \"color\": exactly one of red, blue, green, yellow, purple.\n"
    "- \"shape\": exactly one of circle, square, triangle, star, hexagon.\n"
    "- \"selected\": true if that object is outlined with the thick solid black "
    "border, false otherwise. Objects in OPTIONS are never outlined in black.\n"
    "If a section of the image contains no objects, report no objects for it.")

OBJECTS_SCHEMA = ('"objects":[{"section":"history" or "option","position":"1L",'
                  '"color":"red","shape":"circle","selected":true}, ...],')

PERCEPTION_ONLY_PROMPT = (
    "You are given one image from a category-learning task.\n"
    "The top section of the image, headed OBSERVED CHOICES, shows previous rounds: "
    "one row per round, numbered on the left, with the two objects shown in that "
    "round side by side. The object that was selected is outlined with a thick "
    "solid black border; the object that was not selected is outlined with a thin "
    "dashed grey border. If no rounds were recorded, this section is empty and "
    "reads \"(no observations available)\".\n"
    "The bottom section, headed OPTIONS, shows up to four objects labelled A, B, C "
    "and D from left to right.\n\n"
    + PERCEPTION_INSTRUCTION
    + "\n\nReturn exactly one JSON object and no other text: {" + OBJECTS_SCHEMA[:-1]
    + "}")


def add_perception(prompt: str) -> str:
    """Ask for the visible objects and widen the answer schema to hold them.

    `objects` comes first in the schema so the model commits to what it sees before
    it commits to an answer; the task keys keep their original names and order, so
    score_predictions.py reads the response unchanged.
    """
    m = RE_RETURN.search(prompt)
    if not m:  # unexpected prompt shape -- leave it alone rather than corrupt it
        return prompt
    return (prompt[:m.start()] + PERCEPTION_INSTRUCTION + "\n\n"
            + prompt[m.start():m.end()] + OBJECTS_SCHEMA + prompt[m.end():])


def prompt_variant(row: dict) -> str:
    """Which of the four prompt shapes this request is, for the doc below."""
    txt = row["messages"][1]["content"]
    if row["trial_type"] == "main_joint":
        return "main_joint / TEACHER" if "You are the TEACHER" in txt \
            else "main_joint / IMPOSTER"
    return row["trial_type"]


DOC_NOTES = {
    "main_joint / TEACHER": "Infer the learner's rule, then pick the most "
                            "informative positive example. 50 of the 120 items.",
    "main_joint / IMPOSTER": "Infer the civilian's rule, then satisfy a private "
                             "rule while blending in. 50 of the 120 items. Note "
                             "the actor is called the civilian, not the learner.",
    "control_no_history_choice": "No rows in OBSERVED CHOICES; INSUFFICIENT is "
                                 "the correct answer. 10 items.",
    "control_no_history_diagnosis": "No rows in OBSERVED CHOICES and no OPTIONS "
                                    "section; UNKNOWN is correct. 10 items.",
}


def write_prompt_doc(path: Path, req_rows: list[dict], perception: str) -> None:
    """One readable sample of every prompt shape, so the repo shows what the model
    is actually asked without anyone having to run this script first."""
    by_variant: dict[str, dict] = {}
    for r in req_rows:
        by_variant.setdefault(prompt_variant(r), r)

    L = ["# Vision prompts",
         "",
         "Generated by `scripts/render_vision.py` -- **do not edit by hand**, edit the "
         "string constants in that file and re-run it.",
         "",
         f"Object listing in the task prompt: `--perception {perception}`. "
         "The standalone perception probe is written either way.",
         "",
         "Every prompt is the text-run prompt with the object descriptions replaced by "
         "pointers at the image; the rule and role wording is untouched, which is what "
         "makes the text and vision runs comparable.",
         ""]

    sys_msg = req_rows[0]["messages"][0]["content"] if req_rows else ""
    L += ["## System message", "", "Identical for all 120 items.", "",
          "```text", sys_msg, "```", ""]

    for variant in ("main_joint / TEACHER", "main_joint / IMPOSTER",
                    "control_no_history_choice", "control_no_history_diagnosis"):
        row = by_variant.get(variant)
        if row is None:
            continue
        L += [f"## {variant}", "", DOC_NOTES.get(variant, ""), "",
              f"Example item: `{row['item_id']}` "
              f"(image: `{row['image_path']}`)", "",
              "```text", row["messages"][1]["content"], "```", ""]

    L += ["## Perception-only probe", "",
          "`vision/all_120_perception_requests.jsonl` -- the same 120 images with no "
          "task, role or rule. Use it to measure what the model can see separately "
          "from how it reasons.", "",
          "```text", PERCEPTION_ONLY_PROMPT, "```", ""]

    path.write_text("\n".join(L), encoding="utf-8")


def rewrite_prompt(prompt: str, trial_type: str, perception: bool = True) -> str:
    """Strip every object description, point at the image, keep all else verbatim.

    Surgical replacement rather than regeneration, so the rule/role wording stays
    byte-identical to the text run -- that is what makes the two runs comparable.
    """
    civ = "Civilian selected:" in prompt
    out = RE_HISTORY.sub(IMG_HISTORY_CIV if civ else IMG_HISTORY, prompt)
    out = RE_OPTIONS.sub(IMG_OPTIONS, out)
    if trial_type == "control_no_history_diagnosis":
        out = out.replace(
            "No previous choices from the learner are available.",
            "No previous choices from the learner are available. " + IMG_NONE)
    elif trial_type == "control_no_history_choice":
        out = out.replace(
            "No previous choices from the learner are available, so you do not "
            "know which of the two rules they are using.",
            "No previous choices from the learner are available, so you do not "
            "know which of the two rules they are using. " + IMG_NONE)
    return add_perception(out) if perception else out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def load_items() -> dict:
    """item_id -> record carrying history + target_options, from the gold files."""
    items = {}
    for name in ("benchmark/main_100_full.jsonl",
                 "benchmark/controls_no_history_choice_10_gold.jsonl",
                 "benchmark/controls_no_history_diagnosis_10_gold.jsonl"):
        for line in open(BENCH / name):
            r = json.loads(line)
            items[r["item_id"]] = r
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--outdir", default=str(BENCH / "vision"))
    ap.add_argument("--contact-only", action="store_true",
                    help="render only the legibility contact sheet")
    ap.add_argument("--perception", choices=["inline", "none"], default="inline",
                    help="inline: the task prompt also asks the model to list the "
                         "objects it can see (extra `objects` key in the answer "
                         "JSON). none: task prompt only, byte-comparable with the "
                         "text run. The standalone perception-only request file is "
                         "written either way.")
    args = ap.parse_args()

    bank, idx = make_template_bank()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    sheet = render_contact_sheet(bank, idx, args.size)
    sheet.save(out / "contact_sheet.png")
    print(f"contact sheet -> {out/'contact_sheet.png'}")
    if args.contact_only:
        return 0

    imgdir = out / f"images_{args.size}"
    imgdir.mkdir(exist_ok=True)

    items = load_items()
    requests = [json.loads(l) for l in
                open(BENCH / "requests/all_120_requests.jsonl")]
    public = {json.loads(l)["item_id"]: json.loads(l)
              for l in open(BENCH / "benchmark/all_120_public.jsonl")}

    labels, obj_rows, req_rows, perc_rows = [], [], [], []
    for req in requests:  # preserve the shuffled order of the text run
        iid = req["item_id"]
        item = items[iid]
        tt = public[iid]["trial_type"]

        img, objs = render_trial(item, bank, idx, args.size)
        fname = f"{iid}.png"
        img.save(imgdir / fname)

        vis_prompt = rewrite_prompt(public[iid]["prompt"], tt,
                                    perception=args.perception == "inline")
        sys_msg = next((m["content"] for m in req["messages"]
                        if m["role"] == "system"), "")

        # Relative to the benchmark root when the output lives inside it (the
        # normal case), absolute otherwise -- so --outdir can point anywhere.
        try:
            img_ref = str((imgdir / fname).relative_to(BENCH))
        except ValueError:
            img_ref = str((imgdir / fname).resolve())

        req_rows.append({
            "item_id": iid, "trial_type": tt,
            "image_path": img_ref,
            "messages": [{"role": "system", "content": sys_msg},
                         {"role": "user", "content": vis_prompt}],
        })
        # Perception-only probe over the same image: no task, no role, no rule --
        # just "what do you see". Lets the same stimuli be graded for seeing
        # separately from reasoning, and stays valid when --perception none keeps
        # the task prompt byte-comparable with the text run.
        perc_rows.append({
            "item_id": iid, "trial_type": tt,
            "image_path": img_ref,
            "messages": [{"role": "system", "content": sys_msg},
                         {"role": "user", "content": PERCEPTION_ONLY_PROMPT}],
        })
        labels.append({"item_id": iid, "trial_type": tt, "image_filename": fname,
                       "role": item.get("role", ""), "prompt": vis_prompt,
                       "objects": objs})
        for i, o in enumerate(objs):
            obj_rows.append({"image_id": fname, "object_id": i, **o})

    (out / "all_120_vlm_requests.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in req_rows) + "\n")
    (out / "all_120_perception_requests.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in perc_rows) + "\n")
    (out / "labels.json").write_text(json.dumps(labels, indent=2))
    # Tracked in git, unlike vision/ itself -- the prompts should be readable from
    # a clone without running anything.
    write_prompt_doc(BENCH / "VISION_PROMPTS.md", req_rows, args.perception)

    with open(out / "objects.csv", "w", newline="") as fh:
        cols = ["image_id", "object_id", "region", "slot", "color", "shape",
                "is_selected", "is_option", "label", "branch_type",
                "center_x", "center_y"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(obj_rows)

    (out / "dataset_info.txt").write_text(
        f"Dataset: adaptive_pedagogy_benchmark_v4_final vision stimuli\n"
        f"Rendered: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        f"Image size: {args.size}x{args.size}\n"
        f"Trials: {len(req_rows)} (100 main_joint + 10 choice + 10 diagnosis)\n"
        f"Objects drawn: {len(obj_rows)}\n"
        f"Perception in task prompt: {args.perception}\n"
        f"Perception-only requests: all_120_perception_requests.jsonl\n"
        f"Colors: {', '.join(BENCH_COLORS)}\n"
        f"Shapes: {', '.join(BENCH_SHAPES)} (hexagon built at runtime)\n"
        f"Source: benchmark/main_100_full.jsonl + control gold files\n"
        f"Gold for scoring: benchmark/all_120_gold.jsonl (unchanged)\n"
        f"Drawing stack: scripts/shape_bank.py + assets/shape_templates.npz\n"
        # Rendering is deterministic for a fixed Pillow, but PNG bytes differ
        # across Pillow versions -- pin it here so a re-render is checkable.
        f"Pillow: {PIL.__version__}\n")

    print(f"{len(req_rows)} images -> {imgdir}")
    print(f"{len(obj_rows)} objects -> {out/'objects.csv'}")
    print(f"requests    -> {out/'all_120_vlm_requests.jsonl'} "
          f"(perception={args.perception})")
    print(f"perception  -> {out/'all_120_perception_requests.jsonl'}")
    print(f"prompt doc  -> {BENCH/'VISION_PROMPTS.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
