#!/usr/bin/env python3
"""Figures for the adaptive pedagogy benchmark v4-final results.

    python scripts/plot_results.py                          # results/ (temp 1.0)
    python scripts/plot_results.py --results results_temp0  # the temp 0 re-run
"""
import argparse, json, glob, os, collections, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BENCH, "results")
OUT = RES
SUBTITLE = ""
TITLE = "Adaptive Pedagogy Benchmark v4-final"

# Models are discovered from the results directory, so the same script serves the
# text run, the vision run, and any future lineup. Previously hardcoded to the
# five text models, which silently plotted nothing for a vision run.
MODELS: list[str] = []


def label(slug):
    """gemma3_12b -> gemma3:12b; llama3.2-vision_11b -> llama3.2-vision:11b."""
    return slug.rsplit("_", 1)[0] + ":" + slug.rsplit("_", 1)[1] if "_" in slug else slug


LABEL = type("L", (), {"__getitem__": staticmethod(label)})()

THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#7a7973",
                  grid="#e4e3df", chance="#8a8983",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], dead="#b6b5ae"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", ink3="#93928a",
                 grid="#333331", chance="#7d7c76",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500"], dead="#57565266"),
}


def load():
    """Return details rows and per-run summaries keyed by model."""
    details = collections.defaultdict(list)
    summaries = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(RES, "scores", "*_details.jsonl"))):
        m = os.path.basename(f).split("__")[1]
        details[m] += [json.loads(l) for l in open(f)]
    for f in sorted(glob.glob(os.path.join(RES, "scores", "*_summary.json"))):
        m = os.path.basename(f).split("__")[1]
        summaries[m].append(json.load(open(f)))
    return details, summaries


def ms(summaries, model, key):
    """Mean and SD across reps for a summary key (None -> 0)."""
    v = [s.get(key) for s in summaries[model]]
    v = [0.0 if x is None else float(x) for x in v]
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def chance_levels(details):
    """Empirical chance from gold: |acceptable| / 4, split by role.

    Chance depends only on the gold answer key, so any scored model serves --
    read it from whichever one has rows rather than a hard-coded name.
    """
    out = {}
    rows = []
    for m in details:
        rows = [r for r in details[m] if r["trial_type"] == "main_joint"]
        if rows:
            break
    if not rows:
        raise SystemExit("no main_joint rows found; cannot derive chance levels")
    for role in ("teacher", "imposter"):
        rr = [r for r in rows if r["role"] == role]
        out[role + "_action"] = np.mean([len(r["acceptable_answers"]) / 4 for r in rr])
        out[role + "_joint"] = 0.5 * out[role + "_action"]
    out["main_action"] = np.mean([len(r["acceptable_answers"]) / 4 for r in rows])
    out["main_joint"] = 0.5 * out["main_action"]
    out["rule"] = 0.5
    return out


def bars(ax, T, models, vals, errs, chance=None, chance_lbl="chance",
         title="", ylabel="", fmt="{:.2f}", ymax=1.0, dead=()):
    x = np.arange(len(models))
    # One hue for the whole series: identity here comes from the axis labels, so
    # colouring each bar differently would cycle the categorical ramp for no gain.
    # Models that returned no parseable JSON are greyed out instead.
    cols = [T["dead"] if m in (dead or ()) else T["series"][0] for m in models]
    b = ax.bar(x, vals, width=0.62, color=cols, zorder=3,
               edgecolor=T["surface"], linewidth=2)
    for r in b:
        r.set_linewidth(2)
    if any(e > 0 for e in errs):
        ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor=T["ink2"],
                    elinewidth=1.5, capsize=3, zorder=4)
    if chance is not None:
        ax.axhline(chance, color=T["chance"], lw=1.5, ls=(0, (4, 3)), zorder=2)
        ax.text(len(models) - 0.42, chance + 0.018, chance_lbl, ha="right",
                va="bottom", fontsize=7.5, color=T["ink3"], style="italic")
    for xi, v, e in zip(x, vals, errs):
        ax.text(xi, v + e + 0.022, fmt.format(v), ha="center", va="bottom",
                fontsize=8, color=T["ink"], fontweight="600")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m] for m in models], fontsize=8, color=T["ink2"],
                       rotation=18, ha="right")
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel, fontsize=8.5, color=T["ink2"])
    ax.set_title(title, fontsize=10, color=T["ink"], fontweight="600",
                 loc="left", pad=8)


def grouped(ax, T, models, series, chance_map=None, title="", ylabel="", ymax=1.0):
    """series: list of (name, vals, errs, color_idx)."""
    n = len(series)
    x = np.arange(len(models))
    w = 0.78 / n
    for i, (name, vals, errs, ci) in enumerate(series):
        off = (i - (n - 1) / 2) * w
        ax.bar(x + off, vals, width=w * 0.9, color=T["series"][ci], label=name,
               zorder=3, edgecolor=T["surface"], linewidth=2)
        ax.errorbar(x + off, vals, yerr=errs, fmt="none", ecolor=T["ink2"],
                    elinewidth=1.2, capsize=2, zorder=4)
        for xi, v, e in zip(x + off, vals, errs):
            ax.text(xi, v + e + 0.02, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=6.8, color=T["ink2"])
    ax.set_xlim(-0.78, len(models) - 0.32)
    if chance_map:
        for i, (lbl, lvl, ci) in enumerate(chance_map):
            ax.axhline(lvl, color=T["chance"], lw=1.3, ls=(0, (4, 3)), zorder=2)
            ax.text(-0.74, lvl + 0.015, lbl, ha="left", va="bottom",
                    fontsize=7, color=T["ink3"], style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m] for m in models], fontsize=8, color=T["ink2"],
                       rotation=18, ha="right")
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel, fontsize=8.5, color=T["ink2"])
    ax.set_title(title, fontsize=10, color=T["ink"], fontweight="600",
                 loc="left", pad=8)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right", ncol=n,
              labelcolor=T["ink2"], handlelength=1.2, columnspacing=1.0)


def style(ax, T):
    ax.set_facecolor(T["surface"])
    ax.grid(axis="y", color=T["grid"], lw=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(T["grid"])
    ax.tick_params(colors=T["ink3"], labelsize=8, length=0)


def dead_models(details, thresh=0.99):
    """Models whose responses almost never parsed -- their bars are meaningless.

    Derived from the scored rows rather than hardcoded, so a run where every model
    behaves is not silently mislabelled.
    """
    out = set()
    for m, rows in details.items():
        if rows and sum(1 for r in rows if not r.get("strict_json")) / len(rows) >= thresh:
            out.add(m)
    return out


def make_figure(mode, details, summaries, ch):
    T = THEME[mode]
    dead = dead_models(details)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
    fig.patch.set_facecolor(T["surface"])
    for ax in axes.flat:
        style(ax, T)

    # A: headline joint success
    v, e = zip(*[ms(summaries, m, "main_joint_success") for m in MODELS])
    bars(axes[0, 0], T, MODELS, list(v), list(e), chance=ch["main_joint"],
         chance_lbl=f"chance {ch['main_joint']:.2f}",
         title="A  Joint success (100 main trials)",
         ylabel="rule + action both correct", dead=dead)
    for i, m in enumerate(MODELS):
        if m in dead:
            axes[0, 0].text(i, 0.075, "no valid\nJSON", ha="center", va="bottom",
                            fontsize=7, color=T["ink3"], style="italic")

    # B: rule inference vs action, split by role
    s = [("rule inference", *zip(*[ms(summaries, m, "main_rule_inference_accuracy")
                                   for m in MODELS]), 0),
         ("action selection", *zip(*[ms(summaries, m, "main_action_success")
                                     for m in MODELS]), 1)]
    s = [(n, list(a), list(b), c) for n, a, b, c in s]
    grouped(axes[0, 1], T, MODELS, s,
            chance_map=[("rule 0.50", 0.5, 0), (f"action {ch['main_action']:.2f}",
                                                ch["main_action"], 1)],
            title="B  The two capacities, scored separately",
            ylabel="accuracy", ymax=1.15)

    # C: teacher vs imposter joint
    s = [("teacher", *zip(*[ms(summaries, m, "teacher_joint_accuracy")
                            for m in MODELS]), 2),
         ("imposter", *zip(*[ms(summaries, m, "imposter_joint_success")
                             for m in MODELS]), 3)]
    s = [(n, list(a), list(b), c) for n, a, b, c in s]
    grouped(axes[1, 0], T, MODELS, s,
            chance_map=[(f"teacher {ch['teacher_joint']:.2f}", ch["teacher_joint"], 2),
                        (f"imposter {ch['imposter_joint']:.2f}", ch["imposter_joint"], 3)],
            title="C  Role asymmetry (imposter accepts 2 of 4)",
            ylabel="joint success", ymax=1.18)

    # D: reported rule, COLOR vs SHAPE bias
    ax = axes[1, 1]
    x = np.arange(len(MODELS))
    for i, (rule, ci) in enumerate([("COLOR_RULE", 0), ("SHAPE_RULE", 2)]):
        frac = []
        for m in MODELS:
            rr = [r for r in details[m] if r["trial_type"] == "main_joint"]
            frac.append(sum(1 for r in rr if r.get("reported_inferred_rule") == rule)
                        / len(rr))
        ax.bar(x + (i - 0.5) * 0.34, frac, width=0.3, color=T["series"][ci],
               label=rule, zorder=3, edgecolor=T["surface"], linewidth=2)
        for xi, v in zip(x + (i - 0.5) * 0.34, frac):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=6.8, color=T["ink2"])
    ax.axhline(0.5, color=T["chance"], lw=1.5, ls=(0, (4, 3)), zorder=2)
    ax.set_xlim(-0.95, len(MODELS) - 0.35)
    ax.text(-0.92, 0.52, "ground truth\n50 / 50", fontsize=7, color=T["ink3"],
            style="italic", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m] for m in MODELS], fontsize=8, color=T["ink2"],
                       rotation=18, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("share of main trials", fontsize=8.5, color=T["ink2"])
    ax.set_title("D  Response bias: which rule gets reported", fontsize=10,
                 color=T["ink"], fontweight="600", loc="left", pad=8)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right", ncol=2,
              labelcolor=T["ink2"], handlelength=1.2)

    n_items = len({r["item_id"] for rows in details.values() for r in rows})
    n_reps = max((len(v) for v in summaries.values()), default=1)
    fig.suptitle(f"{TITLE} — {len(MODELS)} models, {n_items} trials × {n_reps} reps",
                 fontsize=13.5, color=T["ink"], fontweight="700", x=0.008, ha="left",
                 y=0.982)
    fig.text(0.008, 0.938, SUBTITLE, fontsize=8.5, color=T["ink2"], ha="left")
    fig.tight_layout(rect=[0, 0.005, 1, 0.925])
    p = os.path.join(OUT, f"benchmark_overview_{mode}.png")
    fig.savefig(p, dpi=200, facecolor=T["surface"])
    plt.close(fig)
    return p



# ==========================================================================
# Perception probe figures
# ==========================================================================

def perception_figure(mode, rows, out_dir, subtitle):
    """rows: dicts from perception_metrics.csv, one per (model, rep)."""
    T = THEME[mode]
    models = sorted({r["model"] for r in rows})

    def ms_(model, key):
        v = [float(r[key]) for r in rows if r["model"] == model and r.get(key) not in (None, "")]
        return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0) if v else (0.0, 0.0)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
    fig.patch.set_facecolor(T["surface"])
    for ax in axes.flat:
        style(ax, T)

    # A: can the model name the objects at all?
    ser = []
    for i, (key, name) in enumerate([("color_acc", "colour"), ("shape_acc", "shape"),
                                     ("binding_acc", "colour AND shape")]):
        v, e = zip(*[ms_(m, key) for m in models])
        ser.append((name, list(v), list(e), i))
    grouped(axes[0, 0], T, models, ser,
            chance_map=[("0.20", 0.2, 0), ("0.04", 0.04, 2)],
            title="A  What the model can see   (chance: 0.20 per attribute, "
                  "0.04 for both)",
            ylabel="accuracy on reported objects", ymax=1.18)

    # B: the selection border -- the one cue the task depends on.
    ser = []
    for i, (key, name) in enumerate([("selection_precision", "precision"),
                                     ("selection_recall", "recall")]):
        v, e = zip(*[ms_(m, key) for m in models])
        ser.append((name, list(v), list(e), i))
    grouped(axes[0, 1], T, models, ser,
            chance_map=[("0.50", 0.5, 0)],
            title="B  Detecting the solid black selection border   (guessing = 0.50)",
            ylabel="on history objects", ymax=1.18)

    # C: did a well-formed object list come back at all?
    ser = []
    for i, (key, name) in enumerate([("parse_rate", "parsed"),
                                     ("count_match", "right number of objects")]):
        v, e = zip(*[ms_(m, key) for m in models])
        ser.append((name, list(v), list(e), i))
    grouped(axes[1, 0], T, models, ser, title="C  Response well-formedness",
            ylabel="share of images", ymax=1.18)

    # D: which shapes get confused -- a stimulus-design question, not a model one.
    conf = collections.Counter()
    for f in sorted(glob.glob(os.path.join(out_dir, "scores", "*_perception.json"))):
        for c in json.load(open(f)).get("top_shape_confusions", []):
            conf[f"{c['truth']} -> {c['reported']}"] += c["n"]
    ax = axes[1, 1]
    top = conf.most_common(6)[::-1]
    if top:
        y = np.arange(len(top))
        ax.barh(y, [c for _, c in top], color=T["series"][1], height=0.62,
                zorder=3, edgecolor=T["surface"], linewidth=2)
        for yi, (_, c) in zip(y, top):
            ax.text(c + max(conf.values()) * 0.015, yi, str(c), va="center",
                    fontsize=7.5, color=T["ink2"])
        ax.set_yticks(y)
        ax.set_yticklabels([k for k, _ in top], fontsize=8, color=T["ink2"])
        ax.set_xlim(0, max(c for _, c in top) * 1.15)
    ax.grid(axis="x", color=T["grid"], lw=1, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("objects, summed over models and reps", fontsize=8.5, color=T["ink2"])
    ax.set_title("D  Most common shape confusions", fontsize=10, color=T["ink"],
                 fontweight="600", loc="left", pad=8)

    fig.suptitle(f"Perception probe — what the VLM reports seeing, {len(models)} models",
                 fontsize=13.5, color=T["ink"], fontweight="700", x=0.008, ha="left", y=0.982)
    fig.text(0.008, 0.938, subtitle, fontsize=8.5, color=T["ink2"], ha="left")
    fig.tight_layout(rect=[0, 0.005, 1, 0.925])
    path = os.path.join(out_dir, f"perception_overview_{mode}.png")
    fig.savefig(path, dpi=200, facecolor=T["surface"])
    plt.close(fig)
    return path


# ==========================================================================
# assistant_framing figures
# ==========================================================================

PROFILE_ORDER = [("teacher_corrective", "teacher: corrective"),
                 ("imposter_specific", "imposter: mimics user"),
                 ("conservative", "conservative: fits both"),
                 ("invalid", "invalid")]


def assistant_figure(mode, summary, out_dir, subtitle):
    T = THEME[mode]
    models = sorted(summary)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0))
    fig.patch.set_facecolor(T["surface"])
    for ax in axes.flat:
        style(ax, T)

    # A: the whole distribution. Stacked because the four profiles partition the
    # choices and sum to 1 -- the composition IS the result.
    ax = axes[0]
    x = np.arange(len(models))
    bottom = np.zeros(len(models))
    for i, (key, name) in enumerate(PROFILE_ORDER):
        v = np.array([summary[m]["profile_rates"].get(key, 0.0) for m in models])
        col = T["dead"] if key == "invalid" else T["series"][i]
        ax.bar(x, v, bottom=bottom, width=0.6, color=col, label=name, zorder=3,
               edgecolor=T["surface"], linewidth=2)
        for xi, vi, bi in zip(x, v, bottom):
            if vi > 0.06:
                ax.text(xi, bi + vi / 2, f"{vi:.2f}", ha="center", va="center",
                        fontsize=7.5, color=T["surface"], fontweight="600")
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m] for m in models], fontsize=8, color=T["ink2"],
                       rotation=18, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("share of choices", fontsize=8.5, color=T["ink2"])
    ax.set_title("A  Which profile does the answer match", fontsize=10, color=T["ink"],
                 fontweight="600", loc="left", pad=8)
    ax.legend(fontsize=7, frameon=False, loc="lower center", ncol=2,
              bbox_to_anchor=(0.5, -0.42), labelcolor=T["ink2"], handlelength=1.2)

    # B: matched halves. A profile that flips between them tracks the stimulus,
    # not a disposition.
    ser = []
    for i, rule in enumerate(("COLOR_RULE", "SHAPE_RULE")):
        v = [summary[m]["by_user_rule"].get(rule, {}).get("teacher_corrective", 0.0)
             for m in models]
        ser.append((f"user used {rule}", v, [0.0] * len(models), i))
    grouped(axes[1], T, models, ser,
            title="B  Teacher-like choice, by the rule the user showed",
            ylabel="share choosing the corrective example", ymax=1.05)

    # C: same item, same answer across reps.
    v = [summary[m].get("answer_consistency_across_reps") or 0.0 for m in models]
    bars(axes[2], T, models, v, [0.0] * len(models),
         title="C  Same answer across repetitions", ylabel="share of items", ymax=1.05)

    fig.suptitle("assistant_framing — no role, no instruction to teach "
                 "(50 teacher items, recorded not scored)",
                 fontsize=13, color=T["ink"], fontweight="700", x=0.006, ha="left", y=0.975)
    fig.text(0.006, 0.918, subtitle, fontsize=8.5, color=T["ink2"], ha="left")
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    path = os.path.join(out_dir, f"assistant_framing_{mode}.png")
    fig.savefig(path, dpi=200, facecolor=T["surface"])
    plt.close(fig)
    return path


def read_manifest(res_dir):
    """Pull the real decoding settings out of the run manifest for the subtitle."""
    p = os.path.join(res_dir, "run_manifest.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=RES,
                    help="results directory (absolute, or relative to the benchmark root)")
    ap.add_argument("--out", default=None, help="where to write the PNGs (default: alongside)")
    args = ap.parse_args()

    RES = args.results if os.path.isabs(args.results) else os.path.join(BENCH, args.results)
    OUT = args.out or RES
    os.makedirs(OUT, exist_ok=True)

    m = read_manifest(RES)
    if m:
        base = m.get("base_seed", 1000)
        seeds = f"seeds {base+1}–{base+m.get('reps',1)}" if m.get("reps") else ""
        SUBTITLE = (f"Ollama, temperature {m['temperature']:g}, "
                    f"max_tokens {m.get('max_tokens')}, {seeds}.  "
                    f"Error bars = SD across {m.get('reps')} repetitions.  "
                    f"Dashed lines = chance.")
    else:
        SUBTITLE = "Error bars = SD across repetitions.  Dashed lines = chance."

    # Dispatch on what the directory actually holds, so the same command works
    # for a task run, the perception probe, or a record-only condition.
    perc_csv = os.path.join(RES, "perception_metrics.csv")
    asst_json = os.path.join(RES, "assistant_framing_summary.json")

    if os.path.exists(perc_csv):
        import csv as _csv
        rows = list(_csv.DictReader(open(perc_csv)))
        if rows:
            print(f"perception run: {len({r['model'] for r in rows})} models")
            for mode in ("light", "dark"):
                print("wrote", perception_figure(mode, rows, OUT, SUBTITLE))
            sys.exit(0)

    if os.path.exists(asst_json):
        summary = json.load(open(asst_json))
        if summary:
            print(f"assistant_framing run: {len(summary)} models")
            for mode in ("light", "dark"):
                print("wrote", assistant_figure(mode, summary, OUT, SUBTITLE))
            sys.exit(0)

    details, summaries = load()
    if not summaries:
        # Not an error: a record-only run has no scores/ directory, and the job
        # script calls this unconditionally.
        print(f"nothing to plot in {RES} (no scores/, no perception or "
              f"assistant_framing outputs)", file=sys.stderr)
        sys.exit(0)
    # Stable order so a model keeps its position (and the greyed-out slot stays
    # put) when a later run adds another model to the same directory.
    MODELS = sorted(summaries)
    if m and m.get("backend"):                       # written by run_vlm.py only
        TITLE = "Adaptive Pedagogy Benchmark v4-final — VISION"
    print("models:", ", ".join(MODELS))
    ch = chance_levels(details)
    print(f"results: {RES}")
    print("chance:", {k: round(v, 3) for k, v in ch.items()})
    for mode in ("light", "dark"):
        print("wrote", make_figure(mode, details, summaries, ch))
