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

MODELS = ["gemma3_12b", "llama3.1_8b", "llama3.2_3b", "mistral_7b", "qwen3_8b"]
LABEL = {"gemma3_12b": "gemma3:12b", "llama3.1_8b": "llama3.1:8b",
         "llama3.2_3b": "llama3.2:3b", "mistral_7b": "mistral:7b",
         "qwen3_8b": "qwen3:8b"}

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
         title="", ylabel="", fmt="{:.2f}", ymax=1.0):
    x = np.arange(len(models))
    cols = []
    for m in models:
        cols.append(T["dead"] if m == "qwen3_8b" else T["series"][models.index(m) % 4])
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
    ax.set_xlim(-0.62, len(models) - 0.38)
    if chance_map:
        for i, (lbl, lvl, ci) in enumerate(chance_map):
            ax.axhline(lvl, color=T["chance"], lw=1.3, ls=(0, (4, 3)), zorder=2)
            ax.text(len(models) - 0.42, lvl + 0.015, lbl, ha="right", va="bottom",
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


def make_figure(mode, details, summaries, ch):
    T = THEME[mode]
    valid = [m for m in MODELS if m != "qwen3_8b"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
    fig.patch.set_facecolor(T["surface"])
    for ax in axes.flat:
        style(ax, T)

    # A: headline joint success
    v, e = zip(*[ms(summaries, m, "main_joint_success") for m in MODELS])
    bars(axes[0, 0], T, MODELS, list(v), list(e), chance=ch["main_joint"],
         chance_lbl=f"chance {ch['main_joint']:.2f}",
         title="A  Joint success (100 main trials)",
         ylabel="rule + action both correct")
    axes[0, 0].text(4, 0.075, "no valid\nJSON", ha="center", va="bottom",
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
    ax.text(-0.48, 0.52, "ground truth 50 / 50", fontsize=7, color=T["ink3"],
            style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m] for m in MODELS], fontsize=8, color=T["ink2"],
                       rotation=18, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("share of main trials", fontsize=8.5, color=T["ink2"])
    ax.set_title("D  Response bias: which rule gets reported", fontsize=10,
                 color=T["ink"], fontweight="600", loc="left", pad=8)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right", ncol=2,
              labelcolor=T["ink2"], handlelength=1.2)

    fig.suptitle("Adaptive Pedagogy Benchmark v4-final — 5 local models, 120 trials × 3 reps",
                 fontsize=13.5, color=T["ink"], fontweight="700", x=0.008, ha="left",
                 y=0.982)
    fig.text(0.008, 0.938, SUBTITLE, fontsize=8.5, color=T["ink2"], ha="left")
    fig.tight_layout(rect=[0, 0.005, 1, 0.925])
    p = os.path.join(OUT, f"benchmark_overview_{mode}.png")
    fig.savefig(p, dpi=200, facecolor=T["surface"])
    plt.close(fig)
    return p


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

    details, summaries = load()
    if not summaries:
        sys.exit(f"no scored runs found in {RES}/scores/")
    ch = chance_levels(details)
    print(f"results: {RES}")
    print("chance:", {k: round(v, 3) for k, v in ch.items()})
    for mode in ("light", "dark"):
        print("wrote", make_figure(mode, details, summaries, ch))
