#!/usr/bin/env python3
"""Figures for EpiSelect Level 2 Advanced Logic v3 results.

    python scripts/plot_results.py --results results_maxtok4096_temp0

Reads <results>/scores/<suite>__<model>__rep<NN>_summary.json (written by
scripts/score_ollama_runs.py) and emits one figure per suite present:

    primary_overview_{light,dark}.png       open inference + Teacher/Imposter
    null_policy_{light,dark}.png            default evidence policy, no role
    nohistory_calibration_{light,dark}.png  UNKNOWN / E calibration

Two Level-2 design constraints are enforced by the figures themselves:

1. Teacher and Imposter have different target-set sizes (1 of 4 vs 2 of 4), so
   their action-success bars carry their OWN reference lines and are never drawn
   against a shared chance line. README section 14.
2. Compatible-positive and Compatible-negative are one policy class with equal
   Imposter credit (README section 9), so they share a hue and differ by hatch
   rather than reading as two unrelated categories.
"""
import argparse, json, glob, os, re, collections, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BENCH, "results")
OUT = RES
SUBTITLE = ""

RUN_RE = re.compile(r"^(?P<suite>.+?)__(?P<model>.+)__rep(?P<rep>\d+)_summary\.json$")

# Chance / reference levels that follow from the design, not from any run.
CHANCE_RULE_EXACT = 1 / 16      # open inference over all 16 Boolean functions
CHANCE_RULE_AGREE = 0.50        # a random truth table agrees on 2 of 4 states
REF_ACTION = {"teacher": 0.25, "imposter": 0.50}   # 1 of 4 vs 2 of 4 targets

# Palette carried over from the v4 benchmark figures so the two families read as
# one system. The three hues were re-validated for CVD separation and contrast
# in both modes; the grays are status slots (a non-strategy), deliberately
# desaturated, and always carry a direct label as relief.
THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#7a7973",
                  grid="#e4e3df", chance="#8a8983",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
                  dead="#b6b5ae", dead2="#d3d2cb", ramp="#2a78d6"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", ink3="#93928a",
                 grid="#333331", chance="#7d7c76",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500"],
                 dead="#57565e", dead2="#3f3e3b", ramp="#3987e5"),
}

# semantic role -> (legend label, hue index or None for a status gray, hatch)
SEMANTIC_ORDER = [
    ("informative",         "informative (T≠H)",   0,    None),
    ("compatible_positive", "compatible +",        2,    None),
    ("compatible_negative", "compatible −",        2,    "///"),
    ("control_invalid",     "invalid label",       None, None),
    ("abstain",             "E (abstain)",         1,    None),
    ("invalid_format",      "unparseable",         None, "xxx"),
]


def label(slug):
    """gemma3_12b -> gemma3:12b."""
    return slug.rsplit("_", 1)[0] + ":" + slug.rsplit("_", 1)[1] if "_" in slug else slug


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load(res_dir):
    """{suite: {model: [summary, ...one per rep]}}"""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in sorted(glob.glob(os.path.join(res_dir, "scores", "*_summary.json"))):
        m = RUN_RE.match(os.path.basename(f))
        if not m:
            continue
        out[m["suite"]][m["model"]].append(json.load(open(f)))
    return out


def ms(summaries, model, key, default=0.0):
    """Mean and SD across reps for a scalar summary key (None -> default)."""
    v = [s.get(key) for s in summaries.get(model, [])]
    v = [default if x is None else float(x) for x in v]
    if not v:
        return default, 0.0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)


def role_share(summaries, model, role, semantic):
    """Share of a role's choices falling in one semantic class, mean over reps.

    Read from semantic_choice_counts_by_role so the composition always sums to
    1 even when a class never occurs.
    """
    vals = []
    for s in summaries.get(model, []):
        counts = (s.get("semantic_choice_counts_by_role") or {}).get(role, {})
        n = sum(counts.values())
        if n:
            vals.append(counts.get(semantic, 0) / n)
    return float(np.mean(vals)) if vals else 0.0


def breakdown(summaries, model, field, key):
    """{group: mean value} from summary['by_<field>'][group][key]."""
    acc = collections.defaultdict(list)
    for s in summaries.get(model, []):
        for g, d in (s.get(f"by_{field}") or {}).items():
            if d.get(key) is not None:
                acc[g].append(float(d[key]))
    return {g: float(np.mean(v)) for g, v in acc.items()}


# --------------------------------------------------------------------------
# drawing primitives
# --------------------------------------------------------------------------

def style(ax, T):
    ax.set_facecolor(T["surface"])
    ax.grid(axis="y", color=T["grid"], lw=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(T["grid"])
    ax.tick_params(colors=T["ink3"], labelsize=8, length=0)


def xlabels(ax, T, models, rotation=18):
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([label(m) for m in models], fontsize=8, color=T["ink2"],
                       rotation=rotation, ha="right")


def title(ax, T, text, sub=None):
    # Constant pad whether or not there is a subtitle, so panel titles in the
    # same row sit on one line.
    ax.set_title(text, fontsize=10, color=T["ink"], fontweight="600", loc="left",
                 pad=16)
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=7.3,
                color=T["ink3"], style="italic", va="bottom")


def bars(ax, T, models, vals, errs, chance=None, chance_lbl="chance",
         ylabel="", ymax=1.0, color_idx=0):
    x = np.arange(len(models))
    # One hue for the whole series: identity comes from the axis labels here, so
    # cycling the categorical ramp per bar would spend colour for nothing.
    ax.bar(x, vals, width=0.62, color=T["series"][color_idx], zorder=3,
           edgecolor=T["surface"], linewidth=2)
    if any(e > 0 for e in errs):
        ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor=T["ink2"],
                    elinewidth=1.5, capsize=3, zorder=4)
    if chance is not None:
        ax.axhline(chance, color=T["chance"], lw=1.5, ls=(0, (4, 3)), zorder=2)
        ax.text(len(models) - 0.42, chance + 0.018, chance_lbl, ha="right",
                va="bottom", fontsize=7.5, color=T["ink3"], style="italic")
    for xi, v, e in zip(x, vals, errs):
        ax.text(xi, v + e + 0.022, f"{v:.2f}", ha="center", va="bottom",
                fontsize=8, color=T["ink"], fontweight="600")
    xlabels(ax, T, models)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel, fontsize=8.5, color=T["ink2"])


def grouped(ax, T, models, series, hlines=(), ylabel="", ymax=1.0, legend=True):
    """series: list of (name, vals, errs, hue_idx)."""
    n = len(series)
    x = np.arange(len(models))
    w = 0.78 / n
    for i, (name, vals, errs, ci) in enumerate(series):
        off = (i - (n - 1) / 2) * w
        ax.bar(x + off, vals, width=w * 0.9, color=T["series"][ci], label=name,
               zorder=3, edgecolor=T["surface"], linewidth=2)
        if any(e > 0 for e in errs):
            ax.errorbar(x + off, vals, yerr=errs, fmt="none", ecolor=T["ink2"],
                        elinewidth=1.2, capsize=2, zorder=4)
        for xi, v, e in zip(x + off, vals, errs):
            ax.text(xi, v + e + 0.02, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=6.8, color=T["ink2"])
    ax.set_xlim(-0.78, len(models) - 0.32)
    for lbl, lvl in hlines:
        ax.axhline(lvl, color=T["chance"], lw=1.3, ls=(0, (4, 3)), zorder=2)
        ax.text(-0.74, lvl + 0.015, lbl, ha="left", va="bottom", fontsize=7,
                color=T["ink3"], style="italic")
    xlabels(ax, T, models)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel, fontsize=8.5, color=T["ink2"])
    if legend:
        ax.legend(fontsize=7.5, frameon=False, loc="upper right", ncol=n,
                  labelcolor=T["ink2"], handlelength=1.2, columnspacing=1.0)


def semantic_stack(ax, T, models, get_share, group_labels=("teacher", "imposter"),
                   ylabel="share of choices", legend_below=False):
    """Stacked composition of the semantic choice classes.

    get_share(model, group, semantic) -> float. One bar per (model, group); the
    classes partition the choices and sum to 1, so composition IS the result.
    """
    ng = len(group_labels)
    x = np.arange(len(models))
    w = 0.74 / ng
    for gi, g in enumerate(group_labels):
        off = (gi - (ng - 1) / 2) * w
        bottom = np.zeros(len(models))
        for key, _, hue, hatch in SEMANTIC_ORDER:
            v = np.array([get_share(m, g, key) for m in models])
            if not v.any():
                continue
            col = T["dead"] if hue is None else T["series"][hue]
            ax.bar(x + off, v, bottom=bottom, width=w * 0.88, color=col,
                   hatch=hatch, zorder=3, edgecolor=T["surface"], linewidth=2)
            for xi, vi, bi in zip(x + off, v, bottom):
                if vi > 0.09:
                    # Stroked: a hatched segment shows surface-coloured lines
                    # through the fill, and plain white text disappears into them.
                    ax.text(xi, bi + vi / 2, f"{vi:.2f}", ha="center", va="center",
                            fontsize=6.4, color=T["surface"], fontweight="700",
                            path_effects=[pe.withStroke(linewidth=1.6,
                                                        foreground=col)])
            bottom += v
        if ng > 1:
            for xi in x:
                ax.text(xi + off, 1.02, g[0].upper(), ha="center", va="bottom",
                        fontsize=6.5, color=T["ink3"])
    xlabels(ax, T, models)
    # Headroom for the legend when it sits inside: the bars always sum to 1.0,
    # so the space above them is free and costs no resolution.
    ax.set_ylim(0, 1.12 if legend_below else 1.62)
    ax.set_ylabel(ylabel, fontsize=8.5, color=T["ink2"])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    handles = [Patch(facecolor=(T["dead"] if h is None else T["series"][h]),
                     hatch=hh, edgecolor=T["surface"], label=name)
               for _, name, h, hh in SEMANTIC_ORDER]
    if legend_below:
        ax.legend(handles=handles, fontsize=6.6, frameon=False, loc="lower center",
                  bbox_to_anchor=(0.5, -0.40), ncol=3, labelcolor=T["ink2"],
                  handlelength=1.1, columnspacing=0.9)
    else:
        ax.legend(handles=handles, fontsize=6.8, frameon=False, loc="upper center",
                  ncol=2, labelcolor=T["ink2"], handlelength=1.1,
                  columnspacing=0.9, labelspacing=0.35)


def heatmap(ax, T, models, groups, table, vmax=1.0, cbar_label=""):
    """table[model][group] -> value. Sequential single hue, light to dark."""
    cmap = LinearSegmentedColormap.from_list("seq", [T["surface"], T["ramp"]])
    M = np.array([[table.get(m, {}).get(g, np.nan) for g in groups] for m in models])
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect="auto", zorder=3)
    for i in range(len(models)):
        for j in range(len(groups)):
            if np.isnan(M[i, j]):
                continue
            # Ink flips on the dark half of the ramp so the value stays legible.
            c = T["surface"] if M[i, j] > 0.55 * vmax else T["ink"]
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.6, color=c, fontweight="600")
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups, fontsize=7, color=T["ink2"], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([label(m) for m in models], fontsize=7.5, color=T["ink2"])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=T["ink3"], labelsize=6.5, length=0)
    cb.set_label(cbar_label, fontsize=7, color=T["ink2"])


def finish(fig, T, path, head, sub):
    # Header positions are computed from the figure height in inches, not fixed
    # fractions: a 5-inch figure and a 9-inch one otherwise get wildly different
    # gaps, and the short ones collide.
    h = fig.get_size_inches()[1]
    fig.suptitle(head, fontsize=13.5, color=T["ink"], fontweight="700",
                 x=0.008, ha="left", y=1 - 0.30 / h)
    fig.text(0.008, 1 - 0.62 / h, sub, fontsize=8.5, color=T["ink2"], ha="left")
    fig.tight_layout(rect=[0, 0.005, 1, 1 - 0.88 / h])
    fig.savefig(path, dpi=200, facecolor=T["surface"])
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def primary_figure(mode, S, out_dir, subtitle, suite="primary"):
    T = THEME[mode]
    models = sorted(S)
    n_reps = max((len(v) for v in S.values()), default=1)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2))
    fig.patch.set_facecolor(T["surface"])
    for ax in axes.flat:
        style(ax, T)

    # A: the headline -- open inference over all 16 Boolean functions.
    ser = [("exact (truth-table equivalent)",
            *zip(*[ms(S, m, "rule_inference_accuracy") for m in models]), 0),
           ("mean 4-state agreement",
            *zip(*[ms(S, m, "mean_rule_truth_table_agreement") for m in models]), 2)]
    ser = [(n, list(a), list(b), c) for n, a, b, c in ser]
    grouped(axes[0, 0], T, models, ser,
            hlines=[(f"exact {CHANCE_RULE_EXACT:.2f}", CHANCE_RULE_EXACT),
                    (f"agreement {CHANCE_RULE_AGREE:.2f}", CHANCE_RULE_AGREE)],
            ylabel="accuracy", ymax=1.18)
    title(axes[0, 0], T, "A  Open rule inference",
          "chance is 1/16: the model writes a formula, no candidates are shown")

    # B: action success. Each role gets its OWN reference line -- the target sets
    # are different sizes, so a shared chance line would be a lie.
    ser = [("teacher (1 of 4)",
            *zip(*[ms(S, m, "teacher_action_success") for m in models]), 0),
           ("imposter (2 of 4)",
            *zip(*[ms(S, m, "imposter_action_success") for m in models]), 2)]
    ser = [(n, list(a), list(b), c) for n, a, b, c in ser]
    grouped(axes[0, 1], T, models, ser,
            hlines=[(f"teacher {REF_ACTION['teacher']:.2f}", REF_ACTION["teacher"]),
                    (f"imposter {REF_ACTION['imposter']:.2f}", REF_ACTION["imposter"])],
            ylabel="action success", ymax=1.18)
    title(axes[0, 1], T, "B  Action selection, by goal",
          "different target-set sizes — NOT comparable as equal accuracies")

    # C: what was actually chosen. The four classes partition the options, so the
    # composition is the result; compatible +/- share a hue (one policy class).
    semantic_stack(axes[0, 2], T, models,
                   lambda m, g, k: role_share(S, m, g, k))
    title(axes[0, 2], T, "C  Which evidence class was chosen",
          "T = teacher bar, I = imposter bar")

    # D: the strict, design-specific measures.
    ser = [("joint (rule + action)",
            *zip(*[ms(S, m, "joint_inference_action_success") for m in models]), 0),
           ("goal-sensitive contrast",
            *zip(*[ms(S, m, "strict_goal_sensitive_role_contrast") for m in models]), 2),
           ("four-cell quartet",
            *zip(*[ms(S, m, "strict_four_cell_quartet") for m in models]), 1)]
    ser = [(n, list(a), list(b), c) for n, a, b, c in ser]
    grouped(axes[1, 0], T, models, ser, ylabel="share", ymax=1.18)
    title(axes[1, 0], T, "D  Strict joint criteria",
          "contrast = both roles right AND choosing differently on the same context")

    # E: which logical structures fail. 10 families x models is a grid, not bars.
    fams = sorted({g for m in models for g in breakdown(S, m, "rule_family",
                                                        "inference_accuracy")})
    heatmap(axes[1, 1], T, models, fams,
            {m: breakdown(S, m, "rule_family", "inference_accuracy") for m in models},
            cbar_label="inference accuracy")
    title(axes[1, 1], T, "E  Rule inference by logical family")

    # F: does more required evidence hurt? PDD 3 = IFF and XOR contexts.
    pdds = sorted({g for m in models for g in breakdown(S, m, "pdd", "inference_accuracy")})
    ser = []
    for i, p in enumerate(pdds):
        v = [breakdown(S, m, "pdd", "inference_accuracy").get(p, 0.0) for m in models]
        ser.append((f"PDD {p} ({'2 obs' if p == '2' else '3 obs'})", v,
                    [0.0] * len(models), 0 if i == 0 else 2))
    if ser:
        grouped(axes[1, 2], T, models, ser,
                hlines=[(f"chance {CHANCE_RULE_EXACT:.2f}", CHANCE_RULE_EXACT)],
                ylabel="inference accuracy", ymax=1.18)
    title(axes[1, 2], T, "F  Diagnostic depth (PDD)",
          "PDD 3 = the IFF and XOR contexts, which need a third observation")

    head = (f"EpiSelect Level 2 Advanced Logic v3 — {len(models)} models, "
            f"240 role items × {n_reps} reps")
    if suite != "primary":
        head += f"  [{suite}]"
    return finish(fig, T, os.path.join(out_dir, f"{suite}_overview_{mode}.png"),
                  head, subtitle)


def null_figure(mode, S, out_dir, subtitle, suite="null"):
    T = THEME[mode]
    models = sorted(S)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2))
    fig.patch.set_facecolor(T["surface"])
    for ax in axes.flat:
        style(ax, T)

    # The null track has no correct answer by design, so the distribution over
    # evidence classes IS the finding -- there is nothing to score against.
    def share(m, _g, key):
        vals = [(s.get("choice_rates") or {}).get(key, 0.0) for s in S.get(m, [])]
        return float(np.mean(vals)) if vals else 0.0

    semantic_stack(axes[0], T, models, share, group_labels=("null",),
                   legend_below=True)
    title(axes[0], T, "A  Default evidence policy",
          "no role, no instruction to teach, correct, or agree")

    ser = [("invalid label", *zip(*[ms(S, m, "invalid_control_rate") for m in models]), 1),
           ("abstain (E)", *zip(*[ms(S, m, "abstention_rate") for m in models]), 0)]
    ser = [(n, list(a), list(b), c) for n, a, b, c in ser]
    # These two should be near zero; scale to the data so a real failure is not
    # flattened against a hardcoded ceiling.
    top = max([v for _, vals, _, _ in ser for v in vals] + [0.2])
    grouped(axes[1], T, models, ser, ylabel="share of choices", ymax=top * 1.45)
    title(axes[1], T, "B  Comprehension failures",
          "the invalid option's label is false under the true rule")

    # Does the choice track the user's rule, or is it a fixed habit? The two
    # hidden user rules of a base share identical options, so a model reading the
    # user must sometimes flip.
    ser = [("choice flips", *zip(*[ms(S, m, "paired_choice_flip_rate") for m in models]), 0),
           ("same policy class",
            *zip(*[ms(S, m, "paired_same_policy_class_rate") for m in models]), 2)]
    ser = [(n, list(a), list(b), c) for n, a, b, c in ser]
    grouped(axes[2], T, models, ser, ylabel="share of matched pairs", ymax=1.18)
    title(axes[2], T, "C  Sensitivity to the user, on matched pairs",
          "same four options, two different hidden user rules")

    return finish(fig, T, os.path.join(out_dir, f"{suite}_policy_{mode}.png"),
                  f"Helpful-assistant framing — default evidence policy, "
                  f"{len(models)} models", subtitle)


def nohistory_figure(mode, S, out_dir, subtitle):
    T = THEME[mode]
    models = sorted(S)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    fig.patch.set_facecolor(T["surface"])
    for ax in axes.flat:
        style(ax, T)

    # With no observations, UNKNOWN and E are the correct answers, so 1.00 is the
    # ceiling here rather than a chance line.
    ser = [("rule = UNKNOWN", *zip(*[ms(S, m, "unknown_rate") for m in models]), 0),
           ("choice = E", *zip(*[ms(S, m, "abstention_rate") for m in models]), 2),
           ("both", *zip(*[ms(S, m, "unknown_and_abstain_rate") for m in models]), 1)]
    ser = [(n, list(a), list(b), c) for n, a, b, c in ser]
    grouped(axes[0], T, models, ser, hlines=[("correct = 1.00", 1.0)],
            ylabel="share of no-history items", ymax=1.18)
    title(axes[0], T, "A  Abstention when nothing can be inferred",
          "12 contexts with no observations: UNKNOWN + E is the only correct answer")

    ser = [("teacher", *zip(*[ms(S, m, "teacher_abstention_rate") for m in models]), 0),
           ("imposter", *zip(*[ms(S, m, "imposter_abstention_rate") for m in models]), 2)]
    ser = [(n, list(a), list(b), c) for n, a, b, c in ser]
    grouped(axes[1], T, models, ser, hlines=[("correct = 1.00", 1.0)],
            ylabel="share choosing E", ymax=1.18)
    title(axes[1], T, "B  Abstention by goal")

    return finish(fig, T, os.path.join(out_dir, f"nohistory_calibration_{mode}.png"),
                  f"No-history calibration control — {len(models)} models", subtitle)


def read_manifest(res_dir):
    p = os.path.join(res_dir, "run_manifest.json")
    return json.load(open(p)) if os.path.exists(p) else None


# --------------------------------------------------------------------------

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
        seeds = f"seeds {base+1}–{base+m.get('reps', 1)}" if m.get("reps") else ""
        SUBTITLE = (f"Ollama, temperature {m['temperature']:g}, "
                    f"max_tokens {m.get('max_tokens')}, {seeds}.  "
                    f"Error bars = SD across {m.get('reps')} repetitions.  "
                    f"Dashed lines = design reference levels.")
    else:
        SUBTITLE = ("Error bars = SD across repetitions.  "
                    "Dashed lines = design reference levels.")

    suites = load(RES)
    if not suites:
        # Not an error: the job script calls this unconditionally, and a run that
        # died before scoring simply has nothing to draw.
        print(f"nothing to plot in {RES} (no scores/*_summary.json)", file=sys.stderr)
        sys.exit(0)

    print(f"results: {RES}")
    for suite, S in sorted(suites.items()):
        models = sorted(S)
        reps = max(len(v) for v in S.values())
        print(f"  {suite}: {len(models)} models x {reps} reps -> {', '.join(models)}")

    # Dispatch on the shape of each suite's summaries rather than its name, so an
    # ablation scored by the same scorer gets the same figure for free.
    for suite, S in sorted(suites.items()):
        probe = next(iter(S.values()))[0]
        for mode in ("light", "dark"):
            if "strict_goal_sensitive_role_contrast" in probe:
                p = primary_figure(mode, S, OUT, SUBTITLE, suite)
            elif "choice_rates" in probe:
                p = null_figure(mode, S, OUT, SUBTITLE, suite)
            elif "unknown_rate" in probe:
                p = nohistory_figure(mode, S, OUT, SUBTITLE)
            else:
                print(f"  no figure for suite {suite!r} (unrecognised summary shape)",
                      file=sys.stderr)
                break
            print("wrote", p)
