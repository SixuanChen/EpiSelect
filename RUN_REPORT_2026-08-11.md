# Adaptive Pedagogy Benchmark v4-final — first full run

**Run date:** 2026-08-11 · **Analysis date:** 2026-08-11

Artifacts:

- Figure: `results/benchmark_overview_light.png` (and `_dark.png`)
- Plot script: `scripts/plot_results.py` (needs `module load anaconda3/2023.09-0-aqbc`)
- Metrics: `results/aggregate_metrics.csv`, `results/per_run_metrics.csv`
- Trial-level: `results/scores/*_details.jsonl`
- Raw model output: `results/raw/*.jsonl`
- Run config: `results/run_manifest.json`

---

## 1. What the benchmark measures

120 trials testing two capacities, scored separately and jointly.

Each trial gives a category rule of the form `COLOR c OR SHAPE s` (25 base rules
from 5 colors × 5 shapes), states that the other person is using either the color
branch or the shape branch, and shows two of their past choices. The model must:

1. **Infer** which branch that person is on (`COLOR_RULE` vs `SHAPE_RULE`), and
2. **Select** one of four objects according to its role.

The four options are one of each semantic class — `intersection_positive`,
`color_only_positive`, `shape_only_positive`, `negative` — in counterbalanced
A/B/C/D order.

| Role | Goal | Correct choice |
|---|---|---|
| **TEACHER** | Correct the learner's too-narrow rule | The unique counterexample: a positive example their current rule excludes (1 of 4 → chance 0.25) |
| **IMPOSTER** | Obey a private rule while blending in with the civilian | Either the specific blend-in *or* the conservative intersection (**2 of 4 → chance 0.50**) |

**Composition:** 100 main trials (25 rules × 2 other-person rules × 2 roles, forming
a fully matched 2×2 quartet per base rule) + 10 no-history "say UNKNOWN" controls
+ 10 no-history "say INSUFFICIENT" controls.

Output is strict JSON (`{"inferred_rule":...,"answer":...}`), auto-scored, no LLM judge.

## 2. Run configuration

| | |
|---|---|
| Models | `llama3.1:8b`, `llama3.2:3b`, `qwen3:8b`, `gemma3:12b`, `mistral:7b` |
| Backend | Ollama @ 127.0.0.1:11466, `format=json` constrained decoding |
| Temperature | **1.0**, `top_p` unset, base seed 1000 → seeds 1001–1003 |
| Budget | `num_predict` 256, `num_ctx` 4096 |
| Scale | 3 reps × 120 items × 5 models = **1800 calls** |
| Compute | Oscar SLURM, `gpu-he` / `carney-tserre-condo2`, 1 GPU, 64 G, 4 CPU |
| Wall clock | 2026-08-11 21:37 → 22:54 UTC (77 min) |

> **Caveat:** `README.md` §"Recommended first-run settings" asks for sampling off
> for a clean capability floor. This run used temperature 1.0, so the reported SDs
> mix sampling noise with genuine run-to-run instability.

## 3. Results (main 100 trials, mean over 3 reps)

| model | joint | rule inf. | action | teacher joint | imposter joint |
|---|---|---|---|---|---|
| gemma3:12b | **0.61** | 0.88 | 0.64 | 0.24 | 0.98 |
| llama3.1:8b | 0.45 | 0.70 | 0.54 | 0.07 | 0.84 |
| llama3.2:3b | 0.27 | 0.57 | 0.44 | 0.01 | 0.53 |
| mistral:7b | 0.29 | 0.32 | 0.65 | 0.08 | 0.49 |
| qwen3:8b | — invalid, see §5 — | | | | |
| **chance** | 0.19 | 0.50 | 0.38 | 0.125 | 0.25 |

Teacher action selection alone (150 trials/model, ignoring rule inference):
gemma3 0.28 · llama3.1 0.15 · llama3.2 0.13 · **mistral 0.55**.

---

## 4. Message for PI

> Subject: Adaptive Pedagogy Benchmark v4 — first full run (5 local models), results + two issues

I finished the first complete run of the v4 benchmark: 5 open-weight models via
Ollama on Oscar (gpu-he), 120 trials × 3 repetitions each = 1800 calls, ~77 min
wall clock. Temperature 1.0, seeds 1001–1003, 256-token budget, JSON-constrained
decoding. Figure attached (`results/benchmark_overview_light.png`); metrics in
`results/aggregate_metrics.csv`.

### Headline numbers (main 100 trials, mean over 3 reps)

```
  model         joint   rule inf.  action   teacher joint  imposter joint
  gemma3:12b     0.61     0.88      0.64        0.24            0.98
  llama3.1:8b    0.45     0.70      0.54        0.07            0.84
  llama3.2:3b    0.27     0.57      0.44        0.01            0.53
  mistral:7b     0.29     0.32      0.65        0.08            0.49
  qwen3:8b        — invalid, see issue 1 —
  chance         0.19     0.50      0.38        0.125           0.25
```

### What I think this shows

1. **The headline is carried almost entirely by the imposter role.** Teacher joint
   accuracy is at or below its 0.125 chance level for three of four models.
   Note the two roles do not have the same chance level: the imposter scorer
   accepts both the specific blend-in and the conservative intersection, so
   2 of 4 options count as correct (chance 0.50) versus 1 of 4 for teacher
   (chance 0.25). Some of the role gap is this scoring asymmetry, not ability.

2. **The mechanism looks like a single default rather than audience reasoning.**
   In the teacher role llama3.1 picks the "intersection" object (satisfies both
   branches) on 79% of trials and gemma3 on 70%. That object is always true but
   never diagnostic — it is exactly the choice that scores as success for the
   imposter and failure for the teacher. So one habit is being graded twice with
   opposite answer keys, which accounts for most of the asymmetry in point 1.

3. **Rule inference is contaminated by a color prior.** Ground truth is 50/50
   COLOR vs SHAPE, but llama3.2:3b reports COLOR_RULE on 93% of trials and
   mistral:7b on 82%. mistral's rule accuracy of 0.32 is below chance, i.e.
   systematically anti-correlated rather than merely noisy.

4. **Diagnosis is not stable across the matched quartet.** gemma3's quartet joint
   consistency is 0.08: the same underlying item, with only the other person's
   behaviour and the model's goal varying, is almost never handled coherently.
   Its rule inference also shifts with role on identical evidence (0.98 as
   imposter, 0.80 as teacher).

5. **One model deserves a caveat in the other direction:** mistral:7b has the best
   teacher action selection (0.55, roughly double gemma3's 0.28) but the worst
   rule inference, so it picks the right teaching example while naming the wrong
   learner rule. The joint metric hides this; it is a different failure than the
   other three.

### Two issues to fix before these numbers go anywhere

**Issue 1 — qwen3:8b produced no usable data, and this is our bug, not the model's.**
114/120 items returned `done_reason="length"` with exactly 256 tokens generated and
an EMPTY content string. Our runner reads only `message.content` and ignores
`message.thinking`, so qwen3's reasoning consumed the whole token budget in a
channel we never captured. The 6 items that did parse were the trivial no-history
controls, answered in 8 tokens with no reasoning. Fix is to pass `think=false`,
capture `message.thinking` as a fallback, and raise `num_predict` to ~1024 for
reasoning models. Until re-run, qwen3's zeros in `aggregate_metrics.csv` should be
read as missing data, not as task failure.

**Issue 2 — the "INSUFFICIENT" control returns 0.00 for every model on every rep.**
A universal floor with zero variance usually indicts the prompt rather than the
models, so I want to verify that instruction actually presents INSUFFICIENT as an
available answer before we report it as a finding. The parallel "UNKNOWN" control
behaves sensibly (gemma3 and mistral both 1.00), which makes the contrast more
suspicious.

### Proposed next steps

- Re-run with temperature 0 (the README's own recommended setting for a capability
  floor — 1.0 mixes sampling noise into the SDs) and with the qwen3 fix.
- Report teacher and imposter separately with their distinct chance levels rather
  than a pooled joint score, and consider reporting imposter specific-blend-in
  rate as the primary imposter measure since the pooled one is lenient.
- Run the `action_given_rule` ablation that already ships with the package. It hands
  the model the other person's rule directly, which would tell us whether the
  teacher failure is a diagnosis bottleneck or a genuine pedagogical-selection
  failure. Given that gemma3 infers the rule at 0.88 but teaches at 0.28, I expect
  it to be the latter, and that is the more interesting result.

Happy to walk through the figure whenever convenient.

---

## 5. Verification status

Flagged for the sender before this goes out:

- **Issue 2 is a suspicion, not a confirmed finding.** The control prompt text has
  not been read; the 0.00 could be a genuine model failure. Worth a five-minute
  check of `requests/controls_no_history_choice_10_requests.jsonl` first.
- **The qwen3 diagnosis is inferred from response logs**, not confirmed live: empty
  `content` + exactly 256 tokens generated + `scripts/run_ollama.py:245` reading only
  `message.content`. The Ollama server has since exited, so the `message.thinking`
  hypothesis was not tested against a running API.
- Everything else is computed directly from `results/scores/*_details.jsonl`.
