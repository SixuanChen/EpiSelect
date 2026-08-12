# Adaptive Pedagogy Benchmark v4-final

A self-contained, automatically generated and automatically scored benchmark for **learner-sensitive rule inference** and **audience-sensitive evidence selection**.

The benchmark starts deliberately simple: text-only colored shapes, OR category rules, two closed candidate hypotheses, and two clear behavioral observations. The goal is to establish a clean capability floor before scaling to material/size, open hypothesis spaces, relational rules, and rendered visual stimuli.

## What the benchmark measures

The main 100 trials measure two capacities **separately and jointly**:

1. **Rule inference**: From two observed choices, does the model infer whether the learner/civilian is using the relevant `COLOR_RULE` or `SHAPE_RULE`?
2. **Evidence/action selection**: Given that inferred rule and a role-specific goal, does the model choose an appropriate A/B/C/D example?

Every main response is strict JSON:

```json
{"inferred_rule":"COLOR_RULE","answer":"B"}
```

The automatic scorer therefore reports rule-inference accuracy and example-choice performance separately, plus joint success.

## Experimental design

### 25 base category rules

Five colors × five shapes:

- colors: red, blue, green, yellow, purple
- shapes: circle, square, triangle, star, hexagon

Every category rule has the form:

> `COLOR == c OR SHAPE == s`

For example:

> RED OR CIRCLE

The exact 25 rules are listed in:

- `BASE_RULES_25.md`
- `BASE_RULES_25.csv`
- `BASE_RULES_25.json`

### Two possible learner/civilian hypotheses

For a base rule such as RED OR CIRCLE, the other person is known to be using exactly one of:

- `COLOR_RULE`: RED objects belong, regardless of shape
- `SHAPE_RULE`: CIRCLE objects belong, regardless of color

The model is told these are the two candidates but is **not told which one is active**. It sees two clear pairwise choices and must infer the active rule.

### Two behavioral observations

Each history trial contrasts a color-only positive against a shape-only positive. The person consistently chooses according to their rule.

The option order is balanced so the person chooses the **left item once and the right item once**. History choices have no A/B labels.

### Four target evidence classes

Every main trial has exactly four target options, in counterbalanced A/B/C/D order:

1. `intersection_positive`: satisfies both COLOR_RULE and SHAPE_RULE
2. `color_only_positive`: satisfies only the color branch
3. `shape_only_positive`: satisfies only the shape branch
4. `negative`: satisfies neither branch and violates the category/private rule

No exact history object is reused as a target object.

### Two roles

#### TEACHER

The model must choose the **unique truthful counterexample** to the learner's current narrower rule.

For a RED learner under RED OR CIRCLE, the best teaching example is a non-red circle.

Teacher scoring has one unique correct A/B/C/D choice.

#### IMPOSTER

The model must obey its private RED OR CIRCLE rule while blending in with the civilian's inferred rule.

Two behaviors are accepted:

- `current_rule_specific`: e.g. a red non-circle for a RED civilian. This is the **specific/preferred** blend-in strategy.
- `conservative_intersection`: e.g. a red circle. This is a valid but less diagnostic blend-in strategy.

The scorer therefore reports both:

- `imposter_blend_in_success`
- `imposter_specific_blend_in_rate`

This avoids falsely treating a conservative intersection choice as incorrect.

## Full matched reversal

Every base rule produces the full 2×2 matched quartet:

| Other person's rule | Teacher target | Imposter specific target |
|---|---|---|
| `COLOR_RULE` | shape-only positive | color-only positive |
| `SHAPE_RULE` | color-only positive | shape-only positive |

Across the quartet, the category/private rule, target objects, and A/B/C/D target order are identical. Only the other person's behavior and the model's goal change.

This supports matched metrics such as:

- `teacher_learner_rule_reversal_consistency`
- `imposter_specific_rule_reversal_consistency`
- `specific_goal_reversal_consistency`
- `quartet_joint_consistency`

## 120 total benchmark trials

The final benchmark contains:

- **100 main joint trials** = 25 base rules × 2 other-person rules × 2 roles
- **10 no-history diagnosis controls**
- **10 no-history targeted-teaching controls**

Total = **120 trials**.

### No-history diagnosis controls

The model sees the category rule and the two candidate hypotheses but no learner behavior. The correct output is:

```json
{"inferred_rule":"UNKNOWN"}
```

This checks that the model does not manufacture a learner state when no evidence is available.

### No-history targeted-teaching controls

The model sees the rule and four targets but no learner behavior. Because no learner hypothesis can be identified, there is no uniquely targeted pedagogical example. The correct output is:

```json
{"answer":"INSUFFICIENT"}
```

These controls are sanity checks, not the main cognitive-capacity measure.

---

# Folder map

```text
adaptive_pedagogy_benchmark_v4_final/
├── README.md                         <- start here
├── BASE_RULES_25.md                  <- human-readable list of all 25 base rules
├── BASE_RULES_25.csv
├── BASE_RULES_25.json
├── ADVERSARIAL_REVIEW.md             <- design threats, fixes, remaining limits
├── generate_benchmark.py             <- regenerate all benchmark files from scratch
├── validate_benchmark.py             <- structural anti-shortcut checks
├── prepare_requests.py               <- build provider-neutral chat requests
├── score_predictions.py              <- automatic scoring + failure decomposition
├── self_test.py                      <- end-to-end package test
├── requirements.txt                  <- core is standard-library only
│
├── benchmark/
│   ├── all_120_public.jsonl           <- all 120 prompts, SAFE TO SEND TO MODEL
│   ├── all_120_gold.jsonl             <- all 120 private gold annotations, DO NOT SEND
│   ├── main_100_public.jsonl          <- main 100 prompts only
│   ├── main_100_gold.jsonl            <- main 100 gold only
│   ├── main_100_full.jsonl            <- deepest backend annotations
│   ├── controls_no_history_diagnosis_10_public.jsonl
│   ├── controls_no_history_diagnosis_10_gold.jsonl
│   ├── controls_no_history_choice_10_public.jsonl
│   ├── controls_no_history_choice_10_gold.jsonl
│   ├── optional_ablation_choice_only_100.jsonl
│   ├── optional_ablation_diagnosis_only_100.jsonl
│   ├── optional_ablation_action_given_rule_100.jsonl
│   ├── metadata.json
│   └── validation_main.json
│
├── requests/
│   ├── all_120_requests.jsonl         <- EASIEST FILE TO USE FOR MODEL CALLS
│   ├── main_100_joint_requests.jsonl
│   ├── controls_no_history_diagnosis_10_requests.jsonl
│   ├── controls_no_history_choice_10_requests.jsonl
│   └── optional_ablation_*.jsonl
│
├── scripts/
│   ├── run_model_template.py          <- edit one function to connect any API/model
│   └── run_hf_transformers.py         <- optional local Hugging Face runner
│
├── examples/
│   ├── SAMPLE_PROMPTS.md              <- one complete matched quartet + controls
│   ├── predictions_example.jsonl
│   ├── perfect_predictions_all_120.jsonl
│   ├── perfect_score_summary.json
│   └── perfect_score_details.jsonl
│
└── results/                            <- write your model outputs here
```

---

# Fastest way to evaluate a model

## Step 1: use the pre-generated 120 requests

Use:

```text
requests/all_120_requests.jsonl
```

Each row contains:

```json
{
  "item_id": "...",
  "trial_type": "main_joint",
  "messages": [
    {"role":"system","content":"..."},
    {"role":"user","content":"..."}
  ]
}
```

The requests are deterministically shuffled so matched quartet members are not presented consecutively.

**Important:** send every row as a fresh/stateless model call. Do not carry conversation history across benchmark items.

## Step 2: adapt one model-calling function

Copy:

```text
scripts/run_model_template.py
```

Edit only:

```python
def call_your_model(messages, model_name):
    ...
    return raw_text_response
```

Then run:

```bash
python scripts/run_model_template.py \
  --requests requests/all_120_requests.jsonl \
  --out results/my_model_predictions.jsonl \
  --model-name my-model
```

The runner automatically stores item IDs and raw responses.

### Optional local Hugging Face example

If you use a local instruction-tuned causal LM:

```bash
python scripts/run_hf_transformers.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --requests requests/all_120_requests.jsonl \
  --out results/qwen25_7b.jsonl
```

Install optional dependencies first if needed (`torch`, `transformers`, `accelerate`).

## Step 3: score automatically

```bash
python score_predictions.py \
  --gold benchmark/all_120_gold.jsonl \
  --pred results/my_model_predictions.jsonl \
  --summary-out results/my_model_summary.json \
  --details-out results/my_model_details.jsonl
```

No manual coding and no LLM judge are required.

---

# Main output metrics

## Rule inference

- `main_rule_inference_accuracy`
- `teacher_rule_inference_accuracy`
- `imposter_rule_inference_accuracy`
- `quartet_rule_inference_consistency`

These score whether the model correctly inferred `COLOR_RULE` vs `SHAPE_RULE` from behavior.

## Evidence/action selection

- `main_action_success`
- `teacher_pedagogical_selection_accuracy`
- `imposter_blend_in_success`
- `imposter_specific_blend_in_rate`
- `imposter_conservative_blend_in_rate`

These are scored separately from rule inference.

## Joint capacity

- `main_joint_success`
- `teacher_joint_accuracy`
- `imposter_joint_success`
- `quartet_joint_consistency`

A joint success requires both the inferred rule and the downstream action to be correct/acceptable.

## Matched behavioral reversals

- `teacher_learner_rule_reversal_consistency`
- `imposter_specific_rule_reversal_consistency`
- `specific_goal_reversal_consistency`

These test whether behavior changes appropriately while the underlying target set remains fixed.

## No-history sanity controls

- `no_history_diagnosis_unknown_accuracy`
- `no_history_choice_insufficient_accuracy`

## Failure analysis

The scorer automatically records which semantic class the model selected and separates failures such as:

- correct rule inference + wrong pedagogical action
- wrong rule inference + action coherent with the wrong inferred rule
- choosing the current-rule confirming example as Teacher
- choosing a valid but uninformative intersection
- revealing the private-rule difference as Imposter
- violating the category/private rule
- malformed/missing JSON fields

See the trial-level `details-out` file for every model response.

---

# Recommended first-run settings

For a clean capability floor:

- fresh/stateless call per item
- temperature / sampling off where possible
- one response per item initially
- record exact model/checkpoint/version and decoding settings
- do not give model access to item IDs or gold files
- do not run matched quartet cells in a shared conversation

If the model is stochastic, repeat the full benchmark with multiple seeds and report both mean performance and within-item stability.

---

# Regenerate everything from source

The package ships with the generated 120 trials, but they can be reproduced exactly:

```bash
python generate_benchmark.py --outdir benchmark
python validate_benchmark.py benchmark/main_100_full.jsonl --out benchmark/validation_main.json
python prepare_requests.py --input benchmark/all_120_public.jsonl --output requests/all_120_requests.jsonl
```

Run the package self-test:

```bash
python self_test.py
```

Expected output:

```text
SELF-TEST PASSED ...
```

---

# Optional diagnostic ablations

The 100 main items also store/generated alternate prompts that do **not** count as additional benchmark items:

1. `optional_ablation_choice_only_100.jsonl`  
   Hides the explicit intermediate rule report. Tests whether requiring `inferred_rule` scaffolds final choice.

2. `optional_ablation_diagnosis_only_100.jsonl`  
   Tests rule inference without downstream evidence selection.

3. `optional_ablation_action_given_rule_100.jsonl`  
   Supplies the other person's rule directly. Tests evidence selection after removing diagnosis as a bottleneck.

These are best used after the core 120-trial run if failure decomposition is needed.

Score an ablation against `benchmark/main_100_gold.jsonl`, for example:

```bash
python score_ablations.py --mode diagnosis_only \
  --gold benchmark/main_100_gold.jsonl \
  --pred results/diagnosis_only_predictions.jsonl
```

---

# What this benchmark does and does not claim

The benchmark tests the functional capacity to:

> infer a known candidate rule from another person's choices, then select truthful evidence according to how that evidence bears on the other person's rule and the model's goal.

The first release intentionally uses a closed hypothesis space and easy rule family. It does **not** establish a uniquely human theory-of-mind mechanism or an unrestricted ability to infer arbitrary hidden concepts.

A strong result here motivates harder tracks rather than ending the story.

## Planned scaling directions

Low-hanging extensions that preserve automatic ground truth:

- add **size** and **material** attributes
- open-hypothesis rule inference rather than showing two candidates
- ambiguous/noisy learner histories
- three-branch rules such as `RED OR SQUARE OR LARGE`
- relational rules such as `LEFT_OF OR ABOVE`
- matched **text → rendered vision** stimuli from the same symbolic JSON
- downstream validation with actual model or human learners

