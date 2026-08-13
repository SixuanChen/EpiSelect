# Coding-agent handoff: run EpiSelect Level 2 Advanced Logic v3

Work entirely inside this folder. It is standalone.

## Goal

Evaluate models on:

1. open Boolean rule inference;
2. Teacher selection of Informative evidence;
3. Imposter selection of either Compatible-positive or Compatible-negative evidence while remaining truthful;
4. the no-role helpful-assistant evidence policy;
5. calibration and framing ablations.

## Before any paid calls

Run:

```bash
python -m pip install -r requirements.txt
python generate_benchmark.py --root .
python validate_benchmark.py --root . --out benchmark/validation.json
python self_test.py
```

Do not proceed unless validation reports `"ok": true` and the self-test passes.

## Credentials

Check environment variables without printing values:

```bash
echo ${OPENROUTER_API_KEY:+OPENROUTER_API_KEY is set}
echo ${OPENAI_API_KEY:+OPENAI_API_KEY is set}
echo ${ANTHROPIC_API_KEY:+ANTHROPIC_API_KEY is set}
echo ${GEMINI_API_KEY:+GEMINI_API_KEY is set}
```

If missing, instruct the user to set the needed key, e.g.:

```bash
export OPENROUTER_API_KEY='PASTE_KEY_HERE'
```

Never print, save, or commit secret keys.

## Files to run

### Primary staged explicit roles

```text
requests/primary_staged_120_contexts.jsonl
```

Each context contains one role-neutral Stage-1 inference call and two Stage-2 branches.

Call Stage 1 exactly once, save the exact response, append it as the assistant turn, then branch independently into Teacher and Imposter.

### Helpful-assistant null

```text
requests/naturalistic_null_120_requests.jsonl
```

### No-history calibration

```text
requests/nohistory_staged_12_contexts.jsonl
```

### Recommended ablations

```text
requests/ablations/action_given_gold.jsonl
requests/ablations/neutral_goals_staged.jsonl
requests/ablations/underdetermined_staged.jsonl
requests/ablations/joint_single_call.jsonl
requests/ablations/choice_only.jsonl
requests/ablations/closed_hypothesis_staged.jsonl
requests/ablations/naturalistic_staged.jsonl
requests/ablations/all_truthful_staged.jsonl
requests/ablations/all_truthful_null.jsonl
```

## Output schemas

Stage 1:

```json
{"rule":"BOOLEAN_EXPRESSION or UNKNOWN"}
```

Stage 2 / flat action:

```json
{"choice":"1 or 2 or 3 or 4 or E"}
```

Do not add a separate `category` output field. The numbered option already contains its proposed BELONGS / DOES NOT BELONG label and the backend stores the true label.

## Formula handling

Do not post-hoc repair model formulas. Preserve raw output.

The scorer accepts semantic equivalents such as:

- `IF(A,B)`
- `A -> B`
- `NOT A OR B`

because they evaluate to the same truth table.

## Structured outputs

Use provider-native JSON schema when available, but do not constrain the `rule` string to one of a finite list. The primary task is open formula production.

Action `choice` may be schema-constrained to `1`, `2`, `3`, `4`, or `E`.

## Save every call

Each result row should include at minimum:

- provider
- exact model ID/version
- repetition
- context_id
- item_id
- role
- request type
- raw Stage-1 response
- raw Stage-2 response
- parsed rule string
- parsed choice
- decoding parameters actually accepted by the provider
- seed if supported
- latency
- token counts/cost if returned
- retry count
- provider errors

Never send private gold files in the prompt.

## Smoke test

Before a full run, execute two primary contexts and verify:

1. Stage 1 happens once per context;
2. exact Stage-1 output is reused for both branches;
3. role is not revealed before Stage 1;
4. formulas and choices parse;
5. raw outputs are stored;
6. no gold annotation enters the model call.

## Primary scoring

```bash
python score_predictions.py \
  --gold benchmark/primary_240_gold.jsonl \
  --pred results/PROVIDER/MODEL/rep0_primary.jsonl \
  --summary-out results/PROVIDER/MODEL/rep0_primary_summary.json \
  --details-out results/PROVIDER/MODEL/rep0_primary_details.jsonl
```

Report separately:

- semantic rule-inference accuracy;
- mean four-state truth-table agreement;
- Teacher Informative success;
- Imposter Compatible success;
- Imposter Compatible-positive rate;
- Imposter Compatible-negative rate;
- Invalid-control rate;
- abstention rate;
- joint inference/action success;
- strict goal-sensitive role contrast;
- strict four-cell quartet;
- rule-family, feature-pair, and PDD breakdowns.

**Do not compare Teacher and Imposter raw action-success percentages as if they had the same chance baseline.** Teacher has 1 target among 1-4; Imposter has 2.

## Null scoring

```bash
python score_naturalistic.py \
  --gold benchmark/naturalistic_null_120_gold.jsonl \
  --pred results/PROVIDER/MODEL/rep0_null.jsonl \
  --out results/PROVIDER/MODEL/rep0_null_summary.json
```

Treat this as a policy distribution, not a single accuracy.

## Ablation scoring

Use `score_ablations.py` with the corresponding gold file in `benchmark/ablations/`.

## Repetitions

After full item-level detail files exist:

```bash
python analyze_repetitions.py \
  results/PROVIDER/MODEL/rep0_primary_details.jsonl \
  results/PROVIDER/MODEL/rep1_primary_details.jsonl \
  results/PROVIDER/MODEL/rep2_primary_details.jsonl \
  --json-out results/PROVIDER/MODEL/repetition_stability.json \
  --csv-out results/PROVIDER/MODEL/repetition_stability.csv
```

## Final consolidated report

Create:

```text
results/LEVEL2_MODEL_REPORT.md
results/level2_model_summary.csv
results/level2_item_level.csv
```

At minimum discuss:

1. whether open rule inference remains near ceiling;
2. which logical families fail;
3. feature-pair effects in text and, if run, vision;
4. PDD=2 versus PDD=3;
5. Teacher disagreement-evidence selection;
6. Imposter agreement-evidence selection;
7. positive versus negative compatibility preference;
8. Invalid-control failures;
9. null/default evidence policy;
10. underdetermined/no-history abstention;
11. action-given-rule gap;
12. named versus neutral framing;
13. all-truthful sensitivity;
14. item-level stability.

Cluster or bootstrap uncertainty by `base_id` rather than pretending all role/user rows are independent conceptual rules.
