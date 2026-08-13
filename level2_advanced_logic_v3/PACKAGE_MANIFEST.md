# Package manifest

## Core documentation

- `README.md`
- `QUICKSTART.txt`
- `COGNITIVE_DESIGN.md`
- `ADVERSARIAL_REVIEW.md`
- `STIMULUS_SPACE.md`
- `RULE_CATALOG.md`
- `RULE_DISTRIBUTION.csv`
- `RULE_DISTRIBUTION.json`
- `REQUEST_COUNTS.md`
- `BUILD_VERIFICATION.md`
- `CODING_AGENT_PROMPT.md`

## Core code

- `logic_core.py`
- `generate_benchmark.py`
- `validate_benchmark.py`
- `score_predictions.py`
- `score_naturalistic.py`
- `score_ablations.py`
- `analyze_repetitions.py`
- `render_stimuli.py`
- `self_test.py`
- `scripts/run_model_template.py`

## Requests

- `requests/primary_staged_120_contexts.jsonl`
- `requests/naturalistic_null_120_requests.jsonl`
- `requests/nohistory_staged_12_contexts.jsonl`
- `requests/ablations/action_given_gold.jsonl`
- `requests/ablations/joint_single_call.jsonl`
- `requests/ablations/choice_only.jsonl`
- `requests/ablations/closed_hypothesis_staged.jsonl`
- `requests/ablations/neutral_goals_staged.jsonl`
- `requests/ablations/naturalistic_staged.jsonl`
- `requests/ablations/underdetermined_staged.jsonl`
- `requests/ablations/all_truthful_staged.jsonl`
- `requests/ablations/all_truthful_null.jsonl`

## Private gold

- `benchmark/primary_240_gold.jsonl`
- `benchmark/naturalistic_null_120_gold.jsonl`
- `benchmark/nohistory_12_gold.jsonl`
- `benchmark/ablations/*_gold.jsonl`
- `benchmark/all_truthful_360_gold.jsonl`
- `benchmark/metadata.json`
- `benchmark/validation.json`

## Stimuli

- `stimuli/object_universe.csv`
- `stimuli/object_universe.jsonl`
- `stimuli/rendered_universe/*.png` (96 files)

## Examples

- `examples/SAMPLE_PROMPTS.md`
- `examples/perfect_primary_predictions.jsonl`
- `examples/perfect_primary_summary.json`
- `examples/perfect_primary_details.jsonl`
- `examples/example_null_informative_policy_predictions.jsonl`
- `examples/example_null_informative_policy_summary.json`

## Outputs

`results/` is intentionally empty except for its README. Create provider/model-specific result folders there.
