# Request counts

| Suite | Context/request rows | Model calls | Scored action rows | Purpose |
|---|---:|---:|---:|---|
| `primary_staged_120_contexts.jsonl` | 120 staged | 360 | 240 | open inference + Teacher/Imposter |
| `naturalistic_null_120_requests.jsonl` | 120 flat | 120 | 120 descriptive | helpful-assistant default policy |
| `nohistory_staged_12_contexts.jsonl` | 12 staged | 36 | 24 actions + 12 inference controls | UNKNOWN/E calibration |
| `ablations/action_given_gold.jsonl` | 240 flat | 240 | 240 | remove inference bottleneck |
| `ablations/joint_single_call.jsonl` | 240 flat | 240 | 240 | role visible before rule report |
| `ablations/choice_only.jsonl` | 240 flat | 240 | 240 | no explicit rule report |
| `ablations/closed_hypothesis_staged.jsonl` | 120 staged | 360 | 240 | old two-candidate inference comparison |
| `ablations/neutral_goals_staged.jsonl` | 120 staged | 240 if Stage 1 reused; 360 otherwise | 240 | remove Teacher/Imposter labels |
| `ablations/naturalistic_staged.jsonl` | 120 staged | 120 if Stage 1 reused; 240 otherwise | 120 descriptive | explicit inference then no-role action |
| `ablations/underdetermined_staged.jsonl` | 120 staged | 360 | 240 | PDD-1, expect UNKNOWN/E |
| `ablations/all_truthful_staged.jsonl` | 120 staged | 240 if Stage 1 reused; 360 otherwise | 240 | restore true label on Invalid state |
| `ablations/all_truthful_null.jsonl` | 120 flat | 120 | 120 descriptive | all-truthful null policy |

The primary three-framing set (Teacher + Imposter + null) is **480 calls** total: 120 Stage-1 inference calls, 240 explicit-role action calls, and 120 independent null calls.
