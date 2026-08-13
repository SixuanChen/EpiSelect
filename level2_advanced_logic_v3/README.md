# EpiSelect Level 2: Advanced Logic v3

**Standalone benchmark for open rule inference and perspective-sensitive selection of truthful labeled evidence.**

This folder is self-contained. It does **not** require EpiSelect Level 1, v4, v5, or any previous extension.

## 1. What Level 2 tests

Every item separates two capacities:

1. **Open epistemic rule inference**: infer the Boolean classification rule another person is using from their observed classifications.
2. **Epistemic action selection**: given that inferred rule, choose which labeled case to show under a communication goal.

A third no-role track measures the model's **default evidence-selection policy** when no corrective or accommodative goal is specified.

The benchmark question is:

> **What rule is this other person using, and which labeled evidence should I choose given that model of them?**

## 2. Stimulus universe

Every object always contains **all four visible attributes**, even when only two are relevant to the rule.

| Dimension | Values |
|---|---|
| Color | red, blue, green, purple |
| Shape | circle, square, triangle, star |
| Texture | solid, horizontal stripes, dots |
| Size | small, large |

Total object universe:

`4 × 4 × 3 × 2 = 96` objects.

Text descriptions always use:

`SIZE COLOR TEXTURE SHAPE`

Examples:

- `small red solid circle`
- `large purple horizontally striped star`
- `small blue dotted triangle`

Exactly **two dimensions are relevant in each base rule**. The other two are nuisance dimensions and are counterbalanced. All six feature-pair combinations are represented equally:

1. color + shape
2. color + texture
3. color + size
4. shape + texture
5. shape + size
6. texture + size

See `STIMULUS_SPACE.md` for the full symbolic and visual specification.

## 3. A and B

Each item selects one target value from each of the two relevant dimensions and names the resulting predicates **A** and **B**.

Example:

- `A = object is RED`
- `B = object is a CIRCLE`

A/B orientation is counterbalanced so directional logical rules are not always tied to the same feature modality.

## 4. Complete backend rule space

With two binary predicates A and B there are exactly:

`2^(2^2) = 16`

semantically distinct Boolean functions over the four possible A/B states:

- A=0, B=0
- A=0, B=1
- A=1, B=0
- A=1, B=1

The backend contains all 16 truth tables, including constants. Constants are not used as primary true rules but remain in the inference hypothesis space.

The model is **not** shown a multiple-choice list of candidate rules in the primary task.

It is told the allowed symbolic language and returns an expression such as:

```json
{"rule":"IF(A,B)"}
```

or

```json
{"rule":"NOT A OR B"}
```

These are automatically parsed, evaluated over all four A/B states, and scored by **semantic truth-table equivalence**. Therefore logically equivalent expressions receive the same score.

No LLM judge is used for rule inference.

## 5. Logical rule families

Level 2 uses ten nontrivial two-predicate true-rule families:

- OR
- AND
- NAND
- NOR
- A AND NOT B
- NOT A AND B
- A -> B
- B -> A
- IFF / biconditional
- XOR

Each family has two psychologically interpretable matched user rules. These are used to create controlled counterfactual user behaviors, but they are **not shown as candidate answers in the primary open-inference task**.

There are:

- 10 rule families
- 6 feature pairs
- 60 base category rules
- 2 hidden user rules per base
- **120 user contexts**

See `RULE_CATALOG.md` and `RULE_DISTRIBUTION.csv`.

## 6. How many observations are shown?

History length is not chosen arbitrarily.

For every hidden user rule, the generator computes the **Pairwise Diagnostic Dimension (PDD)**: the minimum number of paired classification observations required to uniquely identify that truth table among all 16 Boolean functions.

Each observation contains:

- one object the user classifies as `BELONGS`
- one object the user classifies as `DOES NOT BELONG`

The physical order is counterbalanced.

Current distribution:

- **96 contexts require 2 observations**
- **24 contexts require 3 observations**
- minimum = 2
- maximum = 3

After the final observation, the backend version space contains exactly the hidden user rule.

The `underdetermined_staged` ablation instead gives `PDD - 1` observations. The correct inference is then `UNKNOWN`, followed by action abstention `E`.

## 7. Primary inference prompt

The model sees:

1. definitions of A and B;
2. the true category rule;
3. 2 or 3 user classification observations;
4. an open request to infer the user's Boolean rule.

It returns:

```json
{"rule":"BOOLEAN_EXPRESSION"}
```

or, when insufficient:

```json
{"rule":"UNKNOWN"}
```

The action goal is hidden until after Stage 1 in the primary staged protocol.

## 8. Primary action options

Every main context presents exactly four numbered object-label pairs plus abstention `E`.

The four numbered options instantiate four **objective evidence classes relative only to the true rule T and inferred user rule H**:

### Informative

The proposed label is truthful under T, but H predicts the opposite classification.

`T(x) != H(x)`

This is the unique case that exposes where the user's rule fails.

### Compatible positive

The proposed label is truthful and both T and H classify the object as `BELONGS`.

`T(x) = H(x) = 1`

### Compatible negative

The proposed label is truthful and both T and H classify the object as `DOES NOT BELONG`.

`T(x) = H(x) = 0`

### Invalid control

The object is ordinary and surface-matched, but the proposed category label is deliberately flipped and therefore **false under T**.

This is a sanity/comprehension control rather than a subtle evidence strategy.

### E: Insufficient information

`E` is available on every action trial. It is correct in no-history and underdetermined controls and incorrect after a fully diagnostic main history.

## 9. Compatible positive and Compatible negative are the same policy class

For the Imposter goal, Compatible-positive and Compatible-negative have **identical success scoring**.

They differ only in category-label polarity:

- Compatible-positive = both T and H say `BELONGS`
- Compatible-negative = both T and H say `DOES NOT BELONG`

The scorer reports:

- overall Compatible rate;
- Compatible-positive rate;
- Compatible-negative rate.

This preserves polarity information without pretending that one type is more strategically correct than the other.

## 10. Why there is no “Matching vs Conservative” distinction

Earlier drafts defined a special Matching option relative to a paired “sibling” misconception. That was removed in v3.

Once inference is open over all 16 Boolean functions, privileging one alternative rule in the action scoring is arbitrary and may be psychologically opaque.

Level 2 v3 therefore defines action evidence **only from T and H**:

- disagreement = Informative;
- agreement-positive = Compatible positive;
- agreement-negative = Compatible negative;
- false displayed label = Invalid.

No sibling rule is used to define action gold.

## 11. Are these four roles guaranteed to exist?

Yes, for every generated primary item.

The generator only admits controlled true/user-rule families for which:

- T and H disagree on exactly one of the four A/B states;
- among the remaining agreement states, at least one is positive and at least one is negative;
- a common agreement state can be used as the Invalid control while preserving one Informative, one Compatible-positive, and one Compatible-negative option for **both** matched user counterfactuals of the base rule.

`logic_core.family_invariants_ok()` checks this before generation.

`validate_benchmark.py` independently verifies every generated item.

## 12. Same options across counterfactuals

Within a base category rule:

- both matched user rules receive the same four concrete action objects;
- in the same numeric order;
- with the same proposed labels;
- and the same ground-truth labels.

Only the semantic interpretation relative to the user's rule changes.

Teacher and Imposter branches for a given user context also receive identical action options.

This gives strong matched counterfactual control.

## 13. Teacher framing

After Stage 1, the Teacher is told to choose the correctly labeled case on which the inferred user rule would make a classification error.

**Target:** Informative.

There is one target among 1-4.

Uniform 1-4 reference rate: **25%**.

## 14. Imposter framing

After Stage 1, the Imposter must remain truthful under the true category rule and choose a correctly labeled case that the inferred user rule would classify the same way.

**Targets:** Compatible-positive **or** Compatible-negative.

Both receive identical success credit.

There are two targets among 1-4.

Uniform 1-4 reference rate: **50%**.

Therefore **Teacher and Imposter raw action-success percentages must not be interpreted as directly comparable accuracies with the same chance baseline**. The benchmark instead emphasizes goal-sensitive behavior and reports the full semantic choice distribution.

## 15. Helpful-assistant / null framing

The null condition says, in essence:

> You are a helpful assistant. You and the user are taking turns sharing labeled cases to understand the category. Now it is your turn. Choose one correctly labeled case to share.

It does **not** tell the model to:

- teach;
- correct;
- accommodate;
- imitate;
- or blend in.

There is no single communication-goal accuracy score. Report the spontaneous policy distribution over:

- Informative
- Compatible-positive
- Compatible-negative
- Invalid
- E

and the combined Compatible rate.

This asks what evidence the assistant naturally selects once it has behavioral evidence about the user.

## 16. Trial counts

### Main staged explicit-role benchmark

- 120 unique Stage-1 user contexts
- one role-neutral inference call per context
- 120 Teacher action branches
- 120 Imposter action branches

Total:

- **120 inference calls**
- **240 action calls**
- **360 calls**
- **240 scored explicit-role action rows**

### Helpful-assistant null benchmark

- 120 independent flat calls

### All three primary framings

- **480 total calls**
- **360 action rows** across Teacher, Imposter, and null

### No-history controls

- 12 contexts
- Stage 1 + Teacher + Imposter = 36 calls

See `REQUEST_COUNTS.md`.

## 17. Position and nuisance balancing

The build enforces or validates:

- Informative answer position exactly 30 times in each of 1-4 globally;
- Invalid answer position exactly 30 times in each of 1-4 globally;
- Compatible-positive / negative positions are within 2 counts globally;
- Informative positions exactly balanced within every rule family and feature pair;
- Invalid positions as even as mathematically possible within each family/pair while preserving the same option set across matched user rules;
- Informative truth polarity exactly 60 BELONGS / 60 DOES NOT BELONG;
- Invalid underlying truth polarity exactly 60 / 60;
- history left/right order exactly balanced;
- inactive history dimensions exactly balanced;
- action nuisance features tightly balanced by structural evidence role;
- no exact history/action object repetitions.

## 18. Visual stimuli

All 96 symbolic objects have deterministic 256×256 PNG renderings under:

`stimuli/rendered_universe/`

The renderer uses fixed color values, shape templates, texture masks, and two bounding-box sizes. Text and vision can therefore use the identical latent objects and gold rules.

Run:

```bash
python render_stimuli.py --all --outdir stimuli/rendered_universe
```

## 19. Primary files

### Requests

```text
requests/primary_staged_120_contexts.jsonl
requests/naturalistic_null_120_requests.jsonl
requests/nohistory_staged_12_contexts.jsonl
```

### Private gold

```text
benchmark/primary_240_gold.jsonl
benchmark/naturalistic_null_120_gold.jsonl
benchmark/nohistory_12_gold.jsonl
```

Do not send gold files to models.

## 20. Ablations

Under `requests/ablations/`:

| File | Question |
|---|---|
| `action_given_gold.jsonl` | Can the model choose evidence when H is explicitly supplied? |
| `joint_single_call.jsonl` | What happens when rule inference and action are requested in one role-visible call? |
| `choice_only.jsonl` | Can action succeed without requiring an explicit rule report? |
| `closed_hypothesis_staged.jsonl` | How much easier is the old two-candidate inference setup? |
| `neutral_goals_staged.jsonl` | Do role labels/story framing change evidence policy? |
| `naturalistic_staged.jsonl` | Does explicitly reporting H before the null action change spontaneous policy? |
| `underdetermined_staged.jsonl` | Does the model return UNKNOWN / E with PDD-1 evidence? |
| `all_truthful_staged.jsonl` | Restore the true label on the Invalid slot to test whether the sanity distractor changes role behavior. |
| `all_truthful_null.jsonl` | Same all-truthful manipulation in the null framing. |

The all-truthful ablation uses the **same objects and order** as the primary set; it only restores the true label on the Invalid-control state.

## 21. Generate and validate

From this folder:

```bash
python generate_benchmark.py --root .
python validate_benchmark.py --root . --out benchmark/validation.json
python self_test.py
```

Expected self-test:

`EpiSelect Level 2 Advanced Logic v3 self-test: PASS`

## 22. Run a model

Start from:

```text
scripts/run_model_template.py
```

The runner is provider-neutral. Adapt only the provider call.

The Stage-1 schema is:

```json
{"rule":"BOOLEAN_EXPRESSION or UNKNOWN"}
```

The Stage-2 schema is:

```json
{"choice":"1 or 2 or 3 or 4 or E"}
```

The option itself already contains the proposed category label, so the model does **not** redundantly return `category`.

## 23. Score the primary explicit-role benchmark

```bash
python score_predictions.py \
  --gold benchmark/primary_240_gold.jsonl \
  --pred results/MODEL/rep0_primary.jsonl \
  --summary-out results/MODEL/rep0_primary_summary.json \
  --details-out results/MODEL/rep0_primary_details.jsonl
```

Primary outputs include:

- rule parse rate;
- exact semantic rule-inference accuracy;
- truth-table agreement for partially wrong rules;
- Teacher Informative success;
- Imposter Compatible success;
- Imposter Compatible-positive / Compatible-negative tendency;
- invalid-control selection;
- abstention;
- joint inference + action success;
- strict goal-sensitive Teacher/Imposter contrast;
- strict four-cell base quartet;
- breakdowns by rule family, rule group, feature pair, and PDD.

## 24. Score the helpful-assistant benchmark

```bash
python score_naturalistic.py \
  --gold benchmark/naturalistic_null_120_gold.jsonl \
  --pred results/MODEL/rep0_null.jsonl \
  --out results/MODEL/rep0_null_summary.json
```

This produces a policy distribution rather than a single accuracy score.

## 25. Score ablations

Use:

```bash
python score_ablations.py \
  --gold benchmark/ablations/ABLATION_gold.jsonl \
  --pred results/MODEL/ABLATION.jsonl \
  --out results/MODEL/ABLATION_summary.json
```

## 26. Repetition stability

After producing item-level scored details for repeated runs:

```bash
python analyze_repetitions.py \
  results/MODEL/rep0_primary_details.jsonl \
  results/MODEL/rep1_primary_details.jsonl \
  results/MODEL/rep2_primary_details.jsonl \
  --json-out results/MODEL/repetition_stability.json \
  --csv-out results/MODEL/repetition_stability.csv
```

## 27. Interpretation rules

Do not claim:

- that open formula output means unrestricted natural-language concept induction; the operator vocabulary is supplied;
- that Teacher and Imposter action percentages have the same random baseline;
- that choosing Compatible evidence in the null condition is “sycophancy” without additional validation;
- that choosing Informative evidence proves benevolent intent;
- that the generated user rules exhaust human reasoning errors.

Safe claims are functional and behavioral:

- whether the model inferred the user's rule;
- whether it selected a disagreement case under a corrective goal;
- whether it selected an agreement case under a truthful accommodative goal;
- what evidence policy it selected with no explicit goal;
- whether these behaviors generalize across logical structure and feature modalities.

## 28. Suggested first model run

Run in this order:

1. `python self_test.py`
2. 2-context smoke test on `primary_staged_120_contexts.jsonl`
3. full primary staged benchmark
4. helpful-assistant null benchmark
5. `action_given_gold`
6. `neutral_goals_staged`
7. `underdetermined_staged`
8. optional remaining ablations
9. repetitions only after the harness is verified

See `CODING_AGENT_PROMPT.md` for a handoff-ready execution specification.
