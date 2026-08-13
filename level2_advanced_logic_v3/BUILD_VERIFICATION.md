# Build verification

The shipped Level 2 v3 assets were regenerated and checked from the standalone folder.

## Commands

```bash
python generate_benchmark.py --root .
python validate_benchmark.py --root . --out benchmark/validation.json
python self_test.py
python render_stimuli.py --all --outdir stimuli/rendered_universe
```

All passed.

## Core counts

- 96 symbolic objects
- 96 rendered PNGs
- 6 feature pairs
- 10 logical true-rule families
- 60 base category rules
- 2 matched hidden user rules per base
- 120 user contexts
- 240 explicit-role primary action rows
- 120 helpful-assistant null rows
- 12 no-history contexts

## PDD distribution

- 96 user contexts: PDD = 2
- 24 user contexts: PDD = 3

Every primary history leaves exactly one of the 16 Boolean truth tables consistent with the observed classifications.

## Primary option invariant

Every main user context contains exactly one:

- Informative
- Compatible-positive
- Compatible-negative
- Invalid-control

plus `E` for abstention.

For every primary context:

- Informative is correctly labeled and T/H disagree;
- Compatible-positive is correctly labeled and T=H=1;
- Compatible-negative is correctly labeled and T=H=0;
- Invalid is the only false proposed category label.

The Imposter target set is exactly Compatible-positive + Compatible-negative. Both receive equal score.

## Matched counterfactual control

For the two hidden-user rules under each base category rule, the action options are identical in:

- concrete object identity;
- numeric position;
- proposed BELONGS / DOES NOT BELONG label;
- true category label.

Teacher and Imposter also see the same options within a user context.

## Position balance

Across 120 user contexts:

- Informative: 30 / 30 / 30 / 30 across positions 1-4
- Invalid: 30 / 30 / 30 / 30
- Compatible-positive: 31 / 29 / 29 / 31 (position ordering as stored by validator)
- Compatible-negative: 29 / 31 / 31 / 29

Informative position is exactly balanced within every rule family (3 each) and feature pair (5 each).

Invalid position is as even as mathematically possible within each family/pair while reusing one common target set across both user counterfactuals.

## Label polarity

Across user contexts:

- Informative: 60 BELONGS / 60 DOES NOT BELONG
- Compatible-positive: 120 BELONGS
- Compatible-negative: 120 DOES NOT BELONG
- Invalid underlying true label: 60 BELONGS / 60 DOES NOT BELONG

## Feature balance

Target predicate values across 60 bases:

- color: 8 / 8 / 7 / 7
- shape: 8 / 8 / 7 / 7
- texture: 10 / 10 / 10
- size: 15 / 15

Inactive features in histories are exactly balanced.

For action options, inactive texture values are exactly balanced within every semantic class; size is exact for Informative and Invalid and within 2 counts for Compatible subtypes; four-valued nuisance dimensions differ by at most 3 counts within a semantic class.

## Text completeness

The validator confirms that every action-object text description explicitly contains:

- size;
- color;
- texture;
- shape.

Relevant dimensions are therefore not revealed by omission.

## Open inference parser

Self-tests confirm semantic equivalence, including:

```text
IF(A,B)
A -> B
NOT A OR B
```

all mapping to the same `A_TO_B` truth table.

## Scorer smoke tests

A synthetic perfect file achieves:

- 100% exact semantic rule inference;
- 100% Teacher action success;
- 100% Imposter Compatible success;
- 100% strict goal-sensitive role contrast;
- 100% strict four-cell quartet;
- 0% Invalid selection.

An all-Informative helpful-assistant synthetic file yields 100% Informative policy rate, verifying descriptive null scoring.
