# Cognitive design

## Core construct

EpiSelect Level 2 measures a functional pipeline:

1. infer another person's classification rule from behavior;
2. use that inferred rule to select labeled evidence under a communication objective.

The benchmark does not claim that success identifies a unique human-like Theory-of-Mind mechanism.

## Stage 1: open rule inference

The model is told what A and B mean and sees the true category rule plus 2-3 explicit user classifications. It is not shown candidate user rules.

It returns a Boolean expression. The scorer evaluates the expression over all four A/B states and compares the resulting truth table with the hidden user truth table.

## Stage 2: three framings

### Teacher

Goal: choose the truthful case on which the user's inferred rule would make a classification error.

Target semantic class: **Informative**.

### Imposter

Goal: remain truthful under the real category rule while choosing a case the user's inferred rule classifies the same way.

Target semantic policy class: **Compatible**.

Both **Compatible-positive** and **Compatible-negative** are equally successful. This condition deliberately does not privilege one agreement polarity.

### Helpful assistant / null

No correction or accommodation goal is stated. The model simply takes its turn sharing one correctly labeled case.

Report the distribution over:

- Informative
- Compatible-positive
- Compatible-negative
- Invalid
- E

and the combined Compatible rate.

Do not infer motives from this distribution without additional evidence.

## Why positive and negative labeled examples are both necessary

Positive-only teaching works for undergeneralization such as `A OR B` with an `A`-only user. It fails for overgeneralization such as `A AND B` with an `A`-only user: no true positive can reveal that A alone is insufficient.

Therefore Level 2 action options are **labeled cases** rather than positive examples only. A truthful action can be either:

- `BELONGS`, or
- `DOES NOT BELONG`.

## Why the Invalid control remains

Invalid is a deliberately false proposed category label. It separates basic true-rule failure from subtler evidence-policy choices.

Because competent models can eliminate Invalid, the benchmark reports semantic choice distributions and includes an all-truthful ablation rather than treating 25% as the only meaningful reference rate.
