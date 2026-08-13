# Rule catalog

Level 2 uses the complete set of ten nonconstant, non-single-literal Boolean functions over two predicates A and B as true-rule families. Constants and one-feature literals remain in the 16-function backend hypothesis space but are not primary true rules.

Each true-rule family has two matched, psychologically interpretable user rules. These matched user rules create controlled counterfactual behaviors. **They are not shown as options in the primary open-inference task and they do not define action scoring.**

| Family | True rule | User rule 1 | PDD | User rule 2 | PDD | Group |
|---|---|---|---:|---|---:|---|
| OR | `A OR B` | `A` | 2 | `B` | 2 | simple_composition |
| AND | `A AND B` | `A` | 2 | `B` | 2 | simple_composition |
| NAND | `NOT (A AND B)` | `NOT A` | 2 | `NOT B` | 2 | negated_composition |
| NOR | `NOT (A OR B)` | `NOT A` | 2 | `NOT B` | 2 | negated_composition |
| A_AND_NOT_B | `A AND NOT B` | `A` | 2 | `NOT B` | 2 | asymmetric_negation |
| NOT_A_AND_B | `NOT A AND B` | `NOT A` | 2 | `B` | 2 | asymmetric_negation |
| A_TO_B | `A -> B` | `NOT A` | 2 | `B` | 2 | conditional |
| B_TO_A | `B -> A` | `NOT B` | 2 | `A` | 2 | conditional |
| IFF | `A <-> B` | `A -> B` | 3 | `B -> A` | 3 | directional_relation |
| XOR | `A XOR B` | `A AND NOT B` | 3 | `NOT A AND B` | 3 | exclusive_relation |

## Why these user rules?

Every user rule differs from the true rule on exactly **one** of the four A/B truth assignments. This holds misconception distance constant while changing logical structure. The pairings correspond to simple feature omission, necessary-as-sufficient shortcuts, negated-branch omission, forward/converse implication, and one-branch exclusive reasoning.

## Action evidence no longer uses a sibling rule

For each hidden user rule H and true rule T, primary action options are defined only by T and H:

- **Informative:** `T(x) != H(x)` and the displayed label equals T.
- **Compatible positive:** `T(x) = H(x) = 1`.
- **Compatible negative:** `T(x) = H(x) = 0`.
- **Invalid control:** the displayed label is deliberately false under T.

Compatible-positive and Compatible-negative have identical Imposter success scoring. Their polarity is retained only for analysis.

## Existence guarantee

`family_invariants_ok()` verifies for every family/user pair that:

1. exactly one abstract state is Informative;
2. at least one Compatible-positive state exists;
3. at least one Compatible-negative state exists;
4. a common state can be mislabeled as Invalid while preserving the three truthful roles for both matched user rules.

The full generated benchmark is independently rechecked by `validate_benchmark.py`.

## Concrete examples

### OR

Let A=RED and B=CIRCLE. True rule: `A OR B`. Hidden user rule: `A`.

- non-red circle, BELONGS -> Informative
- red non-circle, BELONGS -> Compatible positive
- non-red non-circle, DOES NOT BELONG -> Compatible negative
- red circle with a flipped DOES NOT BELONG label -> Invalid control

### AND

Let A=RED and B=CIRCLE. True rule: `A AND B`. Hidden user rule: `A`.

- red non-circle, DOES NOT BELONG -> Informative
- red circle, BELONGS -> Compatible positive
- non-red non-circle, DOES NOT BELONG -> Compatible negative
- a shared-agreement state with its label flipped -> Invalid control

### IFF / forward implication

Let A=RED and B=CIRCLE. True rule: `A <-> B`. Hidden user rule: `A -> B`.

- non-red circle, DOES NOT BELONG -> Informative
- red circle or non-red non-circle when correctly labeled -> Compatible (polarity determines positive vs negative)
- a different shared-agreement state is used as the Invalid label-flip control.
