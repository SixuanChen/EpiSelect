# Adversarial Review and Design Audit

This file records the main ways an adversarial reviewer could attack v4-final, what is already controlled, and what remains an explicit limitation.

| Reviewer attack | Status / design response |
|---|---|
| Finite behavior does not uniquely identify an unrestricted rule. | **Fixed for the core track.** The model is explicitly told the two candidate hypotheses, COLOR_RULE and SHAPE_RULE. Open hypothesis inference is reserved for a harder extension. |
| The benchmark conflates rule inference with evidence selection. | **Fixed.** Every main response contains both `inferred_rule` and `answer`, and the scorer reports inference, action, and joint performance separately. |
| Requiring an explicit intermediate rule might scaffold the action. | **Measured.** Choice-only ablation prompts are generated for the same 100 items. |
| A model could fail because it understood the other person but not the downstream action. | **Measured.** Action-given-rule ablation removes diagnosis as a bottleneck. |
| A model could choose the first/second history item based on position. | **Fixed.** History choices have no A/B labels; each rule-consistent person selects left once and right once over the two observations. Validator checks this. |
| A/B/C/D answer-position bias could inflate results. | **Fixed.** Semantic target classes rotate through A/B/C/D. Teacher unique gold is 13/13/12/12 across A/B/C/D. |
| Models could copy a learner-selected object into the final answer. | **Fixed.** Exact history objects never reappear among final target options. Validator checks this. |
| Teacher success could reduce to basic category-rule application. | **Controlled.** Three target options satisfy the category rule; only one uniquely challenges the learner's current rule. |
| Imposter has more than one rational blend-in action. | **Fixed in scoring.** Both rule-specific and intersection choices count as blend-in success. The scorer separately reports the stronger audience-sensitive specific strategy. |
| Imposter can always choose the intersection without modeling the civilian. | **Not hidden.** General blend-in success and specific audience-sensitive blend-in are separate metrics. Only the latter supports the stronger perspective-sensitive claim. |
| Teacher/Imposter wording may cue different learned scripts. | **Remaining framing limitation.** A neutral goal-framing control is a natural next ablation; the core benchmark keeps the intuitive roles. |
| “Imposter” may trigger safety/refusal behavior in aligned models. | **Mitigated, not eliminated.** It is framed as a benign category game; malformed/refusal outputs remain visible in strict-format and failure counts. Neutral framing can test this later. |
| The two candidate labels themselves make the task easy. | **Intentional capability floor.** v4-final is the closed-hypothesis track. Harder open-hypothesis inference is planned later. |
| Two behavioral observations make rule inference too easy. | **Intentional capability floor.** This release asks whether the full diagnosis→action capacity is present at all before manipulating uncertainty/noise. |
| 100 trials are not 100 independent conceptual rules. | **Handled in analysis.** There are 25 independent base-rule quartets. Matched/quartet metrics are reported; paper-level inference should treat the base rule as the paired unit. |
| Specific colors/shapes could carry priors or salience. | **Counterbalanced.** The main task uses the full 5×5 Cartesian product, and nuisance values rotate through history/target roles. |
| A stateful evaluation could leak one matched condition into another. | **Operational fix.** Requests are shuffled, and README requires a fresh/stateless call per item. |
| Systematic quartet ordering could leak the next answer. | **Fixed in request files.** Pre-generated request order is deterministically shuffled. |
| Model output prose parsing could produce false positives (e.g. a sentence beginning with A). | **Fixed.** The scorer accepts structured JSON/direct fields only; arbitrary prose is not interpreted as an answer. |
| The negative distractor makes the task partly trivial. | **By design.** It acts as a rule-application check, while the difficult comparison is among three true positives. Backend semantics expose which error type occurred. |
| No-history controls have no unique targeted learner state. | **Handled explicitly.** They use `UNKNOWN` / `INSUFFICIENT`, and are treated as sanity controls rather than core capacity trials. |
| The benchmark could be solved by symbolic set operations rather than “human-like” mental-state reasoning. | **Conceptual limitation, not a bug.** The claim is functional capacity: behaviorally conditioning evidence selection on another person's inferred rule. The benchmark does not claim a unique internal mechanism. |
| Models may have seen classic pedagogical-selection paradigms in training. | **Partially mitigated.** Stimuli are procedural and not copied from canonical child items; the benchmark tests flexible recombination across 25 matched rules. This does not eliminate high-level task familiarity. |
| Text colored-shape stimuli are not a substantive VLM test yet. | **Explicit scope.** v4-final is the text capability floor. The intended next track renders the same underlying JSON visually while keeping gold labels identical. |
| OR-only rules are narrow. | **Explicit scope.** Material/size, three-branch, relational, and open-hypothesis tracks are the intended publication-scale expansion. |
| Deterministic one-shot output may hide instability. | **Operational recommendation.** Use temperature 0 for the capability floor; if sampling, repeat seeds and report stability. |

## Strongest remaining limitation

The core track gives the model a two-hypothesis closed world and two clean observations. That makes rule diagnosis deliberately easy. A positive result establishes a clear **floor** for learner-sensitive evidence selection; it should not be presented as evidence of unrestricted hidden-rule inference.

The most informative next difficulty manipulation is therefore not “more colors.” It is **uncertainty over the other person's hypothesis**: ambiguous choices, noisy choices, more candidate hypotheses, and eventually open inference.
