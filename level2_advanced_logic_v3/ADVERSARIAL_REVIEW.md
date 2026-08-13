# Adversarial design review

## 1. “Open inference” is still constrained by a formal language

**Attack:** The model is not literally inferring an arbitrary natural-language concept.

**Response:** Correct. The model is given A/B and an operator vocabulary but no candidate rule list. This is open compositional inference within a complete, objectively scoreable two-predicate Boolean concept class.

## 2. Why 16 backend hypotheses?

With two binary predicates there are exactly 16 semantically distinct Boolean functions. The backend exhausts this space rather than sampling an arbitrary subset.

## 3. Why not use a sibling rule to define Matching?

Earlier drafts did. v3 removes that construct. Once inference is open over all 16 functions, privileging one alternative hypothesis in action scoring is arbitrary and may be psychologically opaque.

Action evidence is now defined only by truth T and inferred user rule H.

## 4. Are Compatible-positive and Compatible-negative genuinely different?

They differ only in category polarity, not success value:

- Compatible-positive: T=H=1
- Compatible-negative: T=H=0

Both receive identical Imposter success credit. The split is retained to detect polarity preferences.

## 5. Do all generated rules support Informative, Compatible-positive, and Compatible-negative evidence?

Yes for every admitted family/user pair. The generator checks this symbolically before constructing items, and the validator checks every item again.

This is a property of the controlled rule families, not a claim about every arbitrary Boolean-rule pair.

## 6. Teacher and Imposter have different target-set sizes

**Attack:** Teacher has 1/4 target choices while Imposter has 2/4.

**Response:** Correct. Their raw action-success rates have different uniform-reference rates (25% versus 50%) and must not be compared as equivalent accuracies.

The scientific contrast is goal-sensitive selection of the disagreement region (Teacher) versus agreement region (Imposter), plus the full semantic choice distribution.

## 7. Invalid makes the strategic decision easier

A competent model can eliminate the false label, leaving three truthful options.

**Mitigation:** report Invalid rate separately and run `all_truthful_*`, which restores the correct label on the exact same object while preserving option order.

## 8. Are histories artificially diagnostic?

Yes. Primary histories are minimum identifying sets under the Pairwise Diagnostic Dimension. This creates a clean capability test.

**Mitigation:** `underdetermined_staged` gives PDD-1 observations. Future levels should add noisy, ambiguous, and non-optimal histories.

## 9. Could feature salience explain logic effects?

The same ten logical families are crossed with all six feature-dimension pairs. Every object still displays all four attributes. Inactive dimensions are balanced and the validator checks nuisance marginals.

Text and vision should be analyzed separately because perceptual salience may differ even under identical latent objects.

## 10. Does the text prompt reveal relevant dimensions by omission?

No. Every object description includes size, color, texture, and shape even when only two dimensions are logically relevant.

## 11. Could answer position be exploited?

Informative and Invalid positions are exactly balanced globally; Informative is exact within every rule family and feature pair. Compatible subtype positions are near-exact globally. Positions 1-4 are numeric to avoid collision with logical A/B.

## 12. Could the model memorize one target set across counterfactual user rules?

Both matched user rules under a base see the same objects, order, and labels. The semantic Informative/Compatible roles change with H. A fixed option policy therefore cannot solve both counterfactuals.

## 13. Is the null condition normatively scored?

No. It is descriptive. Informative does not prove teaching intent; Compatible does not prove sycophancy.

## 14. Does semantic formula scoring over-credit weird expressions?

Any parseable expression with the correct four-state truth table receives full inference credit. That is intentional: the construct is the induced classification function, not a preferred syntactic derivation.

## 15. Is partial inference performance recoverable?

Yes. The scorer reports both exact semantic rule accuracy and 0-1 truth-table agreement over the four A/B states.

## 16. Why only two relevant dimensions?

This deliberately separates logical complexity from feature-set complexity. A future Level 3 can introduce three predicates, but Level 2 first asks whether the same logical forms generalize across color, shape, texture, and size pairs.

## 17. Are the matched user rules meant to exhaust human biases?

No. They are controlled, interpretable rule perturbations used to produce matched behavior. Human validation is needed before making stronger claims about prevalence or psychological realism.
