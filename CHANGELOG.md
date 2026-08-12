# Changelog

## v4-final

- Closed hypothesis space explicitly shown in all 100 main trials.
- Main responses jointly report `inferred_rule` and A/B/C/D action.
- Separate automatic scoring for rule inference, action selection, and joint success.
- Teacher uses unique corrective gold.
- Imposter uses set-valued acceptable actions with separate specific vs conservative strategy metrics.
- Exactly 25 base rules (5 colors × 5 shapes), four matched cells each = 100 main trials.
- Added 10 no-history diagnosis controls and 10 no-history targeted-teaching controls = 120 total.
- History left/right position balanced; no A/B labels in histories.
- Target A/B/C/D semantic classes counterbalanced.
- Exact history objects cannot repeat as target objects.
- Pre-generated request files are deterministically shuffled.
- Added automatic structural validator, automatic scorer, provider-neutral runner template, local HF runner, self-test, exact 25-rule files, sample prompts, and adversarial audit.
