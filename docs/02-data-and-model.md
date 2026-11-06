# Bank Customer Subscription ML Deployment — Data Card and Model Notes

## Source and attribution

Dataset: Bank Marketing, UCI Machine Learning Repository, dataset 222.

Source: https://archive.ics.uci.edu/dataset/222/bank+marketing

Download: https://archive.ics.uci.edu/static/public/222/bank+marketing.zip

Creators credited by UCI: S. Moro, P. Rita, P. Cortez. Dataset DOI: https://doi.org/10.24432/C5K306

UCI lists the license as Creative Commons Attribution 4.0 International: https://creativecommons.org/licenses/by/4.0/

We use `bank-full.csv`, the original date-ordered 45,211-row version, not the 41,188-row `bank-additional-full.csv`. This project selects features and derives a binary label; it does not redistribute the raw dataset. Keep this attribution when reusing it.

## Prediction timing and features

The score is calculated before the current marketing contact. Features: age, balance, previous contact count, job, marital status, education, credit default, housing loan, personal loan, and previous campaign outcome. `unknown` is an explicit source category, not a missing Python value. New categories are encoded as all-zero indicators for that feature; prediction continues but may be less reliable.

We exclude duration because it is known only after the call. We also omit contact channel, current campaign count, day and month for a conservative first baseline. The source does not establish that every financial field was frozen at the exact scoring time. Historical leakage and repeated-customer effects therefore remain limitations. This is not a causal model.

## Evaluation

First 80% train, final 20% test, preserving source order. Do not shuffle these rows. This better resembles prediction on later campaigns than a random split, but record-level timestamps and customer IDs are absent from our selected source. The same person may appear across periods. No tuning is performed, and threshold 0.5 is only illustrative.

Average precision and ROC AUC describe ranking. Precision/recall at a fixed threshold describe a particular decision rule. The test positive rate is the random-ranking reference for average precision. Accuracy alone is not useful for this imbalanced target. Generated test fixtures validate code, not measured business performance.

## Intended use and limits

Educational demonstration only. No fairness audit, privacy review, probability calibration, modern population validation, or business cost analysis has been performed. Historical Portuguese bank marketing outcomes do not establish generalisation to other banks or countries. Age and marital information can raise important policy and fairness questions that must be assessed before operational use.

`joblib` model files must come from trusted builds because loading them can execute code. Model and preprocessing are serialized together. A different library version may be incompatible, so metrics record versions. Deployment should eventually use pinned dependencies, immutable model versions and explicit approval gates.
