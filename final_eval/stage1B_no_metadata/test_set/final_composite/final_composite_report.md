# Final Composite Report

## Inputs

- Eval-A results: `final_eval/stage1B_no_metadata/test_set/results.csv`
- Eval-B judge results: `final_eval/stage1B_no_metadata/test_set/eval_ab/judge_results_repaired.csv`
- Repair used: `True`

## Formula

`final_weighted_score = 0.70 * decision_score_5 + 0.30 * response_quality_5`

where:

- `decision_score_5 = 5` if `pred_decision == true_decision`, otherwise `0`
- `response_quality_5 = overall_quality`
- missing `overall_quality` is imputed as `1.0`

## Summary

- Number of samples: 4057
- Decision accuracy: 0.9184
- Average response quality: 4.7802
- Raw average overall quality: 4.7802
- Average final weighted score: 4.6485
- Average final weighted percentage: 92.97%
- Judge JSON parse success rate: 1.0000
- Response score imputed count: 0

## Interpretation

This composite score gives 70% weight to decision correctness and 30% weight to response quality. It prevents fluent but incorrectly routed responses from receiving an overly high final score.
