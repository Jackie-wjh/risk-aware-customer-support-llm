# Final Evaluation Report

## 1. Selected Final Model

The selected final model is **Stage 1B Action + Response SFT with LoRA**.

- Route name: `stage1B_no_metadata`
- Eval-A full test results: `final_eval/stage1B_no_metadata/test_set/results_evaluation_summary.csv`
- Eval-B judge results: `final_eval/stage1B_no_metadata/test_set/eval_ab/judge_results_repaired.csv`
- Final composite results: `final_eval/stage1B_no_metadata/test_set/final_composite/final_composite_summary.csv`

Metadata, verifier, and DPO variants were treated as ablation / pilot experiments and were not selected as the final model because they did not provide consistent improvement over the Stage 1B no-metadata model.

---

## 2. Evaluation Setup

Eval-A metrics are computed on the full evaluation split.

Eval-B metrics are computed on the full evaluation split and judged by an LLM. In this run, the judge model was **deepseek-v4-flash** via **DeepSeek API**.

Eval-A+B combines full-split decision-layer evaluation with full-split response-quality evaluation.

The response quality score uses:

`overall_quality = 0.30 × helpfulness + 0.35 × safety + 0.20 × professionalism + 0.15 × hallucination_score`

The final decision-weighted composite score uses:

`final_weighted_score = 0.70 × decision_score_5 + 0.30 × response_quality_5`

where `decision_score_5 = 5` if `pred_decision == true_decision`, otherwise `0`.

---

## 3. Eval-A Decision-Layer Results

### Full Test Set

| Metric | Value |
|---|---:|
| Accuracy | 0.9184 |
| Macro-F1 | 0.8972 |
| ANSWER Recall | 0.9552 |
| ASK_BACK Recall | 0.8679 |
| ESCALATE Recall | 0.8419 |
| REFUSE Recall | 1.0000 |
| Unsafe Answer Rate | 0.0880 |
| JSON Parse Success Rate | 1.0000 |

### High-Risk-Only Set

| Metric | Value |
|---|---:|
| Accuracy | 0.8531 |
| Macro-F1 | 0.8524 |
| ESCALATE Recall | 0.8333 |
| REFUSE Recall | 1.0000 |
| Unsafe Answer Rate | 0.0852 |
| JSON Parse Success Rate | 1.0000 |

---

## 4. Eval-B Response-Layer Results

Eval-B was run on the full test split. The repaired judge result file is used for this final report.

| Metric | Value |
|---|---:|
| Judged Samples | 4057 |
| Judge JSON Parse Success Rate | 1.0000 |
| Avg Decision-Answer Consistency | 4.6803 |
| Avg Helpfulness | 4.4750 |
| Avg Safety | 4.9798 |
| Avg Professionalism | 4.8011 |
| Avg Hallucination Score | 4.8970 |
| Avg Overall Quality | 4.7802 |

A repair pass was applied to malformed judge outputs. The original judge parse success rate was lower, while the repaired version achieved full parse success. The repaired results are used for the final composite analysis.

### Eval-B by True Decision

| true_decision   |   count |   avg_decision_answer_consistency_score |   avg_helpfulness |   avg_safety |   avg_professionalism |   avg_hallucination_score |   avg_overall_quality |
|:----------------|--------:|----------------------------------------:|------------------:|-------------:|----------------------:|--------------------------:|----------------------:|
| ANSWER          |    2410 |                                 4.64232 |           4.49585 |      4.97842 |               4.8     |                   4.8751  |               4.78247 |
| ASK_BACK        |    1363 |                                 4.73001 |           4.45194 |      4.98166 |               4.78944 |                   4.9193  |               4.77494 |
| ESCALATE        |     272 |                                 4.75368 |           4.44485 |      4.98162 |               4.86765 |                   4.97426 |               4.79669 |
| REFUSE          |      12 |                                 5       |           3.58333 |      5       |               4.83333 |                   5       |               4.54167 |

### Eval-B by Predicted Decision

| pred_decision   |   count |   avg_decision_answer_consistency_score |   avg_helpfulness |   avg_safety |   avg_professionalism |   avg_hallucination_score |   avg_overall_quality |
|:----------------|--------:|----------------------------------------:|------------------:|-------------:|----------------------:|--------------------------:|----------------------:|
| ANSWER          |    2414 |                                 4.55012 |           4.48136 |      4.98053 |               4.80365 |                   4.87448 |               4.77949 |
| ASK_BACK        |    1285 |                                 4.8537  |           4.47471 |      4.97432 |               4.78444 |                   4.92296 |               4.77875 |
| ESCALATE        |     346 |                                 4.93353 |           4.46243 |      4.99422 |               4.84393 |                   4.95376 |               4.79855 |
| REFUSE          |      12 |                                 5       |           3.58333 |      5       |               4.83333 |                   5       |               4.54167 |

---

## 5. Final Decision-Weighted Composite

| Metric | Value |
|---|---:|
| Number of Samples | 4057 |
| Decision Accuracy | 0.9184 |
| Avg Response Quality / 5 | 4.7802 |
| Avg Final Weighted Score / 5 | 4.6485 |
| Avg Final Weighted Percentage | 92.97% |
| Judge JSON Parse Success Rate | 1.0000 |
| Response Score Imputed Count | 0 |
| Decision Weight | 0.7000 |
| Response Quality Weight | 0.3000 |

### Final Composite by True Decision

| true_decision   |   count |   decision_accuracy |   avg_response_quality_5 |   avg_final_weighted_score |   avg_final_weighted_pct |
|:----------------|--------:|--------------------:|-------------------------:|---------------------------:|-------------------------:|
| ANSWER          |    2410 |            0.955187 |                  4.78247 |                    4.77789 |                  95.5579 |
| ASK_BACK        |    1363 |            0.867938 |                  4.77494 |                    4.47027 |                  89.4054 |
| ESCALATE        |     272 |            0.841912 |                  4.79669 |                    4.3857  |                  87.714  |
| REFUSE          |      12 |            1        |                  4.54167 |                    4.8625  |                  97.25   |

---

## 6. Interpretation

The final model achieved strong overall performance. Eval-A shows that the decision layer performs well on the full test split, with the main remaining challenge coming from boundary cases between `ASK_BACK` and `ESCALATE`.

Eval-B indicates that the generated customer-facing responses are generally strong, especially in safety and hallucination control. The repaired full-split judge results give an average overall response quality of **4.7802 / 5**.

The final decision-weighted composite score is **4.6485 / 5**, or **92.97%**. This score gives 70% weight to correct action decisions and 30% weight to response quality, so it penalizes fluent responses when the predicted decision is wrong.

The per-decision composite results show that `ANSWER` and `REFUSE` are the strongest categories. `ASK_BACK` and especially `ESCALATE` remain the main areas for future improvement because their response quality is high, but their decision accuracy is lower than `ANSWER`.

---

## 7. Key Output Files

- `final_eval/stage1B_no_metadata/test_set/results.csv`
- `final_eval/stage1B_no_metadata/test_set/eval_ab/judge_results_repaired.csv`
- `final_eval/stage1B_no_metadata/test_set/eval_ab/judge_summary_repaired.csv`
- `final_eval/stage1B_no_metadata/test_set/final_composite/final_composite_summary.csv`
- `final_eval/stage1B_no_metadata/test_set/final_composite/final_composite_by_decision.csv`
- `final_comparison/final_evaluation_summary.csv`
- `final_comparison/final_evaluation_report.md`
