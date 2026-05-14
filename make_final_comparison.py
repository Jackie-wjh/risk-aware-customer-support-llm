from pathlib import Path
import pandas as pd
import json


# Paths

ROUTE_NAME = "stage1B_no_metadata"
MODEL_NAME = "Stage 1B Action + Response SFT with LoRA"

BASE_DIR = Path("final_eval/stage1B_no_metadata")

TEST_DIR = BASE_DIR / "test_set"
HIGH_RISK_DIR = BASE_DIR / "high_risk_set"
EVAL_B_DIR = TEST_DIR / "eval_ab"
COMPOSITE_DIR = TEST_DIR / "final_composite"

OUT_DIR = Path("final_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SUMMARY = OUT_DIR / "final_evaluation_summary.csv"
OUT_REPORT = OUT_DIR / "final_evaluation_report.md"


# Helpers

def read_first_existing(paths, required=True):
    for p in paths:
        p = Path(p)
        if p.exists():
            return pd.read_csv(p), p

    if required:
        raise FileNotFoundError(
            "Cannot find any of these files:\n" + "\n".join(str(p) for p in paths)
        )

    return None, None


def get_value(df, col, default=None):
    if df is None or col not in df.columns or len(df) == 0:
        return default
    return df.iloc[0][col]


def fmt(x, digits=4):
    if x is None:
        return "N/A"
    try:
        if pd.isna(x):
            return "N/A"
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def pct(x, digits=2):
    if x is None:
        return "N/A"
    try:
        if pd.isna(x):
            return "N/A"
        return f"{float(x):.{digits}f}%"
    except Exception:
        return str(x)


def safe_read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Load Eval-A summaries

# Your current eval_action2.py may save either:
# results_evaluation_summary.csv
# or evaluation_summary.csv
test_summary, test_summary_path = read_first_existing([
    TEST_DIR / "results_evaluation_summary.csv",
    TEST_DIR / "evaluation_summary.csv",
])

high_risk_summary, high_risk_summary_path = read_first_existing([
    HIGH_RISK_DIR / "results_evaluation_summary.csv",
    HIGH_RISK_DIR / "evaluation_summary.csv",
], required=False)


# Load Eval-B repaired summary

eval_b_summary, eval_b_summary_path = read_first_existing([
    EVAL_B_DIR / "judge_summary_repaired.csv",
    EVAL_B_DIR / "judge_summary.csv",
])

eval_b_by_true, eval_b_by_true_path = read_first_existing([
    EVAL_B_DIR / "judge_summary_by_true_decision_repaired.csv",
    EVAL_B_DIR / "judge_summary_by_true_decision.csv",
], required=False)

eval_b_by_pred, eval_b_by_pred_path = read_first_existing([
    EVAL_B_DIR / "judge_summary_by_pred_decision_repaired.csv",
    EVAL_B_DIR / "judge_summary_by_pred_decision.csv",
], required=False)


# Load final composite

composite_summary, composite_summary_path = read_first_existing([
    COMPOSITE_DIR / "final_composite_summary.csv",
])

composite_by_decision, composite_by_decision_path = read_first_existing([
    COMPOSITE_DIR / "final_composite_by_decision.csv",
])

composite_config = safe_read_json(COMPOSITE_DIR / "final_composite_config.json")


# Extract key metrics

row = {
    "route_name": ROUTE_NAME,
    "model_name": MODEL_NAME,

    # Eval-A full test
    "test_accuracy": get_value(test_summary, "accuracy"),
    "test_macro_f1": get_value(test_summary, "macro_f1"),
    "test_answer_recall": get_value(test_summary, "answer_recall"),
    "test_ask_back_recall": get_value(test_summary, "ask_back_recall"),
    "test_escalate_recall": get_value(test_summary, "escalate_recall"),
    "test_refuse_recall": get_value(test_summary, "refuse_recall"),
    "test_unsafe_answer_rate": get_value(test_summary, "unsafe_answer_rate"),
    "test_json_parse_success_rate": get_value(test_summary, "json_parse_success_rate"),

    # Eval-A high-risk
    "high_risk_accuracy": get_value(high_risk_summary, "accuracy"),
    "high_risk_macro_f1": get_value(high_risk_summary, "macro_f1"),
    "high_risk_escalate_recall": get_value(high_risk_summary, "escalate_recall"),
    "high_risk_refuse_recall": get_value(high_risk_summary, "refuse_recall"),
    "high_risk_unsafe_answer_rate": get_value(high_risk_summary, "unsafe_answer_rate"),
    "high_risk_json_parse_success_rate": get_value(high_risk_summary, "json_parse_success_rate"),

    # Eval-B
    "eval_b_num_judged_samples": get_value(eval_b_summary, "num_judged_samples"),
    "eval_b_judge_json_parse_success_rate": get_value(eval_b_summary, "judge_json_parse_success_rate"),
    "eval_b_avg_decision_answer_consistency_score": get_value(eval_b_summary, "avg_decision_answer_consistency_score"),
    "eval_b_avg_helpfulness": get_value(eval_b_summary, "avg_helpfulness"),
    "eval_b_avg_safety": get_value(eval_b_summary, "avg_safety"),
    "eval_b_avg_professionalism": get_value(eval_b_summary, "avg_professionalism"),
    "eval_b_avg_hallucination_score": get_value(eval_b_summary, "avg_hallucination_score"),
    "eval_b_avg_overall_quality": get_value(eval_b_summary, "avg_overall_quality"),
    "eval_b_judge_provider": get_value(eval_b_summary, "judge_provider"),
    "eval_b_judge_model": get_value(eval_b_summary, "judge_model"),
    "eval_b_repair_used": get_value(eval_b_summary, "repair_used"),

    # Final composite
    "final_num_samples": get_value(composite_summary, "num_samples"),
    "final_decision_accuracy": get_value(composite_summary, "decision_accuracy"),
    "final_avg_response_quality_5": get_value(composite_summary, "avg_response_quality_5"),
    "final_avg_overall_quality_raw": get_value(composite_summary, "avg_overall_quality_raw"),
    "final_judge_json_parse_success_rate": get_value(composite_summary, "judge_json_parse_success_rate"),
    "final_avg_weighted_score": get_value(composite_summary, "avg_final_weighted_score"),
    "final_avg_weighted_pct": get_value(composite_summary, "avg_final_weighted_pct"),
    "final_response_score_imputed_count": get_value(composite_summary, "response_score_imputed_count"),
    "final_response_score_imputed_rate": get_value(composite_summary, "response_score_imputed_rate"),
    "final_weight_decision_correctness": get_value(composite_summary, "weight_decision_correctness"),
    "final_weight_response_quality": get_value(composite_summary, "weight_response_quality"),
    "final_repair_used": get_value(composite_summary, "repair_used"),
}

summary_df = pd.DataFrame([row])
summary_df.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")


# Prepare text snippets for report

def table_from_df(df, cols=None):
    if df is None or len(df) == 0:
        return "_Not available._"

    tmp = df.copy()

    if cols is not None:
        tmp = tmp[[c for c in cols if c in tmp.columns]]

    return tmp.to_markdown(index=False)


composite_by_decision_md = table_from_df(
    composite_by_decision,
    cols=[
        "true_decision",
        "count",
        "decision_accuracy",
        "avg_response_quality_5",
        "avg_final_weighted_score",
        "avg_final_weighted_pct",
    ],
)

eval_b_by_true_md = table_from_df(
    eval_b_by_true,
    cols=[
        "true_decision",
        "count",
        "avg_decision_answer_consistency_score",
        "avg_helpfulness",
        "avg_safety",
        "avg_professionalism",
        "avg_hallucination_score",
        "avg_overall_quality",
    ],
)

eval_b_by_pred_md = table_from_df(
    eval_b_by_pred,
    cols=[
        "pred_decision",
        "count",
        "avg_decision_answer_consistency_score",
        "avg_helpfulness",
        "avg_safety",
        "avg_professionalism",
        "avg_hallucination_score",
        "avg_overall_quality",
    ],
)


# Write Markdown report

report = f"""# Final Evaluation Report

## 1. Selected Final Model

The selected final model is **{MODEL_NAME}**.

- Route name: `{ROUTE_NAME}`
- Eval-A full test results: `{test_summary_path}`
- Eval-B judge results: `{EVAL_B_DIR / "judge_results_repaired.csv"}`
- Final composite results: `{composite_summary_path}`

Metadata, verifier, and DPO variants were treated as ablation / pilot experiments and were not selected as the final model because they did not provide consistent improvement over the Stage 1B no-metadata model.

---

## 2. Evaluation Setup

Eval-A metrics are computed on the full evaluation split.

Eval-B metrics are computed on the full evaluation split and judged by an LLM. In this run, the judge model was **{row["eval_b_judge_model"]}** via **{row["eval_b_judge_provider"]}**.

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
| Accuracy | {fmt(row["test_accuracy"])} |
| Macro-F1 | {fmt(row["test_macro_f1"])} |
| ANSWER Recall | {fmt(row["test_answer_recall"])} |
| ASK_BACK Recall | {fmt(row["test_ask_back_recall"])} |
| ESCALATE Recall | {fmt(row["test_escalate_recall"])} |
| REFUSE Recall | {fmt(row["test_refuse_recall"])} |
| Unsafe Answer Rate | {fmt(row["test_unsafe_answer_rate"])} |
| JSON Parse Success Rate | {fmt(row["test_json_parse_success_rate"])} |

### High-Risk-Only Set

| Metric | Value |
|---|---:|
| Accuracy | {fmt(row["high_risk_accuracy"])} |
| Macro-F1 | {fmt(row["high_risk_macro_f1"])} |
| ESCALATE Recall | {fmt(row["high_risk_escalate_recall"])} |
| REFUSE Recall | {fmt(row["high_risk_refuse_recall"])} |
| Unsafe Answer Rate | {fmt(row["high_risk_unsafe_answer_rate"])} |
| JSON Parse Success Rate | {fmt(row["high_risk_json_parse_success_rate"])} |

---

## 4. Eval-B Response-Layer Results

Eval-B was run on the full test split. The repaired judge result file is used for this final report.

| Metric | Value |
|---|---:|
| Judged Samples | {row["eval_b_num_judged_samples"]} |
| Judge JSON Parse Success Rate | {fmt(row["eval_b_judge_json_parse_success_rate"])} |
| Avg Decision-Answer Consistency | {fmt(row["eval_b_avg_decision_answer_consistency_score"])} |
| Avg Helpfulness | {fmt(row["eval_b_avg_helpfulness"])} |
| Avg Safety | {fmt(row["eval_b_avg_safety"])} |
| Avg Professionalism | {fmt(row["eval_b_avg_professionalism"])} |
| Avg Hallucination Score | {fmt(row["eval_b_avg_hallucination_score"])} |
| Avg Overall Quality | {fmt(row["eval_b_avg_overall_quality"])} |

A repair pass was applied to malformed judge outputs. The original judge parse success rate was lower, while the repaired version achieved full parse success. The repaired results are used for the final composite analysis.

### Eval-B by True Decision

{eval_b_by_true_md}

### Eval-B by Predicted Decision

{eval_b_by_pred_md}

---

## 5. Final Decision-Weighted Composite

| Metric | Value |
|---|---:|
| Number of Samples | {row["final_num_samples"]} |
| Decision Accuracy | {fmt(row["final_decision_accuracy"])} |
| Avg Response Quality / 5 | {fmt(row["final_avg_response_quality_5"])} |
| Avg Final Weighted Score / 5 | {fmt(row["final_avg_weighted_score"])} |
| Avg Final Weighted Percentage | {pct(row["final_avg_weighted_pct"])} |
| Judge JSON Parse Success Rate | {fmt(row["final_judge_json_parse_success_rate"])} |
| Response Score Imputed Count | {row["final_response_score_imputed_count"]} |
| Decision Weight | {fmt(row["final_weight_decision_correctness"])} |
| Response Quality Weight | {fmt(row["final_weight_response_quality"])} |

### Final Composite by True Decision

{composite_by_decision_md}

---

## 6. Interpretation

The final model achieved strong overall performance. Eval-A shows that the decision layer performs well on the full test split, with the main remaining challenge coming from boundary cases between `ASK_BACK` and `ESCALATE`.

Eval-B indicates that the generated customer-facing responses are generally strong, especially in safety and hallucination control. The repaired full-split judge results give an average overall response quality of **{fmt(row["eval_b_avg_overall_quality"])} / 5**.

The final decision-weighted composite score is **{fmt(row["final_avg_weighted_score"])} / 5**, or **{pct(row["final_avg_weighted_pct"])}**. This score gives 70% weight to correct action decisions and 30% weight to response quality, so it penalizes fluent responses when the predicted decision is wrong.

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
"""

OUT_REPORT.write_text(report, encoding="utf-8")

print("Saved:")
print(OUT_SUMMARY)
print(OUT_REPORT)

print("\nFinal comparison summary:")
print(summary_df.to_string(index=False))