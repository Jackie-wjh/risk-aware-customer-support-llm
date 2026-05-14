from pathlib import Path
import pandas as pd

EVAL_B_DIR = Path("final_eval/stage1B_no_metadata/test_set/eval_ab")

INPUT = EVAL_B_DIR / "judge_results_repaired.csv"
OUTPUT = EVAL_B_DIR / "judge_results_repaired_with_5dim_avg.csv"
SUMMARY_OUTPUT = EVAL_B_DIR / "judge_summary_repaired_5dim_avg.csv"


five_dims = [
    "decision_answer_consistency_score",
    "helpfulness",
    "safety",
    "professionalism",
    "hallucination_score",
]


df = pd.read_csv(INPUT)

for col in five_dims:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["five_dim_avg_score"] = df[five_dims].mean(axis=1)

df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

summary = {
    "num_samples": len(df),
    "avg_decision_answer_consistency_score": df["decision_answer_consistency_score"].mean(),
    "avg_helpfulness": df["helpfulness"].mean(),
    "avg_safety": df["safety"].mean(),
    "avg_professionalism": df["professionalism"].mean(),
    "avg_hallucination_score": df["hallucination_score"].mean(),
    "avg_five_dim_avg_score": df["five_dim_avg_score"].mean(),
    "judge_json_parse_success_rate": df["judge_json_parse_success"].astype(str).str.lower().isin(["true", "1"]).mean(),
}

pd.DataFrame([summary]).to_csv(SUMMARY_OUTPUT, index=False, encoding="utf-8-sig")

print("Saved:")
print(OUTPUT)
print(SUMMARY_OUTPUT)

print("\nSummary:")
print(pd.DataFrame([summary]).to_string(index=False))