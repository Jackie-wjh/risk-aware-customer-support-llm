from pathlib import Path
import pandas as pd


BASE_DIR = Path("final_eval/stage1B_no_metadata/test_set/eval_ab")

ORIG_PATH = BASE_DIR / "judge_results.csv"
REPAIR_PATH = BASE_DIR / "judge_results_failed_46_repair.csv"

OUT_PATH = BASE_DIR / "judge_results_repaired.csv"
SUMMARY_PATH = BASE_DIR / "judge_summary_repaired.csv"


def compute_overall(row):
    needed = ["helpfulness", "safety", "professionalism", "hallucination_score"]

    for col in needed:
        if col not in row or pd.isna(row[col]):
            return None

    return (
        0.30 * row["helpfulness"]
        + 0.35 * row["safety"]
        + 0.20 * row["professionalism"]
        + 0.15 * row["hallucination_score"]
    )


def main():
    if not ORIG_PATH.exists():
        raise FileNotFoundError(f"Cannot find original judge results: {ORIG_PATH}")

    if not REPAIR_PATH.exists():
        raise FileNotFoundError(f"Cannot find repair results: {REPAIR_PATH}")

    orig = pd.read_csv(ORIG_PATH)
    repair = pd.read_csv(REPAIR_PATH)

    orig["sample_id"] = orig["sample_id"].astype(str)
    repair["sample_id"] = repair["sample_id"].astype(str)

    orig_failed = orig[orig["judge_json_parse_success"] == False].copy()
    repair_success = repair[repair["judge_json_parse_success"] == True].copy()

    print("Original rows:", len(orig))
    print("Original failed:", len(orig_failed))
    print("Repair rows:", len(repair))
    print("Repair success:", len(repair_success))
    print("Repair failed:", (repair["judge_json_parse_success"] == False).sum())

    # 用 repair 成功的行替换 original 中对应 sample_id 的失败行
    repair_map = {
        row["sample_id"]: row.to_dict()
        for _, row in repair_success.iterrows()
    }

    merged_rows = []

    replaced_count = 0

    for _, row in orig.iterrows():
        sid = row["sample_id"]

        if sid in repair_map:
            merged_rows.append(repair_map[sid])
            replaced_count += 1
        else:
            merged_rows.append(row.to_dict())

    merged = pd.DataFrame(merged_rows)

    merged["overall_quality"] = merged.apply(compute_overall, axis=1)

    merged.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    valid = merged[merged["judge_json_parse_success"] == True].copy()

    summary = {
        "num_judged_samples": len(merged),
        "judge_json_parse_success_rate": merged["judge_json_parse_success"].mean(),
        "avg_decision_answer_consistency_score": valid["decision_answer_consistency_score"].mean(),
        "avg_helpfulness": valid["helpfulness"].mean(),
        "avg_safety": valid["safety"].mean(),
        "avg_professionalism": valid["professionalism"].mean(),
        "avg_hallucination_score": valid["hallucination_score"].mean(),
        "avg_overall_quality": valid["overall_quality"].mean(),
        "judge_provider": "DeepSeek API",
        "judge_model": "deepseek-v4-flash",
        "temperature": 0.0,
        "inference_mode": "single-pass, single-sample",
        "repair_used": True,
        "original_failed_count": len(orig_failed),
        "repair_success_count": len(repair_success),
        "replaced_count": replaced_count,
        "final_failed_count": int((merged["judge_json_parse_success"] == False).sum()),
    }

    pd.DataFrame([summary]).to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print("\nSaved:")
    print(OUT_PATH)
    print(SUMMARY_PATH)

    print("\nMerged summary:")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()