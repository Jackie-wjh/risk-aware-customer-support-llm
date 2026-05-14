from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Paths

EVAL_B_DIR = Path("final_eval/stage1B_no_metadata/test_set/eval_ab")

# Use repaired judge results
RESULTS_PATH = EVAL_B_DIR / "judge_results_repaired.csv"

# Save repaired summaries
SUMMARY_PATH = EVAL_B_DIR / "judge_summary_repaired.csv"
BY_PRED_PATH = EVAL_B_DIR / "judge_summary_by_pred_decision_repaired.csv"
BY_TRUE_PATH = EVAL_B_DIR / "judge_summary_by_true_decision_repaired.csv"

# Standard Eval-B visuals folder
VISUALS_DIR = EVAL_B_DIR / "visuals"
VISUALS_DIR.mkdir(parents=True, exist_ok=True)


# Helpers

def to_bool_series(s):
    """
    Convert bool/string bool column to real bool.
    """
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


# Main

def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Cannot find repaired judge results: {RESULTS_PATH}")

    df = pd.read_csv(RESULTS_PATH)

    if "judge_json_parse_success" not in df.columns:
        raise ValueError("Missing column: judge_json_parse_success")

    df["judge_json_parse_success_bool"] = to_bool_series(df["judge_json_parse_success"])

    score_cols = [
        "decision_answer_consistency_score",
        "helpfulness",
        "safety",
        "professionalism",
        "hallucination_score",
        "overall_quality",
    ]

    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = df[df["judge_json_parse_success_bool"] == True].copy()

    print("Total rows:", len(df))
    print("Valid judged rows:", len(valid))
    print("Parse success rate:", df["judge_json_parse_success_bool"].mean())

    # 1. Overall repaired judge summary

    summary = {
        "num_judged_samples": len(df),
        "judge_json_parse_success_rate": df["judge_json_parse_success_bool"].mean(),
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
    }

    pd.DataFrame([summary]).to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # 2. Summary by predicted decision

    by_pred = (
        valid.groupby("pred_decision")
        .agg(
            count=("sample_id", "count"),
            avg_decision_answer_consistency_score=("decision_answer_consistency_score", "mean"),
            avg_helpfulness=("helpfulness", "mean"),
            avg_safety=("safety", "mean"),
            avg_professionalism=("professionalism", "mean"),
            avg_hallucination_score=("hallucination_score", "mean"),
            avg_overall_quality=("overall_quality", "mean"),
        )
        .reset_index()
    )

    by_pred.to_csv(
        BY_PRED_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # 3. Summary by true decision

    by_true = (
        valid.groupby("true_decision")
        .agg(
            count=("sample_id", "count"),
            avg_decision_answer_consistency_score=("decision_answer_consistency_score", "mean"),
            avg_helpfulness=("helpfulness", "mean"),
            avg_safety=("safety", "mean"),
            avg_professionalism=("professionalism", "mean"),
            avg_hallucination_score=("hallucination_score", "mean"),
            avg_overall_quality=("overall_quality", "mean"),
        )
        .reset_index()
    )

    by_true.to_csv(
        BY_TRUE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # 4. judge_dimension_means.png

    dimension_cols = [
        "decision_answer_consistency_score",
        "helpfulness",
        "safety",
        "professionalism",
        "hallucination_score",
    ]

    display_names = [
        "Consistency",
        "Helpfulness",
        "Safety",
        "Professionalism",
        "Hallucination",
    ]

    means = valid[dimension_cols].mean()

    plt.figure(figsize=(9, 5))
    bars = plt.bar(display_names, means.values)
    plt.ylim(0, 5)
    plt.ylabel("Average Score")
    plt.title("Eval-B Judge Dimension Means")
    plt.xticks(rotation=20, ha="right")

    for bar, value in zip(bars, means.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.05,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "judge_dimension_means.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5. judge_score_distribution.png


    plt.figure(figsize=(8, 5))
    plt.hist(valid["overall_quality"].dropna(), bins=20)
    plt.xlabel("Overall Quality")
    plt.ylabel("Count")
    plt.title("Eval-B Overall Quality Distribution")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "judge_score_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 6. judge_dimension_radar.png

    radar_values = means.values.tolist()
    radar_values += radar_values[:1]

    angles = np.linspace(0, 2 * np.pi, len(display_names), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, radar_values, linewidth=2)
    ax.fill(angles, radar_values, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display_names)
    ax.set_ylim(0, 5)
    ax.set_title("Eval-B Judge Dimension Radar", pad=20)
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "judge_dimension_radar.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nSaved:")
    print(SUMMARY_PATH)
    print(BY_PRED_PATH)
    print(BY_TRUE_PATH)
    print(VISUALS_DIR / "judge_dimension_means.png")
    print(VISUALS_DIR / "judge_score_distribution.png")
    print(VISUALS_DIR / "judge_dimension_radar.png")

    print("\nOverall repaired summary:")
    print(pd.DataFrame([summary]).to_string(index=False))

    print("\nBy predicted decision:")
    print(by_pred.to_string(index=False))

    print("\nBy true decision:")
    print(by_true.to_string(index=False))


if __name__ == "__main__":
    main()