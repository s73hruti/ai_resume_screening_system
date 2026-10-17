"""Evaluate V2 resume-to-job matching against human-reviewed labels."""

from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)
from model.skill_matching import calculate_skill_overlap
from model.embeddings import load_embedding_model
from model.v2_pipeline import (
    analyze_resume_for_job,
    determine_decision_status,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data/evaluation/v2_eval_candidates.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/evaluation/v2_eval_results.csv"

REQUIREMENT_THRESHOLD = 0.50
CLASSIFICATION_THRESHOLD = 0.20


def main():
    # Read the reviewed candidate pairs.
    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "resume_text",
        "job_description",
        "expected_match",
    }
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    # Exclude rows that were deliberately left unlabeled.
    df = df[df["expected_match"].notna()].copy()

    if df.empty:
        raise ValueError("No labeled evaluation pairs were found.")

    # Validate labels before converting them to integers.
    labels = pd.to_numeric(df["expected_match"], errors="coerce")

    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Labeled expected_match values must contain only 0 or 1.")

    df["expected_match"] = labels.astype(int)

    print(f"Labeled pairs: {len(df)}")
    print("Loading Sentence Transformer model...")

    # Load the embedding model only once for the full evaluation.
    model = load_embedding_model()

    results = []

    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        analysis = analyze_resume_for_job(
            resume=str(row["resume_text"]),
            job_description=str(row["job_description"]),
            model=model,
            threshold=REQUIREMENT_THRESHOLD,
        )

        skill_analysis = calculate_skill_overlap(
            resume=str(row["resume_text"]),
            job_description=(
                f"{row.get('job_title', '')}\n"
                f"{row['job_description']}"
            ),
        )

        score = float(analysis["final_score"])
        predicted_match = int(
            score >= CLASSIFICATION_THRESHOLD
            and analysis["must_haves_satisfied"]
        )
        decision_status = determine_decision_status(
            analysis,
            classification_threshold=CLASSIFICATION_THRESHOLD,
        )


        results.append(
            {
                "source_index": row.name,
                "job_title": row.get("job_title", ""),
                "expected_match": int(row["expected_match"]),
                "predicted_match": predicted_match,
                "decision_status": decision_status,
                "final_score": score,
                "average_similarity": analysis["average_similarity"],
                "requirement_coverage": analysis["requirement_coverage"],

                # Explicit skill-overlap diagnostics
                "matched_skills": ", ".join(skill_analysis["matched_skills"]),
                "missing_skills": ", ".join(skill_analysis["missing_skills"]),
                "skill_overlap": skill_analysis["skill_overlap"],

                "requirements_extracted": len(analysis["requirements"]),
                "strengths_count": len(analysis["strengths"]),
                "potential_gaps_count": len(analysis["potential_gaps"]),
                "must_have_skills": ", ".join(analysis["must_have_skills"]),
                "missing_must_have_skills": ", ".join(analysis["missing_must_have_skills"]),
                "must_haves_satisfied": analysis["must_haves_satisfied"],
            }
        )


        print(
            f"[{row_number}/{len(df)}] "
            f"score={score:.4f}, "
            f"expected={int(row['expected_match'])}, "
            f"predicted={predicted_match}, "
            f"status={decision_status}"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print("\n=== Decision Status Summary ===")
    print(
        results_df["decision_status"]
        .value_counts()
        .reindex(["match", "no_match", "review"], fill_value=0)
        .to_string()
    )

    y_true = results_df["expected_match"]
    y_pred = results_df["predicted_match"]

    print("\n=== V2 Evaluation Results ===")
    print(f"Pairs evaluated: {len(results_df)}")
    print(f"Requirement threshold: {REQUIREMENT_THRESHOLD:.2f}")
    print(f"Classification threshold: {CLASSIFICATION_THRESHOLD:.2f}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(
        "Precision: "
        f"{precision_score(y_true, y_pred, zero_division=0):.4f}"
    )
    print(
        "Recall:    "
        f"{recall_score(y_true, y_pred, zero_division=0):.4f}"
    )
    print(f"F1 score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")

    print("\nConfusion matrix (rows=actual, columns=predicted; labels 0, 1):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    print(f"\nDetailed results saved to: {OUTPUT_PATH}")

    y_score = results_df["final_score"]

    print("\n=== Ranking Metrics ===")

    if y_true.nunique() == 2:
        print(f"ROC-AUC:           {roc_auc_score(y_true, y_score):.4f}")
        print(
            "Average Precision: "
            f"{average_precision_score(y_true, y_score):.4f}"
        )
    else:
        print("ROC-AUC and Average Precision require both label classes.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        raise
