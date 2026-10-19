"""Compare V1 TF-IDF and V2 semantic ranking on identical labeled pairs."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from src.model.loader import load_artifacts
from src.model.predictor import preprocess_resume


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_PATH = PROJECT_ROOT / "data/evaluation/v2_eval_candidates.csv"
V2_RESULTS_PATH = PROJECT_ROOT / "data/evaluation/v2_eval_results.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/evaluation/v1_v2_comparison.csv"


def main():
    candidates = pd.read_csv(CANDIDATES_PATH)
    v2_results = pd.read_csv(V2_RESULTS_PATH)

    # Preserve original row indexes so they align with V2 source_index.
    candidates["source_index"] = candidates.index
    candidates = candidates[candidates["expected_match"].notna()].copy()

    candidates["expected_match"] = pd.to_numeric(
        candidates["expected_match"], errors="raise"
    ).astype(int)

    if not candidates["expected_match"].isin([0, 1]).all():
        raise ValueError("Expected labels must be binary: 0 or 1.")

    required_v2_columns = {
        "source_index", "expected_match", "final_score", "job_title"
    }
    missing = required_v2_columns - set(v2_results.columns)
    if missing:
        raise ValueError(f"Missing V2 result columns: {sorted(missing)}")

    # Ensure both systems are evaluated on exactly the same labeled pairs.
    merged = candidates.merge(
        v2_results[
            ["source_index", "expected_match", "final_score", "job_title"]
        ],
        on="source_index",
        how="outer",
        suffixes=("_candidate", "_v2"),
        indicator=True,
        validate="one_to_one",
    )

    if not (merged["_merge"] == "both").all():
        raise ValueError("Candidate pairs and V2 results do not align.")

    if not (
        merged["expected_match_candidate"]
        == merged["expected_match_v2"]
    ).all():
        raise ValueError("Ground-truth labels differ between candidates and V2.")

    if not (
        merged["job_title_candidate"].fillna("")
        == merged["job_title_v2"].fillna("")
    ).all():
        raise ValueError("Job titles differ between candidates and V2 results.")

    print("Loading existing V1 artifacts...")
    tfidf, _, _ = load_artifacts()

    v1_scores = []

    for row in merged.itertuples(index=False):
        cleaned_resume = preprocess_resume(str(row.resume_text))
        resume_vector = tfidf.transform([cleaned_resume])

        # V1's original workflow transforms job descriptions directly.
        job_vector = tfidf.transform([str(row.job_description)])

        score = cosine_similarity(resume_vector, job_vector)[0, 0]
        v1_scores.append(float(score))

    merged["v1_tfidf_similarity"] = v1_scores
    merged["v2_final_score"] = merged["final_score"]

    y_true = merged["expected_match_candidate"]

    if y_true.nunique() != 2:
        raise ValueError("ROC-AUC and Average Precision require both label classes.")

    metrics = {
        "V1 TF-IDF cosine similarity": {
            "ROC-AUC": roc_auc_score(y_true, merged["v1_tfidf_similarity"]),
            "Average Precision": average_precision_score(
                y_true, merged["v1_tfidf_similarity"]
            ),
        },
        "V2 semantic final score": {
            "ROC-AUC": roc_auc_score(y_true, merged["v2_final_score"]),
            "Average Precision": average_precision_score(
                y_true, merged["v2_final_score"]
            ),
        },
    }

    output_columns = [
        "source_index",
        "job_title_candidate",
        "expected_match_candidate",
        "v1_tfidf_similarity",
        "v2_final_score",
    ]
    merged[output_columns].rename(
        columns={
            "job_title_candidate": "job_title",
            "expected_match_candidate": "expected_match",
        }
    ).to_csv(OUTPUT_PATH, index=False)

    print(f"\nPairs compared: {len(merged)}")
    print("\nRanking metrics (higher is better):")
    for system, values in metrics.items():
        print(f"\n{system}")
        for metric, value in values.items():
            print(f"  {metric}: {value:.4f}")

    print(f"\nPairwise scores saved to: {OUTPUT_PATH}")
    print(
        "\nNote: These are ranking metrics, not a comparison of "
        "calibrated probabilities or binary match decisions."
    )


if __name__ == "__main__":
    main()
