"""Inspect extracted requirements and semantic evidence for selected pairs."""

from pathlib import Path
import sys

import pandas as pd

from model.embeddings import load_embedding_model
from model.v2_pipeline import analyze_resume_for_job
from model.skill_matching import calculate_skill_overlap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data/evaluation/v2_eval_candidates.csv"

TARGET_INDICES = [11, 27]
REQUIREMENT_THRESHOLD = 0.50


def main():
    df = pd.read_csv(INPUT_PATH)
    df.index.name = "source_index"

    missing = sorted(set(TARGET_INDICES) - set(df.index))
    if missing:
        raise ValueError(f"Candidate indices not found: {missing}")

    model = load_embedding_model()

    for index in TARGET_INDICES:
        row = df.loc[index]

        analysis = analyze_resume_for_job(
            resume=str(row["resume_text"]),
            job_description=str(row["job_description"]),
            model=model,
            threshold=REQUIREMENT_THRESHOLD,
        )

        print("\n" + "=" * 90)
        print(f"Candidate index: {index}")
        print(f"Job title: {row.get('job_title', '')}")
        print(f"Expected label: {int(row['expected_match'])}")
        print(f"Final score: {analysis['final_score']:.4f}")
        print(f"Extracted requirements: {len(analysis['requirements'])}")

        
        skill_analysis = calculate_skill_overlap(
            resume=str(row["resume_text"]),
            job_description=(
                f"{row.get('job_title', '')}\n"
                f"{row['job_description']}"
            ),
        )

        print(f"\nResume skills: {skill_analysis['resume_skills']}")
        print(f"Job skills: {skill_analysis['job_skills']}")
        print(f"Matched skills: {skill_analysis['matched_skills']}")
        print(f"Missing skills: {skill_analysis['missing_skills']}")
        print(f"Skill overlap: {skill_analysis['skill_overlap']:.1%}")


        for number, result in enumerate(
            analysis["requirement_matches"], start=1
        ):
            print(f"\n  Requirement {number}: {result['requirement']}")
            print(f"  Similarity: {result['score']:.4f}")
            print(f"  Matched at {REQUIREMENT_THRESHOLD:.2f}: {result['matched']}")
            print(f"  Resume evidence: {result['evidence']}")

        print("\nResume excerpt:")
        print(str(row["resume_text"])[:1200])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Diagnostic failed: {exc}", file=sys.stderr)
        raise
