from .requirements import extract_requirements
from .chunks import split_resume_into_chunks

from .matching import (
    match_requirements_to_resume,
    calculate_v2_score,
    explain_requirement_matches,
)

def analyze_resume_for_job(
    resume: str,
    job_description: str,
    model,
    threshold: float = 0.50,
) -> dict:
    """Analyze a resume against a job description using V2."""

    # Step 1: Extract requirements from the job description.
    requirements = extract_requirements(job_description)

    # Step 2: Split the resume into smaller text chunks.
    resume_chunks = split_resume_into_chunks(resume)

    # Step 3: Match each requirement with the resume.
    if not requirements or not resume_chunks:
        matching_results = [
            {
                "requirement": requirement,
                "score": 0.0,
                "evidence": "",
                "matched": False,
            }
            for requirement in requirements
        ]
    else:
        matching_results = match_requirements_to_resume(
            model,
            requirements,
            resume_chunks,
            threshold=threshold,
        )

    # Step 4: Calculate the overall V2 score.
    score_details = calculate_v2_score(matching_results)

    # Step 5: Explain matched requirements and potential gaps.
    explanations = explain_requirement_matches(matching_results)

    # Step 6: Return the complete analysis.
    return {
        "requirements": requirements,
        "requirement_matches": matching_results,
        **score_details,
        **explanations,
    }