from .requirements import extract_requirements
from .chunks import split_resume_into_chunks
from .skill_matching import check_must_have_skills
from .matching import (
    match_requirements_to_resume,
    calculate_v2_score,
    explain_requirement_matches,
    verify_experience_requirement,
)

def determine_decision_status(
    analysis: dict,
    classification_threshold: float = 0.20,
) -> str:
    """Classify the analysis as match, no_match, or review."""

    # Mandatory-skill failures remain a hard rejection.
    if not analysis["must_haves_satisfied"]:
        return "no_match"

    # A score below the existing classification threshold is a no-match.
    if analysis["final_score"] < classification_threshold:
        return "no_match"

    # A sufficiently high score without any matched requirements
    # is uncertain, so send it for review.
    if analysis["requirement_coverage"] == 0:
        return "review"

    # The score is sufficient, mandatory skills pass, and at least
    # one requirement is matched.
    return "match"

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

    # Step 6: Verify all experience requirements together.
    # This helps detect conflicting numeric experience ranges.
    experience_requirements = [
        requirement
        for requirement in requirements
        if "experience" in requirement.lower()
    ]

    experience_verification = []

    if experience_requirements:
        combined_requirement = "; ".join(experience_requirements)

        verification = verify_experience_requirement(
            requirement=combined_requirement,
            resume=resume,
        )

        experience_verification.append(
            {
                "requirements": experience_requirements,
                **verification,
            }
        )
    # Step 7: Verify explicitly mandatory skills.
    mandatory_skill_check = check_must_have_skills(
        resume=resume,
        job_description=job_description,
    )


    # Step 8: Return the complete analysis and decision status.
    analysis = {
        "requirements": requirements,
        "requirement_matches": matching_results,
        **score_details,
        **explanations,
        "experience_verification": experience_verification,
        **mandatory_skill_check,
    }

    analysis["decision_status"] = determine_decision_status(analysis)

    return analysis
