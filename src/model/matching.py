# This module contains the logic for matching each JD requirement
# against resume chunks and identifying the strongest semantic match.
import re
from sklearn.metrics.pairwise import cosine_similarity

def verify_experience_requirement(
    requirement: str,
    resume: str,
) -> dict:
    """
    Check whether explicit resume text supports a numeric
    years-of-experience requirement.

    This is a conservative text check, not a complete employment
    history calculator or proof of a candidate's actual experience.
    """

    requirement_text = requirement.lower()
    resume_text = resume.lower()

    # Find numeric experience ranges, such as 5 years or 7-10 years.
    range_pattern = (
        r"(?<!\d)(\d{1,2})\s*"
        r"(?:(?:-|–|to)\s*(\d{1,2})\s*)?"
        r"(?:\+?\s*)?(?:years?|yrs?)\b"
    )

    requirement_ranges = list(re.finditer(
        range_pattern,
        requirement_text,
        flags=re.IGNORECASE,
    ))

    # No numeric experience requirement to verify.
    if not requirement_ranges:
        return {
            "status": "not_applicable",
            "required_years": None,
            "reported_years": None,
        }

    # More than one distinct numeric range may indicate conflicting
    # qualification statements. Do not silently choose one.
    ranges = {
        (
            int(match.group(1)),
            int(match.group(2)) if match.group(2) else None,
        )
        for match in requirement_ranges
    }

    if len(ranges) > 1:
        return {
            "status": "ambiguous",
            "required_years": None,
            "reported_years": None,
        }

    minimum_years, maximum_years = next(iter(ranges))

    # For a range such as 5-7 years, use the lower bound as the
    # minimum requirement. For "5+ years", use 5.
    required_years = minimum_years

    # Look for an explicit duration near experience-related wording.
    resume_patterns = [
        r"(\d{1,2})\s*\+?\s*years?\s+"
        r"(?:of\s+)?(?:professional\s+)?experience",
        r"experience\s+(?:of\s+)?(\d{1,2})\s*\+?\s*years?",
        r"(\d{1,2})\s*\+?\s*years?\s+"
        r"(?:in|as)\s+(?:the\s+)?"
        r"(?:software|database|network|data|development|testing)",
    ]

    reported_years = []

    for pattern in resume_patterns:
        for match in re.finditer(pattern, resume_text):
            reported_years.append(int(match.group(1)))

    if not reported_years:
        return {
            "status": "not_verified",
            "required_years": required_years,
            "reported_years": None,
        }

    # Use the largest explicitly stated duration as evidence.
    # This does not calculate total career duration from job dates.
    highest_reported = max(reported_years)

    return {
        "status": (
            "verified"
            if highest_reported >= required_years
            else "not_verified"
        ),
        "required_years": required_years,
        "reported_years": highest_reported,
    }

def find_best_resume_chunk(
    model,
    requirement: str,
    resume_chunks: list[str]
) -> dict:
    """
    Find the resume chunk that has the highest semantic similarity
    to a given job requirement.

    Parameters:
        model: Loaded Sentence Transformer model.
        requirement (str): One job requirement.
        resume_chunks (list[str]): Resume sentences/chunks.

    Returns:
        dict: Requirement, best similarity score, and matching
              resume chunk as evidence.
    """

    # Generate an embedding for the job requirement.
    # We pass the requirement as a list so the result has
    # the same batch-oriented structure used for the chunks.
    requirement_embedding = model.encode([requirement])

    # Generate embeddings for all resume chunks at once.
    # This avoids repeatedly calling the model inside a loop.
    chunk_embeddings = model.encode(resume_chunks)

    # Compare the requirement embedding against every
    # resume chunk embedding using cosine similarity.
    similarity_scores = cosine_similarity(
        requirement_embedding,
        chunk_embeddings
    )[0]

    # Find the position (index) of the highest similarity score.
    best_index = similarity_scores.argmax()

    # Retrieve the highest similarity score.
    best_score = similarity_scores[best_index]

    # Retrieve the resume chunk responsible for that score.
    best_chunk = resume_chunks[best_index]

    # Return the requirement, similarity score, and the
    # matching resume content as evidence.
    return {
        "requirement": requirement,
        "score": float(best_score),
        "evidence": best_chunk
    }

def match_requirements_to_resume(
    model,
    requirements: list[str],
    resume_chunks: list[str],
    threshold: float = 0.50
) -> list[dict]:
    """
    Match every job requirement against the resume chunks.

    Parameters:
        model: Loaded Sentence Transformer model.
        requirements (list[str]): Requirements extracted from the JD.
        resume_chunks (list[str]): Resume sentences/chunks.
        threshold (float): Minimum similarity score required
                           to consider a requirement matched.

    Returns:
        list[dict]: Matching result for every requirement.
    """

    # Create an empty list to store the matching result
    # for each job requirement.
    results = []

    # Process each requirement one at a time.
    for requirement in requirements:

        # Find the resume chunk that has the highest
        # semantic similarity to the current requirement.
        result = find_best_resume_chunk(
            model,
            requirement,
            resume_chunks
        )

        # Get the similarity score from the matching result.
        score = result["score"]

        # Decide whether the requirement is considered matched.
        #
        # IMPORTANT:
        # The threshold is currently configurable.
        # We will evaluate different threshold values later
        # instead of assuming that 0.50 is scientifically correct.
        # Exact evidence may override semantic similarity only for an
        # explicitly extracted, single-skill requirement such as
        # "Required skill: php". Do not apply this to generic sentences.
        import re

        requirement_match = re.fullmatch(
            r"\s*(?:required|mandatory)\s+skill\s*:\s*(.+?)\s*",
            requirement,
            flags=re.IGNORECASE,
        )

        exact_required_skill_match = False

        if requirement_match:
            from .skill_matching import extract_explicit_skills

            required_skills = extract_explicit_skills(
                requirement_match.group(1)
            )
            evidence_skills = extract_explicit_skills(
                result.get("evidence", "")
            )

            exact_required_skill_match = (
                len(required_skills) == 1
                and required_skills.issubset(evidence_skills)
            )

        result["matched"] = (
            float(score) >= threshold or exact_required_skill_match
        )

        # Add the complete result to our final results list.
        results.append(result)

    # Return the matching result for all requirements.
    return results

def calculate_requirement_coverage(
    matching_results: list[dict]
) -> float:
    """
    Calculate the proportion of job requirements matched
    by the resume.

    Returns:
        float: Coverage between 0.0 and 1.0.
    """

    # Avoid division by zero when no requirements are extracted.
    if not matching_results:
        return 0.0

    # Count requirements whose similarity score met the threshold.
    matched_count = sum(
        1 for result in matching_results
        if result["matched"]
    )

    # Return coverage as a proportion, not a percentage.
    return matched_count / len(matching_results)


def calculate_v2_score(
    matching_results: list[dict]
) -> dict:
    """Calculate the V2 resume-to-job ranking score."""

    if not matching_results:
        return {
            "average_similarity": 0.0,
            "requirement_coverage": 0.0,
            "final_score": 0.0,
        }

    scores = [
        float(result["score"])
        for result in matching_results
    ]

    average_similarity = sum(scores) / len(scores)

    requirement_coverage = calculate_requirement_coverage(
        matching_results
    )

    final_score = (
        0.5 * average_similarity
        + 0.5 * requirement_coverage
    )

    return {
        "average_similarity": round(average_similarity, 4),
        "requirement_coverage": round(requirement_coverage, 4),
        "final_score": round(final_score, 4),
    }

def explain_requirement_matches(
    matching_results: list[dict]
) -> dict:
    """
    Summarize matched requirements and potential skill gaps
    using the existing semantic matching results.
    """

    # Store requirements with sufficient semantic matches.
    strengths = [
        {
            "requirement": result["requirement"],
            "similarity_score": result["score"],
            "evidence": result["evidence"]
        }
        for result in matching_results
        if result["matched"]
    ]

    # Store requirements that did not meet the threshold.
    # These are potential gaps, not proof that a skill is absent.
    potential_gaps = [
        {
            "requirement": result["requirement"],
            "similarity_score": result["score"],
            "closest_resume_text": result["evidence"]
        }
        for result in matching_results
        if not result["matched"]
    ]

    return {
        "strengths": strengths,
        "potential_gaps": potential_gaps
    }

