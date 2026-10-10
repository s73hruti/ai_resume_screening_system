# This module contains the logic for matching each JD requirement
# against resume chunks and identifying the strongest semantic match.

from sklearn.metrics.pairwise import cosine_similarity

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
        result["matched"] = score >= threshold

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
    """
    Calculate the V2 resume-to-job ranking score.

    Combines average requirement similarity (60%)
    with requirement coverage (40%).
    """

    # Handle cases where no requirements were extracted.
    if not matching_results:
        return {
            "average_similarity": 0.0,
            "requirement_coverage": 0.0,
            "final_score": 0.0
        }

    # Average the best similarity score for each requirement.
    average_similarity = sum(
        result["score"] for result in matching_results
    ) / len(matching_results)

    # Calculate the proportion of matched requirements.
    coverage = calculate_requirement_coverage(
        matching_results
    )

    # Combine both metrics using the planned weights.
    final_score = (
        0.6 * average_similarity
        + 0.4 * coverage
    )

    # Return individual metrics for explainability.
    return {
        "average_similarity": round(average_similarity, 4),
        "requirement_coverage": round(coverage, 4),
        "final_score": round(final_score, 4)
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

