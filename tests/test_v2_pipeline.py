
from model.v2_pipeline import analyze_resume_for_job


class FakeEmbeddingModel:
    """Return deterministic vectors without downloading a real model."""

    def encode(self, texts, convert_to_tensor=True):
        import numpy as np

        if isinstance(texts, str):
            texts = [texts]

        vectors = []
        for text in texts:
            normalized = text.lower()
            if "python" in normalized:
                vectors.append([1.0, 0.0, 0.0])
            elif "sql" in normalized:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])

        return np.array(vectors, dtype=float)


def test_pipeline_returns_complete_result():
    model = FakeEmbeddingModel()

    result = analyze_resume_for_job(
        resume="Python programming experience.",
        job_description="Requirements:\n- Python programming",
        model=model,
    )

    assert result["requirements"] == ["Python programming"]
    assert len(result["requirement_matches"]) == 1
    assert "final_score" in result
    assert "strengths" in result
    assert "potential_gaps" in result


def test_empty_resume_does_not_crash():
    result = analyze_resume_for_job(
        resume="",
        job_description="Requirements:\n- Python programming",
        model=FakeEmbeddingModel(),
    )

    assert result["requirement_coverage"] == 0.0
    assert result["final_score"] == 0.0
    assert result["potential_gaps"][0]["requirement"] == "Python programming"


def test_no_requirements_does_not_crash():
    result = analyze_resume_for_job(
        resume="Python programming experience.",
        job_description="We are looking for a motivated candidate.",
        model=FakeEmbeddingModel(),
    )

    assert result["requirements"] == []
    assert result["requirement_matches"] == []
    assert result["final_score"] == 0.0
    assert result["strengths"] == []
    assert result["potential_gaps"] == []

from unittest.mock import patch
from model.matching import match_requirements_to_resume


def test_exact_skill_alone_does_not_override_semantic_threshold():
    mocked_result = {
        "requirement": "Angular",
        "score": 0.3568,
        "evidence": "Built applications using Angular",
    }

    with patch(
        "model.matching.find_best_resume_chunk",
        return_value=mocked_result.copy(),
    ):
        results = match_requirements_to_resume(
            model=object(),
            requirements=["Angular"],
            resume_chunks=["Built applications using Angular"],
            threshold=0.50,
        )

    assert results[0]["matched"] is False
    assert results[0]["score"] == 0.3568


def test_partial_exact_skill_overlap_does_not_match_multi_skill_requirement():
    mocked_result = {
        "requirement": "Python and SQL",
        "score": 0.20,
        "evidence": "Python projects",
    }

    with patch(
        "model.matching.find_best_resume_chunk",
        return_value=mocked_result.copy(),
    ):
        results = match_requirements_to_resume(
            model=object(),
            requirements=["Python and SQL"],
            resume_chunks=["Python projects only"],
            threshold=0.50,
        )

    assert results[0]["matched"] is False




def test_explicit_required_skill_matches_when_present_in_evidence():
    mocked_result = {
        "requirement": "Required skill: php",
        "score": 0.3093,
        "evidence": "Experience building applications using PHP and MySQL",
    }

    with patch(
        "model.matching.find_best_resume_chunk",
        return_value=mocked_result.copy(),
    ):
        results = match_requirements_to_resume(
            model=object(),
            requirements=["Required skill: php"],
            resume_chunks=["Experience building applications using PHP and MySQL"],
            threshold=0.50,
        )

    assert results[0]["matched"] is True
    assert results[0]["score"] == 0.3093


def test_explicit_required_skill_does_not_match_without_evidence():
    mocked_result = {
        "requirement": "Required skill: php",
        "score": 0.3093,
        "evidence": "Experience with SQL and database administration",
    }

    with patch(
        "model.matching.find_best_resume_chunk",
        return_value=mocked_result.copy(),
    ):
        results = match_requirements_to_resume(
            model=object(),
            requirements=["Required skill: php"],
            resume_chunks=["Experience with SQL and database administration"],
            threshold=0.50,
        )

    assert results[0]["matched"] is False


