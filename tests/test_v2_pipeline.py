
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
