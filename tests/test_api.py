from fastapi.testclient import TestClient

from src.app import main


def test_home_endpoint():
    client = TestClient(main.app)

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "AI Resume Screening API is running"
    }


def test_v1_predict_keeps_existing_response(monkeypatch):
    expected_recommendations = [
        {"job_title": "Python Developer", "score": 0.85}
    ]

    monkeypatch.setattr(
        main,
        "predict_jobs",
        lambda **kwargs: expected_recommendations,
    )

    client = TestClient(main.app)
    response = client.post(
        "/predict",
        json={"resume": "Python developer with relevant experience"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "recommendations": expected_recommendations
    }


def test_v1_rejects_resume_shorter_than_ten_characters():
    client = TestClient(main.app)

    response = client.post("/predict", json={"resume": "short"})

    assert response.status_code == 422


def test_v2_predict_returns_analysis_and_reuses_model(monkeypatch):
    load_calls = []
    analyze_calls = []

    fake_model = object()

    def fake_loader():
        load_calls.append(True)
        return fake_model

    def fake_analyze_resume_for_job(**kwargs):
        analyze_calls.append(kwargs)
        return {
            "final_score": 0.42,
            "decision_status": "match",
            "requirements": ["Python experience"],
            "requirement_matches": [],
            "strengths": ["Python"],
            "potential_gaps": [],
        }

    monkeypatch.setattr(main, "load_embedding_model", fake_loader)
    monkeypatch.setattr(
        main, "analyze_resume_for_job", fake_analyze_resume_for_job
    )

    # Reset the lazy model cache for this isolated test.
    monkeypatch.setattr(main, "_v2_model", None)

    client = TestClient(main.app)
    payload = {
        "resume": "Python developer with relevant experience",
        "job_description": "Requires Python experience",
    }

    first_response = client.post("/predict-v2", json=payload)
    second_response = client.post("/predict-v2", json=payload)

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json()["decision_status"] == "match"
    assert first_response.json()["final_score"] == 0.42
    assert len(load_calls) == 1
    assert len(analyze_calls) == 2
    assert all(call["model"] is fake_model for call in analyze_calls)


def test_v2_requires_job_description():
    client = TestClient(main.app)

    response = client.post(
        "/predict-v2",
        json={"resume": "Python developer with relevant experience"},
    )

    assert response.status_code == 422
