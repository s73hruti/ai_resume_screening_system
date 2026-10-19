# AI Resume Screening System

An end-to-end Machine Learning application that analyzes a resume and recommends the most relevant job opportunities using **NLP, TF-IDF vectorization, and cosine similarity**.

The trained prediction pipeline is exposed through a **FastAPI REST API**, containerized using **Docker**, and deployed on **Render**.

![CI](https://github.com/s73hruti/ai_resume_screening_system/actions/workflows/ci.yml/badge.svg)

## Live Demo

**Live API:**
https://ai-resume-screening-system-gygn.onrender.com

**Swagger UI:**
https://ai-resume-screening-system-gygn.onrender.com/docs

The Swagger UI can be used to submit a resume and test the `/predict` endpoint directly from the browser.

---

## Project Overview

The system takes a candidate's resume as input and compares it against a collection of job descriptions.

The application:

1. Preprocesses the resume using NLP techniques.
2. Converts the resume into a TF-IDF vector.
3. Compares the resume vector with job-description vectors.
4. Calculates cosine similarity scores.
5. Ranks jobs based on similarity.
6. Returns the top unique job recommendations.

The final prediction pipeline is exposed through a FastAPI endpoint and deployed as a Docker container.

---

## Performance & Scale

The system was benchmarked using the deployed prediction pipeline and API:

| Metric | Result |
|---|---:|
| Resume dataset | 54,933 resumes |
| Job postings | 2,277 |
| Local ML inference (average) | 1.91 ms |
| Local ML inference (P95) | 2.09 ms |
| Deployed API latency (median) | 310.07 ms |
| Deployed API latency (P95) | 819.55 ms |
| Benchmark requests | 100 |

The local inference benchmark measures the complete `predict_jobs()` pipeline, including resume preprocessing, TF-IDF transformation, cosine similarity calculation, and unique job recommendation generation.

The deployed API benchmark measures end-to-end response latency for `POST /predict`, including network and server overhead.

---

## Architecture

```text
                    Candidate Resume
                           │
                           ▼
                  Resume Preprocessing
                           │
                           ▼
                    TF-IDF Vectorizer
                           │
                           ▼
                 Resume TF-IDF Vector
                           │
                           ▼
              Cosine Similarity Calculation
                           │
                           ▼
                  Job Recommendations
                           │
                           ▼
                    FastAPI /predict
                           │
                           ▼
                    Docker Container
                           │
                           ▼
                         Render
                           │
                           ▼
                     Public REST API
```

---

## Machine Learning Approach

### 1. Data Processing

The project uses resume and job-description datasets.

Resume information is combined from multiple sources including:

* Experience
* Education
* Skills
* Abilities

The resulting information is combined into a unified resume text representation.

### 2. NLP Preprocessing

The text preprocessing pipeline includes:

* Tokenization
* Stopword removal
* Lemmatization
* Removal of unnecessary characters
* Removal of numeric-only tokens

NLTK is used for the NLP preprocessing steps.

### 3. TF-IDF

TF-IDF is used to convert text into numerical vectors.

A fitted `TfidfVectorizer` is used for both resume and job-description representations.

The vectorizer is persisted as a Joblib artifact so it can be reused during inference.

### 4. Cosine Similarity

Cosine similarity measures how similar the candidate's resume is to each job description.

The jobs are then ranked according to their similarity score.

### 5. Unique Job Recommendations

The recommendation logic prevents duplicate job titles from appearing multiple times in the final results.

---

## Tech Stack

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* TF-IDF
* Cosine Similarity

### API

* FastAPI
* Pydantic
* Uvicorn

### Model Persistence

* Joblib

### Deployment

* Docker
* Render

### Version Control

* Git
* GitHub

---

## Project Structure

```text
ai_resume_screening_system/
│
├── artifacts/
│   ├── job_data.pkl
│   ├── job_tfidf_matrix.joblib
│   └── tfidf_vectorizer.joblib
│
├── src/
│   ├── app/
│   │   └── main.py
│   │
│   └── model/
│       ├── loader.py
│       └── predictor.py
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

Large training/processing artifacts that are not required during inference are excluded from deployment.

---

## API

### POST `/predict`

Accepts a resume and returns the most relevant job recommendations.

### Example Request

```json
{
  "resume": "Experienced Python Developer with Django, Flask, SQL, AWS and REST APIs.",
  "top_n": 5
}
```

### Example Response

```json
{
  "recommendations": [
    {
      "job_title": "Django Developer",
      "similarity_score": 0.3428,
      "job_description": "..."
    },
    {
      "job_title": "Backend Developer",
      "similarity_score": 0.2809,
      "job_description": "..."
    }
  ]
}
```

The API returns the highest-ranked unique job titles based on cosine similarity.

---

## Running with Docker

### Build the Docker image

```bash
docker build -t ai-resume-screening .
```

### Run the container

```bash
docker run -p 8000:8000 ai-resume-screening
```

The API will then be available locally at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

## Running Locally Without Docker

Install the dependencies:

```bash
pip install -r requirements.txt
```

Download the required NLTK resources:

```bash
python -m nltk.downloader stopwords wordnet
```

Start the FastAPI application:

```bash
uvicorn src.app.main:app --reload
```

Open Swagger:

```text
http://localhost:8000/docs
```

---

## Deployment

The application is deployed using **Docker on Render**.

Deployment flow:

```text
GitHub
   │
   ▼
Render
   │
   ▼
Docker Build
   │
   ▼
Docker Container
   │
   ▼
FastAPI
   │
   ▼
Public API
```

The Docker image contains:

* FastAPI application
* Python dependencies
* NLTK resources
* Required ML inference artifacts

---

## Deployment Artifacts

Only the artifacts required for prediction are deployed:

```text
tfidf_vectorizer.joblib
job_tfidf_matrix.joblib
job_data.pkl
```

Large artifacts that are not required for API inference are excluded from the repository and Docker deployment.

---

## Example Recommendation

For a resume such as:

```text
Experienced Python Developer with Django, Flask, SQL, AWS and REST APIs.
```

the system can return recommendations such as:

```text
1. Django Developer
2. Backend Developer
3. DevOps Engineer
4. Full Stack Developer
5. Node js Developer
```

Each recommendation includes a cosine similarity score and the corresponding job description.

---

## Key Engineering Practices

This project demonstrates:

* Modular Python code
* Reusable ML prediction functions
* Separation of training and inference artifacts
* Model serialization using Joblib
* REST API development using FastAPI
* API documentation using Swagger/OpenAPI
* Docker containerization
* Dependency version management
* Docker build optimization using `.dockerignore`
* Cloud deployment
* Git/GitHub version control

---

## Future Improvements

Possible future enhancements include:

* Transformer-based sentence embeddings
* Semantic similarity using embedding models
* Vector database integration
* Resume skill extraction
* Explainable job recommendations
* Candidate-job skill gap analysis
* Authentication and authorization
* Automated CI/CD pipeline
* Model monitoring
* MLflow integration
* Database-backed job storage

---

## Project Goal

The project demonstrates an end-to-end ML engineering workflow:

```text
Data
 ↓
Preprocessing
 ↓
NLP
 ↓
Feature Engineering
 ↓
Similarity Model
 ↓
Reusable Prediction Pipeline
 ↓
FastAPI
 ↓
Docker
 ↓
Cloud Deployment
 ↓
Production-style REST API
```

This project was built to demonstrate practical **Machine Learning, NLP, API development, Docker, and deployment skills**.

---

## V2 Development Progress

**Status:** Requirement-level semantic matching, explicit skill matching, mandatory-skill gating, and experience-requirement verification are implemented and under evaluation. V2 remains experimental and is not production-validated.

### Requirement-Level Semantic Matching

V2 uses Sentence Transformers to compare individual job requirements against resume chunks and identify supporting evidence, strengths, and potential skill gaps.

The scoring formula is:

`final_score = 0.6 * average_similarity + 0.4 * requirement_coverage`

The requirement-matching threshold is `0.50`. The evaluation classification threshold is `0.20`.

The binary prediction is derived from the decision status:
- `match` maps to `1` (predicted match).
- `no_match` maps to `0` (predicted non-match).
- `review` maps to `0` (not counted as an automatic positive prediction).

The decision status is determined using the existing scoring threshold and mandatory-requirement checks. A `review` result remains available for human review even though its binary evaluation prediction is `0`.

### Current Evaluation Results

The latest evaluation ran against 29 labeled resume–job description pairs: 7 matches and 22 non-matches.

| Metric | Result at classification threshold 0.20 |
|---|---:|
| Accuracy | 93.10% |
| Precision | 85.71% |
| Recall | 85.71% |
| F1 score | 0.8571 |
| ROC-AUC | 0.8506 |
| Average precision | 0.7577 |

Confusion matrix: 21 true negatives, 1 false positive, 1 false negative, and 6 true positives.

Decision-status summary: 5 match, 22 no-match, and 2 review.

### Evaluation Limitations

These metrics describe a small development evaluation set and should not be interpreted as expected performance on unseen resumes. The classification threshold is experimental, not production-validated. A larger, independently labeled dataset is needed for reliable assessment and threshold selection.

### Skill Matching, Experience Verification, and Testing

V2 includes explicit technical-skill matching and mandatory-skill checks. Mandatory-skill satisfaction acts as a gate in binary evaluation predictions; it does not alter the scoring formula above.

Experience-requirement verification distinguishes verified, unverified, ambiguous, and non-applicable cases. Its influence on final decision-making requires further review and testing.

The latest targeted verification run passed **21 tests** across `tests/test_experience_verification.py` and `tests/test_v2_pipeline.py`. The full test suite passed after the binary prediction rule was updated.

### Project Scope and Next Steps

- Branch: `v2-semantic-screening`.
- Candidate dataset: `data/evaluation/v2_eval_candidates.csv`.
- Generated evaluation results: `data/evaluation/v2_eval_results.csv`.
- Previous evaluation backup: `data/evaluation/v2_eval_results_pre_day6.csv`.
- Candidate labels remain separate from generated evaluation results.
- V2 has not been integrated into the existing V1 API.
- Next: review the two errors, validate experience-verification behavior in decision-making, run the full test suite, and assess a larger labeled dataset before considering production use.

## Earlier V2 Evaluation Checkpoint

The current V2 evaluation milestone is frozen. The scoring formula and
thresholds remain unchanged while development proceeds.

- Evaluation examples: 30
- Classification threshold: 0.25
- Requirement-match threshold: 0.50
- Accuracy: 93.10%
- Precision: 100.00%
- Recall: 71.43%
- F1 score: 0.8333
- ROC-AUC: 0.8506
- Average Precision: 0.7577
- Confusion matrix: TN=22, FP=0, FN=2, TP=5

Labels for evaluation rows 16 and 18 were revised to non-match based on
insufficient evidence for essential role-specific requirements. Reviewer
rationales are recorded in the evaluation CSV. The original CSV backup is
retained locally.

These metrics describe a small development evaluation set and do not
establish production readiness. No further threshold tuning is part of this
milestone. V2 remains separate from the V1 application.
