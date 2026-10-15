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

## V2 Development Progress — Day 3

**Status:** Requirement-level semantic matching implemented; evaluation completed provisionally. Further improvements and validation are required before production use.

### Requirement-Level Semantic Matching

V2 uses Sentence Transformers to compare individual job requirements against resume chunks and identify supporting evidence, strengths, and potential skill gaps.

The scoring formula is:

`final_score = 0.6 * average_similarity + 0.4 * requirement_coverage`

The requirement-matching threshold is `0.50`. The classification threshold currently used by the evaluation script is `0.25`.

### Current Evaluation Results

The evaluation uses 30 labeled resume–job description pairs: 10 matches and 20 non-matches.

| Metric | Result at classification threshold 0.25 |
|---|---:|
| Accuracy | 66.67% |
| Precision | 50.00% |
| Recall | 20.00% |
| F1 score | 0.2857 |
| ROC-AUC | 0.7400 |
| Average precision | 0.6297 |

Confusion matrix: 18 true negatives, 2 false positives, 8 false negatives, and 2 true positives.

### Threshold Decision and Limitations

Threshold `0.25` is the current experimental setting, not a validated production threshold. In this evaluation, it identifies 2 of 10 labeled matches and incorrectly classifies 2 non-matches as matches.

Threshold comparisons on this small dataset show the precision–recall trade-off:

| Threshold | Accuracy | Precision | Recall | F1 score |
|---:|---:|---:|---:|---:|
| 0.20 | 76.67% | 71.43% | 50.00% | 0.5882 |
| 0.25 | 66.67% | 50.00% | 20.00% | 0.2857 |
| 0.30 | 70.00% | 66.67% | 20.00% | 0.3077 |
| 0.35 | 70.00% | 100.00% | 10.00% | 0.1818 |

These results are based on only 30 examples and should not be interpreted as expected performance on unseen resumes. A larger, independently labeled dataset is needed for reliable threshold selection.

### Skill Matching and Testing

Explicit skill-overlap and mandatory-skill extraction utilities are being developed and tested separately. Mandatory-skill extraction is currently diagnostic-only and does not affect the V2 scoring formula.

The latest full test run passed: **57 tests**.

### Project Scope and Next Steps

- Branch: `v2-semantic-screening`.
- Evaluation output: `data/evaluation/v2_eval_results.csv`.
- Candidate labels are kept separate from generated evaluation results.
- V2 has not been integrated into the existing V1 API.
- Next: improve requirement matching, investigate false negatives, and evaluate on a larger labeled dataset before considering production use.

## V2 Evaluation Checkpoint — 15 October 2026

The current V2 evaluation milestone is frozen. The scoring formula and
thresholds remain unchanged while development proceeds.

- Evaluation examples: 30
- Classification threshold: 0.25
- Requirement-match threshold: 0.50
- Accuracy: 73.33%
- Precision: 50.00%
- Recall: 25.00%
- F1 score: 0.3333
- ROC-AUC: 0.7330
- Average Precision: 0.5966
- Confusion matrix: TN=20, FP=2, FN=6, TP=2

Labels for evaluation rows 16 and 18 were revised to non-match based on
insufficient evidence for essential role-specific requirements. Reviewer
rationales are recorded in the evaluation CSV. The original CSV backup is
retained locally.

These metrics describe a small development evaluation set and do not
establish production readiness. No further threshold tuning is part of this
milestone. V2 remains separate from the V1 application.
