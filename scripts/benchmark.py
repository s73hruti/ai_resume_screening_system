# Benchmark the inference performance of the job recommendation pipeline.

import statistics
import sys
import time
from pathlib import Path

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import the reusable prediction pipeline.
from src.model.predictor import predict_jobs

# Import the artifact loader.
from src.model.loader import load_artifacts


# Sample resume used for benchmarking.
SAMPLE_RESUME = """
Experienced Python Developer with Django, Flask, SQL, AWS and REST APIs.
Built backend applications and REST APIs using Python and Django.
Experienced with databases, cloud deployment and software development.
"""


# Load the trained artifacts once.
tfidf, job_tfidf_matrix, job_df = load_artifacts()


# Run one prediction before measuring latency.
# This allows libraries and internal operations to warm up.
predict_jobs(
    SAMPLE_RESUME,
    tfidf,
    job_tfidf_matrix,
    job_df,
    top_n=5
)


# Store inference times for multiple runs.
latencies = []

# Number of benchmark runs.
NUM_RUNS = 100


# Measure the complete prediction pipeline.
for _ in range(NUM_RUNS):

    start_time = time.perf_counter()

    recommendations = predict_jobs(
        SAMPLE_RESUME,
        tfidf,
        job_tfidf_matrix,
        job_df,
        top_n=5
    )

    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000

    latencies.append(latency_ms)


# Sort latency values for percentile calculation.
sorted_latencies = sorted(latencies)


# Calculate P95 latency.
p95_index = int(0.95 * len(sorted_latencies)) - 1
p95_latency = sorted_latencies[p95_index]


# Display benchmark results.
print("\n===== Inference Benchmark =====")

print(f"Number of runs       : {NUM_RUNS}")
print(f"Number of job posts  : {len(job_df)}")
print(f"Recommendations/run  : {len(recommendations)}")

print(f"\nAverage latency      : {statistics.mean(latencies):.2f} ms")
print(f"Median latency       : {statistics.median(latencies):.2f} ms")
print(f"Minimum latency      : {min(latencies):.2f} ms")
print(f"Maximum latency      : {max(latencies):.2f} ms")
print(f"P95 latency          : {p95_latency:.2f} ms")