# Benchmark the deployed FastAPI prediction endpoint.

import statistics
import time
import urllib.request
import json


# Deployed FastAPI endpoint.
API_URL = "https://ai-resume-screening-system-gygn.onrender.com/predict"


# Sample resume used for benchmarking.
SAMPLE_RESUME = """
Experienced Python Developer with Django, Flask, SQL, AWS and REST APIs.
Built backend applications and REST APIs using Python and Django.
Experienced with databases, cloud deployment and software development.
"""


# Number of measured API requests.
NUM_RUNS = 100


# Prepare the request body.
payload = json.dumps({
    "resume": SAMPLE_RESUME
}).encode("utf-8")


# HTTP headers required by the FastAPI endpoint.
headers = {
    "Content-Type": "application/json"
}


# Warm-up request.
# This is not included in the benchmark because
# deployed services may have cold-start overhead.
request = urllib.request.Request(
    API_URL,
    data=payload,
    headers=headers,
    method="POST"
)

with urllib.request.urlopen(request, timeout=60) as response:
    if response.status != 200:
        raise RuntimeError(
            f"Warm-up request failed with status {response.status}"
        )


# Store API response times.
latencies = []


# Measure 100 deployed API requests.
for _ in range(NUM_RUNS):

    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers=headers,
        method="POST"
    )

    start_time = time.perf_counter()

    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(
                f"Request failed with status {response.status}"
            )

        response.read()

    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000

    latencies.append(latency_ms)


# Sort latency values for percentile calculation.
sorted_latencies = sorted(latencies)


# Calculate P95 latency.
p95_index = int(0.95 * len(sorted_latencies)) - 1
p95_latency = sorted_latencies[p95_index]


# Display results.
print("\n===== Deployed API Benchmark =====")

print(f"Number of runs       : {NUM_RUNS}")
print(f"Endpoint             : POST /predict")

print(f"\nAverage latency      : {statistics.mean(latencies):.2f} ms")
print(f"Median latency       : {statistics.median(latencies):.2f} ms")
print(f"Minimum latency      : {min(latencies):.2f} ms")
print(f"Maximum latency      : {max(latencies):.2f} ms")
print(f"P95 latency          : {p95_latency:.2f} ms")
