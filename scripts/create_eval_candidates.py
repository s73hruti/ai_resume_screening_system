import random
import re
from pathlib import Path

import pandas as pd

SEED = 42

resumes = pd.read_csv("data/processed/cleaned_resume_dataset.csv")
jobs = pd.read_pickle("artifacts/job_data.pkl")


def normalize(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


categories = [
    "database administrator",
    "full stack developer",
    "java developer",
    "network administrator",
    "software engineer",
]

resumes["category"] = resumes["name"].map(normalize)
jobs["category"] = jobs["Job Title"].map(normalize)

resumes = resumes[
    resumes["category"].isin(categories)
    & resumes["cleaned_resume_text"].notna()
].copy()

jobs = jobs[
    jobs["category"].isin(categories)
    & jobs["Job Description"].notna()
].copy()

pairs = []

# Create 15 same-category candidate pairs.
for category in categories:
    resume_rows = resumes[
        resumes["category"] == category
    ].sample(n=3, random_state=SEED)

    job_rows = jobs[
        jobs["category"] == category
    ].sample(n=3, random_state=SEED)

    for (_, resume), (_, job) in zip(
        resume_rows.iterrows(), job_rows.iterrows()
    ):
        pairs.append({
            "pair_type": "same_category_candidate",
            "resume_category": category,
            "job_title": job["Job Title"],
            "resume_text": resume["cleaned_resume_text"],
            "job_description": job["Job Description"],
            "expected_match": "",
            "reviewer_notes": "",
        })

# Create 15 cross-category candidate pairs.
for i, category in enumerate(categories):
    other_category = categories[(i + 1) % len(categories)]

    resume_rows = resumes[
        resumes["category"] == category
    ].sample(n=3, random_state=SEED + i + 1)

    job_rows = jobs[
        jobs["category"] == other_category
    ].sample(n=3, random_state=SEED + i + 11)

    for (_, resume), (_, job) in zip(
        resume_rows.iterrows(), job_rows.iterrows()
    ):
        pairs.append({
            "pair_type": "cross_category_candidate",
            "resume_category": category,
            "job_title": job["Job Title"],
            "resume_text": resume["cleaned_resume_text"],
            "job_description": job["Job Description"],
            "expected_match": "",
            "reviewer_notes": "",
        })

random.Random(SEED).shuffle(pairs)

output = Path("data/evaluation/v2_eval_candidates.csv")
pd.DataFrame(pairs).to_csv(output, index=False)

print("Created:", output)
print("Total candidate pairs:", len(pairs))
print(pd.Series([p["pair_type"] for p in pairs]).value_counts())
print("Expected labels remain blank for manual review.")
