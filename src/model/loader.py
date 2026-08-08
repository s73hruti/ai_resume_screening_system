# Import Path for constructing reliable file paths.
from pathlib import Path

# Import Joblib for loading serialized ML artifacts.
import joblib

# Import pandas for loading the job dataset.
import pandas as pd


# Determine the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Define the artifacts directory.
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def load_artifacts():
    """
    Load all ML artifacts required for job prediction.

    Returns:
        tuple: TF-IDF vectorizer, job TF-IDF matrix,
               and job dataset.
    """

    # Load the fitted TF-IDF vectorizer.
    tfidf = joblib.load(
        ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
    )

    # Load the TF-IDF matrix containing all job descriptions.
    job_tfidf_matrix = joblib.load(
        ARTIFACTS_DIR / "job_tfidf_matrix.joblib"
    )

    # Load the job dataset containing titles and descriptions.
    job_df = pd.read_pickle(
        ARTIFACTS_DIR / "job_data.pkl"
    )

    return tfidf, job_tfidf_matrix, job_df