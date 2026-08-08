# Import FastAPI to create the REST API application.
from fastapi import FastAPI

# Import BaseModel to define the API request structure.
from pydantic import BaseModel, Field

# Import the artifact loader.
from src.model.loader import load_artifacts

# Import the reusable prediction pipeline.
from src.model.predictor import predict_jobs

# Create the FastAPI application instance.
app = FastAPI(
    title="AI Resume Screening API",
    description="API for recommending jobs based on resume similarity.",
    version="1.0.0"
)

# Load the ML artifacts when the API application starts.
tfidf, job_tfidf_matrix, job_df = load_artifacts()

# Define the structure of the resume prediction request.
class ResumeRequest(BaseModel):
    """
    Request schema for resume-based job prediction.
    """

    # Resume text provided by the API user.
    resume: str = Field(
        ...,
        min_length=10,
        description="Resume text used to generate job recommendations."
    )

# Define the root endpoint to verify that the API is running.
@app.get("/")
def home():
    """
    Return a welcome message when the root endpoint is accessed.
    """

    return {
        "message": "AI Resume Screening API is running"
    }

# Define the prediction endpoint.
@app.post("/predict")
def predict(request: ResumeRequest):
    """
    Generate job recommendations for the submitted resume.
    """

    # Generate job recommendations using the reusable
    # prediction pipeline.
    recommendations = predict_jobs(
        resume=request.resume,
        tfidf=tfidf,
        job_tfidf_matrix=job_tfidf_matrix,
        job_df=job_df,
        top_n=5
    )

    # Return the recommendations as a JSON response.
    return {
        "recommendations": recommendations
    }