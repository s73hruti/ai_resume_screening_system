# Import FastAPI to create the REST API application.
from fastapi import FastAPI


# Create the FastAPI application instance.
app = FastAPI(
    title="AI Resume Screening API",
    description="API for recommending jobs based on resume similarity.",
    version="1.0.0"
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