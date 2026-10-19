# Use a lightweight Python image.
FROM python:3.11-slim

# Set the working directory inside the container.
WORKDIR /app

# Copy the project dependencies file.
COPY requirements.txt .

# Install Python dependencies.
RUN pip install --no-cache-dir --default-timeout=600 --retries=10 torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --default-timeout=600 --retries=10 -r requirements.txt

# Download the NLTK resources required by the prediction pipeline.
RUN python -m nltk.downloader stopwords wordnet

# Copy the application source code.
COPY src ./src

# Copy the serialized ML artifacts.
COPY artifacts ./artifacts

# Expose the FastAPI port.
EXPOSE 8000

# Start the FastAPI application.
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]