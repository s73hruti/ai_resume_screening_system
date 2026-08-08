# This module will contain the ML prediction logic used by the FastAPI application.

# Import the regular expression module for text cleaning.
import re

# Import NLTK components used during resume preprocessing.
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Import cosine similarity to compare the resume vector
# with the job description vectors.
from sklearn.metrics.pairwise import cosine_similarity


# Initialize reusable NLP preprocessing components.
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Resume preprocessing
def preprocess_resume(resume):
    """
    Clean and normalize resume text before TF-IDF transformation.

    Parameters:
        resume (str): Raw resume text.

    Returns:
        str: Cleaned resume text.
    """

    # Convert the resume text to lowercase.
    resume = resume.lower()

    # Remove punctuation and special characters.
    resume = re.sub(r"[^\w\s]", "", resume)

    # Remove numeric tokens from the resume.
    resume = re.sub(r"\b\d+\b", "", resume)

    # Tokenize the resume into individual words.
    tokens = resume.split()

    # Remove stopwords and lemmatize each remaining word.
    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    # Combine the cleaned tokens back into a single string.
    cleaned_resume = " ".join(cleaned_tokens)

    return cleaned_resume

# Job recommendation
def recommend_jobs(similarity_scores, job_df, top_n=5):
    """
    Return the top unique job recommendations based on
    similarity scores.

    Parameters:
        similarity_scores: Cosine similarity scores between
                           the resume and job descriptions.
        job_df: DataFrame containing job information.
        top_n (int): Number of unique recommendations.

    Returns:
        list: Unique job recommendations.
    """

    # Retrieve all job indices sorted by similarity score
    # in descending order.
    sorted_indices = similarity_scores.argsort()[0][::-1]

    # Store job titles that have already been recommended.
    recommended_titles = set()

    # Store the final unique recommendations.
    top_recommendations = []

    # Iterate through jobs from highest to lowest similarity.
    for index in sorted_indices:

        # Retrieve the current job title.
        job_title = job_df.iloc[index]["Job Title"]

        # Skip the job if its title has already been recommended.
        if job_title in recommended_titles:
            continue

        # Store the job title to prevent duplicates.
        recommended_titles.add(job_title)

        # Store the recommendation details.
        top_recommendations.append({
            "job_title": job_title,
            "similarity_score": float(similarity_scores[0][index]),
            "job_description": job_df.iloc[index]["Job Description"]
        })

        # Stop once the required number of unique jobs is collected.
        if len(top_recommendations) == top_n:
            break

    return top_recommendations


# Complete prediction pipeline
def predict_jobs(resume, tfidf, job_tfidf_matrix, job_df, top_n=5):
    """
    Generate job recommendations for a given resume.

    Parameters:
        resume (str): Raw resume text.
        tfidf: Fitted TF-IDF vectorizer.
        job_tfidf_matrix: TF-IDF matrix of all job descriptions.
        job_df: DataFrame containing job information.
        top_n (int): Number of unique job recommendations.

    Returns:
        list: Top unique job recommendations.
    """

    # Preprocess the raw resume using the same
    # preprocessing steps used during model development.
    cleaned_resume = preprocess_resume(resume)

    # Transform the cleaned resume into its TF-IDF representation.
    resume_vector = tfidf.transform([cleaned_resume])

    # Calculate cosine similarity between the resume
    # and all available job descriptions.
    similarity_scores = cosine_similarity(
        resume_vector,
        job_tfidf_matrix
    )

    # Generate the top unique job recommendations.
    recommendations = recommend_jobs(
        similarity_scores,
        job_df,
        top_n=top_n
    )

    return recommendations