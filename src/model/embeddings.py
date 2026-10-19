#embeddings.py
#→ text → embeddings → similarity

# This module will contain the semantic embedding and similarity
# logic used by the AI Resume Screening System.

# Import Sentence Transformer to convert text into
# numerical embeddings that capture semantic meaning.
from sentence_transformers import SentenceTransformer

# Import cosine similarity to measure the similarity
# between two numerical embeddings.
from sklearn.metrics.pairwise import cosine_similarity


# Define the pre-trained Sentence Transformer model
# that will be used to generate text embeddings.
MODEL_NAME = "all-MiniLM-L6-v2"


# Load the embedding model
def load_embedding_model():
    """
    Load and return the Sentence Transformer embedding model.

    Returns:
        SentenceTransformer: Loaded embedding model.
    """

    # Load the pre-trained Sentence Transformer model
    # using the model name defined above.
    return SentenceTransformer(MODEL_NAME)


# Generate text embedding
def generate_embedding(model, text: str):
    """
    Convert input text into a numerical embedding.

    Parameters:
        model: Loaded Sentence Transformer model.
        text (str): Text that needs to be converted into an embedding.

    Returns:
        array: Numerical embedding representing the meaning of the text.
    """

    # Convert the input text into a numerical embedding
    # using the loaded Sentence Transformer model.
    return model.encode(text)


# Calculate cosine similarity
def calculate_similarity(embedding_a, embedding_b):
    """
    Calculate cosine similarity between two embeddings.

    Parameters:
        embedding_a: Numerical embedding of the first text.
        embedding_b: Numerical embedding of the second text.

    Returns:
        float: Cosine similarity score between the two embeddings.
    """

    # Calculate the cosine similarity between the two embeddings.
    # The embeddings are placed inside lists because
    # cosine_similarity expects 2D input.
    return cosine_similarity(
        [embedding_a],
        [embedding_b]
    )[0][0]


# Calculate requirement-level semantic similarity
def calculate_requirement_similarity(model, requirement, resume):
    """
    Calculate semantic similarity between one job requirement
    and the resume.

    Parameters:
        model: Loaded Sentence Transformer model.
        requirement (str): One job requirement.
        resume (str): Resume text.

    Returns:
        float: Semantic similarity score.
    """

    # Convert the job requirement into a numerical embedding
    # that represents its semantic meaning.
    requirement_embedding = generate_embedding(
        model,
        requirement
    )

    # Convert the resume into a numerical embedding
    # that represents its overall semantic meaning.
    resume_embedding = generate_embedding(
        model,
        resume
    )

    # Compare the requirement embedding with the resume embedding
    # using cosine similarity.
    similarity_score = calculate_similarity(
        requirement_embedding,
        resume_embedding
    )

    # Convert the similarity score into a standard Python float
    # so it can be easily returned by the application or API.
    return float(similarity_score)