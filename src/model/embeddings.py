from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"

def load_embedding_model():
    """
    Load and return the Sentence Transformer embedding model.
    """
    return SentenceTransformer(MODEL_NAME)

def generate_embedding(model, text: str):
    """
    Convert input text into a numerical embedding.
    """
    return model.encode(text)

def calculate_similarity(embedding_a, embedding_b):
    """
    Calculate cosine similarity between two embeddings.
    """
    return cosine_similarity(
        [embedding_a],
        [embedding_b]
    )[0][0]
