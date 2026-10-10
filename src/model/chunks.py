import re


def split_resume_into_chunks(resume: str) -> list[str]:
    """
    Split resume text into meaningful sentence-level chunks.

    Parameters:
        resume (str): Raw resume text.

    Returns:
        list[str]: Clean, non-empty resume chunks.
    """

    # Split the resume whenever we encounter sentence-ending
    # punctuation followed by whitespace.
    chunks = re.split(r"(?<=[.!?])\s+", resume.strip())

    # Remove empty chunks and surrounding whitespace.
    chunks = [
        chunk.strip()
        for chunk in chunks
        if chunk.strip() #Keep this chunk only if chunk.strip() is not empty.
    ]

    return chunks