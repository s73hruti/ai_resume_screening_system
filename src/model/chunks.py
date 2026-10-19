
import re


def split_resume_into_chunks(
    resume: str,
    max_words: int = 80,
    overlap_words: int = 20,
) -> list[str]:
    """
    Split resume text into manageable chunks.

    First split by sentence boundaries. If a long resume has no
    useful sentence boundaries, fall back to overlapping word windows.
    """

    if not isinstance(resume, str) or not resume.strip():
        return []

    if max_words <= 0:
        raise ValueError("max_words must be greater than zero")

    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError(
            "overlap_words must be non-negative and less than max_words"
        )

    # Try sentence-level splitting first.
    chunks = [
        chunk.strip()
        for chunk in re.split(r"(?<=[.!?])\s+", resume.strip())
        if chunk.strip()
    ]

    # Keep sentence chunks when the text already has useful boundaries.
    if len(chunks) > 1:
        return chunks

    words = resume.split()

    # Short resumes do not need additional splitting.
    if len(words) <= max_words:
        return [resume.strip()]

    # Flattened resumes need overlapping word windows.
    step = max_words - overlap_words
    chunks = []

    for start in range(0, len(words), step):
        window = words[start : start + max_words]

        if not window:
            break

        chunks.append(" ".join(window))

        if start + max_words >= len(words):
            break

    return chunks
