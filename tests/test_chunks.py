
from model.chunks import split_resume_into_chunks


def test_splits_long_flattened_resume_into_overlapping_chunks():
    resume = " ".join(f"skill{i}" for i in range(200))

    chunks = split_resume_into_chunks(resume)

    assert len(chunks) == 3
    assert len(chunks[0].split()) == 80
    assert len(chunks[1].split()) == 80
    assert len(chunks[2].split()) == 80

    # The first chunk ends with skill79; the second starts at skill60.
    assert chunks[0].split()[-1] == "skill79"
    assert chunks[1].split()[0] == "skill60"


def test_keeps_short_flattened_resume_as_one_chunk():
    resume = "Python SQL Java REST API"

    assert split_resume_into_chunks(resume) == [resume]


def test_empty_resume_returns_no_chunks():
    assert split_resume_into_chunks("") == []
    assert split_resume_into_chunks("   ") == []
