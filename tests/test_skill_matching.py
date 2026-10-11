from model.skill_matching import (
    calculate_skill_overlap,
    extract_explicit_skills,
)


def test_extracts_explicit_technical_skills():
    text = "Python, Java, Spring Boot, and REST API development."
    skills = extract_explicit_skills(text)

    assert "python" in skills
    assert "java" in skills
    assert "spring boot" in skills
    assert "rest api" in skills


def test_skill_matching_is_case_insensitive():
    skills = extract_explicit_skills("PYTHON and JavaScript")
    assert "python" in skills
    assert "javascript" in skills


def test_short_skill_names_do_not_match_inside_words():
    skills = extract_explicit_skills("I can write and read code.")
    assert "c" not in skills
    assert "r" not in skills


def test_calculates_overlap_and_missing_skills():
    result = calculate_skill_overlap(
        resume="Python and Java development with Spring Boot",
        job_description="Required: Python, Java, React, and Spring Boot",
    )

    assert result["matched_skills"] == [
        "java",
        "python",
        "spring boot",
    ]
    assert result["missing_skills"] == ["react"]
    assert result["skill_overlap"] == 0.75


def test_empty_inputs_return_zero_overlap():
    result = calculate_skill_overlap("", "Python and Java")

    assert result["matched_skills"] == []
    assert result["skill_overlap"] == 0.0
