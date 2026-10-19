from model.matching import verify_experience_requirement


def test_verifies_explicitly_sufficient_experience():
    result = verify_experience_requirement(
        requirement="Minimum Work Experience: 5 years",
        resume="Software engineer with 7 years of professional experience.",
    )

    assert result["status"] == "verified"
    assert result["required_years"] == 5
    assert result["reported_years"] == 7


def test_does_not_verify_when_resume_has_no_explicit_duration():
    result = verify_experience_requirement(
        requirement="Minimum Work Experience: 5 years",
        resume="Experienced software engineer skilled in Python.",
    )

    assert result["status"] == "not_verified"
    assert result["required_years"] == 5
    assert result["reported_years"] is None


def test_flags_conflicting_experience_ranges():
    result = verify_experience_requirement(
        requirement=(
            "Minimum Work Experience: 7 - 10 year; "
            "Minimum Experience Required: 3-5 year"
        ),
        resume="Database administrator with SQL experience.",
    )

    assert result["status"] == "ambiguous"
    assert result["required_years"] is None
    assert result["reported_years"] is None

def test_single_experience_range_is_not_ambiguous():
    result = verify_experience_requirement(
        requirement="Minimum Work Experience: 7 - 10 year",
        resume="Software engineer with 8 years of professional experience.",
    )

    assert result["status"] == "verified"
    assert result["required_years"] == 7
    assert result["reported_years"] == 8


def test_experience_below_minimum_is_not_verified():
    result = verify_experience_requirement(
        requirement="Minimum Work Experience: 5 years",
        resume="Software engineer with 3 years of professional experience.",
    )

    assert result["status"] == "not_verified"
    assert result["required_years"] == 5
    assert result["reported_years"] == 3


def test_conflicting_experience_ranges_are_ambiguous():
    result = verify_experience_requirement(
        requirement=(
            "Minimum Work Experience: 7 - 10 year; "
            "Minimum Experience Required: 3-5 year"
        ),
        resume="Database administrator with 8 years of experience.",
    )

    assert result["status"] == "ambiguous"
    assert result["required_years"] is None


def test_unrelated_numbers_do_not_count_as_experience():
    result = verify_experience_requirement(
        requirement="Minimum Work Experience: 5 years",
        resume="Completed 3 projects using SQL and Python.",
    )

    assert result["status"] == "not_verified"
    assert result["reported_years"] is None
