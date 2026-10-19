from model.skill_matching import (
    calculate_skill_overlap,
    extract_explicit_skills,
    extract_must_have_skills,
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


def test_extracts_explicit_must_have_skills():
    from model.skill_matching import extract_must_have_skills

    result = extract_must_have_skills(
        "Must have PHP. PostgreSQL is required."
    )

    assert result == {"php", "postgresql"}


def test_excludes_preferred_skills_from_must_haves():
    from model.skill_matching import extract_must_have_skills

    result = extract_must_have_skills(
        "PHP is required. JavaScript is preferred. "
        "Docker is nice to have."
    )

    assert result == {"php"}


def test_does_not_infer_must_have_from_skill_mention_alone():
    from model.skill_matching import extract_must_have_skills

    result = extract_must_have_skills(
        "Experience with PHP and PostgreSQL is desirable."
    )

    assert result == set()


def test_missing_must_have_skill_is_detected():
    from model.skill_matching import check_must_have_skills

    result = check_must_have_skills(
        resume="Java and JavaScript development",
        job_description="PHP is required",
    )

    assert result["must_have_skills"] == ["php"]
    assert result["missing_must_have_skills"] == ["php"]
    assert result["must_haves_satisfied"] is False


def test_separates_required_and_preferred_skills_in_same_sentence():
    from model.skill_matching import extract_must_have_skills

    result = extract_must_have_skills(
        "PHP is required, while JavaScript is preferred."
    )

    assert result == {"php"}


def test_does_not_treat_desirable_skills_as_mandatory():
    from model.skill_matching import extract_must_have_skills

    result = extract_must_have_skills(
        "Experience with PHP and PostgreSQL is desirable."
    )

    assert result == set()


def test_detects_required_label_before_skill():
    from model.skill_matching import extract_must_have_skills

    result = extract_must_have_skills(
        "Required: PHP. Preferred: JavaScript."
    )

    assert result == {"php"}


def test_detects_must_have_wording_before_skill():
    from model.skill_matching import extract_must_have_skills

    result = extract_must_have_skills(
        "Must have PostgreSQL. Docker is optional."
    )

    assert result == {"postgresql"}


def test_mandatory_marker_does_not_apply_to_unrelated_skills():
    from model.skill_matching import extract_must_have_skills

    result = extract_must_have_skills(
        "Mandatory skill: Node.js. "
        "Other technologies include HTML, JavaScript, jQuery, and CSS."
    )

    assert result == {"node.js"}


def test_detects_explicit_sql_admin_requirement():
    from model.skill_matching import extract_must_have_skills

    result = extract_must_have_skills(
        "Mandatory skill: SQL admin. "
        "Desirable skill: English."
    )

    assert result == {"sql"}


def test_does_not_promote_preferred_network_experience_to_mandatory():
    from model.skill_matching import extract_must_have_skills

    result = extract_must_have_skills(
        "Network Administration: 4 years (Preferred)."
    )

    assert result == set()

def test_extracts_skills_from_concatenated_htmlcss_token():
    skills = extract_explicit_skills(
        "JavaScript HTMLCSS PHP AJAX MySQL"
    )

    assert "html" in skills
    assert "css" in skills
    assert "php" in skills
    assert "javascript" in skills
    assert "mysql" in skills

def test_extracts_core_requirements_from_full_stack_php_job():
    job_description = (
        "Excellent knowledge of object-oriented PHP. "
        "Required: 2+ years of professional experience. "
        "Strong knowledge of JavaScript and HTML/CSS. "
        "Strong knowledge of relational databases such as MySQL."
    )

    must_haves = extract_must_have_skills(job_description)

    assert "php" in must_haves
    assert "javascript" in must_haves
    assert "html" in must_haves
    assert "mysql" in must_haves

def test_detects_all_skills_in_shared_mandatory_phrase():
    result = extract_must_have_skills(
        "Must have Python and SQL."
    )

    assert result == {"python", "sql"}


def test_detects_php_in_excellent_knowledge_phrase():
    assert extract_must_have_skills(
        "Excellent knowledge of object-oriented PHP."
    ) == {"php"}


def test_detects_mandatory_technology_list():
    result = extract_must_have_skills(
        "Proficient in the following technologies: "
        "C#, .NET, HTML5, CSS3, MySQL, ReactJS."
    )

    assert {"c#", ".net", "html5", "css3", "mysql"}.issubset(result)


def test_preferred_sentence_does_not_create_mandatory_skills():
    assert extract_must_have_skills(
        "PHP and MySQL are preferred technologies."
    ) == set()


def test_missing_mandatory_skill_blocks_eligibility():
    from model.skill_matching import check_must_have_skills

    result = check_must_have_skills(
        resume="JavaScript and HTML development experience.",
        job_description="Excellent knowledge of PHP is required.",
    )

    assert result["must_haves_satisfied"] is False
    assert "php" in result["missing_must_have_skills"]


def test_no_detected_mandatory_skills_does_not_block_eligibility():
    from model.skill_matching import check_must_have_skills

    result = check_must_have_skills(
        resume="Python development experience.",
        job_description="Python experience is preferred.",
    )

    assert result["must_have_skills"] == []
    assert result["must_haves_satisfied"] is True


def test_preferably_with_multiple_skills_does_not_make_them_mandatory():
    result = extract_must_have_skills(
        "2 - 6 years of programming experience required, "
        "preferably with ASP.NET and SQL."
    )

    assert result == set()

