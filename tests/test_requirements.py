
"""Tests for job-description requirement extraction."""

import pytest

from model.requirements import extract_requirements


def test_extracts_requirements_from_uppercase_heading():
    description = """
    REQUIREMENTS
    Strong knowledge of Python programming.
    Experience with REST API development.
    """

    requirements = extract_requirements(description)

    assert any("Python programming" in item for item in requirements)
    assert any("REST API development" in item for item in requirements)


def test_extracts_content_after_inline_qualifications_heading():
    description = """
    Qualifications: Must have experience with Linux administration.
    JOB ROLES AND RESPONSIBILITIES
    Troubleshoot network connectivity issues.
    """

    requirements = extract_requirements(description)

    assert any("Linux administration" in item for item in requirements)
    assert not any(
        item.lower().startswith("qualifications:")
        for item in requirements
    )


def test_extracts_standalone_bullet_points():
    description = """
    Job Description
    - Experience with Python programming.
    - Knowledge of SQL databases.
    """

    requirements = extract_requirements(description)

    assert any("Python programming" in item for item in requirements)
    assert any("SQL databases" in item for item in requirements)


def test_does_not_return_company_footer_as_requirement():
    description = """
    Requirements
    Experience with automated testing frameworks.
    Knowledge of continuous integration tools.

    Equal Opportunity Employer. Visit www.example.com to learn more.
    """

    requirements = extract_requirements(description)

    assert any("automated testing frameworks" in item for item in requirements)
    assert not any(
        "Equal Opportunity Employer" in item
        for item in requirements
    )
    assert not any(
        "www.example.com" in item
        for item in requirements
    )


@pytest.mark.parametrize("description", ["", "   ", None])
def test_empty_description_returns_no_requirements(description):
    assert extract_requirements(description) == []


def test_unstructured_generic_prose_returns_no_requirements():
    description = "We are looking for a motivated candidate."

    assert extract_requirements(description) == []

def test_flattened_description_is_not_returned_as_one_requirement():
    description = (
        "job title java full stack developer job description "
        "4+ year web development using html javascript jquery "
        "mandatory skill java spring framework"
    )

    requirements = extract_requirements(description)

    # The full flattened description must not be returned as one item.
    assert requirements != [description]

    # Every extracted item should be a smaller, meaningful clause.
    assert all(len(item) < len(description) for item in requirements)

def test_extracts_key_skills_from_separate_lines():
    description = """
    Job Title: Java Full Stack Developer
    Key Skills:
    Angular
    Spring boot
    Mysql
    Job Types: Full-time
    Salary: Not specified
    """

    requirements = extract_requirements(description)

    assert "Angular" in requirements
    assert "Spring boot" in requirements
    assert "Mysql" in requirements
    assert not any("Salary" in item for item in requirements)


def test_extracts_requirements_after_candidate_skill_intro():
    description = """
    Candidates need to demonstrate experience in the following skillsets:
    - Proven technical expertise in enterprise software development.
    - Experience working with relational databases.
    """

    requirements = extract_requirements(description)

    assert any("enterprise software development" in item for item in requirements)
    assert any("relational databases" in item for item in requirements)


def test_extracts_explicit_experience_and_certification_fields():
    description = """
    Experience - 4 - 5 years.
    Certification - CCNA & CCNP (CISCO).
    Location - Andaman.
    Salary: Not specified.
    """

    requirements = extract_requirements(description)

    assert any("4 - 5 years" in item for item in requirements)
    assert any("CCNA & CCNP" in item for item in requirements)
    assert not any("Andaman" in item for item in requirements)
    assert not any("Salary" in item for item in requirements)


def test_extracts_required_skills_section():
    description = """
    Required Skills:
    Bachelor or Master of Engineering degree.
    Experience with Test Automation.
    """

    requirements = extract_requirements(description)

    assert any("Engineering degree" in item for item in requirements)
    assert any("Test Automation" in item for item in requirements)

def test_extracts_requirements_from_flattened_technical_description():
    description = (
        "job title java full stack developer job description "
        "4+ year web development using html html5 javascript jquery ajax "
        "2+ year web application development using java j2ee technology "
        "mvc spring framework good knowledge problem solution "
        "cross browser compatibility"
    )

    requirements = extract_requirements(description)

    assert requirements
    combined = " ".join(requirements).lower()
    assert "javascript" in combined
    assert "java" in combined
    assert "spring" in combined

def test_extracts_generic_software_experience_when_no_skills_are_listed():
    description = (
        "position description demonstrates up-to-date expertise software "
        "engineering applies development execution improvement action plan "
        "minimum qualification bachelor's degree computer science related "
        "field 4 year experience building scalable ecommerce application "
        "mobile software additional preferred qualification company summary "
        "walmart ecommerce team"
    )

    requirements = extract_requirements(description)
    combined = " ".join(requirements).lower()

    assert requirements
    assert "scalable ecommerce application" in combined
    assert "computer science" in combined


def test_does_not_extract_metadata_headings_as_requirements():
    description = """
    Knowledge, Skills, & Abilities:
    Experience with Python development.
    Other details
    Pay Type
    Salary information
    """

    requirements = extract_requirements(description)
    combined = " ".join(requirements).lower()

    assert any("python" in item.lower() for item in requirements)
    assert "other details" not in combined
    assert "pay type" not in combined


def test_rejects_incomplete_requirement_fragments():
    description = """
    Requirements:
    Knowledge of core and service platforms availability as per the
    Ensure the usage of all processes and tools as per guidelines
    Must have Knowledge of
    Experience with Python development.
    """

    requirements = extract_requirements(description)
    combined = " ".join(requirements).lower()

    assert any("python development" in item.lower() for item in requirements)
    assert "knowledge of core and service platforms availability as per the" not in combined
    assert "must have knowledge of" not in combined


def test_rejects_recruitment_spam_in_requirement_section():
    description = """
    Requirements:
    Experience with network administration.
    WHATSAPP YOUR CV ON (+9I-72-52-88-87-95) HR TEAM
    """

    requirements = extract_requirements(description)
    combined = " ".join(requirements).lower()

    assert any("network administration" in item.lower() for item in requirements)
    assert "whatsapp" not in combined
    assert "your cv" not in combined

def test_rejects_generic_responsibilities_disclaimer():
    description = """
    Responsibilities:
    Develop and maintain Java applications.
    *NOTE: Responsibilities of this role are not limited to the details above.
    """

    requirements = extract_requirements(description)
    combined = " ".join(requirements).lower()

    assert any("java applications" in item.lower() for item in requirements)
    assert "responsibilities of this role are not limited" not in combined


def test_stops_before_job_posting_metadata_and_company_overview():
    description = """
    Required Skills:
    Experience with test automation.
    Python and PowerShell knowledge.
    Category: Engineering and Technology
    Subcategory: Software Engineering
    Remote: No
    Posted Date: 2020-08-26

    VMware Company Overview:
    Our software helps organizations deliver digital innovation.
    """

    requirements = extract_requirements(description)
    combined = " ".join(requirements).lower()

    assert any("test automation" in item.lower() for item in requirements)
    assert any("python" in item.lower() for item in requirements)
    assert "category" not in combined
    assert "subcategory" not in combined
    assert "posted date" not in combined
    assert "company overview" not in combined
    assert "digital innovation" not in combined


def test_extracts_networking_requirements_from_opening_paragraph():
    description = (
        "A requirement for an experienced Network engineer. "
        "Candidate must have at least 4 years of experience in "
        "Networking (Router & Switches, Wireless) and CCNA & CCNP Certified. "
        "Immediate joining."
    )

    requirements = extract_requirements(description)
    combined = " ".join(requirements).lower()

    assert "network" in combined
    assert "ccna" in combined
    assert "ccnp" in combined


def test_extracts_talents_needed_for_success_without_company_overview():
    from model.requirements import extract_requirements

    description = """
    Your Responsibilities:
    - Participate in code reviews and sprint meetings.
    - Passionate about building an innovative culture.

    Talents Needed For Success:
    - Java (Core & EE, Spring Boot, Spring MVC)
    - Testing tools & frameworks (JUnit, Selenium)
    - Experience with Cloud Platforms (Docker, Kubernetes)

    Who We Are:
    We are a global company focused on innovation and financial services.
    """

    requirements = extract_requirements(description)
    combined = " ".join(requirements).lower()

    assert any("java" in req.lower() for req in requirements)
    assert any("selenium" in req.lower() for req in requirements)
    assert any("kubernetes" in req.lower() for req in requirements)
    assert "global company focused on innovation" not in combined


def test_rejects_generic_process_instructions_but_keeps_real_requirements():
    description = """
    Requirements:
    Ensure the usage of all processes and tools as per guidelines.
    Troubleshoot network connectivity issues.
    Experience with Python development.
    """

    requirements = extract_requirements(description)
    combined = " ".join(requirements).lower()

    # Generic process instructions should not count as core qualifications.
    assert "ensure the usage of all processes and tools" not in combined

    # Meaningful role-related requirements must still be retained.
    assert any(
        "network connectivity" in item.lower()
        for item in requirements
    )
    assert any(
        "python development" in item.lower()
        for item in requirements
    )

def test_rejects_requirement_ending_with_as_per_the_agreed():
    description = """
    Requirements:
    Knowledge of core and service platforms availability as per the agreed
    Experience with Python development.
    """

    requirements = extract_requirements(description)
    combined = " ".join(requirements).lower()

    # The incomplete fragment must not become a requirement.
    assert not any(
        "core and service platforms availability" in item.lower()
        for item in requirements
    )

    # A valid technical requirement must still be retained.
    assert any(
        "python development" in item.lower()
        for item in requirements
    )


def test_flattened_description_extracts_skills_and_minimum_experience():
    description = (
        "mandatory skill rdbms database development "
        "minimum work experience 7 - 10 year "
        "good knowledge sql oracle "
        "experience with data analysis and database design"
    )

    requirements = extract_requirements(description)
    combined = " ".join(requirements).lower()

    # Existing technical-skill extraction must continue working.
    assert any("oracle" in item.lower() for item in requirements)
    assert any("sql" in item.lower() for item in requirements)

    # The minimum experience must not be lost when skills are found.
    assert "7 - 10 year" in combined or "7-10 year" in combined
    assert any(
        "experience" in item.lower()
        and ("7" in item or "10" in item)
        for item in requirements
    )


def test_flattened_description_does_not_invent_experience_requirement():
    description = (
        "required skills python sql database development "
        "good communication and teamwork"
    )

    requirements = extract_requirements(description)
    combined = " ".join(requirements).lower()

    assert any("python" in item.lower() for item in requirements)
    assert any("sql" in item.lower() for item in requirements)
    assert not any(
        "years experience" in item.lower()
        or "minimum experience" in item.lower()
        for item in requirements
    )

def test_extracts_explicit_required_skill_after_job_metadata():
    description = """
    Job Title: Java Full Stack Developer
    Experience: 1yrs to 3yrs
    Key Skills:
    Angular
    Spring boot
    Mysql
    Job Types: Full-time, Walk-In
    Salary: ₹376,484.00 - ₹1,457,529.00 per year
    Schedule:
    Day shift
    Experience:
    total work: 1 year (Preferred)
    JSP: 1 year (Preferred)
    java: 1 year (Required)
    Education:
    Bachelor's (Preferred)
    """

    requirements = extract_requirements(description)
    combined = " ".join(requirements).lower()

    assert any("java" in item.lower() for item in requirements)
    assert "salary" not in combined
    assert "day shift" not in combined


def test_preferred_skills_are_not_extracted_as_requirements():
    description = (
        "Minimum of 2 - 6 years of programming experience required, "
        "preferably with ASP.NET and SQL."
    )

    requirements = extract_requirements(description)
    combined = " ".join(requirements).lower()

    assert not any(
        item.lower() in {"required skill: asp.net", "required skill: sql"}
        for item in requirements
    )
    assert "2 - 6 years" in combined or "2-6 years" in combined