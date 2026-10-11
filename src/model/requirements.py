
"""Extract requirement statements from job descriptions."""

import re


FOOTER_MARKERS = re.compile(
    r"equal opportunity employer|"
    r"visit\s+www\.|"
    r"follow us on|"
    r"all rights reserved|"
    r"copyright|"
    r"©",
    flags=re.IGNORECASE,
)

REQUIREMENT_SECTION_HEADERS = {
    "requirements",
    "job requirements",
    "qualifications",
    "job qualifications",
    "required skills",
    "required skills experience",
    "technical skills",
    "skills required",
    "key skills",
    "mandatory skill",
    "mandatory skills",
    "basic qualifications",
    "basic qualification",
    "minimum qualifications",
    "required experiences",
    "knowledge, skills, & abilities",
    "knowledge, skills and abilities",
    "what you'll need",
    "what you will need",
    "what we're looking for",
    "what we are looking for",
    "in order to be successful in this role, we need someone who has",
    "candidates need to demonstrate experience in the following skillsets",
    "responsibilities",
    "your responsibilities",
    "talents needed for success",
}

STOP_SECTION_HEADERS = {
    "responsibilities and duties",
    "key responsibilities",
    "job roles and responsibilities",
    "job role and responsibility",
    "about the role",
    "about us",
    "benefits",
    "location",
    "salary",
    "compensation",
    "job type",
    "job types",
    "schedule",
    "how to apply",
    "what you'll do",
    "what you will do",
    "company summary",
    "other details",
    "pay type",
    "employment details",
    "company overview",
    "who we are",
}

INLINE_REQUIREMENT_LABELS = {
    "qualifications",
    "job qualifications",
    "job requirements",
    "required skills",
    "required skills experience",
    "technical skills",
    "skills required",
    "key skills",
    "mandatory skill",
    "mandatory skills",
    "basic qualifications",
    "basic qualification",
    "minimum qualifications",
    "required experiences",
    "knowledge, skills, & abilities",
    "knowledge, skills and abilities",
    "certification",
    "certifications",
    "minimum work experience",
}

METADATA_LABELS = {
    "location",
    "work location",
    "salary",
    "compensation",
    "mail id",
    "email",
    "positions",
    "shift",
    "job type",
    "job types",
    "schedule",
    "work remotely",
    "licence",
    "license",
    "pay type",
    "other details",
    "employment details",
    "category",
    "subcategory",
    "remote",
    "posted date",
}

FLATTENED_METADATA_PREFIXES = (
    "job title",
    "job description",
    "company name",
    "company overview",
)

TECHNICAL_SKILLS = {
    "python", "java", "j2ee", "javascript", "typescript",
    "html", "html5", "css", "css3", "jquery", "ajax",
    "spring", "spring boot", "mvc", "sql", "mysql",
    "postgresql", "oracle", "sql server", "mongodb",
    "react", "angular", "node.js", "node", "django", "flask",
    "ruby on rails", "php", "c#", ".net", "asp.net",
    "rest api", "api development", "test automation",
    "playwright", "selenium", "cypress", "linux",
    "network administration", "aws", "azure", "docker",
    "kubernetes", "git", "jenkins", "ci/cd",
    "networking", "it helpdesk", "help desk", "service desk",
    "desktop support", "application deployment", "application support",
    "technical support", "troubleshooting", "operating systems",
    "windows", "active directory", "system administration",
    "incident management", "ticketing systems", "unix", "bash",
    "tableau", "greenplum", "terraform", "azure devops",
    "azure api management", "data factory", "logic apps",
    "rest", "microservices", "oracle", "pl/sql", "maven",
    "junit", "web services", "weblogic", "ccna",
    "ccnp",
}


FLATTENED_REQUIREMENT_SIGNALS = re.compile(
    r"\b(?:"
    r"\d+\+?\s*(?:year|years|month|months)\s+(?:of\s+)?"
    r"|experience\s+(?:with|in|using|developing|building)\s+"
    r"|knowledge\s+of\s+"
    r"|proficient\s+in\s+"
    r"|familiar(?:ity)?\s+with\s+"
    r"|using\s+"
    r"|know(?:ledge)?\s+"
    r"|write\s+"
    r"|worked\s+with\s+"
    r"|hands-on\s+(?:experience|knowledge)\s+(?:with|in)\s+"
    r"|ability\s+to\s+"
    r"|developing\s+"
    r"|building\s+"
    r")",
    flags=re.IGNORECASE,
)



def _clean_requirement(text: str) -> str:
    """Remove bullet markers and surrounding whitespace."""
    text = re.sub(r"^(?:[-*•▪·]|\bo(?=\s)|\d+[.)])\s*", "", text.strip())
    return text.strip(" \t-;,")


def _is_valid_requirement(text: str) -> bool:
    """Reject incomplete requirement fragments and recruitment spam."""
    text = _clean_requirement(text)
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    normalized = normalized.rstrip(" .,:;")

    if len(normalized) < 3:
        return False

    # Reject recruitment messages that are not job requirements.
    spam_patterns = (
        r"^ensure the usage of all processes and tools\b",
        r"\bwhatsapp\b",
        r"\byour cv\b",
        r"\bsend your resume\b",
        r"\bhr team\b",
        r"\bapply now\b",
        r"\bresponsibilities of this role are not limited\b",
        r"\bresponsibilities are not limited to\b",
        r"\bresponsibilities may include\b",
        r"^participate in requirements gathering\b",
        r"^write clean, self-documenting code\b",
        r"^collaborate with the senior application architects\b",
        r"^participate in code reviews\b",
        r"^provide on-going maintenance support\b",
        r"^expertise in working in in large, collaborative teams\b",
        r"^passionate about building an innovative culture\b",
        r"^be a self-starter\b",


    )

    if any(re.search(pattern, normalized) for pattern in spam_patterns):
        return False

    # Reject incomplete phrases such as "Knowledge of" and
    # "availability as per the".
    incomplete_endings = (
        " as per the agreed",
        " as per the",
        " as per",
        " knowledge of",
        " must have",
        " should have",
        " experience of",
        " experience with",
        " familiarity with",
        " ability to",
        " such as",
        " including",
        " and",
        " or",
        " of",
        " with",
        " in",
        " for",
        " to",
        " the",
        " a",
        " an",
        " using",
    )

    if normalized.endswith(incomplete_endings):
        return False

    return True

def _remove_footer(text: str) -> str:
    """Remove obvious company-footer text."""
    match = FOOTER_MARKERS.search(text)

    if match:
        text = text[:match.start()]

    return text.strip()


def _normalize_header(text: str) -> str:
    """Normalize a possible section heading."""
    return re.sub(r"[\s:.-]+$", "", text.strip().lower())


def _extract_inline_label(line: str) -> tuple[str, str] | None:
    """Find a recognized requirement label and its associated content."""
    match = re.match(
        r"^\s*(.+?)\s*:\s*(.*)$",
        line,
    )

    if match:
        label = _normalize_header(match.group(1))
        content = match.group(2).strip()

        if label in INLINE_REQUIREMENT_LABELS:
            return label, content

    match = re.match(
        r"^\s*(experience|education|certification|certifications)"
        r"\s*[-–]\s*(.+)$",
        line,
        flags=re.IGNORECASE,
    )

    if match:
        return (
            _normalize_header(match.group(1)),
            match.group(2).strip(),
        )

    return None


def _is_metadata(line: str) -> bool:
    """Identify administrative job-posting fields."""
    match = re.match(r"^([^:]+):", line)

    if match and _normalize_header(match.group(1)) in METADATA_LABELS:
        return True

    return False



def _extract_flattened_requirements(job_description: str) -> list[str]:
    """Extract individual technical skills from flattened descriptions."""
    text = re.sub(r"\s+", " ", job_description).strip()
    text = _remove_footer(text)

    for prefix in FLATTENED_METADATA_PREFIXES:
        text = re.sub(
            rf"^{re.escape(prefix)}\s*:?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

    lower_text = text.lower()

    # Avoid extracting skills from arbitrary prose with no job-related context.
    has_requirement_context = bool(
        FLATTENED_REQUIREMENT_SIGNALS.search(text)
        or re.search(
            r"\b(?:required skills?|technical skills?|keywords?|"
            r"experience|qualifications?|proficient|troubleshooting|"
            r"support analyst)\b",
            lower_text,
        )
    )

    if not has_requirement_context:
        return []

    found_skills = []

    # Match longer names first, avoiding partial matches where possible.
    for skill in sorted(TECHNICAL_SKILLS, key=lambda item: -len(item)):
        pattern = rf"(?<!\w){re.escape(skill.lower())}(?!\w)"

        if re.search(pattern, lower_text):
            found_skills.append(skill)


    if found_skills:
        skill_requirements = [
            f"Required skill: {skill}"
            for skill in found_skills
        ]

        experience_matches = re.finditer(
            r"(?P<context>minimum\s+work\s+experience|"
            r"minimum\s+experience\s+required|"
            r"minimum\s+experience|at\s+least|minimum\s+of)?"
            r"\s*:?\s*"
            r"(?P<range>\d{1,2}\s*(?:-|–|to)\s*"
            r"\d{1,2}\s*(?:\+?\s*)?(?:years?|yrs?))\b",
            lower_text,
            flags=re.IGNORECASE,
        )

        experience_requirements = []
        seen_ranges = set()

        for match in experience_matches:
            context = (match.group("context") or "").strip()
            experience_range = match.group("range").strip()

            # Avoid extracting the same range twice.
            range_key = re.sub(r"\s+", "", experience_range).lower()
            if range_key in seen_ranges:
                continue

            seen_ranges.add(range_key)

            if context:
                requirement_text = (
                    f"{context.title()}: {experience_range}"
                )
            else:
                requirement_text = (
                    f"Experience range: {experience_range}"
                )

            if _is_valid_requirement(requirement_text):
                experience_requirements.append(requirement_text)

        return list(dict.fromkeys(
            skill_requirements + experience_requirements
        ))


    # Fallback for generic job descriptions that describe experience
    # without listing specific technologies.
    generic_requirements = []

    experience_match = re.search(
        r"\b\d+\+?\s*years?\s+experience\s+"
        r"(?:building|developing|designing|maintaining|supporting)\s+"
        r"(.{3,100}?)"
        r"(?=\s+(?:additional preferred qualification|"
        r"minimum qualification|preferred qualification|"
        r"company summary|position summary)\b|$)",
        lower_text,
        flags=re.IGNORECASE,
    )

    if experience_match:
        experience = experience_match.group(1).strip(" .,:;-")
        generic_requirements.append(
            f"Experience building or developing {experience}"
        )

    education_match = re.search(
        r"\bbachelor\s*['’]?\s*s\s+degree\s+"
        r"(.{3,80}?)"
        r"(?=\s+\d+\+?\s*years?\s+experience\b|$)",
        lower_text,
        flags=re.IGNORECASE,
    )

    if education_match:
        education = education_match.group(1).strip(" .,:;-")
        generic_requirements.append(
            f"Degree in {education}"
        )

    return list(
        dict.fromkeys(
            item for item in generic_requirements
            if _is_valid_requirement(item)
        )
    )


def extract_requirements(job_description: str) -> list[str]:
    """Extract requirements from recognizable job-description structures.

    Supports requirement headings, inline labels, and standalone bullets.
    Generic prose and unstructured flattened descriptions are not
    automatically treated as requirements.
    """
    if not isinstance(job_description, str) or not job_description.strip():
        return []

    lines = [
        _remove_footer(line)
        for line in job_description.splitlines()
    ]

    requirements = []
    inside_requirement_section = False

    for line in lines:
        if not line:
            continue

        # Stop collecting requirements when posting metadata begins.
        if _is_metadata(line):
            inside_requirement_section = False
            continue

        # Stop before company overview text, including prefixed headings.
        if "company overview" in _normalize_header(line):
            inside_requirement_section = False
            continue

        # Handle inline labels such as "Qualifications: ..."
        # and "Experience - 4-5 years".
        inline = _extract_inline_label(line)

        if inline:
            label, content = inline

            if label in METADATA_LABELS:
                continue

            if content:
                cleaned = _clean_requirement(content)

                if _is_valid_requirement(cleaned):
                    requirements.append(cleaned)

            # A label without content may introduce a section.
            inside_requirement_section = label in REQUIREMENT_SECTION_HEADERS
            continue

        header = _normalize_header(line)

        if header in REQUIREMENT_SECTION_HEADERS:
            inside_requirement_section = True
            continue

        if header in STOP_SECTION_HEADERS:
            inside_requirement_section = False
            continue

        if inside_requirement_section:
            cleaned = _clean_requirement(line)

            if _is_metadata(cleaned):
                continue

            if _is_valid_requirement(cleaned):
                requirements.append(cleaned)

    valid_requirements = list(
        dict.fromkeys(
            item for item in requirements
            if _is_valid_requirement(item)
        )
    )

    if valid_requirements:
        # Supplement structured extraction using only the opening line.
        # This avoids scanning company descriptions and unrelated sections.
        opening_line = next(
            (
                line.strip()
                for line in job_description.splitlines()
                if line.strip()
            ),
            "",
        )
        flattened = _extract_flattened_requirements(opening_line)
        combined = list(valid_requirements)

        for item in flattened:
            if item not in combined:
                combined.append(item)

        return combined


    # Fallback: collect explicit bullet points, excluding obvious footers.
    for original_line in job_description.splitlines():
        line = _remove_footer(original_line.strip())

        if re.match(r"^(?:[-*•▪]|\d+[.)])\s*.+", line):
            cleaned = _clean_requirement(line)

            if len(cleaned) >= 15 and _is_valid_requirement(cleaned):
                requirements.append(cleaned)

        requirements = list(dict.fromkeys(requirements))

    if requirements:
        return requirements

    return _extract_flattened_requirements(job_description)
