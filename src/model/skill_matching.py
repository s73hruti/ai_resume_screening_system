"""Explicit technical-skill overlap for resume-to-job matching."""

import re

from .requirements import TECHNICAL_SKILLS


# Match longer skill names first so compound terms are considered first.


SKILL_ALIASES = {
    "network administrator": "network administration",
}

SKILLS = sorted(
    TECHNICAL_SKILLS | set(SKILL_ALIASES),
    key=len,
    reverse=True,
)




def extract_explicit_skills(text: str) -> set[str]:
    """Return known technical skills, preferring longer overlapping matches."""
    if not isinstance(text, str) or not text.strip():
        return set()

    normalized = re.sub(r"\s+", " ", text.lower())
    normalized = re.sub(r"\bhtmlcss\b", "html css", normalized)
    matches = []

    # Collect every occurrence of every known skill.
    for skill in SKILLS:
        escaped_skill = re.escape(skill.lower()).replace(r"\ ", r"\s+")
        pattern = rf"(?<!\w){escaped_skill}(?!\w)"

        for match in re.finditer(pattern, normalized):
            matches.append((match.start(), match.end(), skill))

    # Prefer longer matches when their text spans overlap.
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))

    selected = []
    for start, end, skill in matches:
        if any(
            start < selected_end and selected_start < end
            for selected_start, selected_end, _ in selected
        ):
            continue

        selected.append((start, end, skill))

    return {
    SKILL_ALIASES.get(skill, skill)
    for _, _, skill in selected
}



def calculate_skill_overlap(
    resume: str,
    job_description: str,
) -> dict:
    """Compare explicitly mentioned resume skills with job skills."""
    resume_skills = extract_explicit_skills(resume)
    job_skills = extract_explicit_skills(job_description)

    matched_skills = sorted(resume_skills & job_skills)
    missing_skills = sorted(job_skills - resume_skills)

    overlap = (
        len(matched_skills) / len(job_skills)
        if job_skills
        else 0.0
    )

    return {
        "resume_skills": sorted(resume_skills),
        "job_skills": sorted(job_skills),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_overlap": round(overlap, 4),
    }

def extract_must_have_skills(job_description: str) -> set[str]:
    """Extract explicitly mandatory skills without promoting nearby skills."""
    if not isinstance(job_description, str) or not job_description.strip():
        return set()

    normalized = re.sub(r"\s+", " ", job_description.lower())
    must_have_skills = set()

    before_patterns = [
        r"\bmust[\s-]+have\b",
        r"\brequired\b",
        r"\bmandatory\b",
        r"\bessential\b",
    ]

    after_patterns = [
        r"\bis\s+required\b",
        r"\bare\s+required\b",
        r"\bis\s+mandatory\b",
        r"\bare\s+mandatory\b",
        r"\bis\s+essential\b",
        r"\bare\s+essential\b",
    ]

    core_requirement_patterns = [
        r"\bexcellent\s+knowledge\s+of\b",
        r"\bstrong\s+knowledge\s+of\b",
        r"\bgood\s+knowledge\s+of\b",
        r"\bsolid\s+knowledge\s+of\b",
        r"\bsound\s+knowledge\s+of\b",
    ]

    optional_patterns = [
        r"\bpreferred\b",
        r"\bnice\s+to\s+have\b",
        r"\bdesirable\b",
        r"\boptional\b",
    ]

    # Split at sentence and contrasting-clause boundaries.
    sentences = re.split(r"(?<=[.!?;])\s+", normalized)

    for sentence in sentences:
        clauses = re.split(
            r"\b(?:while|whereas|but|however|although)\b|,",
            sentence,
        )

        for clause in clauses:
            # Find skill occurrences, preferring longer names at the same
            # location (e.g. Node.js instead of both Node.js and Node).
            occurrences = []

            for skill in SKILLS:
                escaped = re.escape(skill.lower()).replace(r"\ ", r"\s+")
                pattern = rf"(?<!\w){escaped}(?!\w)"

                for match in re.finditer(pattern, clause):
                    occurrences.append(
                        (match.start(), match.end(), skill)
                    )

            occurrences.sort(
                key=lambda item: (item[0], -(item[1] - item[0]))
            )

            selected = []
            for start, end, skill in occurrences:
                if any(
                    start < selected_end and selected_start < end
                    for selected_start, selected_end, _ in selected
                ):
                    continue
                selected.append((start, end, skill))

            # Evaluate each skill using only its local text, bounded by
            # neighboring skill mentions in the same clause.
            for index, (start, end, skill) in enumerate(selected):
                previous_end = (
                    selected[index - 1][1] if index > 0 else 0
                )
                next_start = (
                    selected[index + 1][0]
                    if index + 1 < len(selected)
                    else len(clause)
                )

                before = clause[:start]
                after = clause[end:next_start]

                if any(
                    re.search(pattern, before)
                    for pattern in optional_patterns
                ):
                    continue

                if any(
                    re.search(pattern, after)
                    for pattern in optional_patterns
                ):
                    continue


                is_mandatory = (
                    any(
                        re.search(pattern, before)
                        for pattern in before_patterns
                    )
                    or any(
                        re.search(pattern, after)
                        for pattern in after_patterns
                    )
                    or any(
                        re.search(pattern, clause[:start])
                        for pattern in core_requirement_patterns
                    )
                )


                if is_mandatory:
                    must_have_skills.add(
                        SKILL_ALIASES.get(skill, skill)
                    )

    return must_have_skills




def check_must_have_skills(
    resume: str,
    job_description: str,
) -> dict:
    """Report whether the resume explicitly mentions every mandatory skill."""
    must_have_skills = extract_must_have_skills(job_description)
    resume_skills = extract_explicit_skills(resume)

    missing_skills = sorted(must_have_skills - resume_skills)

    return {
        "must_have_skills": sorted(must_have_skills),
        "missing_must_have_skills": missing_skills,
        "must_haves_satisfied": len(missing_skills) == 0,
    }
