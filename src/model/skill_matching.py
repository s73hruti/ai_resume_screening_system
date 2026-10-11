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
