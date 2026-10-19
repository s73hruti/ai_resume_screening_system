"""Explicit technical-skill overlap for resume-to-job matching."""

import re

from .requirements import TECHNICAL_SKILLS


# Match longer skill names first so compound terms are considered first.


SKILL_ALIASES = {
    "network administrator": "network administration",
}

SKILL_ALIASES.update({
    "reactjs": "react",
    "mssql": "sql server",
})

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
    """Extract clearly mandatory skills without promoting preferred skills."""
    if not isinstance(job_description, str) or not job_description.strip():
        return set()

    normalized = re.sub(r"\s+", " ", job_description.lower())
    mandatory = set()

    optional_patterns = [
        r"\bpreferably\b",
        r"\bpreferred\b",
        r"\bnice\s+to\s+have\b",
        r"\bdesirable\b",
        r"\boptional\b",
    ]

    mandatory_patterns = [
        r"\bmust[\s-]+have\b",
        r"\brequired\b",
        r"\bmandatory\b",
        r"\bessential\b",
        r"\bexcellent\s+knowledge\s+of\b",
        r"\bstrong\s+knowledge\s+of\b",
        r"\bgood\s+knowledge\s+of\b",
        r"\bsolid\s+knowledge\s+of\b",
        r"\bsound\s+knowledge\s+of\b",
    ]

    # Split sentences, but retain commas so a shared introductory phrase
    # can apply to a list of technologies.
    sentences = re.split(r"(?<=[.!?;])\s+", normalized)

    for sentence in sentences:
        occurrences = []
        for skill in SKILLS:
            escaped = re.escape(skill.lower()).replace(r"\ ", r"\s+")
            pattern = rf"(?<!\w){escaped}(?!\w)"

            for match in re.finditer(pattern, sentence):
                occurrences.append(
                    (match.start(), match.end(), skill)
                )

        # Prefer longer names when skill names overlap.
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

        for index, (start, end, skill) in enumerate(selected):
            previous_end = selected[index - 1][1] if index else 0
            next_start = (
                selected[index + 1][0]
                if index + 1 < len(selected)
                else len(sentence)
            )

            before = sentence[previous_end:start]
            local_before = sentence[max(0, start - 100):start]
            after = sentence[end:next_start]

            # Ignore skills locally marked as optional.
            if any(re.search(p, before) for p in optional_patterns):
                continue
            if any(re.search(p, after) for p in optional_patterns):
                continue

            explicit_mandatory = any(
                re.search(p, before)
                for p in mandatory_patterns[:4]
            ) or any(
                re.search(
                    p,
                    after,
                )
                for p in [
                    r"^\s*(?:is|are)\s+required\b",
                    r"^\s*(?:is|are)\s+mandatory\b",
                    r"^\s*(?:is|are)\s+essential\b",
                ]
            )

            # Carry mandatory wording forward to skills in the same list.
            if index > 0:
                earlier_text = sentence[:start]

                last_optional = max(
                    (
                        match.start()
                        for pattern in optional_patterns
                        for match in re.finditer(pattern, earlier_text)
                    ),
                    default=-1,
                )

                last_mandatory = max(
                    (
                        match.start()
                        for pattern in mandatory_patterns[:4]
                        for match in re.finditer(pattern, earlier_text)
                    ),
                    default=-1,
                )

                if last_mandatory > last_optional:
                    explicit_mandatory = True
                    
            core_knowledge = any(
                re.search(p, local_before)
                for p in mandatory_patterns[4:]
            )

            # A shared list introduction applies to the technologies
            # following it, provided the sentence isn't optional.
            list_requirement = bool(
                re.search(
                    r"\b(?:proficient|expert|experienced)\s+in"
                    r"\s+(?:the\s+following\s+)?"
                    r"(?:technologies|skills|tools)\s*:",
                    sentence[:start],
                )
            )

            if explicit_mandatory or core_knowledge or list_requirement:
                mandatory.add(SKILL_ALIASES.get(skill, skill))

    return mandatory

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
