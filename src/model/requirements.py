# requirements.py
# → JD → requirements

# This module will contain the logic used to extract
# individual job requirements from a job description.

# Import the regular expression module for matching
# different bullet-point formats.
import re


# Define the common section headers that can contain
# job requirements in a job description.
REQUIREMENT_SECTION_HEADERS = [
    "requirements",
    "qualifications",
    "required skills",
    "technical skills",
    "skills required",
    "must have",
]


# Extract requirements from a job description
def extract_requirements(job_description: str) -> list[str]:
    """
    Extract bullet-point requirements from a job description.

    Parameters:
        job_description (str): Complete job description text.

    Returns:
        list[str]: List of individual job requirements.
    """

    # Split the complete job description into individual lines
    # so that each line can be checked separately.
    lines = job_description.splitlines()

    # Store the requirements extracted from the job description.
    requirements = []

    # Keep track of whether we are currently inside
    # a requirement section.
    inside_requirement_section = False

    # Process each line of the job description one by one.
    for line in lines:

        # Remove leading and trailing whitespace from the line.
        cleaned_line = line.strip()

        # Skip the line if it is empty.
        if not cleaned_line:
            continue

        # Convert the current line to lowercase and remove
        # a trailing colon so that headers like "Requirements:"
        # can be matched with "requirements".
        header = cleaned_line.lower().rstrip(":")

        # Check whether the current line is a requirement
        # section header.
        if header in REQUIREMENT_SECTION_HEADERS:

            # Mark that we have entered a requirement section.
            inside_requirement_section = True

            # Skip the section header itself because
            # it is not an actual requirement.
            continue

        # Check whether another obvious section begins
        # while we are inside the requirement section.
        if inside_requirement_section:

            # Convert the current line to lowercase and remove
            # a trailing colon before comparing it with known
            # section headers.
            if cleaned_line.lower().rstrip(":") in {
                "responsibilities",
                "responsibilities and duties",
                "about the role",
                "benefits",
                "about us",
            }:

                # Mark that we have exited the requirement section.
                inside_requirement_section = False

                # Skip the new section header.
                continue

        # Process the line only when we are currently
        # inside a requirement section.
        if inside_requirement_section:

            # Match common bullet-point formats such as:
            # -, *, •, numbered lists such as 1. or 2).
            match = re.match(
                r"^(?:[-*•]|\d+[.)])\s*(.+)$",
                cleaned_line
            )

            # Continue only if the current line matches
            # one of the supported bullet formats.
            if match:

                # Extract the actual requirement text from
                # the first captured group of the regular expression.
                requirement = match.group(1).strip()

                # Make sure the extracted requirement is not empty
                # before adding it to the final list.
                if requirement:
                    requirements.append(requirement)

    # Return the complete list of extracted job requirements.
    return requirements