"""Default criteria-set definitions and registry.

Ships with one built-in criteria set:
    - **default**: General-purpose instruction-following evaluation

Custom criteria sets can be registered at runtime or loaded from JSON.
"""

from __future__ import annotations

from openjury.criteria.schema import Criterion, Criteria


# ── Default Criteria ────────────────────────────────────────────────


def _refs(
    s10: str,
    s7: str,
    s4: str,
    s1: str,
) -> dict[int, str]:
    """Sparse score anchors for the default 1-10 scale."""
    return {
        10: s10,
        7: s7,
        4: s4,
        1: s1,
    }

DEFAULT_CRITERIA = [
    Criterion(
        name="adherence",
        description=(
            "Follows the user's instructions and constraints precisely: required format, scope, "
            "style constraints, and any do/don't requirements. Penalize missing requested parts "
            "or deviating from constraints."
        ),
        score_references=_refs(
            "Fully follows the user's instructions and constraints, including format and scope.",
            "Mostly follows the request but misses some details or adds minor unnecessary content.",
            "Follows only part of the request and misses important constraints or requested parts.",
            "Does not follow the user's request in any meaningful way.",
        ),
    ),
    Criterion(
        name="helpfulness",
        description=(
            "Advances the user's goal with relevant, actionable content. Provides useful steps, "
            "options, or explanations tailored to the request. Penalize generic filler or "
            "non-responsive content."
        ),
        score_references=_refs(
            "Directly solves the user's problem with highly useful, actionable, and relevant content.",
            "Generally helpful and relevant, but misses some useful detail or optimization.",
            "Partially helpful but vague, generic, or missing key actionable guidance.",
            "Unhelpful or non-responsive to the user's goal.",
        ),
    ),
    Criterion(
        name="factuality",
        description=(
            "Information is correct and appropriately qualified. Avoids hallucinations and "
            "unwarranted specifics. If uncertain, expresses uncertainty and does not fabricate "
            "sources, citations, or details."
        ),
        score_references=_refs(
            "Accurate and well-qualified throughout, with no fabricated or unsupported claims.",
            "Mostly accurate, with only minor imprecision or insufficient qualification.",
            "Contains multiple factual errors, overclaims, or likely hallucinated details.",
            "Largely incorrect, fabricated, or misleading.",
        ),
    ),
    Criterion(
        name="completeness",
        description=(
            "Covers the key aspects of the request without major omissions. Addresses all "
            "sub-questions and important constraints. Penalize partial answers or skipped items."
        ),
        score_references=_refs(
            "Covers all major parts of the request with no important omissions.",
            "Covers the main request but misses some secondary details or sub-parts.",
            "Only partially addresses the request; multiple important pieces are missing.",
            "Fails to address the requested task in a complete or usable way.",
        ),
    ),
    Criterion(
        name="clarity",
        description=(
            "Well-organized, easy to follow, and unambiguous. Uses logical structure, headings/"
            "lists when helpful, and clear references. Penalize confusing organization or ambiguity."
        ),
        score_references=_refs(
            "Clear, logically structured, and easy to follow with no ambiguity.",
            "Mostly clear and organized, but has a few awkward transitions or ambiguities.",
            "Noticeably hard to follow due to weak structure or unclear phrasing.",
            "Confusing, disorganized, or difficult to understand.",
        ),
    ),
    Criterion(
        name="fluency",
        description=(
            "Language and presentation quality: fluent, readable, appropriately concise, and "
            "well-formatted. Tone is appropriate for the user/context."
        ),
        score_references=_refs(
            "Fluent, natural, polished language with strong readability and appropriate tone.",
            "Generally fluent and readable, with some awkward phrasing or minor disfluencies.",
            "Frequent disfluencies or formatting issues that make reading noticeably harder.",
            "Severely disfluent or poorly formatted to the point of harming comprehension.",
        ),
    ),
]

DEFAULT_CRITERIA_SET = Criteria(
    name="default",
    criteria=DEFAULT_CRITERIA,
    description="General-purpose criteria for instruction-following evaluation.",
)


# ── Registry ────────────────────────────────────────────────────────

CRITERIA_REGISTRY: dict[str, Criteria] = {
    "default": DEFAULT_CRITERIA_SET,
}


def get_criteria(name: str) -> Criteria:
    """Look up a criteria set by name.

    Args:
        name: Criteria-set identifier (e.g. "default").

    Returns:
        The Criteria object.

    Raises:
        KeyError: If the criteria-set name is not registered.
    """
    if name not in CRITERIA_REGISTRY:
        available = ", ".join(sorted(CRITERIA_REGISTRY.keys()))
        raise KeyError(f"Unknown criteria '{name}'. Available: {available}")
    return CRITERIA_REGISTRY[name]


def register_criteria(criteria: Criteria) -> None:
    """Register a custom criteria set so it can be looked up by name.

    Args:
        criteria: Criteria object with a unique ``name`` attribute.
    """
    CRITERIA_REGISTRY[criteria.name] = criteria
