"""Default rubric definitions and registry.

Ships with three built-in rubrics:
    - **default**: General-purpose instruction-following evaluation
    - **coding**: Code generation quality evaluation
    - **translation**: Translation quality evaluation

Custom rubrics can be registered at runtime or loaded from JSON.
"""

from __future__ import annotations

from openjury.rubrics.schema import Rubric, RubricDimension


# ── Default Dimensions ──────────────────────────────────────────────


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

DEFAULT_DIMENSIONS = [
    RubricDimension(
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
    RubricDimension(
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
    RubricDimension(
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
    RubricDimension(
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
    RubricDimension(
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
    RubricDimension(
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

DEFAULT_RUBRIC = Rubric(
    name="default",
    dimensions=DEFAULT_DIMENSIONS,
    description="General-purpose rubric for instruction-following evaluation.",
)

CODING_RUBRIC = Rubric(
    name="coding",
    dimensions=[
        RubricDimension(
            name="correctness",
            description="Does the code compile/run and produce the correct output?",
        ),
        RubricDimension(
            name="adherence",
            description="Does the code follow the user's specifications exactly?",
        ),
        RubricDimension(
            name="readability",
            description="Is the code well-structured, documented, and easy to understand?",
        ),
        RubricDimension(
            name="efficiency",
            description="Is the solution reasonably efficient in time and space complexity?",
        ),
        RubricDimension(
            name="explanation",
            description="Is the accompanying explanation clear and helpful?",
        ),
    ],
    description="Rubric for evaluating code generation tasks.",
)

TRANSLATION_RUBRIC = Rubric(
    name="translation",
    dimensions=[
        RubricDimension(
            name="fluency",
            description="Is the translation natural-sounding in the target language?",
        ),
        RubricDimension(
            name="accuracy",
            description="Does the translation preserve the meaning of the source text?",
        ),
        RubricDimension(
            name="terminology",
            description="Are domain-specific terms translated correctly and consistently?",
        ),
        RubricDimension(
            name="style",
            description="Does the translation preserve the tone and register of the source?",
        ),
    ],
    description="Rubric for evaluating translation quality.",
)

OVERALL_RUBRIC = Rubric(
    name="overall",
    dimensions=[
        RubricDimension(
            name="overall",
            description=(
                "Overall quality of the response, considering all relevant "
                "factors. This is a holistic judgment of how well the response "
                "meets the user's needs and expectations."
            ),
            score_references=_refs(
                "Excellent response overall; highly effective across all important dimensions.",
                "Good response overall; useful and solid, with some noticeable weaknesses.",
                "Weak response overall; partially useful but with significant problems.",
                "Very poor response overall; fails to meet the user's needs.",
            ),
        )
    ],
    description="Single-dimension rubric for overall quality assessment.",
)


# ── Registry ────────────────────────────────────────────────────────

RUBRIC_REGISTRY: dict[str, Rubric] = {
    "default": DEFAULT_RUBRIC,
    "coding": CODING_RUBRIC,
    "translation": TRANSLATION_RUBRIC,
    "overall": OVERALL_RUBRIC,
}

# Convenience alias
DEFAULT_RUBRICS = RUBRIC_REGISTRY


def get_rubric(name: str) -> Rubric:
    """Look up a rubric by name.

    Args:
        name: Rubric identifier (e.g. "default", "coding", "translation").

    Returns:
        The Rubric object.

    Raises:
        KeyError: If the rubric name is not registered.
    """
    if name not in RUBRIC_REGISTRY:
        available = ", ".join(sorted(RUBRIC_REGISTRY.keys()))
        raise KeyError(f"Unknown rubric '{name}'. Available: {available}")
    return RUBRIC_REGISTRY[name]


def register_rubric(rubric: Rubric) -> None:
    """Register a custom rubric so it can be looked up by name.

    Args:
        rubric: Rubric object with a unique ``name`` attribute.
    """
    RUBRIC_REGISTRY[rubric.name] = rubric
