"""Public rubrics API for rubric scoring and custom rubric loading."""

from openjury.rubrics.defaults import get_rubric, register_rubric
from openjury.rubrics.io import load_rubric_from_json, register_rubric_from_json, resolve_rubric
from openjury.rubrics.scorer import RubricScorer

__all__ = [
    "RubricScorer",
    "get_rubric",
    "register_rubric",
    "load_rubric_from_json",
    "register_rubric_from_json",
    "resolve_rubric",
]
