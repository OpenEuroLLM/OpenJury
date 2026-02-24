"""Multi-criteria rubric evaluation for OpenJury.

Provides structured, interpretable scoring of LLM completions along
configurable dimensions (fluency, usefulness, clarity, style, adherence, etc.)
using an LLM judge with custom rubrics.

Quick start::

    from openjury.rubrics import RubricScorer, get_rubric
    from openjury.utils import make_model

    judge = make_model("VLLM/Qwen/Qwen2.5-32B-Instruct")
    rubric = get_rubric("default")
    scorer = RubricScorer(judge_model=judge, rubric=rubric)

    scores = scorer.score(
        instructions=["Write a poem"],
        completions=["Roses are red..."],
        model_name="test",
    )
"""

from openjury.rubrics.schema import Rubric, RubricDimension, RubricScore, PairwiseRubricResult
from openjury.rubrics.defaults import DEFAULT_RUBRICS, RUBRIC_REGISTRY, get_rubric, register_rubric
from openjury.rubrics.io import load_rubric_from_json, register_rubric_from_json, resolve_rubric
from openjury.rubrics.scorer import RubricScorer

__all__ = [
    "Rubric",
    "RubricDimension",
    "RubricScore",
    "PairwiseRubricResult",
    "DEFAULT_RUBRICS",
    "RUBRIC_REGISTRY",
    "RubricScorer",
    "get_rubric",
    "register_rubric",
    "load_rubric_from_json",
    "register_rubric_from_json",
    "resolve_rubric",
]
