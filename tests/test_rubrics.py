import json

import pytest

import openjury.rubrics.scorer as scorer_module
from openjury.rubrics.defaults import get_rubric
from openjury.rubrics.io import load_rubric_from_json
from openjury.rubrics.schema import Criterion, Rubric, RubricDimension
from openjury.rubrics.scorer import (
    RubricScorer,
    _build_example_json_strings,
    _build_example_scores,
)


@pytest.fixture
def fake_prompt_loader(monkeypatch):
    loaded_names: list[str] = []

    templates = {
        "rubric_pairwise_system": (
            "PAIRWISE\n{rubric_block}\n{explanation_block}\n{example_json_pairwise}"
        ),
        "rubric_pairwise_user": (
            "Instruction: {instruction}\nA: {completion_A}\nB: {completion_B}"
        ),
        "rubric_samplewise_system": (
            "SAMPLEWISE\n{rubric_block}\n{reference_block}\n"
            "{explanation_block}\n{example_json}"
        ),
        "rubric_samplewise_user": (
            "Instruction: {instruction}\n{reference_section}\nCompletion: {completion}"
        ),
    }

    def fake_load_prompt(name: str) -> str:
        loaded_names.append(name)
        try:
            return templates[name]
        except KeyError as exc:
            raise AssertionError(f"Unexpected prompt requested: {name}") from exc

    monkeypatch.setattr(scorer_module, "load_prompt", fake_load_prompt)
    return loaded_names


def test_get_default_rubric():
    rubric = get_rubric("default")
    assert rubric.name == "default"
    assert len(rubric.criteria) > 0
    assert rubric.dimensions == rubric.criteria
    assert rubric.dimension_names == rubric.criterion_names
    assert rubric.k == rubric.num_criteria


def test_schema_compat_aliases():
    assert RubricDimension is Criterion

    rubric = Rubric(
        name="toy",
        criteria=[
            Criterion(name="c1", description="criterion 1"),
            Criterion(name="c2", description="criterion 2"),
        ],
    )
    assert [c.name for c in rubric.criteria] == ["c1", "c2"]
    assert [c.name for c in rubric.dimensions] == ["c1", "c2"]
    assert rubric.criterion_names == ["c1", "c2"]
    assert rubric.dimension_names == ["c1", "c2"]
    assert rubric.num_criteria == 2
    assert rubric.k == 2


def test_rubric_accepts_legacy_dimensions_keyword():
    rubric = Rubric(
        name="legacy",
        dimensions=[Criterion(name="overall", description="Overall quality")],
    )
    assert rubric.criterion_names == ["overall"]


def test_load_rubric_from_json_supports_legacy_dimensions_key(tmp_path):
    path = tmp_path / "legacy_dimensions.json"
    path.write_text(
        json.dumps(
            {
                "name": "legacy_dims",
                "dimensions": [
                    {"name": "overall", "description": "Overall quality"},
                ],
            }
        )
    )

    rubric = load_rubric_from_json(path)
    assert rubric.name == "legacy_dims"
    assert rubric.criterion_names == ["overall"]


def test_load_rubric_from_json_supports_criteria_key(tmp_path):
    path = tmp_path / "criteria.json"
    path.write_text(
        json.dumps(
            {
                "name": "criteria_rubric",
                "criteria": [
                    {"name": "clarity", "description": "Clarity"},
                    {"name": "correctness", "description": "Correctness"},
                ],
            }
        )
    )

    rubric = load_rubric_from_json(path)
    assert rubric.name == "criteria_rubric"
    assert rubric.criterion_names == ["clarity", "correctness"]


def test_build_example_scores_uses_criterion_names_and_clamps_to_scale():
    rubric = Rubric(
        name="toy",
        criteria=[
            Criterion(name="tiny", description="tiny scale", scale_min=0, scale_max=2),
            Criterion(name="tight", description="tight scale", scale_min=8, scale_max=9),
            Criterion(name="wide", description="wide scale", scale_min=1, scale_max=10),
        ],
    )

    scores = _build_example_scores(rubric)
    scores_dec = _build_example_scores(rubric, decrement=10)

    assert set(scores.keys()) == {"tiny", "tight", "wide"}
    assert scores["tiny"] == 2  # seed value clamps to scale_max
    assert scores["tight"] == 8  # seed value lands in the tight range
    assert 1 <= scores["wide"] <= 10

    assert scores_dec["tiny"] == 0  # decremented value clamps to scale_min
    assert scores_dec["tight"] == 8
    assert scores_dec["wide"] == 1


def test_build_example_json_strings_pairwise_shape():
    rubric = Rubric(
        name="toy",
        criteria=[
            Criterion(name="overall", description="overall quality"),
            Criterion(name="clarity", description="clarity"),
        ],
    )

    example_json, example_pairwise_json = _build_example_json_strings(rubric)
    sample = json.loads(example_json)
    pairwise = json.loads(example_pairwise_json)

    assert set(sample.keys()) == {"overall", "clarity"}
    assert pairwise["preference"] == "A"
    assert set(pairwise["scores_A"].keys()) == {"overall", "clarity"}
    assert set(pairwise["scores_B"].keys()) == {"overall", "clarity"}


def test_rubric_scorer_mode_loads_only_needed_prompts(fake_prompt_loader):
    rubric = get_rubric("default")

    pairwise_scorer = RubricScorer(
        judge_model=object(),
        rubric=rubric,
        mode="pairwise",
    )
    assert fake_prompt_loader == ["rubric_pairwise_system", "rubric_pairwise_user"]
    assert set(pairwise_scorer.system_prompt.keys()) == {"pairwise"}

    fake_prompt_loader.clear()

    samplewise_scorer = RubricScorer(
        judge_model=object(),
        rubric=rubric,
        mode="samplewise",
    )
    assert fake_prompt_loader == ["rubric_samplewise_system", "rubric_samplewise_user"]
    assert set(samplewise_scorer.system_prompt.keys()) == {
        "samplewise",
        "samplewise_with_ref",
    }


def test_rubric_scorer_mode_guards(fake_prompt_loader):
    rubric = get_rubric("default")

    pairwise_scorer = RubricScorer(
        judge_model=object(),
        rubric=rubric,
        mode="pairwise",
    )
    with pytest.raises(RuntimeError, match="configured for pairwise mode"):
        pairwise_scorer.score([], [], model_name="dummy")

    samplewise_scorer = RubricScorer(
        judge_model=object(),
        rubric=rubric,
        mode="samplewise",
    )
    with pytest.raises(RuntimeError, match="configured for samplewise mode"):
        samplewise_scorer.score_pairwise([], [], [])
