import json

import pytest

import openjury.criteria.scorer as scorer_module
from openjury.criteria.defaults import get_criteria
from openjury.criteria.io import load_criteria_from_json
from openjury.criteria.schema import Criterion, Criteria
from openjury.criteria.scorer import (
    CriteriaScorer,
    _build_example_json_strings,
    _build_example_scores,
)


@pytest.fixture
def fake_prompt_loader(monkeypatch):
    loaded_names: list[str] = []

    templates = {
        "criteria_pairwise_system": (
            "PAIRWISE\n{criteria_block}\n{explanation_block}\n{example_json_pairwise}"
        ),
        "criteria_pairwise_user": (
            "Instruction: {instruction}\nA: {completion_A}\nB: {completion_B}"
        ),
        "criteria_samplewise_system": (
            "SAMPLEWISE\n{criteria_block}\n{reference_block}\n"
            "{explanation_block}\n{example_json}"
        ),
        "criteria_samplewise_user": (
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


def test_get_default_criteria():
    criteria = get_criteria("default")
    assert criteria.name == "default"
    assert len(criteria.criteria) > 0
    assert len(criteria.criterion_names) == criteria.num_criteria


def test_load_criteria_from_json_requires_criteria_key(tmp_path):
    path = tmp_path / "missing_criteria.json"
    path.write_text(
        json.dumps(
            {
                "name": "missing_criteria",
                "dimensions": [
                    {"name": "overall", "description": "Overall quality"},
                ],
            }
        )
    )

    with pytest.raises(KeyError, match="criteria"):
        load_criteria_from_json(path)


def test_load_criteria_from_json_supports_criteria_key(tmp_path):
    path = tmp_path / "criteria.json"
    path.write_text(
        json.dumps(
            {
                "name": "criteria_criteria",
                "criteria": [
                    {"name": "clarity", "description": "Clarity"},
                    {"name": "correctness", "description": "Correctness"},
                ],
            }
        )
    )

    criteria = load_criteria_from_json(path)
    assert criteria.name == "criteria_criteria"
    assert criteria.criterion_names == ["clarity", "correctness"]


def test_build_example_scores_uses_criterion_names_and_clamps_to_scale():
    criteria = Criteria(
        name="toy",
        criteria=[
            Criterion(name="tiny", description="tiny scale", scale_min=0, scale_max=2),
            Criterion(name="tight", description="tight scale", scale_min=8, scale_max=9),
            Criterion(name="wide", description="wide scale", scale_min=1, scale_max=10),
        ],
    )

    scores = _build_example_scores(criteria)
    scores_dec = _build_example_scores(criteria, decrement=10)

    assert set(scores.keys()) == {"tiny", "tight", "wide"}
    assert scores["tiny"] == 2  # seed value clamps to scale_max
    assert scores["tight"] == 8  # seed value lands in the tight range
    assert 1 <= scores["wide"] <= 10

    assert scores_dec["tiny"] == 0  # decremented value clamps to scale_min
    assert scores_dec["tight"] == 8
    assert scores_dec["wide"] == 1


def test_build_example_json_strings_pairwise_shape():
    criteria = Criteria(
        name="toy",
        criteria=[
            Criterion(name="overall", description="overall quality"),
            Criterion(name="clarity", description="clarity"),
        ],
    )

    example_json, example_pairwise_json = _build_example_json_strings(criteria)
    sample = json.loads(example_json)
    pairwise = json.loads(example_pairwise_json)

    assert set(sample.keys()) == {"overall", "clarity"}
    assert pairwise["preference"] == "A"
    assert set(pairwise["scores_A"].keys()) == {"overall", "clarity"}
    assert set(pairwise["scores_B"].keys()) == {"overall", "clarity"}


def test_criteria_scorer_mode_loads_only_needed_prompts(fake_prompt_loader):
    criteria = get_criteria("default")

    pairwise_scorer = CriteriaScorer(
        judge_model=object(),
        criteria=criteria,
        mode="pairwise",
    )
    assert fake_prompt_loader == ["criteria_pairwise_system", "criteria_pairwise_user"]
    assert set(pairwise_scorer.system_prompt.keys()) == {"pairwise"}

    fake_prompt_loader.clear()

    samplewise_scorer = CriteriaScorer(
        judge_model=object(),
        criteria=criteria,
        mode="samplewise",
    )
    assert fake_prompt_loader == ["criteria_samplewise_system", "criteria_samplewise_user"]
    assert set(samplewise_scorer.system_prompt.keys()) == {
        "samplewise",
        "samplewise_with_ref",
    }


def test_criteria_scorer_mode_guards(fake_prompt_loader):
    criteria = get_criteria("default")

    pairwise_scorer = CriteriaScorer(
        judge_model=object(),
        criteria=criteria,
        mode="pairwise",
    )
    with pytest.raises(RuntimeError, match="configured for pairwise mode"):
        pairwise_scorer.score([], [], model_name="dummy")

    samplewise_scorer = CriteriaScorer(
        judge_model=object(),
        criteria=criteria,
        mode="samplewise",
    )
    with pytest.raises(RuntimeError, match="configured for samplewise mode"):
        samplewise_scorer.score_pairwise([], [], [])
