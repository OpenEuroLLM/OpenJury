import json

import pytest

import openjury.criteria.scorer as scorer_module
from openjury.criteria.defaults import CRITERIA_BY_NAME
from openjury.criteria.io import (
    load_criteria_from_file,
    load_criteria_from_json,
    load_criteria_from_yaml,
)
from openjury.criteria.schema import SCALE_MAX, SCALE_MIN, Criterion, criterion_names
from openjury.criteria.scorer import (
    CriteriaScorer,
    _build_example_json_strings,
    _build_example_scores,
)


@pytest.fixture
def fake_prompt_loader(monkeypatch):
    loaded_names: list[str] = []

    templates = {
        "criteria_samplewise_system": (
            "SAMPLEWISE\n{criteria_block}\n"
            "{explanation_block}\n{example_json}"
        ),
        "criteria_samplewise_user": (
            "Instruction: {instruction}\nCompletion: {completion}"
        ),
    }

    def fake_load_prompt(name: str) -> str:
        loaded_names.append(name)
        try:
            return templates[name]
        except KeyError as exc:
            raise AssertionError(f"Unexpected prompt requested: {name}") from exc

    monkeypatch.setattr(scorer_module, "_load_prompt_text", fake_load_prompt)
    return loaded_names


def test_get_default_criteria():
    criteria = CRITERIA_BY_NAME["default"]
    assert len(criteria) > 0
    assert criterion_names(criteria) == [
        "adherence",
        "helpfulness",
        "factuality",
        "completeness",
        "clarity",
        "fluency",
    ]


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
    assert criterion_names(criteria) == ["clarity", "correctness"]


def test_load_criteria_from_yaml_supports_criteria_key(tmp_path):
    path = tmp_path / "criteria.yaml"
    path.write_text(
        """
name: yaml_criteria
criteria:
  - name: clarity
    description: Clarity
  - name: correctness
    description: Correctness
"""
    )

    criteria = load_criteria_from_yaml(path)
    assert criterion_names(criteria) == ["clarity", "correctness"]


def test_load_criteria_from_file_supports_yaml_and_yml(tmp_path):
    content = """
name: file_loader
criteria:
  - name: overall
    description: Overall quality
"""
    yaml_path = tmp_path / "criteria.yaml"
    yml_path = tmp_path / "criteria.yml"
    yaml_path.write_text(content)
    yml_path.write_text(content)

    criteria_yaml = load_criteria_from_file(yaml_path)
    criteria_yml = load_criteria_from_file(yml_path)

    assert criterion_names(criteria_yaml) == ["overall"]
    assert criterion_names(criteria_yml) == ["overall"]


def test_load_criteria_from_file_rejects_unknown_extension(tmp_path):
    path = tmp_path / "criteria.txt"
    path.write_text("{}")
    with pytest.raises(ValueError, match="Unsupported criteria file format"):
        load_criteria_from_file(path)


def test_build_example_scores_uses_criterion_names_and_clamps_to_scale():
    criteria = [
        Criterion(name="tiny", description="tiny scale"),
        Criterion(name="tight", description="tight scale"),
        Criterion(name="wide", description="wide scale"),
    ]

    scores = _build_example_scores(criteria)
    scores_dec = _build_example_scores(criteria, decrement=10)

    assert set(scores.keys()) == {"tiny", "tight", "wide"}
    assert scores["tiny"] == 7
    assert scores["tight"] == 8
    assert SCALE_MIN <= scores["wide"] <= SCALE_MAX

    assert scores_dec["tiny"] == SCALE_MIN
    assert scores_dec["tight"] == SCALE_MIN
    assert scores_dec["wide"] == SCALE_MIN


def test_build_example_json_string_shape():
    criteria = [
        Criterion(name="overall", description="overall quality"),
        Criterion(name="clarity", description="clarity"),
    ]

    example_json = _build_example_json_strings(criteria)
    sample = json.loads(example_json)

    assert set(sample.keys()) == {"overall", "clarity"}


def test_criteria_scorer_loads_samplewise_prompts(fake_prompt_loader):
    criteria = CRITERIA_BY_NAME["default"]

    scorer = CriteriaScorer(
        judge_model=object(),
        criteria=criteria,
    )
    assert fake_prompt_loader == ["criteria_samplewise_system", "criteria_samplewise_user"]
    assert set(scorer.system_prompt.keys()) == {
        "samplewise",
    }


def test_criteria_scorer_validates_lengths(fake_prompt_loader):
    criteria = CRITERIA_BY_NAME["default"]

    scorer = CriteriaScorer(
        judge_model=object(),
        criteria=criteria,
    )
    with pytest.raises(AssertionError, match="must have the same length"):
        scorer.score(["i1", "i2"], ["c1"], model_name="dummy")
