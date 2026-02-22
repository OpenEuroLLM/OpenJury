import json

import pytest

from openjury.evaluate import RubricScore, RUBRIC_CRITERIA, VALID_RUBRIC_SCORES


# --- RubricScore parser tests ---

VALID_RUBRIC_JSON = json.dumps({
    "instruction_following_rationale": "The response addresses all aspects.",
    "instruction_following_score": 7,
    "naturalness_rationale": "Fluent and natural text.",
    "naturalness_score": 7,
    "coherence_rationale": "Well-organized response.",
    "coherence_score": 5,
    "accuracy_rationale": "Mostly accurate with minor issues.",
    "accuracy_score": 5,
})


def test_rubric_score_parse_bare_json():
    score = RubricScore()
    result = score.parse_model_raw(VALID_RUBRIC_JSON)
    assert result is not None
    assert result["instruction_following_score"] == 7
    assert result["naturalness_score"] == 7
    assert result["coherence_score"] == 5
    assert result["accuracy_score"] == 5


def test_rubric_score_parse_markdown_wrapped():
    wrapped = f"Here is my evaluation:\n```json\n{VALID_RUBRIC_JSON}\n```"
    score = RubricScore()
    result = score.parse_model_raw(wrapped)
    assert result is not None
    assert result["instruction_following_score"] == 7
    assert result["naturalness_score"] == 7


def test_rubric_score_parse_with_extra_text():
    text = f"Let me evaluate this response.\n\n{VALID_RUBRIC_JSON}\n\nThat's my assessment."
    score = RubricScore()
    result = score.parse_model_raw(text)
    assert result is not None
    assert result["accuracy_score"] == 5


def test_rubric_composite_calculation():
    """Composite = (mean - 1) / 6, where mean is average of 4 scores."""
    score = RubricScore()
    result = score.parse_model_raw(VALID_RUBRIC_JSON)
    assert result is not None
    # mean = (7 + 7 + 5 + 5) / 4 = 6.0
    assert abs(result["mean_score"] - 6.0) < 1e-6
    # composite = (6.0 - 1) / 6 = 0.8333...
    assert abs(result["composite_score"] - 0.833333) < 1e-4


def test_rubric_composite_all_ones():
    """All scores = 1 should give composite = 0."""
    data = {}
    for c in RUBRIC_CRITERIA:
        data[f"{c}_rationale"] = "bad"
        data[f"{c}_score"] = 1
    score = RubricScore()
    result = score.parse_model_raw(json.dumps(data))
    assert result is not None
    assert result["composite_score"] == 0.0


def test_rubric_composite_all_sevens():
    """All scores = 7 should give composite = 1."""
    data = {}
    for c in RUBRIC_CRITERIA:
        data[f"{c}_rationale"] = "perfect"
        data[f"{c}_score"] = 7
    score = RubricScore()
    result = score.parse_model_raw(json.dumps(data))
    assert result is not None
    assert result["composite_score"] == 1.0


def test_rubric_score_missing_field():
    incomplete = json.dumps({"instruction_following_score": 5})
    score = RubricScore()
    result = score.parse_model_raw(incomplete)
    assert result is None


def test_rubric_score_invalid_json():
    score = RubricScore()
    result = score.parse_model_raw("This is not JSON at all")
    assert result is None


def test_rubric_score_snaps_to_valid():
    """Scores are snapped to nearest valid anchor {1, 3, 5, 7}."""
    # 10 -> snaps to 7 (nearest valid)
    data = {}
    for c in RUBRIC_CRITERIA:
        data[f"{c}_rationale"] = "test"
        data[f"{c}_score"] = 10
    score = RubricScore()
    result = score.parse_model_raw(json.dumps(data))
    assert result is not None
    for c in RUBRIC_CRITERIA:
        assert result[f"{c}_score"] == 7

    # 2 -> snaps to 1 (equidistant, min picks first)
    data2 = {}
    for c in RUBRIC_CRITERIA:
        data2[f"{c}_rationale"] = "test"
        data2[f"{c}_score"] = 2
    result2 = score.parse_model_raw(json.dumps(data2))
    assert result2 is not None
    for c in RUBRIC_CRITERIA:
        assert result2[f"{c}_score"] in VALID_RUBRIC_SCORES

    # 4 -> snaps to 3 or 5 (equidistant)
    data4 = {}
    for c in RUBRIC_CRITERIA:
        data4[f"{c}_rationale"] = "test"
        data4[f"{c}_score"] = 4
    result4 = score.parse_model_raw(json.dumps(data4))
    assert result4 is not None
    for c in RUBRIC_CRITERIA:
        assert result4[f"{c}_score"] in VALID_RUBRIC_SCORES

    # 6 -> snaps to 5 or 7 (equidistant)
    data6 = {}
    for c in RUBRIC_CRITERIA:
        data6[f"{c}_rationale"] = "test"
        data6[f"{c}_score"] = 6
    result6 = score.parse_model_raw(json.dumps(data6))
    assert result6 is not None
    for c in RUBRIC_CRITERIA:
        assert result6[f"{c}_score"] in VALID_RUBRIC_SCORES


def test_rubric_score_string_numbers():
    """Scores as strings should still be parsed."""
    data = {}
    for c in RUBRIC_CRITERIA:
        data[f"{c}_rationale"] = "test"
        data[f"{c}_score"] = "5"  # string instead of int
    score = RubricScore()
    result = score.parse_model_raw(json.dumps(data))
    assert result is not None
    for c in RUBRIC_CRITERIA:
        assert result[f"{c}_score"] == 5


def test_rubric_rationales_preserved():
    score = RubricScore()
    result = score.parse_model_raw(VALID_RUBRIC_JSON)
    assert result is not None
    assert result["instruction_following_rationale"] == "The response addresses all aspects."
    assert result["naturalness_rationale"] == "Fluent and natural text."


# --- End-to-end test with Dummy models ---

def test_generate_and_evaluate_rubric(tmp_path):
    from openjury.generate_and_evaluate import main as main_generate_and_eval, CliArgs

    # The Dummy model returns its name (after "Dummy/") as the judge completion.
    # Use valid Likert anchor scores {1, 3, 5, 7} only.
    j = '{"instruction_following_score":5,"naturalness_score":7,"coherence_score":5,"accuracy_score":3}'

    results = main_generate_and_eval(
        CliArgs(
            dataset="alpaca-eval",
            model_A="Dummy/a",
            model_B="Dummy/b",
            judge_model=f"Dummy/{j}",
            eval_mode="rubric",
            n_instructions=3,
            result_folder=str(tmp_path),
        )
    )

    assert results["eval_mode"] == "rubric"
    assert results["num_instructions"] == 3
    assert results["model_A_scores"]["instruction_following_score"] == 5
    assert results["model_A_scores"]["naturalness_score"] == 7
    assert results["model_A_scores"]["coherence_score"] == 5
    assert results["model_A_scores"]["accuracy_score"] == 3
    # composite = (5+7+5+3)/4 = 5.0, mapped: (5-1)/6 = 0.6667
    assert abs(results["model_A_scores"]["composite_score"] - 0.6667) < 0.01
    assert results["model_A_parse_failures"] == 0
    assert results["model_B_parse_failures"] == 0
