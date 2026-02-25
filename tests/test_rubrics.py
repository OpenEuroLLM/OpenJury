import openjury.rubrics as rubrics


def test_get_default_rubric():
    rubric = rubrics.get_rubric("default")
    assert rubric.name == "default"
    assert len(rubric.dimensions) > 0


def test_rubric_scorer_initializes_and_renders_prompts():
    rubric = rubrics.get_rubric("default")
    scorer = rubrics.RubricScorer(judge_model=object(), rubric=rubric)

    prompts = scorer.system_prompt
    assert set(prompts.keys()) == {
        "samplewise",
        "samplewise_with_ref",
        "pairwise",
    }
    assert "preference" in prompts["pairwise"].lower()
