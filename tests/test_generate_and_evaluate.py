import pytest
from llmjudgeeval.generate_and_evaluate import (
    main as main_generate_and_eval,
    CliArgs,
)


@pytest.mark.parametrize(
    "dataset", ["alpaca-eval", "french-contexts", "m-arena-hard-EU"]
)
def test_generate_and_evaluate_context_completion(dataset: str):
    main_generate_and_eval(
        CliArgs(
            dataset=dataset,
            model_A="Dummy",
            model_B="Dummy",
            judge_model="Dummy",
            n_instructions=5,
        )
    )
