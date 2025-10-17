import pytest
from openjury.generate_and_evaluate import (
    main as main_generate_and_eval,
    CliArgs,
)


@pytest.mark.parametrize(
    "dataset", ["alpaca-eval", "fluency-french", "m-arena-hard-EU"]
)
def test_generate_and_evaluate_context_completion(dataset: str):
    main_generate_and_eval(
        CliArgs(
            dataset=dataset,
            model_A="Dummy/no answer",
            model_B="Dummy/open is better than close isnt'it",
            judge_model="Dummy/score A: 0 score B: 10",
            n_instructions=5,
        )
    )
