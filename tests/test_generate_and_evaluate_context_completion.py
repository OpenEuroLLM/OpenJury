from llmjudgeeval.generate_and_evaluate_context_completion import (
    main as main_generate_and_eval_context_compl,
    CliArgs,
)


def test_generate_and_evaluate_context_completion():
    main_generate_and_eval_context_compl(
        CliArgs(
            dataset="french-contexts",
            generation_model_A="Dummy",
            generation_model_B="Dummy",
            judge_model="Dummy",
            n_instructions=5,
        )
    )
