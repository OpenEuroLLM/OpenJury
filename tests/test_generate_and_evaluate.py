import pandas as pd
import pytest
import json

import openjury.generate_and_evaluate as generate_and_evaluate
from openjury.generate_and_evaluate import (
    main as main_generate_and_eval,
    CliArgs,
)


@pytest.fixture(autouse=True)
def mock_external_data_and_cache(monkeypatch):
    instructions = pd.DataFrame(
        {
            "instruction": [f"Synthetic instruction {i}" for i in range(20)],
        },
        index=pd.Index(range(20), name="instruction_index"),
    )

    monkeypatch.setattr(
        generate_and_evaluate,
        "load_instructions",
        lambda dataset, n_instructions=None: (
            instructions.head(n_instructions)
            if n_instructions is not None
            else instructions
        ),
    )
    monkeypatch.setattr(
        generate_and_evaluate,
        "load_contexts",
        lambda dataset: instructions.loc[:, "instruction"],
    )

    monkeypatch.setattr(
        generate_and_evaluate,
        "try_load_dataset_completions",
        lambda dataset, model, n_instructions: None,
    )

    def _run_without_cache(fun, **_kwargs):
        return fun()

    monkeypatch.setattr(
        generate_and_evaluate, "cache_function_dataframe", _run_without_cache
    )


@pytest.mark.parametrize(
    "dataset", ["alpaca-eval", "fluency-french", "m-arena-hard-EU"]
)
def test_generate_and_evaluate_context_completion(dataset: str, tmp_path):
    prefs = main_generate_and_eval(
        CliArgs(
            dataset=dataset,
            model_A="Dummy/no answer",
            model_B="Dummy/open is better than close isnt'it",
            judge_model="Dummy/score A: 0 score B: 10",
            n_instructions=5,
            result_folder=str(tmp_path),
            # default for swap_mode is "fixed"
        )
    )

    avg_pref = sum(prefs) / len(prefs)
    assert avg_pref >= 0.9


def test_generate_and_evaluate_correct_order_bias(tmp_path):
    """Test the correction for model order bias.
    
    In this test, a judge that is totally biased towards model B should be corrected to be neutral.
    Since the judge favors model B regardless of the order and the completions, the average
    preference should be 0.5.
    """
    prefs = main_generate_and_eval(
        CliArgs(
            dataset="alpaca-eval",
            model_A="Dummy/no answer",
            model_B="Dummy/open is better than close isnt'it",
            judge_model="Dummy/score A: 0 score B: 10",
            n_instructions=5,
            swap_mode="both",
            result_folder=str(tmp_path),
        )
    )

    avg_pref = sum(prefs) / len(prefs)
    assert avg_pref == 0.5


def test_generate_and_evaluate_rubric_outputs_with_rubric_file(tmp_path, monkeypatch):
    """Smoke test rubric hook and rubric_file forwarding without real judge calls."""
    rubric_file_path = tmp_path / "custom_rubric.json"
    rubric_file_path.write_text(
        json.dumps(
            {
                "name": "my_custom",
                "dimensions": [
                    {
                        "name": "overall",
                        "description": "Overall quality",
                    }
                ],
            }
        )
    )

    def fake_run_pairwise_rubric_pipeline(**kwargs):
        assert kwargs["rubric_file"] == str(rubric_file_path)
        # rubric_file should override rubric_name later in the shared helper
        assert kwargs["rubric_name"] == "overall"
        out_dir = kwargs["output_folder"]
        prefix = f"{kwargs['output_prefix']}-rubric-my_custom"
        pd.DataFrame({"instruction_index": kwargs["instruction_index"], "preference": [0.0] * len(kwargs["instruction_index"])}).to_csv(
            out_dir / f"{prefix}-preferences.csv",
            index=False,
        )
        with open(out_dir / f"{prefix}-summary.json", "w") as f:
            json.dump(
                {
                    "rubric_name": "my_custom",
                },
                f,
            )
        return {"prefix": prefix}

    monkeypatch.setattr(
        generate_and_evaluate,
        "run_pairwise_rubric_pipeline",
        fake_run_pairwise_rubric_pipeline,
    )

    prefs = main_generate_and_eval(
        CliArgs(
            dataset="alpaca-eval",
            model_A="Dummy/no answer",
            model_B="Dummy/open is better than close isnt'it",
            judge_model="Dummy/score A: 10 score B: 0",
            n_instructions=4,
            enable_rubrics=True,
            rubric_name="overall",
            rubric_file=str(rubric_file_path),
            result_folder=str(tmp_path),
        )
    )

    assert len(prefs) == 4

    summary_files = list(tmp_path.rglob("*-rubric-my_custom-summary.json"))
    pref_files = list(tmp_path.rglob("*-rubric-my_custom-preferences.csv"))
    assert len(summary_files) == 1
    assert len(pref_files) == 1

    rubric_summary = json.loads(summary_files[0].read_text())
    assert rubric_summary["rubric_name"] == "my_custom"
