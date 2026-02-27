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


def test_generate_and_evaluate_criteria_outputs_with_criteria_file(tmp_path, monkeypatch):
    """Smoke test criteria hook and criteria_file forwarding without real judge calls."""
    criteria_file_path = tmp_path / "custom_criteria.json"
    criteria_file_path.write_text(
        json.dumps(
            {
                "name": "my_custom",
                "criteria": [
                    {
                        "name": "overall",
                        "description": "Overall quality",
                    }
                ],
            }
        )
    )

    def fake_run_samplewise_criteria_pipeline(**kwargs):
        assert kwargs["criteria_file"] == str(criteria_file_path)
        # criteria_file should override criteria_name later in the shared helper
        assert kwargs["criteria_name"] == "overall"
        out_dir = kwargs["output_folder"]
        prefs = [0.0] * len(kwargs["instruction_index"])
        prefix = f"{kwargs['output_prefix']}-criteria-my_custom"
        pd.DataFrame({"instruction_index": kwargs["instruction_index"], "preference": prefs}).to_csv(
            out_dir / f"{prefix}-preferences.csv",
            index=False,
        )
        with open(out_dir / f"{prefix}-summary.json", "w") as f:
            json.dump(
                {
                    "criteria_name": "my_custom",
                },
                f,
            )
        return {"prefix": prefix, "preferences": prefs}

    monkeypatch.setattr(
        generate_and_evaluate,
        "run_samplewise_criteria_pipeline",
        fake_run_samplewise_criteria_pipeline,
    )
    monkeypatch.setattr(
        generate_and_evaluate,
        "annotate_battles",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("annotate_battles should not run when enable_criteria=True")
        ),
    )

    prefs = main_generate_and_eval(
        CliArgs(
            dataset="alpaca-eval",
            model_A="Dummy/no answer",
            model_B="Dummy/open is better than close isnt'it",
            judge_model="Dummy/score A: 10 score B: 0",
            n_instructions=4,
            enable_criteria=True,
            criteria_name="overall",
            criteria_file=str(criteria_file_path),
            result_folder=str(tmp_path),
        )
    )

    assert len(prefs) == 4

    summary_files = list(tmp_path.rglob("*-criteria-my_custom-summary.json"))
    pref_files = list(tmp_path.rglob("*-criteria-my_custom-preferences.csv"))
    assert len(summary_files) == 1
    assert len(pref_files) == 1

    criteria_summary = json.loads(summary_files[0].read_text())
    assert criteria_summary["criteria_name"] == "my_custom"
