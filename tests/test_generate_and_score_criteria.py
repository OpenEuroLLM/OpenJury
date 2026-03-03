import json

import pandas as pd
import pytest

import openjury.generate_and_score_criteria as generate_and_score_criteria
from openjury.generate_and_score_criteria import CliArgs, main as main_score_criteria


@pytest.fixture(autouse=True)
def mock_external_data_and_cache(monkeypatch):
    instructions = pd.DataFrame(
        {
            "instruction": [f"Synthetic instruction {i}" for i in range(20)],
        },
        index=pd.Index(range(20), name="instruction_index"),
    )

    monkeypatch.setattr(
        generate_and_score_criteria,
        "load_instructions",
        lambda dataset, n_instructions=None: (
            instructions.head(n_instructions)
            if n_instructions is not None
            else instructions
        ),
    )
    monkeypatch.setattr(
        generate_and_score_criteria,
        "load_contexts",
        lambda dataset: instructions.loc[:, "instruction"],
    )

    monkeypatch.setattr(
        generate_and_score_criteria,
        "try_load_dataset_completions",
        lambda dataset, model, n_instructions: None,
    )

    def _run_without_cache(fun, **_kwargs):
        return fun()

    monkeypatch.setattr(
        generate_and_score_criteria, "cache_function_dataframe", _run_without_cache
    )
    monkeypatch.setattr(
        generate_and_score_criteria,
        "generate_instructions",
        lambda instructions, model, use_tqdm=False, **kwargs: pd.DataFrame(
            {
                "instruction_index": instructions.index.tolist(),
                "completion": [f"{model} completion" for _ in instructions.index],
            }
        ),
    )
    monkeypatch.setattr(
        generate_and_score_criteria,
        "generate_base",
        lambda instructions, model, use_tqdm=False, **kwargs: pd.DataFrame(
            {
                "instruction_index": instructions.index.tolist(),
                "completion": [f"{model} completion" for _ in instructions.index],
            }
        ),
    )
    monkeypatch.setattr(
        generate_and_score_criteria,
        "make_model",
        lambda **kwargs: object(),
    )


def test_generate_and_score_criteria_outputs_with_criteria_file(tmp_path, monkeypatch):
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
        assert kwargs["criteria_name"] == "overall"
        out_dir = kwargs["output_folder"]
        prefs = [0.0] * len(kwargs["instruction_index"])
        prefix = f"{kwargs['output_prefix']}-criteria-my_custom"
        pd.DataFrame(
            {"instruction_index": kwargs["instruction_index"], "preference": prefs}
        ).to_csv(
            out_dir / f"{prefix}-preferences.csv",
            index=False,
        )
        with open(out_dir / f"{prefix}-summary.json", "w") as f:
            json.dump({"criteria_name": "my_custom"}, f)
        return {"prefix": prefix, "preferences": prefs, "criteria_name": "my_custom"}

    monkeypatch.setattr(
        generate_and_score_criteria,
        "run_samplewise_criteria_pipeline",
        fake_run_samplewise_criteria_pipeline,
    )

    prefs = main_score_criteria(
        CliArgs(
            dataset="alpaca-eval",
            model_A="Dummy/no answer",
            model_B="Dummy/open is better than close isnt'it",
            judge_model="Dummy/judge",
            n_instructions=4,
            criteria_name="overall",
            criteria_file=str(criteria_file_path),
            result_folder=str(tmp_path),
        )
    )

    assert len(prefs) == 4

    summary_files = list(tmp_path.rglob("*-criteria-my_custom-summary.json"))
    pref_files = list(tmp_path.rglob("*-criteria-my_custom-preferences.csv"))
    result_files = list(tmp_path.rglob("results-*.json"))
    assert len(summary_files) == 1
    assert len(pref_files) == 1
    assert len(result_files) == 1

    criteria_summary = json.loads(summary_files[0].read_text())
    assert criteria_summary["criteria_name"] == "my_custom"

    results = json.loads(result_files[0].read_text())
    assert results["criteria_name"] == "my_custom"
    assert results["scoring_mode"] == "criteria_samplewise"
