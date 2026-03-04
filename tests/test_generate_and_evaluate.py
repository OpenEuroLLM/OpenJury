import json

import pandas as pd
import pytest

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


def test_generate_and_evaluate_writes_run_metadata(tmp_path):
    _ = main_generate_and_eval(
        CliArgs(
            dataset="alpaca-eval",
            model_A="Dummy/no answer",
            model_B="Dummy/open is better than close isnt'it",
            judge_model="Dummy/score A: 10 score B: 0",
            n_instructions=3,
            result_folder=str(tmp_path),
        )
    )

    metadata_files = list(tmp_path.rglob("run-metadata.v1.json"))
    assert len(metadata_files) == 1

    metadata = json.loads(metadata_files[0].read_text())
    assert metadata["schema_version"] == "openjury-run-metadata/v1"
    assert metadata["entrypoint"] == "openjury.generate_and_evaluate.main"
    assert metadata["run"]["dataset"] == "alpaca-eval"
    assert "command" not in metadata
    assert "cli_args" not in metadata
    assert "inputs" not in metadata
    assert metadata["dataset_statistics"]["instruction_index_count"] == 3
    assert metadata["dataset_statistics"]["instructions_count"] == 3
    assert metadata["dataset_statistics"]["completions_A_count"] == 3
    assert metadata["dataset_statistics"]["completions_B_count"] == 3
    assert len(metadata["instruction_indices_sha256"]) == 64
    assert len(metadata["judge_system_prompt_sha256"]) == 64
    assert len(metadata["judge_user_prompt_template_sha256"]) == 64
    assert "date" not in metadata["results"]
    assert "user" not in metadata["results"]
    assert "preferences" not in metadata["results"]
    assert metadata["results"]["preferences_count"] == 3
    assert "hostname" not in metadata["environment"]
    assert "user" not in metadata["environment"]
    assert "git" not in metadata
    if "git_hash" in metadata:
        assert len(metadata["git_hash"]) == 40
    artifact_paths = {artifact["path"] for artifact in metadata["artifacts"]}
    assert not any(path.startswith("args-") for path in artifact_paths)
    assert any(path.endswith("-annotations.csv") for path in artifact_paths)
    assert any(
        path.startswith("results-") and path.endswith(".json")
        for path in artifact_paths
    )
    assert (
        metadata["extras"]["files"]["results"]["relative_path"] in artifact_paths
    )
    assert "dependencies" in metadata
    assert "artifacts" in metadata
