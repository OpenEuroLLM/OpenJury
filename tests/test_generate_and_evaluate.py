import pandas as pd
import pytest
from types import SimpleNamespace
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


def test_generate_and_evaluate_rubric_and_bt_outputs(tmp_path, monkeypatch):
    """Smoke test optional rubric + Bradley-Terry integration without real judge calls."""
    import openjury.rubrics as rubrics_mod
    import openjury.bradley_terry as bt_mod

    fake_rubric = SimpleNamespace(name="overall", dimension_names=["overall"])

    class FakeRubricScorer:
        def __init__(self, judge_model, rubric, provide_explanation=False):
            self.rubric = rubric

        def score_pairwise(
            self,
            instructions,
            completions_A,
            completions_B,
            swap_to_debias=True,
            use_tqdm=False,
        ):
            return [
                SimpleNamespace(
                    instruction_index=i,
                    scores_A={"overall": 4.0 + i},
                    scores_B={"overall": 3.0 + i},
                    preference=0.0,  # A wins
                    raw_judge_output='{"ok": true}',
                    raw_judge_output_swapped='{"ok": true}' if swap_to_debias else None,
                )
                for i in range(len(instructions))
            ]

        def pairwise_to_dataframes(self, results, model_A_name, model_B_name):
            rows_a = []
            rows_b = []
            prefs = []
            for r in results:
                rows_a.append(
                    {
                        "instruction_index": r.instruction_index,
                        "model": model_A_name,
                        "overall": r.scores_A["overall"],
                    }
                )
                rows_b.append(
                    {
                        "instruction_index": r.instruction_index,
                        "model": model_B_name,
                        "overall": r.scores_B["overall"],
                    }
                )
                prefs.append(r.preference)
            return pd.DataFrame(rows_a), pd.DataFrame(rows_b), pd.Series(prefs)

    class FakeFeatureBradleyTerry:
        def __init__(self, dimension_names, regularization=0.01):
            self.dimension_names = dimension_names
            self.regularization = regularization
            self.intercept = 0.123

        def fit(self, scores_A, scores_B, preferences, verbose=True, **kwargs):
            return self

        def weight_dict(self):
            return {name: 1.0 for name in self.dimension_names}

    monkeypatch.setattr(rubrics_mod, "get_rubric", lambda name: fake_rubric)
    monkeypatch.setattr(rubrics_mod, "RubricScorer", FakeRubricScorer)
    monkeypatch.setattr(bt_mod, "FeatureBradleyTerry", FakeFeatureBradleyTerry)

    prefs = main_generate_and_eval(
        CliArgs(
            dataset="alpaca-eval",
            model_A="Dummy/no answer",
            model_B="Dummy/open is better than close isnt'it",
            judge_model="Dummy/score A: 10 score B: 0",
            n_instructions=4,
            enable_rubrics=True,
            rubric_name="overall",
            fit_bradley_terry=True,
            result_folder=str(tmp_path),
        )
    )

    assert len(prefs) == 4

    summary_files = list(tmp_path.rglob("*-rubric-overall-summary.json"))
    bt_files = list(tmp_path.rglob("*-rubric-overall-bt-weights.json"))
    pref_files = list(tmp_path.rglob("*-rubric-overall-preferences.csv"))
    assert len(summary_files) == 1
    assert len(bt_files) == 1
    assert len(pref_files) == 1

    rubric_summary = json.loads(summary_files[0].read_text())
    assert rubric_summary["rubric_name"] == "overall"
    assert "bradley_terry" in rubric_summary
    assert rubric_summary["bradley_terry"]["weights"]["overall"] == 1.0
