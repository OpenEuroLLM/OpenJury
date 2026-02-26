"""Shared pairwise criteria scoring pipeline helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from openjury.criteria.io import resolve_criteria
from openjury.criteria.scorer import CriteriaScorer


def _compute_pref_summary(prefs: pd.Series) -> dict[str, float | int]:
    prefs = pd.Series(prefs, dtype="float64")
    valid = prefs.dropna()
    num_wins = int((valid < 0.5).sum())
    num_losses = int((valid > 0.5).sum())
    num_ties = int((valid == 0.5).sum())
    num_battles = int(len(prefs))
    denom = num_wins + num_losses + num_ties
    winrate = float((num_wins + 0.5 * num_ties) / denom) if denom > 0 else float("nan")
    return {
        "num_battles": num_battles,
        "winrate": winrate,
        "num_wins": num_wins,
        "num_losses": num_losses,
        "num_ties": num_ties,
        "num_missing": int(num_battles - denom),
    }


def run_pairwise_criteria_pipeline(
    *,
    output_folder: str | Path,
    output_prefix: str,
    judge_model: Any,
    instructions: list[str],
    completions_A: list[str],
    completions_B: list[str],
    instruction_index: list[int | str],
    model_A_name: str,
    model_B_name: str,
    provide_explanation: bool = False,
    use_tqdm: bool = False,
    criteria_name: str = "default",
    criteria_file: str | Path | None = None,
    swap_to_debias: bool = False,
    fit_bradley_terry: bool = False,
    bt_regularization: float = 0.01,
    bt_tie_epsilon: float = 0.05,
    summary_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run pairwise criteria scoring and save outputs."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    criteria = resolve_criteria(criteria_name=criteria_name, criteria_file=criteria_file)
    scorer = CriteriaScorer(
        judge_model=judge_model,
        criteria=criteria,
        provide_explanation=provide_explanation,
        mode="pairwise",
    )
    pairwise_results = scorer.score_pairwise(
        instructions=instructions,
        completions_A=completions_A,
        completions_B=completions_B,
        swap_to_debias=swap_to_debias,
        use_tqdm=use_tqdm,
    )
    df_A, df_B, prefs = scorer.pairwise_to_dataframes(
        pairwise_results,
        model_A_name=model_A_name,
        model_B_name=model_B_name,
    )
    df_A.loc[:, "instruction_index"] = instruction_index
    df_B.loc[:, "instruction_index"] = instruction_index

    prefix_base = f"{output_prefix}-criteria-{criteria.name}" if output_prefix else f"criteria-{criteria.name}"
    df_A.to_csv(output_folder / f"{prefix_base}-scores-A.csv", index=False)
    df_B.to_csv(output_folder / f"{prefix_base}-scores-B.csv", index=False)
    pd.DataFrame(
        {"instruction_index": instruction_index, "preference": prefs.tolist()}
    ).to_csv(output_folder / f"{prefix_base}-preferences.csv", index=False)

    pairwise_rows = []
    for inst_idx, r in zip(instruction_index, pairwise_results):
        row = {
            "instruction_index": inst_idx,
            "preference": r.preference,
            "raw_judge_output": r.raw_judge_output,
            "raw_judge_output_swapped": r.raw_judge_output_swapped,
        }
        for criterion_name, value in r.scores_A.items():
            row[f"A_{criterion_name}"] = value
        for criterion_name, value in r.scores_B.items():
            row[f"B_{criterion_name}"] = value
        pairwise_rows.append(row)
    pd.DataFrame(pairwise_rows).to_csv(output_folder / f"{prefix_base}-pairwise.csv", index=False)

    bt_result: dict[str, Any] | None = None
    if fit_bradley_terry:
        try:
            from openjury.bradley_terry import FeatureBradleyTerry
        except Exception as e:  # pragma: no cover - import error depends on optional deps
            raise RuntimeError(
                "Bradley-Terry fitting requires the optional 'sklearn' dependency. "
                "Install with `uv sync --extra sklearn` or `pip install -e '.[sklearn]'`."
            ) from e

        bt_model = FeatureBradleyTerry(
            criterion_names=criteria.criterion_names,
            regularization=bt_regularization,
            tie_epsilon=bt_tie_epsilon,
        )
        bt_model.fit(
            scores_A=df_A,
            scores_B=df_B,
            preferences=prefs,
            verbose=False,
        )

        bt_result = {
            "criterion_names": criteria.criterion_names,
            "weights": bt_model.weight_dict(),
            "intercept": float(bt_model.intercept),
            "regularization": float(bt_regularization),
            "tie_epsilon": float(bt_tie_epsilon),
        }
        with open(output_folder / f"{prefix_base}-bt-weights.json", "w") as f:
            json.dump(bt_result, f, indent=2)

    summary = {
        **(summary_fields or {}),
        "criteria_name": criteria.name,
        "criterion_names": criteria.criterion_names,
        "swap_debiasing": swap_to_debias,
        "fit_bradley_terry": fit_bradley_terry,
        **_compute_pref_summary(prefs),
        "preferences": prefs.tolist(),
        "date": str(datetime.now().isoformat()),
    }
    if bt_result is not None:
        summary["bt_weights_file"] = f"{prefix_base}-bt-weights.json"

    with open(output_folder / f"{prefix_base}-summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "criteria": criteria,
        "summary": summary,
        "prefix": prefix_base,
        "scores_A": df_A,
        "scores_B": df_B,
        "preferences": prefs,
        "bt_result": bt_result,
    }
