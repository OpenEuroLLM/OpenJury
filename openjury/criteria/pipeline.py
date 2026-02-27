"""Shared criteria scoring pipeline helpers."""

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


def run_samplewise_criteria_pipeline(
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
    summary_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run samplewise criteria scoring and save outputs.

    This helper currently runs sample-wise scoring for model A and model B
    independently, then derives preferences from weighted criterion averages.
    The function name is kept stable for call-site compatibility.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    criteria = resolve_criteria(criteria_name=criteria_name, criteria_file=criteria_file)
    scorer = CriteriaScorer(
        judge_model=judge_model,
        criteria=criteria,
        provide_explanation=provide_explanation,
    )
    scores_A = scorer.score(
        instructions=instructions,
        completions=completions_A,
        model_name=model_A_name,
        use_tqdm=use_tqdm,
    )
    scores_B = scorer.score(
        instructions=instructions,
        completions=completions_B,
        model_name=model_B_name,
        use_tqdm=use_tqdm,
    )
    df_A, df_B, prefs = scorer.samplewise_to_dataframes(
        scores_A=scores_A,
        scores_B=scores_B,
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

    comparison_rows = []
    for inst_idx, row_A, row_B, pref in zip(instruction_index, scores_A, scores_B, prefs.tolist()):
        row = {
            "instruction_index": inst_idx,
            "preference": pref,
            "raw_judge_output_A": row_A.raw_judge_output,
            "raw_judge_output_B": row_B.raw_judge_output,
        }
        for criterion_name in criteria.criterion_names:
            row[f"A_{criterion_name}"] = row_A.scores.get(criterion_name, float("nan"))
            row[f"B_{criterion_name}"] = row_B.scores.get(criterion_name, float("nan"))
        comparison_rows.append(row)
    # Comparison table derived from independent samplewise scores.
    pd.DataFrame(comparison_rows).to_csv(
        output_folder / f"{prefix_base}-comparison.csv",
        index=False,
    )

    summary = {
        **(summary_fields or {}),
        "criteria_name": criteria.name,
        "criterion_names": criteria.criterion_names,
        "scoring_mode": "samplewise",
        "swap_debiasing": False,
        "swap_debiasing_requested": bool(swap_to_debias),
        **_compute_pref_summary(prefs),
        "preferences": prefs.tolist(),
        "date": str(datetime.now().isoformat()),
    }

    with open(output_folder / f"{prefix_base}-summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "criteria": criteria,
        "summary": summary,
        "prefix": prefix_base,
        "scores_A": df_A,
        "scores_B": df_B,
        "preferences": prefs,
    }
