import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from langchain.prompts import ChatPromptTemplate

from openjury.evaluate import PairScore
from openjury.instruction_dataset.mt_bench_101 import (
    MT_BENCH_101_REFERENCE_TASKS,
    MT_BENCH_101_TASK_TO_ABILITY,
)
from openjury.utils import do_inference, safe_text

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts" / "mt_bench_101"
DOUBLE_BRACKET_PATTERN = re.compile(r"\[\[(\d+)\]\]")

TASK_PROMPT_FILES = {
    "CM": "CM.txt",
    "AR": "AR.txt",
    "SI": "SI.txt",
    "TS": "TS.txt",
    "CC": "CC.txt",
    "CR": "rephrasing.txt",
    "FR": "rephrasing.txt",
    "SC": "SC.txt",
    "SA": "SA.txt",
    "MR": "MR.txt",
    "GR": "GR.txt",
    "IC": "IC.txt",
    "PI": "PI.txt",
}


@lru_cache(maxsize=1)
def load_mt_bench_101_prompts() -> dict[str, object]:
    global_system = (PROMPTS_DIR / "global_system.txt").read_text()
    scoring_format = (PROMPTS_DIR / "scoring_format.txt").read_text()
    task_prompts = {
        task: (PROMPTS_DIR / prompt_file).read_text()
        for task, prompt_file in TASK_PROMPT_FILES.items()
    }
    return {
        "global_system": global_system,
        "scoring_format": scoring_format,
        "task_prompts": task_prompts,
    }


def parse_mt_bench_101_rating(judge_completion: str) -> float | None:
    for match in DOUBLE_BRACKET_PATTERN.finditer(judge_completion):
        score = int(match.group(1))
        if 1 <= score <= 10:
            return float(score)
    return None


def format_mt_bench_101_dialogue(
    *,
    golden_context: list[dict[str, str]],
    user_message: str,
    assistant_message: str,
) -> str:
    chunks: list[str] = []
    for turn in golden_context:
        chunks.append(
            f"\n\n Human: {turn.get('user', '')}\n\nAssistant: {turn.get('bot', '')}"
        )
    chunks.append(f"\n\n Human: {user_message}\n\nAssistant: {assistant_message}")
    return "".join(chunks)


def judge_mt_bench_101_single(
    *,
    judge_chat_model,
    eval_items: pd.DataFrame,
    completions: pd.DataFrame,
    truncate_input_chars: int | None = 8192,
    use_tqdm: bool = False,
) -> pd.DataFrame:
    prompts = load_mt_bench_101_prompts()
    completion_by_idx = (
        completions
        if "instruction_index" not in completions.columns
        else completions.set_index("instruction_index")
    )

    rows: list[dict[str, object]] = []
    for idx in eval_items.index:
        eval_row = eval_items.loc[idx]
        completion_row = completion_by_idx.loc[idx]
        task = str(eval_row["task"])
        model_response = safe_text(
            completion_row.get("completion", ""),
            truncate_input_chars,
        )

        dialogue = format_mt_bench_101_dialogue(
            golden_context=eval_row.get("golden_context") or [],
            user_message=safe_text(eval_row.get("user_message", ""), truncate_input_chars),
            assistant_message=model_response,
        )

        user_prompt = (
            "The dialogue need to be judged is: \n *** \n "
            f"{dialogue} \n ***"
        )
        if task in MT_BENCH_101_REFERENCE_TASKS:
            reference_answer = safe_text(
                eval_row.get("reference_answer"),
                truncate_input_chars,
            )
            user_prompt += (
                "\n\nThe reference solution is: \n ### \n "
                f"{reference_answer} \n ###\n\n"
            )

        system_prompt = (
            f"{prompts['global_system']}\n\n"
            f"{prompts['task_prompts'][task]}\n\n"
            f"{prompts['scoring_format']}"
        ).strip()

        rows.append(
            {
                "instruction_index": idx,
                "dialogue_id": eval_row["dialogue_id"],
                "dialogue_uid": eval_row["dialogue_uid"],
                "task": task,
                "ability": eval_row.get("ability", MT_BENCH_101_TASK_TO_ABILITY[task]),
                "turn_index": eval_row["turn_index"],
                "model_completion": model_response,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", "{system_prompt}"), ("user", "{user_prompt}")]
    )
    inputs = prompt_template.batch(
        [
            {"system_prompt": row["system_prompt"], "user_prompt": row["user_prompt"]}
            for row in rows
        ]
    )
    judge_completions = do_inference(
        chat_model=judge_chat_model,
        inputs=inputs,
        use_tqdm=use_tqdm,
    )

    for row, judge_completion in zip(rows, judge_completions):
        row["judge_completion"] = judge_completion
        row["score"] = parse_mt_bench_101_rating(judge_completion)

    return pd.DataFrame(rows)


def compute_mt_bench_101_dialogue_scores(scored_turns: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        scored_turns.groupby(["dialogue_uid", "dialogue_id", "task", "ability"], as_index=False)[
            "score"
        ].min()
    )
    grouped = grouped.rename(columns={"score": "dialogue_score"})
    return grouped


def summarize_mt_bench_101_absolute_scores(scored_turns: pd.DataFrame) -> dict[str, object]:
    dialogue_scores = compute_mt_bench_101_dialogue_scores(scored_turns=scored_turns)
    per_task_series = dialogue_scores.groupby("task")["dialogue_score"].mean().sort_index()
    per_ability_series = (
        dialogue_scores.groupby("ability")["dialogue_score"].mean().sort_index()
    )
    overall = per_task_series.mean() if len(per_task_series) else float("nan")

    return {
        "num_turns": int(len(scored_turns)),
        "num_scored_turns": int(scored_turns["score"].notna().sum()),
        "per_task": {
            task: float(score)
            for task, score in per_task_series.items()
            if pd.notna(score)
        },
        "per_ability": {
            ability: float(score)
            for ability, score in per_ability_series.items()
            if pd.notna(score)
        },
        "overall": float(overall) if pd.notna(overall) else None,
    }


def derive_mt_bench_101_pairwise_preferences(
    scored_a: pd.DataFrame,
    scored_b: pd.DataFrame,
) -> pd.DataFrame:
    cols = ["instruction_index", "dialogue_uid", "dialogue_id", "task", "ability", "turn_index"]
    merged = scored_a.loc[:, cols + ["score"]].rename(columns={"score": "score_A"}).merge(
        scored_b.loc[:, cols + ["score"]].rename(columns={"score": "score_B"}),
        on=cols,
        how="inner",
    )

    scorer = PairScore()
    preferences = []
    for _, row in merged.iterrows():
        score_a = row["score_A"]
        score_b = row["score_B"]
        if pd.isna(score_a) or pd.isna(score_b):
            preferences.append(None)
            continue
        preferences.append(float(scorer.preference_from_scores(score_a, score_b)))
    merged["preference"] = preferences
    return merged


def _compute_preference_stats(preferences: pd.Series) -> dict[str, float | int]:
    tie_tol = 1e-12
    valid = preferences.dropna()
    num_wins = int(sum(valid < 0.5 - tie_tol))
    num_losses = int(sum(valid > 0.5 + tie_tol))
    num_ties = int(sum((valid >= 0.5 - tie_tol) & (valid <= 0.5 + tie_tol)))
    num_battles = len(preferences)
    num_missing = num_battles - (num_wins + num_losses + num_ties)
    denom = num_wins + num_losses + num_ties
    winrate = float((num_wins + 0.5 * num_ties) / denom) if denom else 0.0
    return {
        "num_battles": num_battles,
        "num_wins": num_wins,
        "num_losses": num_losses,
        "num_ties": num_ties,
        "num_missing": num_missing,
        "winrate": winrate,
    }


def _grouped_preference_stats(
    pairwise_df: pd.DataFrame,
    group_by: str,
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[float]] = {}
    for _, row in pairwise_df.iterrows():
        key = row[group_by]
        grouped.setdefault(key, []).append(row["preference"])
    return {
        key: _compute_preference_stats(pd.Series(values))
        for key, values in grouped.items()
    }


def summarize_mt_bench_101_pairwise(pairwise_turns: pd.DataFrame) -> dict[str, object]:
    turn_preferences = pairwise_turns["preference"]
    turn_level = {
        **_compute_preference_stats(turn_preferences),
        "per_task": _grouped_preference_stats(pairwise_turns, "task"),
        "per_ability": _grouped_preference_stats(pairwise_turns, "ability"),
    }

    dialogue_scores = (
        pairwise_turns.groupby(["dialogue_uid", "dialogue_id", "task", "ability"], as_index=False)[
            ["score_A", "score_B"]
        ].min()
    )
    scorer = PairScore()
    dialogue_scores["preference"] = [
        float(scorer.preference_from_scores(score_a, score_b))
        if pd.notna(score_a) and pd.notna(score_b)
        else None
        for score_a, score_b in zip(dialogue_scores["score_A"], dialogue_scores["score_B"])
    ]
    dialogue_level = {
        **_compute_preference_stats(dialogue_scores["preference"]),
        "per_task": _grouped_preference_stats(dialogue_scores, "task"),
        "per_ability": _grouped_preference_stats(dialogue_scores, "ability"),
    }

    return {
        "turn_level": turn_level,
        "dialogue_level": dialogue_level,
        "preferences": [None if pd.isna(x) else float(x) for x in turn_preferences],
    }
