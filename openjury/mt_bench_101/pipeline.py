"""MT-Bench-101 evaluation pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from openjury.mt_bench_101.evaluate import (
    derive_mt_bench_101_pairwise_preferences,
    judge_mt_bench_101_single,
    summarize_mt_bench_101_absolute_scores,
    summarize_mt_bench_101_pairwise,
)
from openjury.mt_bench_101.generate import generate_mt_bench_101_completions
from openjury.utils import cache_function_dataframe, make_model

if TYPE_CHECKING:
    from openjury.generate_and_evaluate import CliArgs


def _generate_mt_bench_101_completions(
    args: CliArgs,
    eval_items_df: pd.DataFrame,
    ignore_cache: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def _run_generation(model_name: str) -> pd.DataFrame:
        return generate_mt_bench_101_completions(
            eval_items=eval_items_df,
            model=model_name,
            truncate_input_chars=args.truncate_all_input_chars,
            max_tokens=args.max_out_tokens_models,
            use_tqdm=args.use_tqdm,
            max_model_len=args.max_model_len,
            chat_template=args.chat_template,
        )

    completions_a = cache_function_dataframe(
        lambda: _run_generation(args.model_A),
        ignore_cache=ignore_cache,
        cache_name=f"mt-bench-101_{args.model_A}_{args.n_instructions}",
    ).set_index("instruction_index")

    completions_b = cache_function_dataframe(
        lambda: _run_generation(args.model_B),
        ignore_cache=ignore_cache,
        cache_name=f"mt-bench-101_{args.model_B}_{args.n_instructions}",
    ).set_index("instruction_index")
    return completions_a, completions_b


def _build_mt_bench_101_result_name(args: CliArgs, suffix: str | None = None) -> str:
    name = f"{args.dataset}-{args.model_A}-{args.model_B}-{args.judge_model}"
    if suffix:
        name += f"-{suffix}"
    return name.replace("/", "_")


def _save_mt_bench_101_results(
    *,
    args: CliArgs,
    results: dict[str, object],
    annotations_df: pd.DataFrame,
    name_suffix: str | None = None,
) -> None:
    name = _build_mt_bench_101_result_name(args, suffix=name_suffix)
    res_folder = Path(args.result_folder) / name
    res_folder.mkdir(parents=True, exist_ok=True)

    with open(res_folder / f"args-{name}.json", "w") as f:
        json.dump(asdict(args), f, indent=2)

    annotations_df.to_csv(res_folder / f"{name}-annotations.csv", index=False)

    with open(res_folder / f"results-{name}.json", "w") as f:
        json.dump(results, f, indent=2)


def run_mt_bench_101(args: CliArgs, ignore_cache: bool) -> pd.Series:
    """MT-Bench-101 pipeline with single-answer grading."""
    if args.mt_bench_compatibility or args.mt_bench_turns:
        print(
            "MT-Bench-101 is a different benchmark from original MT-Bench. "
            "--mt_bench_turns and --mt_bench_compatibility have no effect for this dataset, "
        )
    if args.swap_mode:
        print(
            "--swap_mode has no effect for mt-bench-101 since it does single answer grading before comparing the models"
        )

    from openjury import generate_and_evaluate as gae

    eval_items_df = gae.load_instructions(
        "mt-bench-101", n_instructions=args.n_instructions
    )
    print(
        "Generating completions from golden context for MT-Bench-101 with "
        f"{args.model_A} and {args.model_B}."
    )
    completions_a, completions_b = _generate_mt_bench_101_completions(
        args=args,
        eval_items_df=eval_items_df,
        ignore_cache=ignore_cache,
    )

    judge_chat_model = make_model(
        model=args.judge_model,
        max_tokens=args.max_out_tokens_judge,
        temperature=0.6,
        max_model_len=args.max_model_len,
        chat_template=args.chat_template,
    )
    scored_a = judge_mt_bench_101_single(
        judge_chat_model=judge_chat_model,
        eval_items=eval_items_df,
        completions=completions_a,
        truncate_input_chars=args.truncate_all_input_chars,
        use_tqdm=args.use_tqdm,
    )
    scored_b = judge_mt_bench_101_single(
        judge_chat_model=judge_chat_model,
        eval_items=eval_items_df,
        completions=completions_b,
        truncate_input_chars=args.truncate_all_input_chars,
        use_tqdm=args.use_tqdm,
    )

    absolute_a = summarize_mt_bench_101_absolute_scores(scored_turns=scored_a)
    absolute_b = summarize_mt_bench_101_absolute_scores(scored_turns=scored_b)
    pairwise_turns = derive_mt_bench_101_pairwise_preferences(
        scored_a=scored_a,
        scored_b=scored_b,
    )
    pairwise_summary = summarize_mt_bench_101_pairwise(pairwise_turns=pairwise_turns)
    dialogue_pairwise = pairwise_summary["dialogue_level"]

    print(f"{args.model_A} vs {args.model_B} judged by {args.judge_model}")
    print(
        "MT-Bench-101 dialogue-level pairwise winrate(A): "
        f"{dialogue_pairwise['winrate']:.1%}"
    )

    ann_cols = [
        "instruction_index",
        "dialogue_uid",
        "dialogue_id",
        "task",
        "ability",
        "turn_index",
        "model_completion",
        "judge_completion",
        "score",
    ]
    annotations_a = scored_a.loc[:, ann_cols].copy()
    annotations_a["evaluated_model"] = args.model_A
    annotations_b = scored_b.loc[:, ann_cols].copy()
    annotations_b["evaluated_model"] = args.model_B
    annotations_df = pd.concat([annotations_a, annotations_b], ignore_index=True)
    annotations_df = annotations_df.merge(
        pairwise_turns.loc[
            :, ["instruction_index", "score_A", "score_B", "preference"]
        ],
        on="instruction_index",
        how="left",
        validate="many_to_one",
    )

    results = {
        "dataset": args.dataset,
        "model_A": args.model_A,
        "model_B": args.model_B,
        "judge_model": args.judge_model,
        "judge_temperature": 0.6,
        "evaluation_mode": "single_answer_grading",
        "num_battles": dialogue_pairwise["num_battles"],
        "winrate": dialogue_pairwise["winrate"],
        "num_wins": dialogue_pairwise["num_wins"],
        "num_losses": dialogue_pairwise["num_losses"],
        "num_ties": dialogue_pairwise["num_ties"],
        "num_missing": dialogue_pairwise["num_missing"],
        "per_category": dialogue_pairwise["per_task"],
        "model_A_scores": absolute_a,
        "model_B_scores": absolute_b,
        "pairwise": pairwise_summary,
        "preferences": pairwise_summary["preferences"],
        "date": str(datetime.now().isoformat()),
        "user": os.getenv("USER", ""),
    }

    _save_mt_bench_101_results(
        args=args,
        results=results,
        annotations_df=annotations_df,
        name_suffix="mtbench_101",
    )
    return pd.Series(pairwise_summary["preferences"])
