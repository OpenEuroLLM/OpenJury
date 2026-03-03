"""
This script generates completions for two models and scores them samplewise
with a criteria-based judge.
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd

from openjury.criteria.pipeline import run_samplewise_criteria_pipeline
from openjury.generate import generate_base, generate_instructions
from openjury.generate_and_evaluate import load_contexts, try_load_dataset_completions
from openjury.instruction_dataset import load_instructions
from openjury.utils import cache_function_dataframe, make_model


@dataclass
class CliArgs:
    dataset: str
    model_A: str
    model_B: str
    judge_model: str

    n_instructions: int | None = None
    provide_explanation: bool = False
    ignore_cache: bool = False
    use_tqdm: bool = False
    truncate_all_input_chars: int = 8192
    max_out_tokens_models: int = 32768
    max_out_tokens_judge: int = 32768
    max_model_len: int | None = None
    chat_template: str | None = None
    criteria_name: str = "default"
    criteria_file: str | None = None
    result_folder: str = "results"

    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser(
            prog="Generate completion and score with criteria",
        )
        parser.add_argument("--dataset")
        parser.add_argument("--model_A", required=True)
        parser.add_argument("--model_B", required=True)
        parser.add_argument("--judge_model", required=True)
        parser.add_argument("--n_instructions", type=int, required=False)
        parser.add_argument("--provide_explanation", action="store_true")
        parser.add_argument("--ignore_cache", action="store_true")
        parser.add_argument("--use_tqdm", action="store_true")
        parser.add_argument("--result_folder", type=str, default="results")
        parser.add_argument(
            "--truncate_all_input_chars",
            type=int,
            default=8192,
            help="Character-level truncation applied before tokenization for generation.",
        )
        parser.add_argument(
            "--max_out_tokens_models",
            type=int,
            default=32768,
            help="Generation token budget for each model A/B response.",
        )
        parser.add_argument(
            "--max_out_tokens_judge",
            type=int,
            default=32768,
            help="Generation token budget for the criteria judge response.",
        )
        parser.add_argument("--max_model_len", type=int, default=None)
        parser.add_argument("--chat_template", type=str, default=None)
        parser.add_argument(
            "--criteria_name",
            type=str,
            default="default",
            help="Built-in criteria name to use. Ignored if --criteria_file is provided.",
        )
        parser.add_argument(
            "--criteria_file",
            type=str,
            default=None,
            help="Optional path to a custom criteria file (.json/.yaml/.yml).",
        )
        args = parser.parse_args()

        return cls(
            dataset=args.dataset,
            model_A=args.model_A,
            model_B=args.model_B,
            judge_model=args.judge_model,
            n_instructions=args.n_instructions,
            provide_explanation=args.provide_explanation,
            ignore_cache=args.ignore_cache,
            use_tqdm=args.use_tqdm,
            truncate_all_input_chars=args.truncate_all_input_chars,
            max_out_tokens_models=args.max_out_tokens_models,
            max_out_tokens_judge=args.max_out_tokens_judge,
            max_model_len=args.max_model_len,
            chat_template=args.chat_template,
            criteria_name=args.criteria_name,
            criteria_file=args.criteria_file,
            result_folder=args.result_folder,
        )


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


def main(args: CliArgs):
    print(
        f"Using dataset {args.dataset} and scoring models {args.model_A} and {args.model_B} with criteria."
    )

    is_fluency_task = "fluency" in args.dataset
    if is_fluency_task:
        lang = args.dataset.split("-")[-1]
        instructions = load_contexts(f"{lang}-contexts.csv")
    else:
        instructions = load_instructions(
            dataset=args.dataset,
            n_instructions=args.n_instructions,
        ).loc[:, "instruction"]

    n_instructions = args.n_instructions if args.n_instructions else len(instructions)
    if args.n_instructions is not None:
        instructions = instructions[:n_instructions]

    gen_fun = (
        partial(
            generate_base,
            truncate_input_chars=args.truncate_all_input_chars,
            max_tokens=args.max_out_tokens_models,
            max_model_len=args.max_model_len,
            chat_template=args.chat_template,
        )
        if is_fluency_task
        else partial(
            generate_instructions,
            truncate_input_chars=args.truncate_all_input_chars,
            max_tokens=args.max_out_tokens_models,
            chat_template=args.chat_template,
            max_model_len=args.max_model_len,
        )
    )

    dataset_completions_A = try_load_dataset_completions(
        args.dataset, args.model_A, n_instructions
    )
    if dataset_completions_A is not None:
        completions_A = dataset_completions_A.set_index("instruction_index").loc[:, "completion"]
    else:
        completions_A = cache_function_dataframe(
            lambda: gen_fun(
                instructions=instructions,
                model=args.model_A,
                use_tqdm=args.use_tqdm,
            ),
            ignore_cache=args.ignore_cache,
            cache_name=f"{args.dataset}_{args.model_A}_{args.n_instructions}",
        ).set_index("instruction_index").loc[:, "completion"]

    dataset_completions_B = try_load_dataset_completions(
        args.dataset, args.model_B, n_instructions
    )
    if dataset_completions_B is not None:
        completions_B = dataset_completions_B.set_index("instruction_index").loc[:, "completion"]
    else:
        completions_B = cache_function_dataframe(
            lambda: gen_fun(
                instructions=instructions,
                model=args.model_B,
                use_tqdm=args.use_tqdm,
            ),
            ignore_cache=args.ignore_cache,
            cache_name=f"{args.dataset}_{args.model_B}_{args.n_instructions}",
        ).set_index("instruction_index").loc[:, "completion"]

    judge_chat_model = make_model(
        model=args.judge_model,
        max_tokens=args.max_out_tokens_judge,
        max_model_len=args.max_model_len,
        chat_template=args.chat_template,
    )

    name = f"{args.dataset}-{args.model_A}-{args.model_B}-{args.judge_model}"
    name = name.replace("/", "_")

    res_folder = Path(args.result_folder) / name
    res_folder.mkdir(parents=True, exist_ok=True)

    with open(res_folder / f"args-{name}.json", "w") as f:
        json.dump(asdict(args), f, indent=2)

    run_info = run_samplewise_criteria_pipeline(
        output_folder=res_folder,
        output_prefix=name,
        judge_model=judge_chat_model,
        instructions=instructions.head(n_instructions).tolist(),
        completions_A=completions_A.head(n_instructions).tolist(),
        completions_B=completions_B.head(n_instructions).tolist(),
        instruction_index=instructions.head(n_instructions).index.tolist(),
        model_A_name=args.model_A,
        model_B_name=args.model_B,
        provide_explanation=args.provide_explanation,
        use_tqdm=args.use_tqdm,
        criteria_name=args.criteria_name,
        criteria_file=args.criteria_file,
        summary_fields={
            "dataset": args.dataset,
            "model_A": args.model_A,
            "model_B": args.model_B,
            "judge_model": args.judge_model,
        },
    )

    prefs = pd.Series(run_info["preferences"], dtype="float64")
    results = {
        "dataset": args.dataset,
        "model_A": args.model_A,
        "model_B": args.model_B,
        "judge_model": args.judge_model,
        **_compute_pref_summary(prefs),
        "preferences": prefs.tolist(),
        "date": str(datetime.now().isoformat()),
        "user": os.getenv("USER", ""),
        "scoring_mode": "criteria_samplewise",
        "criteria_name": run_info["criteria_name"],
    }

    with open(res_folder / f"results-{name}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"Saved criteria outputs to {res_folder} "
        f"(prefix: {run_info['prefix']})."
    )
    return prefs


def cli():
    args = CliArgs.parse_args()
    print(f"Running with CLI args: {args.__dict__}")
    main(args)


if __name__ == "__main__":
    cli()
