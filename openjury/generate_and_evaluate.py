"""
This script generates completions for a given dataset and model,
and then evaluates them using a judge model.
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from openjury.evaluate import annotate_battles, PairScore
from openjury.generate import generate_instructions, generate_base
from openjury.instruction_dataset import load_instructions
from openjury.utils import data_root
from openjury.utils import make_model, cache_function_dataframe


@dataclass
class CliArgs:
    dataset: str
    model_A: str
    model_B: str
    judge_model: str

    n_instructions: int | None = None
    provide_explanation: bool = False
    swap_mode: str = "fixed"
    ignore_cache: bool = False
    use_tqdm: bool = False
    max_len: int = 8192
    max_tokens: int = 32768

    result_folder: str = "results"

    def __post_init__(self):
        supported_modes = ["fixed", "both"]
        assert (
            self.swap_mode in supported_modes
        ), f"Only {supported_modes} modes are supported but got {self.swap_mode}."

    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser(
            prog="Generate completion and evaluate with a judge",
        )
        parser.add_argument(
            "--dataset",
            help="The dataset to use. For instance `alpaca-eval`, `arena-hard`, `m-arena-hard-EU` for instruction "
            "tuning cases or `french-contexts`, `spanish-contexts` for base models.",
        )
        parser.add_argument(
            "--model_A",
            required=True,
            help="Name of the LLM to use for a generation, must be a valid choice for `generation_provider`",
        )
        parser.add_argument(
            "--model_B",
            required=True,
            help="Name of the LLM to use for a generation, must be a valid choice for `generation_provider`",
        )
        parser.add_argument(
            "--judge_model",
            required=True,
            help="Name of the LLM to use, for instance `Together/meta-llama/Meta-Llama-3-70B-Instruct-Turbo`, "
            "`VLLM/meta-llama/Meta-Llama-3-70B-Instruct-Turbo`, `LangChain/LocalPath` etc",
        )
        parser.add_argument(
            "--n_instructions",
            type=int,
            required=False,
        )
        parser.add_argument(
            "--provide_explanation",
            action="store_true",
            help="If specified, judge will provide explanation before making a judgement. Does not necessarily improve"
            "the accuracy of the judge but enables some result interpretation.",
        )
        parser.add_argument(
            "--swap_mode",
            type=str,
            choices=["fixed", "both"],
            default="fixed",
            help="Model comparison order mode. 'fixed': always use model order A-B. 'both': correct for model order "
            "bias by evaluating each instruction twice, once as A-B and once as B-A, and average. This helps account "
            "for judge position bias. Default is 'fixed'.",
        )
        parser.add_argument(
            "--ignore_cache",
            action="store_true",
            help="If specified, ignore cache of previous completions.",
        )
        parser.add_argument(
            "--use_tqdm",
            action="store_true",
            help="If specified, use tqdm, does not work with all model providers, vLLM in particular.",
        )
        parser.add_argument(
            "--result_folder",
            type=str,
            required=False,
            default="results",
            help="The folder to save the results. Defaults to `results`. Evaluation results will be saved in"
            " `[result_folder]/[evaluation_name]`.",
        )
        parser.add_argument(
            "--max_len",
            type=int,
            required=False,
            default=8192,
            help="Maximum character length for truncating input text (instructions, completions) "
            "before sending to the model. Useful to avoid exceeding context limits. Defaults to 8192.",
        )
        parser.add_argument(
            "--max_tokens",
            type=int,
            required=False,
            default=32768,
            help="Maximum number of tokens all models (A, B, and judge) can generate in their responses. "
            "Defaults to 32768.",
        )
        args = parser.parse_args()

        return cls(
            dataset=args.dataset,
            model_A=args.model_A,
            model_B=args.model_B,
            judge_model=args.judge_model,
            n_instructions=args.n_instructions,
            provide_explanation=args.provide_explanation,
            swap_mode=args.swap_mode,
            ignore_cache=args.ignore_cache,
            use_tqdm=args.use_tqdm,
            max_len=args.max_len,
            max_tokens=args.max_tokens,
            result_folder=args.result_folder,
        )


def load_contexts(dataset: str) -> pd.Series:
    path = data_root / "contexts" / dataset
    return pd.read_csv(path).loc[:, "instruction"]


def print_results(results):
    """Print battle results in a nice formatted way"""

    print("\n" + "=" * 60)
    print("🏆 MODEL BATTLE RESULTS 🏆".center(60))
    print(f"📊 Dataset: {results['dataset']}")
    print(
        f"🤖 Competitors: Model A: {results['model_A']} vs Model B: {results['model_B']}"
    )
    print(f"⚖️ Judge: {results['judge_model']}")
    print(f"📈 Results Summary:")
    print(f"   Total Battles: {results['num_battles']}")
    print(f"   Win Rate (A): {results['winrate']:.1%}")
    print(f"   ✅ Wins:   {results['num_wins']}")
    print(f"   ❌ Losses: {results['num_losses']}")
    print(f"   🤝 Ties:   {results['num_ties']}")
    print("=" * 60 + "\n")


def main(args: CliArgs):
    """
    1) take as input:
     * dataset, make sure instruct-completion works
     * model to generate output from
     * llm used for judge
     * number of annotations
     * path to save annotations
    2) create completions
    3) create annotations
    """

    print(
        f"Using dataset {args.dataset} and evaluating models {args.model_A} and {args.model_B}."
    )

    # Not working with vllm, not detecting model changes and serving the same cache for two different models...
    # if not args.ignore_cache:
    #     set_langchain_cache()
    ignore_cache = args.ignore_cache

    # Currrently, we run context evaluation
    is_fluency_task = "fluency" in args.dataset
    if is_fluency_task:
        # if args.dataset = "fluency-french", we map to "french-contexts.csv"
        # to match files in https://huggingface.co/datasets/geoalgo/multilingual-contexts-to-be-completed
        lang = args.dataset.split("-")[-1]
        instructions = load_contexts(f"{lang}-contexts.csv")
    else:
        instructions = load_instructions(
            dataset=args.dataset, n_instructions=args.n_instructions
        ).loc[:, "instruction"]

    n_instructions = args.n_instructions if args.n_instructions else len(instructions)
    if args.n_instructions is not None:
        instructions = instructions[:n_instructions]

    print(
        f"Generating completions for dataset {args.dataset} with model {args.model_A} and "
        f"{args.model_B} (or loading them directly if present)"
    )

    # TODO currently we just support base models for fluency, we could also support instruction-tuned models
    gen_fun = (
        partial(generate_base, max_len=args.max_len, max_tokens=args.max_tokens)
        if is_fluency_task
        else partial(generate_instructions, max_len=args.max_len, max_tokens=args.max_tokens)
    )
    completions_A = cache_function_dataframe(
        lambda: gen_fun(
            instructions=instructions,
            model=args.model_A,
            use_tqdm=args.use_tqdm,
        ),
        ignore_cache=ignore_cache,
        cache_name=f"{args.dataset}_{args.model_A}_{args.n_instructions}",
    ).set_index("instruction_index")
    completions_A = completions_A.loc[:, "completion"]

    completions_B = cache_function_dataframe(
        lambda: gen_fun(
            instructions=instructions,
            model=args.model_B,
            use_tqdm=args.use_tqdm,
        ),
        ignore_cache=ignore_cache,
        cache_name=f"{args.dataset}_{args.model_B}_{args.n_instructions}",
    ).set_index("instruction_index")
    completions_B = completions_B.loc[:, "completion"]
    print(f"\nFirst instruction/context: {instructions.values[0]}")

    print(f"\nFirst completion of {args.model_A}")
    print(completions_A.values[0])
    print(f"\nFirst completion of {args.model_B}")
    print(completions_B.values[0])
    print(f"Evaluating completions with judge {args.judge_model}.")

    judge_chat_model = make_model(
        model=args.judge_model,
        max_tokens=args.max_tokens,
    )
    if is_fluency_task:
        system_prompt = """You are a highly efficient assistant, who evaluates and selects the best large language \
        model based on the quality of completion of a sentence. You will see a sentence to be completed and two \
        completions from Assistant A and Assistant B and will have to decide which one is best. Make sure to not \
        over-confidently prefer one assistant or the other and also make sure to not bias your preference based on \
        the ordering or on the length of the answers."""
    else:
        # the default system prompt of annotate is to compare instruction tuned models.

        system_prompt = None
    annotations = annotate_battles(
        judge_chat_model=judge_chat_model,
        instructions=instructions.head(n_instructions).tolist(),
        completions_A=completions_A.head(n_instructions).tolist(),
        completions_B=completions_B.head(n_instructions).tolist(),
        provide_explanation=args.provide_explanation,
        system_prompt=system_prompt,
        max_len=args.max_len,
        use_tqdm=args.use_tqdm,
    )

    if args.swap_mode == "both":
        print("Correction for judge bias towards a certain model position is set.")
        print(
            f"Evaluating completions with models reversed with judge {args.judge_model}."
        )
        annotations_reversed = annotate_battles(
            judge_chat_model=judge_chat_model,
            instructions=instructions.head(n_instructions).tolist(),
            completions_A=completions_B.head(n_instructions).tolist(),
            completions_B=completions_A.head(n_instructions).tolist(),
            provide_explanation=args.provide_explanation,
            system_prompt=system_prompt,
            max_len=args.max_len,
            use_tqdm=args.use_tqdm,
        )

    name = f"{args.dataset}-{args.model_A}-{args.model_B}-{args.judge_model}"
    name += f"-{args.swap_mode}"
    name = name.replace("/", "_")

    res_folder = Path(args.result_folder) / name
    res_folder.mkdir(parents=True, exist_ok=True)

    # save argument for results analysis
    with open(res_folder / f"args-{name}.json", "w") as f:
        json.dump(asdict(args), f, indent=2)

    print(f"Saving results to {res_folder}")
    df = pd.DataFrame(annotations)
    df["instruction_index"] = instructions.head(n_instructions).index.tolist()
    df["model_A"] = args.model_A
    df["model_B"] = args.model_B
    df["judge"] = args.judge_model

    if args.swap_mode == "both":
        df_reversed = pd.DataFrame(annotations_reversed)
        df_reversed["instruction_index"] = instructions.head(
            n_instructions
        ).index.tolist()
        df_reversed["model_A"] = args.model_B
        df_reversed["model_B"] = args.model_A
        df_reversed["judge"] = args.judge_model
        df = pd.concat([df, df_reversed])

    df.to_csv(res_folder / f"{name}-annotations.csv", index=False)

    # compute preferences between A and B
    score_parser = PairScore()
    prefs = pd.Series(
        [
            score_parser.parse_model_raw(annotation.judge_completion)
            for annotation in annotations
        ]
    )

    if args.swap_mode == "both":
        prefs_reversed = pd.Series(
            [
                score_parser.parse_model_raw(annotation.judge_completion)
                for annotation in annotations_reversed
            ]
        )
        prefs = pd.concat([prefs, (1 - prefs_reversed)]).reset_index(drop=True)

    # compute and report statistics
    num_wins = sum(prefs < 0.5)
    num_losses = sum(prefs > 0.5)
    num_ties = sum([1 if not x or x == 0.5 or x == np.nan else 0 for x in prefs])
    num_battles = len(prefs)
    winrate = float((num_wins + 0.5 * num_ties) / (num_ties + num_wins + num_losses))

    results = {
        "dataset": args.dataset,
        "model_A": args.model_A,
        "model_B": args.model_B,
        "judge_model": args.judge_model,
        "num_battles": num_battles,
        "winrate": winrate,
        "num_wins": num_wins,
        "num_losses": num_losses,
        "num_ties": num_ties,
        "num_missing": num_battles - (num_losses + num_wins + num_ties),
        "preferences": prefs.tolist(),
        "date": str(datetime.now().isoformat()),
        "user": os.getenv("USER", ""),
    }
    print(f"{args.model_A} vs {args.model_B} judged by {args.judge_model}")
    print_results(results)

    with open(res_folder / f"results-{name}.json", "w") as f:
        json.dump(results, f, indent=2)

    return prefs


if __name__ == "__main__":
    args = CliArgs.parse_args()

    print(f"Running with CLI args: {args.__dict__}")

    main(args)
