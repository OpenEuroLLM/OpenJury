"""
This script generates completions for a given dataset and model,
and then evaluates them using a judge model.
"""
import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from openjury.evaluate import annotate
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
    ignore_cache: bool = False
    use_tqdm: bool = False

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
            "--ignore_cache",
            action="store_true",
            help="If specified, ignore cache of previous completions.",
        )
        parser.add_argument(
            "--use_tqdm",
            action="store_true",
            help="If specified, use tqdm, does not work with all model providers.",
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
        )


def load_contexts(dataset: str) -> pd.Series:
    path = data_root / "contexts" / dataset
    return pd.read_csv(path).loc[:, "instruction"]


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

    is_base_model = "context" in args.dataset
    if is_base_model:
        print(
            f"Using dataset {args.dataset} and evaluating base models {args.model_A} and {args.model_B}."
        )
    else:
        print(
            f"Using dataset {args.dataset} and instruction-tuned models {args.model_A} and {args.model_B}."
        )

    # Not working with vllm, not detecting model changes and serving the same cache for two different models...
    # # if not args.ignore_cache:
    # #     set_langchain_cache()
    ignore_cache = args.ignore_cache

    if is_base_model:
        instructions = load_contexts(args.dataset + ".csv")
    else:
        instructions = load_instructions(
            dataset=args.dataset, n_instructions=args.n_instructions
        )

    n_instructions = args.n_instructions if args.n_instructions else len(instructions)
    if args.n_instructions is not None:
        instructions = instructions[:n_instructions]

    print(
        f"Generating completions for dataset {args.dataset} with model {args.model_A} and "
        f"{args.model_B} (or loading them directly if present)"
    )

    gen_fun = generate_base if is_base_model else generate_instructions
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
        max_tokens=512,
    )
    if is_base_model:
        system_prompt = """You are a highly efficient assistant, who evaluates and selects the best large language \
        model based on the quality of completion of a sentence. You will see a sentence to be completed and two \
        completions from Assistant A and Assistant B and will have to decide which one is best. Make sure to not \
        over-confidently prefer one assistant or the other and also make sure to not bias your preference based on \
        the ordering or on the length of the answers."""
    else:
        # the default system prompt of annotate is to compare instruction tuned models.

        system_prompt = None
    annotations = annotate(
        judge_chat_model=judge_chat_model,
        user_prompts=instructions.head(n_instructions).tolist(),
        completions_A=completions_A.head(n_instructions).tolist(),
        completions_B=completions_B.head(n_instructions).tolist(),
        provide_explanation=args.provide_explanation,
        system_prompt=system_prompt,
        max_len=200,
        use_tqdm=args.use_tqdm,
    )

    name = f"{args.dataset}-{args.model_A}-{args.model_B}-{args.judge_model}".replace(
        "/", "_"
    )

    date = datetime.now().isoformat()

    res_folder = data_root / "results" / date
    res_folder.mkdir(parents=True, exist_ok=True)

    print(f"Saving results to {res_folder}")
    df = pd.DataFrame(annotations)
    df["instruction_index"] = instructions.head(n_instructions).index.tolist()
    df["model_A"] = args.model_A
    df["model_B"] = args.model_B
    df.to_csv(res_folder / f"{name}-annotations.csv", index=False)

    prefs = pd.Series([annotation.preference for annotation in annotations])
    num_wins = sum(prefs < 0.5)
    num_losses = sum(prefs > 0.5)
    num_ties = sum(prefs == 0.5)
    num_battles = len(prefs)
    winrate = float(num_wins / num_battles)

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
        "preferences": prefs.tolist(),
        "date": date,
        "user": os.getenv("USER", ""),
    }
    print(f"{args.model_A} vs {args.model_B} judged by {args.judge_model}")
    print(results)

    with open(res_folder / f"args-{name}.json", "w") as f:
        json.dump(asdict(args), f, indent=2)

    with open(res_folder / f"results-{name}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = CliArgs.parse_args()

    print(f"Running with CLI args: {args.__dict__}")

    main(args)
