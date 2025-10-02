"""
This script generates completions for a given dataset and model,
and then evaluates them using a judge model.
"""
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from llmjudgeeval.evaluate import annotate
from llmjudgeeval.utils import data_root, set_langchain_cache
from llmjudgeeval.utils import make_model, cache_function_dataframe


@dataclass
class CliArgs:
    dataset: str
    generation_model_A: str
    generation_model_B: str
    judge_model: str

    n_instructions: int | None = None
    provide_explanation: bool = False
    ignore_cache: bool = False

    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser(
            prog="Generate completion and evaluate with a judge",
        )
        parser.add_argument(
            "--dataset",
            help="The dataset to use",
        )
        parser.add_argument(
            "--generation_model_A",
            required=True,
            help="Name of the LLM to use for a generation, must be a valid choice for `generation_provider`",
        )
        parser.add_argument(
            "--generation_model_B",
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

        args = parser.parse_args()

        return cls(
            dataset=args.dataset,
            generation_model_A=args.generation_model_A,
            generation_model_B=args.generation_model_B,
            judge_model=args.judge_model,
            n_instructions=args.n_instructions,
            provide_explanation=args.provide_explanation,
            ignore_cache=args.ignore_cache,
        )


def load_contexts(dataset: str) -> pd.Series:
    path = data_root / "contexts" / dataset
    return pd.read_csv(path).loc[:, "instruction"]


def generate_base(
    instructions: pd.Series,
    model: str,
    n_instructions: int | None = None,
    max_len: int | None = 2000,
) -> pd.DataFrame:
    model = make_model(model, max_tokens=200)

    if n_instructions is not None:
        instructions = instructions[:n_instructions]

    def truncate(s: str, max_len: int | None = None):
        if max_len is not None:
            return s[:max_len]
        else:
            return s

    inputs = [truncate(instruction, max_len=max_len) for instruction in instructions]

    completions = model.batch(
        inputs=inputs,
        max_tokens=max_len,
    )

    df_outputs = pd.DataFrame(
        data={
            "completion": completions,
            "instruction_index": instructions.index.tolist(),
        },
    )

    return df_outputs


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

    # Not working with vllm, not detecting model changes and serving the same cache for two different models...
    if not args.ignore_cache:
        set_langchain_cache()

    instructions = load_contexts(args.dataset + ".csv")

    n_instructions = args.n_instructions if args.n_instructions else len(instructions)
    if args.n_instructions is not None:
        instructions = instructions[:n_instructions]

    print(
        f"Generating completions for dataset {args.dataset} with model {args.generation_model_A} and "
        f"{args.generation_model_B} (or loading them directly if present)"
    )

    ignore_cache = False
    completions_A = cache_function_dataframe(
        lambda: generate_base(
            instructions=instructions,
            model=args.generation_model_A,
            n_instructions=n_instructions,
            # system_prompt="Please complete the following text. Just output the completion and nothing else. Do not output more than 3 sentences.",
        ),
        ignore_cache=ignore_cache,
        cache_name=f"{args.dataset}_{args.generation_model_A}_{args.n_instructions}",
    ).set_index("instruction_index")
    completions_A = completions_A.loc[:, "completion"]
    completions_B = cache_function_dataframe(
        lambda: generate_base(
            instructions=instructions,
            model=args.generation_model_B,
            n_instructions=n_instructions,
            # system_prompt="Please complete the following text. Do not output more than 3 sentences.",
        ),
        ignore_cache=ignore_cache,
        cache_name=f"{args.dataset}_{args.generation_model_B}_{args.n_instructions}",
    ).set_index("instruction_index")
    completions_B = completions_B.loc[:, "completion"]
    print(f"\nFirst context: {instructions.values[0]}")

    print(f"\nFirst completion of {args.generation_model_A}")
    print(completions_A.values[0])
    print(f"\nFirst completion of {args.generation_model_B}")
    print(completions_B.values[0])
    print(f"Evaluating completions with judge {args.judge_model}.")

    judge_chat_model = make_model(
        model=args.judge_model,
        max_tokens=512,
    )
    system_prompt = """You are a highly efficient assistant, who evaluates and selects the best large language \
    model based on the quality of completion of a sentence. You will see a sentence to be completed and two \
    completions from Assistant A and Assistant B and will have to decide which one is best. Make sure to not \
    over-confidently prefer one assistant or the other and also make sure to not bias your preference based on \
    the ordering or on the length of the answers."""

    annotations = annotate(
        judge_chat_model=judge_chat_model,
        user_prompts=instructions.head(n_instructions).tolist(),
        completions_A=completions_A.head(n_instructions).tolist(),
        completions_B=completions_B.head(n_instructions).tolist(),
        provide_explanation=args.provide_explanation,
        system_prompt=system_prompt,
        max_len=200,
    )

    name = f"{args.dataset}-{args.generation_model_A}-{args.generation_model_B}-{args.judge_model}".replace(
        "/", "_"
    )
    output_path = Path("results") / f"{name}-annotations.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving annotations to {output_path}")
    df = pd.DataFrame(annotations)
    df["generation_model_A"] = args.generation_model_A
    df["generation_model_B"] = args.generation_model_B
    df.to_csv(output_path, index=False)

    prefs = pd.Series([annotation.preference for annotation in annotations])
    num_wins = sum(prefs < 0.5)
    num_losses = sum(prefs > 0.5)
    num_ties = sum(prefs == 0.5)
    num_battles = len(prefs)
    winrate = float(num_wins / num_battles)

    results = {
        "num_battles": num_battles,
        "winrate": winrate,
        "num_wins": num_wins,
        "num_losses": num_losses,
        "num_ties": num_ties,
        "preferences": prefs.tolist(),
    }
    print(
        f"{args.generation_model_A} vs {args.generation_model_B} judged by {args.judge_model}"
    )
    print(results)

    with open(output_path.parent / "args.json", "w") as f:
        json.dump(asdict(args), f, indent=2)

    with open(output_path.parent / "results-summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = CliArgs.parse_args()

    main(args)
