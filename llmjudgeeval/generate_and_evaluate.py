"""
This script generates completions for a given dataset and model,
and then evaluates them using a judge model.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from llmjudgeeval.evaluate import annotate
from llmjudgeeval.generate import generate, generate_base
from llmjudgeeval.utils import make_model, set_langchain_cache, cache_function_dataframe


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


def generate_judge_annotations(
    instructions: list[str],
    completions_A: list[str],
    completions_B: list[str],
    judge_model: str,
    n_instructions: int | None = None,
    provide_explanation: bool = False,
) -> list:
    print(f"Generating annotations with judge {judge_model}")

    # Generate completions for a reference model

    judge_chat_model = make_model(model=judge_model)

    annotations = annotate(
        judge_chat_model=judge_chat_model,
        user_prompts=instructions,
        completions_A=completions_A,
        completions_B=completions_B,
        num_annotations=n_instructions,
        provide_explanation=provide_explanation,
    )
    return annotations


def load_contexts(dataset: str) -> pd.Series:
    path = Path(__file__).parent.parent / "data" / dataset
    return pd.read_csv(path).loc[:, "instruction"]


def main():
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
    args = CliArgs.parse_args()

    # Not working with vllm, not detecting model changes and serving the same cache for two different models...
    # if not args.ignore_cache:
    #     set_langchain_cache()

    instructions = load_contexts(args.dataset + ".csv")

    n_instructions = args.n_instructions if args.n_instructions else len(instructions)
    if args.n_instructions is not None:
        instructions = instructions[:n_instructions]

    print(
        f"Generating completions for dataset {args.dataset} with model {args.generation_model_A} and "
        f"{args.generation_model_B} (or loading them directly if present)"
    )
    # TODO check if local file present
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

    judge_chat_model = make_model(model=args.judge_model, max_len=200)
    system_prompt = """You are a highly efficient assistant, who evaluates and selects the best large language \
    model based on the quality of completion of a sentence. You will be a sentence to be completed and two \
    completions from Assistant A and Assistant B and will have to decide which one was best. Make sure to not \
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
    pd.DataFrame(annotations).to_csv(output_path, index=False)

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
    }
    print(
        f"{args.generation_model_A} vs {args.generation_model_B} judged by {args.judge_model}"
    )
    print(results)
    print([annotation.preference for annotation in annotations])


if __name__ == "__main__":
    main()
