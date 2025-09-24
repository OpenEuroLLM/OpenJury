"""
This script generates completions for a given dataset and model,
and then evaluates them using a judge model.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from llmjudgeeval.evaluate import annotate
from llmjudgeeval.generate import generate
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

    if not args.ignore_cache:
        set_langchain_cache()

    instructions = load_contexts(args.dataset + ".csv")
    if args.n_instructions is not None:
        instructions = instructions[: args.n_instructions]

    print(
        f"Generating completions for dataset {args.dataset} with model {args.generation_model_A} and "
        f"{args.generation_model_B} (or loading them directly if present)"
    )
    # TODO check if local file present
    completions_A = cache_function_dataframe(
        lambda: generate(
            instructions=instructions,
            model=args.generation_model_A,
            n_instructions=args.n_instructions,
            use_tqdm=False,
        ),
        ignore_cache=True,
        cache_name=args.dataset + "_" + args.generation_model_A,
    ).set_index("instruction_index")

    completions_B = cache_function_dataframe(
        lambda: generate(
            instructions=instructions,
            model=args.generation_model_B,
            n_instructions=args.n_instructions,
            use_tqdm=False,
        ),
        ignore_cache=True,
        cache_name=args.dataset + "_" + args.generation_model_B,
    ).set_index("instruction_index")

    print(f"Evaluating completions with judge {args.judge_model}.")
    annotations = generate_judge_annotations(
        instructions=instructions.tolist(),
        completions_A=completions_A.loc[instructions.index, "completion"].tolist(),
        completions_B=completions_B.loc[instructions.index, "completion"].tolist(),
        judge_model=args.judge_model,
        n_instructions=args.n_instructions,
        provide_explanation=args.provide_explanation,
    )

    name = (
        f"{args.dataset}-{args.generation_model_A}-{args.generation_model_B}".replace(
            "/", "_"
        )
    )
    output_path = Path("results") / f"{name}-annotations.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving annotations to {output_path}")
    pd.DataFrame(annotations).to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
