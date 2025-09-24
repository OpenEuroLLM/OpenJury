"""
TODOs:
* function that generate predictions given dataset, model provider and model
* main with CLI
Done:
"""
import argparse
from pathlib import Path

import pandas as pd
from langchain.prompts import ChatPromptTemplate

from llmjudgeeval.instruction_dataset import load_instructions
from llmjudgeeval.utils import (
    do_inference,
    set_langchain_cache,
    make_model,
)


def generate(
    instructions: pd.Series,
    model: str,
    n_instructions: int | None = None,
    max_len: int | None = 2000,
    use_tqdm: bool = True,
    system_prompt: str | None = None,
) -> pd.DataFrame:
    chat_model = make_model(model)

    if n_instructions is not None:
        instructions = instructions[:n_instructions]

    # TODO improve prompt to generate instructions
    if system_prompt is None:
        system_prompt = (
            "You are an helpful assistant that answer queries asked by users."
        )
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{user_prompt}")]
    )

    def truncate(s: str, max_len: int | None = None):
        if max_len is not None:
            return s[:max_len]
        else:
            return s

    inputs = prompt_template.batch(
        [
            {
                "user_prompt": truncate(user_prompt, max_len=max_len),
            }
            for user_prompt in instructions
        ]
    )

    completions = do_inference(
        chat_model=chat_model,
        inputs=inputs,
        use_tqdm=use_tqdm,
    )
    df_outputs = pd.DataFrame(
        data={
            "completion": completions,
            "instruction_index": instructions.index.tolist(),
        },
    )
    return df_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="alpaca-eval")
    parser.add_argument(
        "--model", type=str, default="Together/meta-llama/Llama-3.2-3B-Instruct-Turbo"
    )
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--n_instructions", type=int, default=10)
    parser.add_argument(
        "--ignore_cache",
        action="store_true",
        help="If specified, will ignore langchain cache and regenerate all requests even those which were previously"
        " submitted",
    )
    parser.add_argument(
        "--use_tqdm",
        action="store_true",
        help="Option to activate tqdm, does not always work with some backend like LlamaCpp",
    )
    args = parser.parse_args()

    if not args.ignore_cache:
        set_langchain_cache()
    instructions = load_instructions(
        dataset=args.dataset, n_instructions=args.n_instructions
    )

    generate(
        instructions=instructions,
        model=args.model,
        n_instructions=args.n_instructions,
        use_tqdm=args.use_tqdm,
    )

    # TODO save in output_path


if __name__ == "__main__":
    main()
