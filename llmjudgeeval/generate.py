"""
TODOs:
* function that generate predictions given dataset, model provider and model
* main with CLI
Done:
"""
import argparse
import json
from ast import literal_eval
from pathlib import Path

import pandas as pd
from langchain.prompts import ChatPromptTemplate

from llmjudgeeval.instruction_dataset import load_instructions
from llmjudgeeval.utils import (
    do_inference,
    data_root,
    set_langchain_cache,
    make_model,
)


def generate(
    dataset: str,
    model_provider: str,
    model_kwargs: dict[str, str],
    output_path: Path = None,
    n_instructions: int | None = None,
    max_len: int | None = 2000,
    use_tqdm: bool = True,
    system_prompt: str | None = None,
):
    instructions = load_instructions(dataset=dataset)
    chat_model = make_model(model_provider=model_provider, **model_kwargs)

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

    print(completions)
    if output_path is None:
        output_path = (
            data_root / "model-completions" / model_provider / "model_output.csv.zip"
        )

    # TODO store model_kwargs and other in metadata.json in path?
    print(
        f"Saving {len(completions)} completions from {model_provider} {model_kwargs} to {output_path}."
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_outputs = pd.DataFrame(
        data={"output": completions, "instruction_index": instructions.index.tolist()},
    )
    df_outputs.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument(
        "--model_provider",
        type=str,
        choices=["LlamaCpp", "Together", "ChatOpenAI", "VLLM"],
    )
    parser.add_argument(
        "--model_kwargs",
        nargs="+",
        help="List of key-value pairs, e.g., --model_kwargs model_path=~/Llama-3.2-3B-Instruct-q8_0.gguf",
    )
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--n_instructions", type=int, default=None)
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

    model_kwargs_dict = {}
    if args.model_kwargs:
        for pair in args.model_kwargs:
            key, value = pair.split("=")
            model_kwargs_dict[key] = value

    if not args.ignore_cache:
        set_langchain_cache()

    generate(
        dataset=args.dataset,
        model_provider=args.model_provider,
        model_kwargs=model_kwargs_dict,
        output_path=args.output_path,
        n_instructions=args.n_instructions,
        use_tqdm=args.use_tqdm,
    )


if __name__ == "__main__":
    main()
