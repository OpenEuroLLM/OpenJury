"""
TODOs:
* function that generate predictions given dataset, model provider and model
* main with CLI
Done:
"""
from pathlib import Path

import pandas as pd
from langchain_community.llms import LlamaCpp, Together
from langchain.prompts import ChatPromptTemplate
from llmjudgeeval.utils import (
    load_instructions,
    do_inference,
    data_root,
    set_langchain_cache,
)


def make_model(model_provider: str, **kwargs):
    assert model_provider in ["LlamaCpp", "Together"]
    model_classes = [LlamaCpp, Together]
    model_cls_dict = {model_cls.__name__: model_cls for model_cls in model_classes}
    return model_cls_dict[model_provider](**kwargs)


def generate(
    dataset: str,
    model_provider: str,
    model_kwargs: dict[str, str],
    output_path: Path = None,
    n_instructions: int | None = None,
):
    max_len = 2000
    use_tqdm = False

    chat_model = make_model(model_provider=model_provider, **model_kwargs)
    instructions = load_instructions(dataset=dataset)

    if n_instructions is not None:
        instructions = instructions[:n_instructions]

    # TODO prompt to generate instructions
    system_prompt = "You are an helpful assistant that answer queries asked by users."
    user_prompt_template = "{user_prompt}"
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", user_prompt_template)]
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
        output_path = data_root / "model-completions"
    output_file = output_path / f"output.csv.zip"
    # TODO store model_kwargs and other in metadata.json in path?
    print(
        f"Saving {len(completions)} completions from {model_provider} {model_kwargs} to {output_file}."
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        data={"output": completions, "instruction_index": instructions.index.tolist()},
    ).to_csv(output_file, index=False)


def main():
    # TODO parse dataset, model_provider, model_kwargs, output_path, n_instructions from CLI with argparse.
    set_langchain_cache()

    model_path = "/Users/salinasd/Library/Caches/llama.cpp/hugging-quants_Llama-3.2-3B-Instruct-Q8_0-GGUF_llama-3.2-3b-instruct-q8_0.gguf"
    # model_path = "/Users/salinasd/Library/Caches/llama.cpp/jwiggerthale_Llama-3.2-3B-Q8_0-GGUF_llama-3.2-3b-q8_0.gguf"
    generate(
        dataset="alpaca-eval",
        model_provider="LlamaCpp",
        model_kwargs={
            "model_path": model_path,
        },
        output_path=Path(
            "results/hugging-quants_Llama-3.2-3B-Instruct-Q8_0-GGUF_llama-3.2-3b-instruct-q8_0.csv.zip"
        ),
        n_instructions=10,
    )


if __name__ == "__main__":
    main()
