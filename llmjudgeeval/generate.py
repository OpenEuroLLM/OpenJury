import pandas as pd
from langchain.prompts import ChatPromptTemplate

from llmjudgeeval.utils import (
    do_inference,
    make_model,
)


def truncate(s: str, max_len: int | None = None):
    if max_len is not None:
        return s[:max_len]
    else:
        return s


def generate_instructions(
    instructions: pd.Series,
    model: str,
    max_len: int | None = 2000,
    use_tqdm: bool = True,
    system_prompt: str | None = None,
) -> pd.DataFrame:
    chat_model = make_model(model, max_tokens=200)

    # TODO improve prompt to generate instructions
    if system_prompt is None:
        system_prompt = (
            "You are an helpful assistant that answer queries asked by users."
        )
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{user_prompt}")]
    )

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
    print(completions[0])
    return df_outputs


def generate_base(
    instructions: pd.Series,
    model: str,
    max_len: int | None = 2000,
) -> pd.DataFrame:
    model = make_model(model, max_tokens=200)

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
