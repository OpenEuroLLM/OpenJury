import pandas as pd
from langchain.prompts import ChatPromptTemplate

from openjury.utils import (
    do_inference,
    make_model,
    truncate,
)


def generate_instructions(
    instructions: pd.Series,
    model: str,
    truncate_input_chars: int | None = 8192,
    max_tokens: int | None = 32768,
    use_tqdm: bool = True,
    system_prompt: str | None = None,
    **model_kwargs,
) -> pd.DataFrame:
    chat_model = make_model(model, max_tokens=max_tokens, **model_kwargs)

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
                "user_prompt": truncate(user_prompt, max_len=truncate_input_chars),
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


def generate_multiturn(
    questions: pd.DataFrame,
    model: str,
    truncate_input_chars: int | None = 8192,
    max_tokens: int | None = 8192,
    use_tqdm: bool = True,
    **model_kwargs,
) -> pd.DataFrame:
    """Generate two-turn completions for MT-Bench style questions.

    Generates turn 1 answers first, then uses them as conversation context
    to generate turn 2 answers.

    Args:
        questions: DataFrame with columns turn_1, turn_2, and index instruction_index.
        model: Model specification string (e.g. "VLLM/model-name").
        **model_kwargs: Provider-specific options forwarded to make_model
            (e.g. max_model_len, chat_template for VLLM).
    Returns:
        DataFrame with columns: instruction_index, completion_turn_1, completion_turn_2
    """
    chat_model = make_model(model, max_tokens=max_tokens, **model_kwargs)

    system_prompt = "You are a helpful assistant."
    turn1_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{user_prompt}")]
    )

    turn1_inputs = turn1_template.batch(
        [
            {"user_prompt": truncate(row["turn_1"], max_len=truncate_input_chars)}
            for _, row in questions.iterrows()
        ]
    )

    print(f"Generating turn 1 completions ({len(turn1_inputs)} questions).")
    completions_turn_1 = do_inference(
        chat_model=chat_model,
        inputs=turn1_inputs,
        use_tqdm=use_tqdm,
    )

    turn2_inputs = []
    for (_, row), t1_answer in zip(questions.iterrows(), completions_turn_1):
        if row["turn_2"] is None:
            turn2_inputs.append(
                turn1_template.invoke(
                    {"user_prompt": "No follow-up question."}
                )
            )
        else:
            multi_turn_template = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("user", "{turn_1}"),
                    ("assistant", "{turn_1_answer}"),
                    ("user", "{turn_2}"),
                ]
            )
            turn2_inputs.append(
                multi_turn_template.invoke(
                    {
                        "turn_1": truncate(row["turn_1"], max_len=truncate_input_chars),
                        "turn_1_answer": truncate(str(t1_answer), max_len=truncate_input_chars),
                        "turn_2": truncate(row["turn_2"], max_len=truncate_input_chars),
                    }
                )
            )

    print(f"Generating turn 2 completions ({len(turn2_inputs)} questions).")
    completions_turn_2 = do_inference(
        chat_model=chat_model,
        inputs=turn2_inputs,
        use_tqdm=use_tqdm,
    )

    df_outputs = pd.DataFrame(
        data={
            "instruction_index": questions.index.tolist(),
            "completion_turn_1": completions_turn_1,
            "completion_turn_2": completions_turn_2,
        },
    )
    return df_outputs


def generate_base(
    instructions: pd.Series,
    model: str,
    truncate_input_chars: int | None = 8192,
    max_tokens: int | None = 32768,
    use_tqdm: bool = False,
    **model_kwargs,
) -> pd.DataFrame:
    model = make_model(model, max_tokens=max_tokens, **model_kwargs)

    inputs = [
        truncate(instruction, max_len=truncate_input_chars)
        for instruction in instructions
    ]

    completions = model.batch(
        inputs=inputs,
        max_tokens=max_tokens,
    )
    completions = [x.content if hasattr(x, "content") else x for x in completions]

    df_outputs = pd.DataFrame(
        data={
            "completion": completions,
            "instruction_index": instructions.index.tolist(),
        },
    )

    return df_outputs
