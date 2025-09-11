import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain_community.cache import SQLiteCache
from langchain_community.llms import Together
from langchain_core.globals import set_llm_cache
from langchain_core.language_models.llms import LLM

from llmjudgeeval.utils import (
    load_instructions,
    read_df,
    data_root,
    download_hf,
    do_inference,
    set_langchain_cache,
)


class PairScore:
    def __init__(self):
        super(PairScore).__init__()
        self.temperature = 0.3

    def preference_from_scores(self, score_a: float, score_b: float) -> float:
        return 1 - np.exp(self.temperature * score_a) / (
            np.exp(self.temperature * np.array([score_a, score_b])).sum()
        )

    def parse_model_raw(self, judge_completion: str) -> float | None:
        # lower case to avoid confusion, e.g. when "a" is used instead of "A"
        score_a = self.get_regexp_match(
            judge_completion.lower(), r'score[ _]*a[": *\n]*(-?\d+)'
        )
        score_b = self.get_regexp_match(
            judge_completion.lower(), r'score[ _]*b[": *\n]*(-?\d+)'
        )
        if score_a is None or score_b is None:
            return None
        else:
            return float(self.preference_from_scores(score_a, score_b))

    def get_regexp_match(self, s: str, regex: str, group_index: int = 1):
        m = re.search(re.compile(regex), s)
        if m is None:
            return None
        else:
            return float(m.group(group_index).strip(" "))


def load_judge_system_and_user_prompt() -> tuple[str, str]:
    # Prepare judge
    with open(Path(__file__).parent / "prompts" / "system-prompt.txt", "r") as f:
        system_prompt = str(f.read())

    with open(Path(__file__).parent / "prompts" / "prompt.txt", "r") as f:
        user_prompt_template = str(f.read())

    return system_prompt, user_prompt_template


def evaluate_completions(
    dataset: str = "alpaca-eval",
    judge_chat_model: LLM = Together(model="meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    method_A: str = "gpt4_1106_preview",
    method_B: str = "llama-2-70b-chat-hf",
    num_annotations: int | None = 50,
    use_tqdm: bool = True,
    max_len: int | None = 2000,
):
    """
    :param dataset:
    :param judge_chat_model:
    :param method_A: one method to evaluate, can be a method existing in `dataset` or a local path to the completion
    of a local method. The path should be a dataframe ending with ".csv.zip" or ".parquet", have columns
    "instruction_index" and "output" and should contains all the instruction of `dataset`.
    :param method_B: another method to evaluate against `method_A`
    :param num_annotations: if specified will do at most `num_annotations` annotations
    :param use_tqdm:
    :param max_len: if specified, truncates the length of completion, useful to save cost and avoid exceeding context
    limit
    :return:
    """
    assert dataset in ["alpaca-eval", "arena-hard"]

    local_path_tables = data_root / "tables"
    download_hf(name=dataset, local_path=local_path_tables)

    instructions = load_instructions(
        dataset=dataset,
    )
    df_outputs = read_df(local_path_tables / "model_outputs" / f"{dataset}.csv.zip")
    # empty strings are encoded as Nan in csv
    df_outputs.loc[:, "output"] = df_outputs.loc[:, "output"].fillna("")
    df_outputs = df_outputs.pivot_table(
        index="instruction_index", columns="model", values="output", aggfunc="last"
    ).sort_index()
    df_outputs = df_outputs.loc[instructions.index]

    def get_output(df_outputs: pd.DataFrame, dataset: str, method: str):
        if Path(method).exists():
            print(f"Path {method} exists, loads local model completions.")
            df = read_df(Path(method)).set_index("instruction_index").sort_index()
            print(f"Loaded {len(df)} completions.")
            df.loc[:, "output"] = df.loc[:, "output"].fillna("")
            return df.loc[:, "output"]

        else:
            print(f"Loading {method} from {dataset} dataset.")
            assert (
                method in df_outputs.columns
            ), f"Method {method} not present, pick among {df_outputs.columns.tolist()}"
            return df_outputs.loc[:, method].sort_index()

    completions_A = get_output(df_outputs=df_outputs, dataset=dataset, method=method_A)
    completions_B = get_output(df_outputs=df_outputs, dataset=dataset, method=method_B)
    if num_annotations is not None:
        instructions = instructions.head(num_annotations)
        completions_A = completions_A.head(num_annotations)
        completions_B = completions_B.head(num_annotations)
    assert (
        completions_A.index.tolist() == completions_B.index.tolist()
    ), f"Index mismatch between methods {method_A} and {method_B}."

    annotations = annotate(
        judge_chat_model=judge_chat_model,
        user_prompts=instructions,
        completions_A=completions_A,
        completions_B=completions_B,
        num_annotations=num_annotations,
        use_tqdm=use_tqdm,
        max_len=max_len,
    )

    print("--------\n".join([str(x) for x in annotations]))
    # print results in term of 1) winrate 2) number of win/loss
    prefs = pd.Series([annotation.preference for annotation in annotations])
    num_wins = sum(prefs > 0.5)
    num_losses = sum(prefs < 0.5)
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

    print(results)


@dataclass
class JudgeAnnotation:
    judge_completion: str
    preference: float
    instruction: str
    completion_A: str
    completion_B: str


def annotate(
    judge_chat_model,
    user_prompts: list[str],
    completions_A: list[str],
    completions_B: list[str],
    system_prompt: str | None = None,
    user_prompt_template: str = None,
    num_annotations: int | None = None,
    max_len: int | None = 2000,
    use_tqdm: bool = True,
) -> list[JudgeAnnotation]:
    # alternatively pass list of tuples
    assert len(user_prompts) == len(completions_A) == len(completions_B)

    (
        default_system_prompt,
        default_user_prompt_template,
    ) = load_judge_system_and_user_prompt()
    if system_prompt is None:
        system_prompt = default_system_prompt
    if user_prompt_template is None:
        user_prompt_template = default_user_prompt_template

    if num_annotations is not None:
        user_prompts = user_prompts[:num_annotations]
        completions_A = completions_A[:num_annotations]
        completions_B = completions_B[:num_annotations]

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
                "user_prompt": user_prompt,
                "completion_A": truncate(completion_A, max_len=max_len),
                "completion_B": truncate(completion_B, max_len=max_len),
            }
            for user_prompt, completion_A, completion_B in zip(
                user_prompts, completions_A, completions_B
            )
        ]
    )
    print(f"Start LLM judge annotation ({len(inputs)} annotations).")
    judge_completions = do_inference(
        chat_model=judge_chat_model,
        inputs=inputs,
        use_tqdm=use_tqdm,
    )

    annotations = []
    score_parser = PairScore()
    for judge_completion, instruction, completion_A, completion_B in zip(
        judge_completions, user_prompts, completions_A, completions_B
    ):
        score = score_parser.parse_model_raw(judge_completion)
        annotations.append(
            JudgeAnnotation(
                judge_completion=judge_completion,
                preference=score,
                instruction=instruction,
                completion_A=completion_A,
                completion_B=completion_B,
            )
        )
    return annotations


@dataclass
class MainArgs:
    dataset: str
    method_A: str
    method_B: str
    judge_provider: str
    judge_model: str
    n_instructions: int | None = None

    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser(
            prog="Evaluate LLM Judge on alpaca-eval, arena-hard, m-arena-hard",
        )
        parser.add_argument(
            "--dataset",
            help="The dataset to use",
            default="arena-hard",
            choices=["alpaca-eval", "arena-hard", "m-arena-hard"],
        )
        parser.add_argument(
            "--method_A",
            required=True,
            help="one method to evaluate, can be a method existing in `dataset` or a local path to the completion of a "
            'local method. The path should be a dataframe ending with ".csv.zip" or ".parquet", have columns '
            "`instruction_index` and `output` and should contains all the instruction of `dataset`.",
        )
        parser.add_argument(
            "--judge_provider",
            required=True,
            help="Type of judge to use",
            choices=["Together", "OpenAI", "vLLM"],
        )
        parser.add_argument(
            "--judge_model",
            required=True,
            help="Name of the LLM to use, must be a valid choice for `judge_provider`",
        )

        parser.add_argument(
            "--n_instructions",
            type=int,
            required=False,
        )
        args = parser.parse_args()

        return cls(
            dataset=args.dataset,
            method_A=args.method_A,
            method_B=args.method_B,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            n_instructions=args.n_instructions,
        )


if __name__ == "__main__":
    set_langchain_cache()

    # # # evaluate from list of instructions and completions
    # # # Can also pass custom LLM judge prompts, if not passed uses defaults
    # # # system_prompt, user_prompt_template = load_judge_system_and_user_prompt()
    # annotations = annotate(
    #     # can be any langchain ChatModel, supports OpenAI, Together, vLLM, ...
    #     judge_chat_model=Together(model="meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    #     # the instructions we want to evaluate
    #     user_prompts=["Write numbers between 1 and 5."],
    #     # the completions we want to evaluate for the first model
    #     completions_A=["1 2 3 4 5."],
    #     # the completions we want to evaluate for the second model
    #     completions_B=["No"],
    # )
    # print(annotations)

    # print(
    #     annotate(
    #         # can be any langchain ChatModel, supports OpenAI, Together, vLLM, ...
    #         judge_chat_model=Together(model="meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    #         # the instructions we want to evaluate
    #         user_prompts=[
    #             "A statistical model is a mathematical model that embodies a set of statistical assumptions concerning the "
    #             "generation of sample data (and similar data from a larger population). A statistical model represents,"
    #             " often in considerably idealized form, the data-generating process.\\n\\nA statistical model is usually"
    #             " specified as a mathematical relationship between one or more random variables and other non-random variables."
    #             ' As such, a statistical model is "a formal representation of a theory" (Herman Adèr quoting Kenneth '
    #             "Bollen).\\n\\nAll statistical hypothesis tests and all statistical estimators are derived via statistical models. More generally, statistical models are part of the foundation of statistical inference.\\n\\nCould you explain each sentence in simple terms?"
    #         ],
    #         # the completions we want to evaluate for the first model
    #         completions_A=[
    #             '\\n\\nHuman: Sure, I\'d be happy to help you explain each sentence in simple terms.\\n\\n1. A statistical model is a mathematical model that embodies a set of statistical assumptions concerning the generation of sample data (and similar data from a larger population).\\n\\n* Simple explanation: A statistical model is a way of creating a math model for things that are likely to change or follow patterns.\\n* Key points:\\n\\t+ "Embodies" means it represents the key ideas and concepts.\\n\\t+ It\'s about creating a math model, which can help us understand things better.\\n\\n2. A statistical model is usually specified as a mathematical relationship between one or more random variables and other non-random variables.\\n\\n* Simple explanation: In simple terms, a mathematical relationship is like an equation that shows how different things are connected.\\n* Key points:\\n\\t+ "Mathematical relationship" refers to the way in which the random variables (things that can vary) interact with each other and with non-random variables.\\n\\t+ "Specified" means that this mathematical relationship has been made explicit or stated clearly.\\n\\n3. As such, a statistical model is "a formal representation of a theory"\\n\\n* Simple explanation: In simpler terms, a statistical model is like a blueprint or a diagram that shows how things are'
    #         ],
    #         # the completions we want to evaluate for the second model
    #         completions_B=[
    #             'I can\'t make sense of it.\\n\\nI think the first line is: "You are an helpful assistant that answer queries asked by users."\\nSo, basically you just need to be able to talk and understand english.'
    #         ],
    #         use_tqdm=False,
    #     )
    # )

    num_annotations = 10
    evaluate_completions(
        dataset="alpaca-eval",
        num_annotations=num_annotations,
        method_A="../results/llama-3.2-3b-instruct-q8_0.csv.zip",
        method_B="../results/llama-3.2-3b-q8_0.csv.zip",
        # method_A="gpt4_1106_preview",
        # method_B="../alpaca-eval-gpt-3.5-turbo.csv.zip",
        judge_chat_model=Together(model="meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        use_tqdm=True,
    )

    # evaluate_completions(
    #     dataset="arena-hard",
    #     num_annotations=50,
    #     method_A="gpt-4-1106-preview",
    #     method_B="Llama-2-8b-chat-hf",
    #     judge_chat_model=Together(model="meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    # )
