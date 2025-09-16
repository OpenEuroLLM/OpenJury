import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models.llms import LLM
from langchain_together.llms import Together

from llmjudgeeval.instruction_dataset import load_instructions
from llmjudgeeval.utils import (
    read_df,
    data_root,
    download_hf,
    do_inference,
    set_langchain_cache,
    make_model,
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


def load_judge_system_and_user_prompt(
    provide_explanation: bool = True,
) -> tuple[str, str]:
    # Prepare judge
    with open(Path(__file__).parent / "prompts" / "system-prompt.txt", "r") as f:
        system_prompt = str(f.read())

    prompt_filename = (
        "prompt-with-explanation.txt" if provide_explanation else "prompt.txt"
    )
    with open(Path(__file__).parent / "prompts" / prompt_filename, "r") as f:
        user_prompt_template = str(f.read())

    return system_prompt, user_prompt_template


def evaluate_completions(
    dataset: str = "alpaca-eval",
    judge_chat_model: LLM = None,
    method_A: str = "gpt4_1106_preview",
    method_B: str = "llama-2-70b-chat-hf",
    num_annotations: int | None = 50,
    use_tqdm: bool = True,
    max_len: int | None = 2000,
    provide_explanation: bool = False,
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

    if judge_chat_model is None:
        judge_chat_model = Together(model="meta-llama/Llama-3.3-70B-Instruct-Turbo")

    local_path_tables = data_root / "tables"
    download_hf(name=dataset, local_path=local_path_tables)

    instructions = load_instructions(
        dataset=dataset,
    )

    # A bit ugly, only loads if local path exist as we do not have a local path of completion for cases such as
    # m-arena-hard.
    dataset_output_path = local_path_tables / "model_outputs" / f"{dataset}.csv.zip"
    if dataset_output_path.exists():
        df_outputs = read_df(dataset_output_path)
        # empty strings are encoded as Nan in csv
        df_outputs.loc[:, "output"] = df_outputs.loc[:, "output"].fillna("")
        df_outputs = df_outputs.pivot_table(
            index="instruction_index", columns="model", values="output", aggfunc="last"
        ).sort_index()
        df_outputs = df_outputs.loc[instructions.index]
    else:
        df_outputs = None

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
        provide_explanation=provide_explanation,
    )

    # print("--------\n".join([str(x) for x in annotations]))
    # print results in term of 1) winrate 2) number of win/loss
    prefs = pd.Series([annotation.preference for annotation in annotations])
    print([annotation.judge_completion for annotation in annotations])
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

    print(f"{method_A} against {method_B}:\n{results}")

    unique_string = dataset + "-" + datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = data_root / "judge-evals" / unique_string
    print(f"Saving results in {output_folder}")
    output_folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(annotations).to_csv(output_folder / "annotations.csv", index=False)
    with open(output_folder / "results.json", "w") as f:
        json.dump(results, f)


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
    provide_explanation: bool = True,
) -> list[JudgeAnnotation]:
    """
    Directly evaluate from list of instructions and completions
    Can also pass custom LLM judge prompts, if not passed uses defaults
    `system_prompt, user_prompt_template = load_judge_system_and_user_prompt()`
    Example usage:
    ```python
    annotations = annotate(
        # can be any langchain ChatModel, supports OpenAI, Together, vLLM, ...
        judge_chat_model=Together(model="meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        # the instructions we want to evaluate
        user_prompts=["Write numbers between 1 and 5."],
        # the completions we want to evaluate for the first model
        completions_A=["1 2 3 4 5."],
        # the completions we want to evaluate for the second model
        completions_B=["No"],
    )
    ```
    :param provide_explanation:
    :param judge_chat_model:
    :param user_prompts:
    :param completions_A:
    :param completions_B:
    :param system_prompt:
    :param user_prompt_template:
    :param num_annotations:
    :param max_len:
    :param use_tqdm:
    :return:
    """
    # alternatively pass list of tuples
    assert len(user_prompts) == len(completions_A) == len(completions_B)

    (
        default_system_prompt,
        default_user_prompt_template,
    ) = load_judge_system_and_user_prompt(provide_explanation=provide_explanation)
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
class EvalArgs:
    dataset: str
    method_A: str
    method_B: str
    judge_provider: str
    judge_model: str
    n_instructions: int | None = None
    provide_explanation: bool = False

    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser(
            prog="Evaluate LLM Judge on alpaca-eval, arena-hard, m-arena-hard",
        )
        parser.add_argument(
            "--dataset",
            help="The dataset to use",
            default="arena-hard",
            # choices=["alpaca-eval", "arena-hard", "m-arena-hard"],
        )
        parser.add_argument(
            "--method_A",
            required=True,
            help="one method to evaluate, can be a method existing in `dataset` or a local path to the completion of a "
            'local method. The path should be a dataframe ending with ".csv.zip" or ".parquet", have columns '
            "`instruction_index` and `output` and should contains all the instruction of `dataset`.",
        )
        parser.add_argument(
            "--method_B",
            required=True,
            help="another method to evaluate against `method_A`",
        )
        parser.add_argument(
            "--judge_provider",
            required=True,
            help="Type of judge to use",
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
            method_A=args.method_A,
            method_B=args.method_B,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            n_instructions=args.n_instructions,
            provide_explanation=args.provide_explanation,
        )


def main():
    args = EvalArgs.parse_args()

    if args.ignore_cache:
        set_langchain_cache()

    judge_chat_model = make_model(
        model_provider=args.judge_provider, model=args.judge_model
    )

    print(judge_chat_model)
    evaluate_completions(
        dataset=args.dataset,
        num_annotations=args.n_instructions,
        method_A=args.method_A,
        method_B=args.method_B,
        judge_chat_model=judge_chat_model,
        provide_explanation=args.provide_explanation,
    )


if __name__ == "__main__":
    main()
