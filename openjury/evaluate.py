import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models.llms import LLM

from openjury.instruction_dataset import load_instructions
from openjury.utils import (
    read_df,
    data_root,
    download_hf,
    do_inference,
)


RUBRIC_CRITERIA = ["instruction_following", "naturalness", "coherence", "accuracy"]

# Valid scores on the Likert scale (anchor points defined in the rubric prompt).
# Adapted from the Tiny Aya tech report, Appendix B.3 (Aryabumi et al., 2025).
VALID_RUBRIC_SCORES = [1, 3, 5, 7]


class RubricScore:
    """Parse rubric-based judge output (JSON with 4 criteria, 1/3/5/7 Likert scale).

    Adapted from the Tiny Aya tech report, Appendix B.3
    (Aryabumi et al., 2025). Scores are snapped to the nearest valid
    anchor point {1, 3, 5, 7} if the judge outputs an intermediate value.

    Composite score: (mean - 1) / 6, linearly mapping [1, 7] to [0, 1].
    """

    min_score = 1
    max_score = 7

    def parse_model_raw(self, judge_completion: str) -> dict | None:
        """Extract 4 criterion scores + rationales from JSON in judge response.

        Returns dict with keys like 'instruction_following_score',
        'instruction_following_rationale', etc., plus 'composite_score' (0-1)
        and 'mean_score' (1-7). Returns None on parse failure.
        """
        json_str = self._extract_json(judge_completion)
        if json_str is None:
            return None

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return None

        scores = {}
        for criterion in RUBRIC_CRITERIA:
            score_key = f"{criterion}_score"
            rationale_key = f"{criterion}_rationale"
            if score_key not in data:
                return None
            score = data[score_key]
            if not isinstance(score, (int, float)):
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    return None
            score = float(score)
            # Snap to nearest valid anchor point {1, 3, 5, 7}
            snapped = min(VALID_RUBRIC_SCORES, key=lambda v: abs(v - score))
            if snapped != score:
                scores.setdefault("_snapped_scores", [])
                scores["_snapped_scores"].append(
                    (criterion, score, snapped)
                )
            scores[score_key] = snapped
            scores[rationale_key] = data.get(rationale_key, "")

        mean_score = sum(
            scores[f"{c}_score"] for c in RUBRIC_CRITERIA
        ) / len(RUBRIC_CRITERIA)
        scores["composite_score"] = (mean_score - self.min_score) / (
            self.max_score - self.min_score
        )
        scores["mean_score"] = mean_score
        return scores

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """Extract JSON object from text, handling markdown code blocks."""
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        return None


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
            judge_completion.lower(), r'score.*?a[": *\n]*(-?\d+)'
        )
        score_b = self.get_regexp_match(
            judge_completion.lower(), r'score.*?b[": *\n]*(-?\d+)'
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
    use_tqdm: bool = False,
    truncate_input_chars: int | None = 8192,
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
    :param truncate_input_chars: if specified, truncates the length of completion, useful to save cost and avoid
    exceeding context limit
    :return:
    """
    local_path_tables = data_root / "tables"
    download_hf(name=dataset, local_path=local_path_tables)

    instructions = load_instructions(
        dataset=dataset,
    ).loc[:, "instruction"]

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

    if judge_chat_model is None:
        from langchain_together.llms import Together

        judge_chat_model = Together(model="meta-llama/Llama-3.3-70B-Instruct-Turbo")

    annotations = annotate_battles(
        judge_chat_model=judge_chat_model,
        instructions=instructions.tolist(),
        completions_A=completions_A.loc[instructions.index].tolist(),
        completions_B=completions_B.loc[instructions.index].tolist(),
        use_tqdm=use_tqdm,
        truncate_input_chars=truncate_input_chars,
        provide_explanation=provide_explanation,
    )

    # print("--------\n".join([str(x) for x in annotations]))
    # print results in term of 1) winrate 2) number of win/loss
    prefs = pd.Series([annotation.preference for annotation in annotations])
    num_wins = sum(prefs < 0.5)
    num_losses = sum(prefs > 0.5)
    num_ties = sum([1 if not x or x == 0.5 or x == np.nan else 0 for x in prefs])
    num_battles = len(prefs)
    winrate = float((num_wins + 0.5 * num_ties) / (num_ties + num_wins + num_losses))

    results = {
        "num_battles": num_battles,
        "winrate": winrate,
        "num_wins": num_wins,
        "num_losses": num_losses,
        "num_ties": num_ties,
    }

    print(f"{method_A} against {method_B}:\n{results}")
    print([annotation.preference for annotation in annotations])

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
    instruction: str
    completion_A: str
    completion_B: str


def annotate_battles(
    judge_chat_model,
    instructions: list[str],
    completions_A: list[str],
    completions_B: list[str],
    system_prompt: str | None = None,
    user_prompt_template: str = None,
    truncate_input_chars: int | None = 8192,
    use_tqdm: bool = False,
    provide_explanation: bool = False,
) -> list[JudgeAnnotation]:
    """
    Directly evaluate from list of instructions and completions
    Can also pass custom LLM judge prompts, if not passed uses defaults
    `system_prompt, user_prompt_template = load_judge_system_and_user_prompt()`
    Example usage:
    ```python
    annotations = annotate_battles(
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
    :param instructions:
    :param completions_A:
    :param completions_B:
    :param system_prompt:
    :param user_prompt_template:
    :param truncate_input_chars: Max characters to truncate completions before sending to judge.
    :param use_tqdm:
    :return:
    """
    # alternatively pass list of tuples
    assert len(instructions) == len(completions_A) == len(completions_B)

    (
        default_system_prompt,
        default_user_prompt_template,
    ) = load_judge_system_and_user_prompt(provide_explanation=provide_explanation)
    if system_prompt is None:
        system_prompt = default_system_prompt
    if user_prompt_template is None:
        user_prompt_template = default_user_prompt_template

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", user_prompt_template)]
    )

    def truncate(s: str, max_len: int | None = None):
        if not isinstance(s, str):
            return ""
        if max_len is not None:
            return s[:max_len]
        else:
            return s

    inputs = prompt_template.batch(
        [
            {
                "user_prompt": user_prompt,
                "completion_A": truncate(completion_A, max_len=truncate_input_chars),
                "completion_B": truncate(completion_B, max_len=truncate_input_chars),
            }
            for user_prompt, completion_A, completion_B in zip(
                instructions, completions_A, completions_B
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
    for judge_completion, instruction, completion_A, completion_B in zip(
        judge_completions, instructions, completions_A, completions_B
    ):
        annotations.append(
            JudgeAnnotation(
                judge_completion=judge_completion,
                instruction=instruction,
                completion_A=completion_A,
                completion_B=completion_B,
            )
        )
    return annotations


def load_rubric_prompts() -> tuple[str, str]:
    """Load system and user prompts for rubric evaluation."""
    with open(Path(__file__).parent / "prompts" / "rubric-system-prompt.txt", "r") as f:
        system_prompt = str(f.read())
    with open(Path(__file__).parent / "prompts" / "rubric-prompt.txt", "r") as f:
        user_prompt_template = str(f.read())
    return system_prompt, user_prompt_template


@dataclass
class RubricAnnotation:
    judge_completion: str
    instruction: str
    completion: str
    model: str


def annotate_rubric(
    judge_chat_model,
    instructions: list[str],
    completions: list[str],
    model_name: str,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
    truncate_input_chars: int | None = 8192,
    use_tqdm: bool = False,
) -> list[RubricAnnotation]:
    """Evaluate completions from a single model using rubric criteria.

    Unlike annotate_battles(), this evaluates one response at a time
    on 4 criteria (Instruction Following, Naturalness, Coherence, Accuracy)
    using a 1-7 Likert scale.

    Adapted from the Tiny Aya tech report, Appendix B.3.

    :param judge_chat_model: LangChain-compatible chat model for the judge.
    :param instructions: List of instruction strings.
    :param completions: List of completion strings from a single model.
    :param model_name: Name of the model being evaluated.
    :param system_prompt: Override system prompt. Defaults to rubric system prompt.
    :param user_prompt_template: Override user prompt. Defaults to rubric prompt.
    :param truncate_input_chars: Max characters to truncate completions.
    :param use_tqdm: Whether to show progress bar.
    :return: List of RubricAnnotation objects.
    """
    assert len(instructions) == len(completions)

    default_system_prompt, default_user_prompt_template = load_rubric_prompts()
    if system_prompt is None:
        system_prompt = default_system_prompt
    if user_prompt_template is None:
        user_prompt_template = default_user_prompt_template

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", user_prompt_template)]
    )

    def truncate(s: str, max_len: int | None = None):
        if not isinstance(s, str):
            return ""
        if max_len is not None:
            return s[:max_len]
        else:
            return s

    inputs = prompt_template.batch(
        [
            {
                "user_prompt": user_prompt,
                "completion": truncate(completion, max_len=truncate_input_chars),
            }
            for user_prompt, completion in zip(instructions, completions)
        ]
    )
    print(f"Start rubric evaluation for {model_name} ({len(inputs)} annotations).")
    judge_completions = do_inference(
        chat_model=judge_chat_model,
        inputs=inputs,
        use_tqdm=use_tqdm,
    )

    annotations = []
    for judge_completion, instruction, completion in zip(
        judge_completions, instructions, completions
    ):
        annotations.append(
            RubricAnnotation(
                judge_completion=judge_completion,
                instruction=instruction,
                completion=completion,
                model=model_name,
            )
        )
    return annotations
