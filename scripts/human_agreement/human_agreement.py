import numpy as np
from fast_langdetect import detect_language
from collections import defaultdict
from pathlib import Path
import pandas as pd
from huggingface_hub import snapshot_download
from sklearn.metrics import f1_score, accuracy_score

from openjury.evaluate import PairScore
from openjury.utils import set_langchain_cache
from openjury.evaluate import annotate_battles
from openjury.utils import cache_function_dataframe, make_model


def load_df() -> pd.DataFrame:
    # load LMSys
    path = snapshot_download(
        repo_id="lmarena-ai/arena-human-preference-100k",
        repo_type="dataset",
        allow_patterns="*parquet",
        force_download=False,
    )
    df_lmsys = pd.read_parquet(
        Path(path) / "data" / "arena-explorer-preference-100k.parquet"
    )
    df_lmsys["date"] = pd.to_datetime(df_lmsys["tstamp"], unit="s")
    df_lmsys["benchmark"] = "LMSys"

    # load ComparIA
    path = snapshot_download(
        repo_id="ministere-culture/comparia-votes",
        repo_type="dataset",
        allow_patterns="*",
        force_download=False,
    )

    df_comparia = pd.read_parquet(Path(path) / "votes.parquet")

    # unify schema
    df_comparia["tstamp"] = df_comparia["timestamp"]
    df_comparia["model_a"] = df_comparia["model_a_name"]
    df_comparia["model_b"] = df_comparia["model_b_name"]

    def get_winner(
        chosen_model_name: str, model_a: str, model_b: str, both_equal: bool, **kwargs
    ):
        if both_equal or chosen_model_name is None:
            return "tie"
        else:
            assert chosen_model_name in [
                model_a,
                model_b,
            ], f"Chosen model: {chosen_model_name} but model_a: {model_a} and model_b: {model_b}"
            return "model_a" if chosen_model_name == model_a else "model_b"
        return "model_b"

    df_comparia["winner"] = df_comparia.apply(lambda row: get_winner(**row), axis=1)
    df_comparia["benchmark"] = "ComparIA"
    df_comparia["question_id"] = df_comparia["id"]
    cols = [
        "question_id",
        "tstamp",
        "model_a",
        "model_b",
        "winner",
        "conversation_a",
        "conversation_b",
        "benchmark",
    ]
    df = pd.concat([df_lmsys.loc[:, cols], df_comparia.loc[:, cols]], ignore_index=True)

    # keep only one turn conversation for now as they are easier to evaluate
    df["turns"] = df.apply(lambda row: len(row["conversation_a"]) - 1, axis=1)
    df = df.loc[df.turns == 1]

    df["lang"] = df.apply(
        lambda row: detect_language(row["conversation_a"][0]["content"]).lower(), axis=1
    )

    return df


def generate_judge_annotations(
    df: pd.DataFrame,
    judge_name: str,
    max_len: int = 4096,
    provide_explanation: bool = False,
) -> pd.DataFrame:
    judge_chat_model = make_model(judge_name)

    def get_conv(conversation: list):
        return conversation[1]["content"] if len(conversation) > 1 else ""

    instructions = df.apply(lambda row: row["conversation_a"][0]["content"], axis=1)
    completions_A = df.apply(lambda row: get_conv(row["conversation_a"]), axis=1)
    completions_B = df.apply(lambda row: get_conv(row["conversation_b"]), axis=1)

    annotations = annotate_battles(
        judge_chat_model=judge_chat_model,
        instructions=instructions,
        completions_A=completions_A,
        completions_B=completions_B,
        use_tqdm=False,
        max_len=max_len,
        provide_explanation=provide_explanation,
    )

    df_res = pd.DataFrame(annotations)
    cols_to_copy = ["winner", "question_id", "model_a", "model_b", "benchmark"]
    df_res.loc[:, cols_to_copy] = df.loc[:, cols_to_copy].values.copy()
    return df_res


def report_results(result_dict: dict):
    def judge_length(x, y):
        # judge that chose the longest
        if x is None and y is None:
            return "tie"
        elif x is None:
            return "model_b"
        elif y is None:
            return "model_a"
        elif len(x) == len(y):
            return "tie"
        else:
            return "model_b" if len(x) < len(y) else "model_a"

    def proj(x):
        eps = 0.1
        if x is None or np.isnan(x) or abs(x - 0.5) < eps:
            return "tie"
        elif x > 0.5 + eps:
            return "model_b"
        elif x < 0.5 - eps:
            return "model_a"
        else:
            assert False, f"Unexpected value: {x}"

    rows = []
    for language, language_result in result_dict.items():
        # result_dict[language][judge] = df_input, df_annotations
        df_input_first, df_annotations_first = next(iter(language_result.values()))
        y_true = df_input_first.apply(
            lambda row: ("tie" if "tie" in row["winner"] else row["winner"]),
            axis=1,
        ).values

        # add a few baselines
        methods = {
            "random": y_true[np.random.permutation(len(y_true))],
            "constant-tie": np.array(["tie"] * len(y_true)),
            "length": np.array(
                [
                    judge_length(x, y)
                    for x, y in zip(
                        df_annotations_first.completion_A.fillna(
                            "",
                        ),
                        df_annotations_first.completion_B.fillna(
                            "",
                        ),
                    )
                ]
            ),
        }

        for judge, (df_input, df_annotations) in language_result.items():
            assert len(df_input) == len(df_annotations) == len(df_input_first)

            score_parser = PairScore()
            y_pred = df_annotations.apply(
                lambda row: score_parser.parse_model_raw(row["judge_completion"]),
                axis=1,
            )
            y_pred = np.array([proj(x) for x in y_pred.values])
            methods[judge] = y_pred

        # TODO compute y_pred and add scores
        for method, pred in methods.items():
            f1_value = f1_score(
                np.astype(y_true, str), np.astype(pred, str), average="weighted"
            )
            accuracy_value = accuracy_score(
                np.astype(y_true, str), np.astype(pred, str)
            )
            for metric, value in zip(
                ["weighted-F1", "Accuracy"], [f1_value, accuracy_value]
            ):
                rows.append(
                    {
                        "method": method,
                        "lang": language,
                        "metric": metric,
                        "value": value,
                    }
                )

    df_results = pd.DataFrame(rows)
    return df_results


if __name__ == "__main__":
    print("Load preference dataset")
    n_instructions = 200
    # df = cache_function_dataframe(
    #     lambda: load_df(n_instructions=n_instructions),
    #     cache_name=f"preference_{n_instructions}",
    # )
    df = load_df()
    print(f"Loaded {len(df)} instructions")

    judges = [
        "OpenRouter/meta-llama/llama-3.3-70b-instruct",
        "OpenRouter/deepseek/deepseek-chat-v3.1",
        "OpenRouter/qwen/qwen3-30b-a3b-instruct-2507",
        "OpenRouter/qwen/qwen3-next-80b-a3b-instruct",
        "OpenRouter/qwen/qwen3-235b-a22b-2507",
    ]

    languages = [
        "en",
        "fr",
        "cs",
        "es",
        "sv",
        "fi",
        "ca",
    ]

    ignore_cache = True

    # cache llm client inputs/outputs does not work with vllm
    set_langchain_cache()

    result_dict = defaultdict(dict)
    for judge in judges:
        for language in languages:
            # filter by language and sample
            df_input = df.loc[df["lang"] == language]
            df_input = df_input.sample(
                n=min(n_instructions, len(df_input)), random_state=0
            )

            print(f"Generating data or loading cache for {judge, language}")
            df_annotations = cache_function_dataframe(
                lambda: generate_judge_annotations(
                    df=df_input,
                    judge_name=judge,
                    max_len=4096,
                    provide_explanation=False,
                ),
                ignore_cache=ignore_cache,
                cache_name=f"{judge}_{language}_{n_instructions}",
            )
            # print(df_annotations.head(n=n_instructions))
            # todo maybe no need to duplicate columns of df into df_annotations if we pass both
            result_dict[language][judge] = df_input, df_annotations

    # TODO dump result_dict raw results so that we can compute error-bars
    # evaluate accuracy of llm-judgements
    df_results = report_results(result_dict)

    print(
        df_results.pivot_table(
            index="method", columns=["lang", "metric"], values="value"
        ).to_string()
    )

    name_file = (
        f'results-{"_".join(judges)}_{"_".join(languages)}_{n_instructions}.csv.zip'
    ).replace("/", "_")

    print(f"Saving results to {name_file}")
    df_results.to_csv(Path(__file__).parent / name_file, index=False)
