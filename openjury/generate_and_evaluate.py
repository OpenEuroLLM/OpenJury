"""
This script generates completions for a given dataset and model,
and then evaluates them using a judge model.
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from openjury.evaluate import annotate_battles, PairScore
from openjury.generate import generate_instructions, generate_base
from openjury.instruction_dataset import load_instructions
from openjury.rubrics.pipeline import run_pairwise_rubric_pipeline
from openjury.utils import data_root, read_df, download_hf
from openjury.utils import make_model, cache_function_dataframe


def try_load_dataset_completions(
    dataset: str, model: str, n_instructions: int | None
) -> pd.DataFrame | None:
    """Try loading pre-existing completions from the dataset.

    Some datasets (e.g. alpaca-eval) ship with completions for well-known
    models such as ``gpt4_1106_preview``.  When ``model`` matches a column in
    ``model_outputs/{dataset}.csv.zip``, those completions are returned
    directly so that no model instantiation / generation is needed.

    Returns a DataFrame with columns ``completion`` and ``instruction_index``,
    or ``None`` when no pre-existing completions are found.
    """
    local_path_tables = data_root / "tables"
    download_hf(name=dataset, local_path=local_path_tables)
    output_path = local_path_tables / "model_outputs" / f"{dataset}.csv.zip"
    if not output_path.exists():
        return None
    df_outputs = read_df(output_path)
    df_outputs.loc[:, "output"] = df_outputs.loc[:, "output"].fillna("")
    df_outputs = df_outputs.pivot_table(
        index="instruction_index", columns="model", values="output", aggfunc="last"
    ).sort_index()
    if model not in df_outputs.columns:
        return None
    print(f"Found pre-existing completions for '{model}' in {dataset} dataset.")
    completions = df_outputs.loc[:, model]
    if n_instructions is not None:
        completions = completions.head(n_instructions)
    return pd.DataFrame(
        {
            "completion": completions.values,
            "instruction_index": completions.index.tolist(),
        }
    )


@dataclass
class CliArgs:
    dataset: str
    model_A: str
    model_B: str
    judge_model: str

    n_instructions: int | None = None
    provide_explanation: bool = False
    swap_mode: str = "fixed"
    ignore_cache: bool = False
    use_tqdm: bool = False
    truncate_all_input_chars: int = 8192
    max_out_tokens_models: int = 32768
    max_out_tokens_judge: int = 32768
    max_model_len: int | None = None
    chat_template: str | None = None
    enable_rubrics: bool = False
    rubric_name: str = "default"
    rubric_json: str | None = None

    result_folder: str = "results"

    def __post_init__(self):
        supported_modes = ["fixed", "both"]
        assert (
            self.swap_mode in supported_modes
        ), f"Only {supported_modes} modes are supported but got {self.swap_mode}."

    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser(
            prog="Generate completion and evaluate with a judge",
        )
        parser.add_argument(
            "--dataset",
            help="The dataset to use. For instance `alpaca-eval`, `arena-hard`, `m-arena-hard-EU` for instruction "
            "tuning cases or `french-contexts`, `spanish-contexts` for base models.",
        )
        parser.add_argument(
            "--model_A",
            required=True,
            help="Name of the LLM to use for a generation, must be a valid choice for `generation_provider`",
        )
        parser.add_argument(
            "--model_B",
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
            "--swap_mode",
            type=str,
            choices=["fixed", "both"],
            default="fixed",
            help="Model comparison order mode. 'fixed': always use model order A-B. 'both': correct for model order "
            "bias by evaluating each instruction twice, once as A-B and once as B-A, and average. This helps account "
            "for judge position bias. Default is 'fixed'.",
        )
        parser.add_argument(
            "--ignore_cache",
            action="store_true",
            help="If specified, ignore cache of previous completions.",
        )
        parser.add_argument(
            "--use_tqdm",
            action="store_true",
            help="If specified, use tqdm, does not work with all model providers, vLLM in particular.",
        )
        parser.add_argument(
            "--result_folder",
            type=str,
            required=False,
            default="results",
            help="The folder to save the results. Defaults to `results`. Evaluation results will be saved in"
            " `[result_folder]/[evaluation_name]`.",
        )
        parser.add_argument(
            "--truncate_all_input_chars",
            type=int,
            required=False,
            default=8192,
            help="Character-level truncation applied before tokenization: truncates each instruction "
            "before model A/B generation and truncates each completion before judge evaluation.",   
        )
        parser.add_argument(
            "--max_out_tokens_models",
            type=int,
            required=False,
            default=32768,
            help=(
                "Generation token budget for each model A/B response. For VLLM, keep this <= "
                "--max_model_len (if provided)."
            ),
        )
        parser.add_argument(
            "--max_out_tokens_judge",
            type=int,
            required=False,
            default=32768,
            help=(
                "Generation token budget for the judge response (reasoning + scores). For "
                "VLLM, keep this <= --max_model_len (if provided)."
            ),
        )
        parser.add_argument(
            "--max_model_len",
            type=int,
            required=False,
            default=None,
            help=(
                "Optional total context window for VLLM models (prompt + generation). This is "
                "independent from --max_out_tokens_models/--max_out_tokens_judge, which only cap "
                "generated tokens. This is useful on smaller GPUs to avoid OOM."
            ),
        )
        parser.add_argument(
            "--chat_template",
            type=str,
            required=False,
            default=None,
            help="Jinja2 chat template string to use instead of the model's tokenizer template. "
            "If not provided, ChatML is used as fallback for models without a chat template.",
        )
        parser.add_argument(
            "--enable_rubrics",
            action="store_true",
            help="If specified, run rubric-based pairwise scoring in addition to the legacy pairwise judge output.",
        )
        parser.add_argument(
            "--rubric_name",
            type=str,
            default="default",
            help="Rubric to use when --enable_rubrics is set (e.g. default, coding, translation, overall).",
        )
        parser.add_argument(
            "--rubric_json",
            type=str,
            default=None,
            help="Optional path to a custom rubric JSON file. If provided, this overrides --rubric_name.",
        )
        args = parser.parse_args()

        return cls(
            dataset=args.dataset,
            model_A=args.model_A,
            model_B=args.model_B,
            judge_model=args.judge_model,
            n_instructions=args.n_instructions,
            provide_explanation=args.provide_explanation,
            swap_mode=args.swap_mode,
            ignore_cache=args.ignore_cache,
            use_tqdm=args.use_tqdm,
            truncate_all_input_chars=args.truncate_all_input_chars,
            max_out_tokens_models=args.max_out_tokens_models,
            max_out_tokens_judge=args.max_out_tokens_judge,
            max_model_len=args.max_model_len,
            chat_template=args.chat_template,
            enable_rubrics=args.enable_rubrics,
            rubric_name=args.rubric_name,
            rubric_json=args.rubric_json,
            result_folder=args.result_folder,
        )


def load_contexts(dataset: str) -> pd.Series:
    path = data_root / "contexts" / dataset
    return pd.read_csv(path).loc[:, "instruction"]


def print_results(results):
    """Print battle results in a nice formatted way"""

    print("\n" + "=" * 60)
    print("🏆 MODEL BATTLE RESULTS 🏆".center(60))
    print(f"📊 Dataset: {results['dataset']}")
    print(
        f"🤖 Competitors: Model A: {results['model_A']} vs Model B: {results['model_B']}"
    )
    print(f"⚖️ Judge: {results['judge_model']}")
    print(f"📈 Results Summary:")
    print(f"   Total Battles: {results['num_battles']}")
    print(f"   Win Rate (A): {results['winrate']:.1%}")
    print(f"   ✅ Wins:   {results['num_wins']}")
    print(f"   ❌ Losses: {results['num_losses']}")
    print(f"   🤝 Ties:   {results['num_ties']}")
    print("=" * 60 + "\n")


def _compute_pref_summary(prefs: pd.Series) -> dict[str, float | int]:
    """Compute win/loss/tie stats for preference series (0=A, 0.5=tie, 1=B)."""
    prefs = pd.Series(prefs, dtype="float64")
    valid = prefs.dropna()
    num_wins = int((valid < 0.5).sum())
    num_losses = int((valid > 0.5).sum())
    num_ties = int((valid == 0.5).sum())
    num_battles = int(len(prefs))
    denom = num_wins + num_losses + num_ties
    winrate = float((num_wins + 0.5 * num_ties) / denom) if denom > 0 else float("nan")
    return {
        "num_battles": num_battles,
        "winrate": winrate,
        "num_wins": num_wins,
        "num_losses": num_losses,
        "num_ties": num_ties,
        "num_missing": int(num_battles - denom),
    }


def main(args: CliArgs):
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

    print(
        f"Using dataset {args.dataset} and evaluating models {args.model_A} and {args.model_B}."
    )

    # Not working with vllm, not detecting model changes and serving the same cache for two different models...
    # if not args.ignore_cache:
    #     set_langchain_cache()
    ignore_cache = args.ignore_cache

    # Currrently, we run context evaluation
    is_fluency_task = "fluency" in args.dataset
    if is_fluency_task:
        # if args.dataset = "fluency-french", we map to "french-contexts.csv"
        # to match files in https://huggingface.co/datasets/geoalgo/multilingual-contexts-to-be-completed
        lang = args.dataset.split("-")[-1]
        instructions = load_contexts(f"{lang}-contexts.csv")
    else:
        instructions = load_instructions(
            dataset=args.dataset, n_instructions=args.n_instructions
        ).loc[:, "instruction"]

    n_instructions = args.n_instructions if args.n_instructions else len(instructions)
    if args.n_instructions is not None:
        instructions = instructions[:n_instructions]

    print(
        f"Generating completions for dataset {args.dataset} with model {args.model_A} and "
        f"{args.model_B} (or loading them directly if present)"
    )

    # TODO currently we just support base models for fluency, we could also support instruction-tuned models
    gen_fun = (
        partial(generate_base, truncate_input_chars=args.truncate_all_input_chars, max_tokens=args.max_out_tokens_models, max_model_len=args.max_model_len, chat_template=args.chat_template)
        if is_fluency_task
        else partial(generate_instructions, truncate_input_chars=args.truncate_all_input_chars, max_tokens=args.max_out_tokens_models, chat_template=args.chat_template, max_model_len=args.max_model_len)
    )
    dataset_completions_A = try_load_dataset_completions(
        args.dataset, args.model_A, n_instructions
    )
    if dataset_completions_A is not None:
        completions_A = dataset_completions_A.set_index("instruction_index").loc[:, "completion"]
    else:
        completions_A = cache_function_dataframe(
            lambda: gen_fun(
                instructions=instructions,
                model=args.model_A,
                use_tqdm=args.use_tqdm,
            ),
            ignore_cache=ignore_cache,
            cache_name=f"{args.dataset}_{args.model_A}_{args.n_instructions}",
        ).set_index("instruction_index")
        completions_A = completions_A.loc[:, "completion"]

    dataset_completions_B = try_load_dataset_completions(
        args.dataset, args.model_B, n_instructions
    )
    if dataset_completions_B is not None:
        completions_B = dataset_completions_B.set_index("instruction_index").loc[:, "completion"]
    else:
        completions_B = cache_function_dataframe(
            lambda: gen_fun(
                instructions=instructions,
                model=args.model_B,
                use_tqdm=args.use_tqdm,
            ),
            ignore_cache=ignore_cache,
            cache_name=f"{args.dataset}_{args.model_B}_{args.n_instructions}",
        ).set_index("instruction_index")
        completions_B = completions_B.loc[:, "completion"]
    print(f"\nFirst instruction/context: {instructions.values[0]}")

    print(f"\nFirst completion of {args.model_A}")
    print(completions_A.values[0])
    print(f"\nFirst completion of {args.model_B}")
    print(completions_B.values[0])
    print(f"Evaluating completions with judge {args.judge_model}.")

    judge_chat_model = make_model(
        model=args.judge_model,
        max_tokens=args.max_out_tokens_judge,
        max_model_len=args.max_model_len,
        chat_template=args.chat_template,
    )
    if is_fluency_task:
        system_prompt = """You are a highly efficient assistant, who evaluates and selects the best large language \
        model based on the quality of completion of a sentence. You will see a sentence to be completed and two \
        completions from Assistant A and Assistant B and will have to decide which one is best. Make sure to not \
        over-confidently prefer one assistant or the other and also make sure to not bias your preference based on \
        the ordering or on the length of the answers."""
    else:
        # the default system prompt of annotate is to compare instruction tuned models.

        system_prompt = None
    annotations = annotate_battles(
        judge_chat_model=judge_chat_model,
        instructions=instructions.head(n_instructions).tolist(),
        completions_A=completions_A.head(n_instructions).tolist(),
        completions_B=completions_B.head(n_instructions).tolist(),
        provide_explanation=args.provide_explanation,
        system_prompt=system_prompt,
        truncate_input_chars=args.truncate_all_input_chars,
        use_tqdm=args.use_tqdm,
    )

    if args.swap_mode == "both":
        print("Correction for judge bias towards a certain model position is set.")
        print(
            f"Evaluating completions with models reversed with judge {args.judge_model}."
        )
        annotations_reversed = annotate_battles(
            judge_chat_model=judge_chat_model,
            instructions=instructions.head(n_instructions).tolist(),
            completions_A=completions_B.head(n_instructions).tolist(),
            completions_B=completions_A.head(n_instructions).tolist(),
            provide_explanation=args.provide_explanation,
            system_prompt=system_prompt,
            truncate_input_chars=args.truncate_all_input_chars,
            use_tqdm=args.use_tqdm,
        )

    name = f"{args.dataset}-{args.model_A}-{args.model_B}-{args.judge_model}"
    name += f"-{args.swap_mode}"
    name = name.replace("/", "_")

    res_folder = Path(args.result_folder) / name
    res_folder.mkdir(parents=True, exist_ok=True)

    # save argument for results analysis
    with open(res_folder / f"args-{name}.json", "w") as f:
        json.dump(asdict(args), f, indent=2)

    print(f"Saving results to {res_folder}")
    df = pd.DataFrame(annotations)
    df["instruction_index"] = instructions.head(n_instructions).index.tolist()
    df["model_A"] = args.model_A
    df["model_B"] = args.model_B
    df["judge"] = args.judge_model

    if args.swap_mode == "both":
        df_reversed = pd.DataFrame(annotations_reversed)
        df_reversed["instruction_index"] = instructions.head(
            n_instructions
        ).index.tolist()
        df_reversed["model_A"] = args.model_B
        df_reversed["model_B"] = args.model_A
        df_reversed["judge"] = args.judge_model
        df = pd.concat([df, df_reversed])

    df.to_csv(res_folder / f"{name}-annotations.csv", index=False)

    # compute preferences between A and B
    score_parser = PairScore()
    prefs = pd.Series(
        [
            score_parser.parse_model_raw(annotation.judge_completion)
            for annotation in annotations
        ]
    )

    if args.swap_mode == "both":
        prefs_reversed = pd.Series(
            [
                score_parser.parse_model_raw(annotation.judge_completion)
                for annotation in annotations_reversed
            ]
        )
        prefs = pd.concat([prefs, (1 - prefs_reversed)]).reset_index(drop=True)

    # compute and report statistics
    summary = _compute_pref_summary(prefs)

    results = {
        "dataset": args.dataset,
        "model_A": args.model_A,
        "model_B": args.model_B,
        "judge_model": args.judge_model,
        **summary,
        "preferences": prefs.tolist(),
        "date": str(datetime.now().isoformat()),
        "user": os.getenv("USER", ""),
    }
    print(f"{args.model_A} vs {args.model_B} judged by {args.judge_model}")
    print_results(results)

    if args.enable_rubrics:
        rubric_label = args.rubric_json if args.rubric_json is not None else args.rubric_name
        print(
            f"Running rubric pairwise scoring with rubric '{rubric_label}' "
            f"(swap debiasing={'on' if args.swap_mode == 'both' else 'off'})."
        )

        try:
            eval_instruction_index = instructions.head(n_instructions).index.tolist()
            eval_instructions = instructions.head(n_instructions).tolist()
            eval_completions_A = completions_A.head(n_instructions).tolist()
            eval_completions_B = completions_B.head(n_instructions).tolist()

            run_info = run_pairwise_rubric_pipeline(
                output_folder=res_folder,
                output_prefix=name,
                judge_model=judge_chat_model,
                instructions=eval_instructions,
                completions_A=eval_completions_A,
                completions_B=eval_completions_B,
                instruction_index=eval_instruction_index,
                model_A_name=args.model_A,
                model_B_name=args.model_B,
                provide_explanation=args.provide_explanation,
                use_tqdm=args.use_tqdm,
                rubric_name=args.rubric_name,
                rubric_json=args.rubric_json,
                swap_to_debias=(args.swap_mode == "both"),
                summary_fields={
                    "dataset": args.dataset,
                    "model_A": args.model_A,
                    "model_B": args.model_B,
                    "judge_model": args.judge_model,
                },
            )
            print(
                f"Saved rubric outputs to {res_folder} "
                f"(prefix: {run_info['prefix']})."
            )
        except Exception as e:
            msg = f"Rubric scoring failed: {e}"
            print(msg)

    with open(res_folder / f"results-{name}.json", "w") as f:
        json.dump(results, f, indent=2)

    return prefs


def cli():
    args = CliArgs.parse_args()
    print(f"Running with CLI args: {args.__dict__}")
    main(args)


if __name__ == "__main__":
    cli()
