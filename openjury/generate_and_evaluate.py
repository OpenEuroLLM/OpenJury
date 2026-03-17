"""
This script generates completions for a given dataset and model,
and then evaluates them using a judge model.
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd

from openjury.evaluate import (
    annotate_battles,
    PairScore,
)
from openjury.generate import generate_instructions, generate_base
from openjury.instruction_dataset import load_instructions
from openjury.mt_bench.pipeline import (
    format_mt_bench_for_evaluation,
    run_mt_bench,
)
from openjury.utils import (
    cache_function_dataframe,
    data_root,
    download_hf,
    make_model,
    read_df,
)


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
    mt_bench_turns: str = "both"
    mt_bench_compatibility: str = "openjury"
    result_folder: str = "results"
    engine_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        supported_modes = ["fixed", "both"]
        assert (
            self.swap_mode in supported_modes
        ), f"Only {supported_modes} modes are supported but got {self.swap_mode}."
        supported_mt_bench_modes = ["openjury", "fastchat"]
        assert (
            self.mt_bench_compatibility in supported_mt_bench_modes
        ), f"Only {supported_mt_bench_modes} are supported but got {self.mt_bench_compatibility}."

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
            "--mt_bench_turns",
            type=str,
            choices=["both", "single", "multi"],
            default="both",
            help="Which MT-Bench turns to evaluate. 'single': only turn 1, "
            "'multi': only turn 2 (with full conversation context), "
            "'both' (default): evaluate both turns.",
        )
        parser.add_argument(
            "--mt_bench_compatibility",
            type=str,
            choices=["openjury", "fastchat"],
            default="openjury",
            help=(
                "MT-Bench evaluation/generation mode. "
                "'openjury' (default): OpenJury score_A/score_B prompt + softmax preference. "
                "'fastchat': use FastChat/MT-Bench pairwise prompts with [[A]]/[[B]]/[[C]] verdict parsing, "
                "conservative position-bias handling, judge temperature=0, and MT-Bench category temperatures."
            ),
        )
        parser.add_argument(
            "--engine_kwargs",
            type=str,
            required=False,
            default="{}",
            help=(
                "JSON dict of engine-specific kwargs forwarded to the underlying engine. "
                "Example for vLLM: '{\"tensor_parallel_size\": 2, \"gpu_memory_utilization\": 0.9}'."
            ),
        )
        args = parser.parse_args()

        try:
            engine_kwargs = (
                json.loads(args.engine_kwargs) if args.engine_kwargs else {}
            )
            if not isinstance(engine_kwargs, dict):
                raise ValueError("engine_kwargs must be a JSON object")
        except Exception as e:
            raise SystemExit(f"Failed to parse --engine_kwargs: {e}")

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
            mt_bench_turns=args.mt_bench_turns,
            mt_bench_compatibility=args.mt_bench_compatibility,
            result_folder=args.result_folder,
            engine_kwargs=engine_kwargs,
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
    if results.get("num_missing", 0) > 0:
        print(f"   ❓ Missing: {results['num_missing']}")

    per_category = results.get("per_category")
    if per_category:
        print("\nPer-Category Breakdown:")
        print(
            f"  {'Category':<14} | {'Win Rate(A)':>11} | {'Wins':>4} | {'Losses':>6} | {'Ties':>4}"
        )
        print(f"  {'-' * 14}-+-{'-' * 11}-+-{'-' * 4}-+-{'-' * 6}-+-{'-' * 4}")
        for cat, stats in sorted(per_category.items()):
            print(
                f"  {cat:<14} | {stats['winrate']:>11.1%} | "
                f"{stats['num_wins']:>4} | {stats['num_losses']:>6} | {stats['num_ties']:>4}"
            )

    per_turn = results.get("per_turn")
    if per_turn:
        print("\nPer-Turn Breakdown:")
        for turn, stats in sorted(per_turn.items()):
            print(
                f"  Turn {turn} Win Rate(A): {stats['winrate']:.1%} "
                f"(W:{stats['num_wins']} L:{stats['num_losses']} T:{stats['num_ties']})"
            )
    print("=" * 60 + "\n")


def compute_preference_stats(prefs: pd.Series) -> dict:
    """Derive win/loss/tie counts and winrate from a Series of preferences.

    Preference < 0.5 means model A wins, > 0.5 means model B wins,
    exactly 0.5 is a tie.  None/NaN values are counted as missing.
    """
    num_battles = len(prefs)
    num_wins = int(sum(prefs < 0.5))
    num_losses = int(sum(prefs > 0.5))
    num_ties = int(sum(prefs == 0.5))
    num_missing = num_battles - (num_wins + num_losses + num_ties)
    denom = num_wins + num_losses + num_ties
    winrate = float((num_wins + 0.5 * num_ties) / denom) if denom else 0.0
    return {
        "num_battles": num_battles,
        "num_wins": num_wins,
        "num_losses": num_losses,
        "num_ties": num_ties,
        "num_missing": num_missing,
        "winrate": winrate,
    }


def _compute_grouped_stats(
    preferences: pd.Series,
    metadata: list[dict[str, object]],
    group_by: str,
) -> dict[object, dict[str, float | int]]:
    grouped: dict[object, list[float]] = {}
    for meta, pref in zip(metadata, preferences):
        key = meta.get(group_by)
        if key is None:
            continue
        grouped.setdefault(key, []).append(pref)
    return {
        key: compute_preference_stats(pd.Series(vals))
        for key, vals in grouped.items()
    }


def _parse_preferences_from_annotations(
    annotations: list,
    score_parser: PairScore,
) -> pd.Series:
    return pd.Series(
        [
            score_parser.parse_model_raw(annotation.judge_completion)
            for annotation in annotations
        ]
    )


def _judge_turn(
    *,
    judge_chat_model,
    instructions: list[str],
    completions_A: list[str],
    completions_B: list[str],
    metadata: list[dict[str, object]],
    score_parser: PairScore,
    provide_explanation: bool,
    swap_mode: str,
    truncate_input_chars: int | None,
    use_tqdm: bool,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
) -> tuple[
    list,
    list,
    list[dict[str, object]],
    list[dict[str, object]],
    pd.Series,
    list[dict[str, object]],
]:
    if not instructions:
        return [], [], [], [], pd.Series(dtype=float), []

    annotations = annotate_battles(
        judge_chat_model=judge_chat_model,
        instructions=instructions,
        completions_A=completions_A,
        completions_B=completions_B,
        provide_explanation=provide_explanation,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        truncate_input_chars=truncate_input_chars,
        use_tqdm=use_tqdm,
    )
    preference_parts = [_parse_preferences_from_annotations(annotations, score_parser)]

    annotations_reversed: list = []
    metadata_for_reversed_annotations: list[dict[str, object]] = []
    combined_metadata = list(metadata)

    if swap_mode == "both":
        print("Correction for judge bias towards a certain model position is set.")
        print("Evaluating completions with models reversed.")
        annotations_reversed = annotate_battles(
            judge_chat_model=judge_chat_model,
            instructions=instructions,
            completions_A=completions_B,
            completions_B=completions_A,
            provide_explanation=provide_explanation,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            truncate_input_chars=truncate_input_chars,
            use_tqdm=use_tqdm,
        )
        prefs_reversed = _parse_preferences_from_annotations(
            annotations_reversed, score_parser
        )
        preference_parts.append(1 - prefs_reversed)
        metadata_for_reversed_annotations = list(metadata)
        combined_metadata.extend(metadata)

    preferences = pd.concat(preference_parts).reset_index(drop=True)
    return (
        annotations,
        annotations_reversed,
        list(metadata),
        metadata_for_reversed_annotations,
        preferences,
        combined_metadata,
    )


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

    # MT-Bench has its own pipeline: multi-turn generation + category-aware judging
    if args.dataset == "mt-bench":
        return run_mt_bench(args, ignore_cache)

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
        partial(
            generate_base,
            truncate_input_chars=args.truncate_all_input_chars,
            max_tokens=args.max_out_tokens_models,
            max_model_len=args.max_model_len,
            chat_template=args.chat_template,
            use_tqdm=args.use_tqdm,
            **args.engine_kwargs,
        )
        if is_fluency_task
        else partial(
            generate_instructions,
            truncate_input_chars=args.truncate_all_input_chars,
            max_tokens=args.max_out_tokens_models,
            max_model_len=args.max_model_len,
            chat_template=args.chat_template,
            use_tqdm=args.use_tqdm,
            **args.engine_kwargs,
        )
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
        **args.engine_kwargs,
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

    instruction_subset = instructions.head(n_instructions)
    instruction_indices = instruction_subset.index.tolist()
    metadata = [{"instruction_index": idx} for idx in instruction_indices]
    score_parser = PairScore()
    (
        annotations,
        annotations_reversed,
        metadata_for_annotations,
        metadata_for_reversed_annotations,
        prefs,
        _combined_metadata,
    ) = _judge_turn(
        judge_chat_model=judge_chat_model,
        instructions=instruction_subset.tolist(),
        completions_A=completions_A.head(n_instructions).tolist(),
        completions_B=completions_B.head(n_instructions).tolist(),
        metadata=metadata,
        score_parser=score_parser,
        provide_explanation=args.provide_explanation,
        swap_mode=args.swap_mode,
        truncate_input_chars=args.truncate_all_input_chars,
        use_tqdm=args.use_tqdm,
        system_prompt=system_prompt,
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
    df["instruction_index"] = [
        meta["instruction_index"] for meta in metadata_for_annotations
    ]
    df["model_A"] = args.model_A
    df["model_B"] = args.model_B
    df["judge"] = args.judge_model

    if args.swap_mode == "both":
        df_reversed = pd.DataFrame(annotations_reversed)
        df_reversed["instruction_index"] = [
            meta["instruction_index"] for meta in metadata_for_reversed_annotations
        ]
        df_reversed["model_A"] = args.model_B
        df_reversed["model_B"] = args.model_A
        df_reversed["judge"] = args.judge_model
        df = pd.concat([df, df_reversed])

    df.to_csv(res_folder / f"{name}-annotations.csv", index=False)

    stats = compute_preference_stats(prefs)
    results = {
        "dataset": args.dataset,
        "model_A": args.model_A,
        "model_B": args.model_B,
        "judge_model": args.judge_model,
        **stats,
        "preferences": prefs.tolist(),
        "date": str(datetime.now().isoformat()),
        "user": os.getenv("USER", ""),
    }
    print(f"{args.model_A} vs {args.model_B} judged by {args.judge_model}")
    print_results(results)

    with open(res_folder / f"results-{name}.json", "w") as f:
        json.dump(results, f, indent=2)

    return prefs


def cli():
    args = CliArgs.parse_args()
    print(f"Running with CLI args: {args.__dict__}")
    main(args)


if __name__ == "__main__":
    cli()
