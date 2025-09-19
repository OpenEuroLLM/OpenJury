import asyncio
import os
from pathlib import Path

from huggingface_hub import snapshot_download
import pandas as pd
from tqdm.asyncio import tqdm
from langchain_community.llms import LlamaCpp, VLLM
from langchain_together.llms import Together
from langchain_openai import ChatOpenAI, OpenAI
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

from llmjudgeeval.instruction_dataset import load_instructions

data_root = Path(
    os.environ.get("LLM_JUDGE_EVAL_DATA", Path("~/llm-judge-eval-data/").expanduser())
).expanduser()


def set_langchain_cache():
    set_llm_cache(SQLiteCache(database_path=str(data_root / ".langchain.db")))


def download_hf(name: str, local_path: Path):
    local_path.mkdir(exist_ok=True, parents=True)
    # downloads the model from huggingface into `local_path` folder
    snapshot_download(
        repo_id="geoalgo/llmjudge",
        repo_type="dataset",
        allow_patterns=f"*{name}*",
        local_dir=local_path,
        force_download=False,
    )


def read_df(filename: Path, **pandas_kwargs) -> pd.DataFrame:
    assert filename.exists(), f"Dataframe file not found at {filename}"
    if filename.name.endswith(".csv.zip") or filename.name.endswith(".csv"):
        return pd.read_csv(filename, **pandas_kwargs)
    else:
        assert filename.name.endswith(".parquet"), f"Unsupported extension {filename}"
        return pd.read_parquet(filename, **pandas_kwargs)


def do_inference(chat_model, inputs, use_tqdm: bool = True):
    invoke_kwargs = {
        # "stop": ["```"],
        # "max_tokens": 100,
    }
    if use_tqdm:
        # perform inference asynchronously to be able to update tqdm, chat_model.batch does not work as it blocks until
        # all requests are received
        async def process_with_real_progress(chat_model, inputs):
            async def process_single(input_item):
                return await chat_model.ainvoke(input_item, **invoke_kwargs)

            # Create all tasks
            tasks = [asyncio.create_task(process_single(inp)) for inp in inputs]
            results = []

            # Track progress as tasks complete
            with tqdm(total=len(inputs)) as pbar:
                for task in asyncio.as_completed(tasks):
                    result = await task
                    # Not sure why the API of Langchain returns sometime a string and sometimes an AIMessage object
                    # is it because of using Chat and barebones models?
                    # when using OpenAI, the output is AIMessage not a string...
                    if hasattr(result, "content"):
                        result = result.content
                    results.append(result)
                    pbar.update(1)

            return results

        return asyncio.run(
            process_with_real_progress(chat_model=chat_model, inputs=inputs)
        )
    else:
        return chat_model.batch(inputs=inputs, **invoke_kwargs)


def make_model(model_provider: str, **kwargs):
    # TODO get the list of classes programmatically rather
    model_classes = [
        LlamaCpp,
        Together,
        # OpenAI,
        ChatOpenAI,
        VLLM,
    ]
    model_cls_dict = {model_cls.__name__: model_cls for model_cls in model_classes}
    assert model_provider in model_cls_dict
    return model_cls_dict[model_provider](**kwargs)


def download_all():
    for dataset in ["alpaca-eval", "arena-hard", "m-arena-hard"]:
        local_path_tables = data_root / "tables"
        download_hf(name=dataset, local_path=local_path_tables)
        instructions = load_instructions(
            dataset=dataset,
        )
