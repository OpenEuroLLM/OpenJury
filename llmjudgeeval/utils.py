import time
import asyncio
import os
from pathlib import Path
from typing import Callable

from huggingface_hub import snapshot_download
import pandas as pd
from tqdm.asyncio import tqdm
from langchain_community.llms import LlamaCpp, VLLM
from langchain_openai import ChatOpenAI
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

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
                    results.append(result)
                    pbar.update(1)

            return results

        res = asyncio.run(
            process_with_real_progress(chat_model=chat_model, inputs=inputs)
        )
    else:
        res = chat_model.batch(inputs=inputs, **invoke_kwargs)

    # Not sure why the API of Langchain returns sometime a string and sometimes an AIMessage object
    # is it because of using Chat and barebones models?
    # when using OpenAI, the output is AIMessage not a string...
    res = [x.content if hasattr(x, "content") else x for x in res]
    return res


class DummyModel:
    def batch(self, inputs, **invoke_kwargs) -> list[str]:
        return ["Dummy output"] * len(inputs)

    def invoke(self, input, **invoke_kwargs) -> str:
        return "Dummy"

    async def ainvoke(self, input, **invoke_kwargs):
        return "Dummy"


def make_model(model: str, max_tokens: int | None = 200):
    if model == "Dummy":
        return DummyModel()

    model_provider = model.split("/")[0]
    model_kwargs = {}
    if max_tokens is not None:
        if model_provider == "Together":
            # TODO allow to specify kwargs in model string
            model_kwargs["max_tokens"] = max_tokens
        if model_provider == "VLLM":
            model_kwargs["max_model_len"] = max_tokens

    model_name = "/".join(model.split("/")[1:])
    print(f"Loading {model_provider}(model={model_name})")
    model_classes = [
        LlamaCpp,
        ChatOpenAI,
        VLLM,
    ]
    model_kwargs["model"] = model_name
    try:
        from langchain_together.llms import Together

        model_classes.append(Together)
    except ImportError:
        pass
    try:
        from langchain_openai.llms import OpenAI

        model_classes.append(OpenAI)
    except ImportError:
        pass
    model_cls_dict = {model_cls.__name__: model_cls for model_cls in model_classes}
    assert (
        model_provider in model_cls_dict
    ), f"{model_provider} not available, choose among {list(model_cls_dict.keys())}"
    return model_cls_dict[model_provider](**model_kwargs)


def download_all():
    print(f"Downloading all dataset in {data_root}")
    for dataset in ["alpaca-eval", "arena-hard", "m-arena-hard"]:
        local_path_tables = data_root / "tables"
        download_hf(name=dataset, local_path=local_path_tables)

    snapshot_download(
        repo_id="geoalgo/multilingual-contexts-to-be-completed",
        repo_type="dataset",
        allow_patterns="*",
        local_dir=data_root / "contexts",
        force_download=False,
    )


class Timeblock:
    """Timer context manager"""

    def __init__(self, name: str | None = None, verbose: bool = True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start = time.time()
        return self

    def __exit__(self, *args):
        """Stop the context manager timer"""
        self.end = time.time()
        self.duration = self.end - self.start
        if self.verbose:
            print(self)

    def __str__(self):
        name = self.name if self.name else "block"
        msg = f"{name} took {self.duration} seconds"
        return msg


def cache_function_dataframe(
    fun: Callable[[], pd.DataFrame],
    cache_name: str,
    ignore_cache: bool = False,
    cache_path: Path | None = None,
):
    f"""
    :param fun: a function whose dataframe result obtained `fun()` will be cached
    :param cache_name: the cache of the function result is written into `{cache_path}/{cache_name}.csv.zip`
    :param ignore_cache: whether to recompute even if the cache is present
    :param cache_path: folder where to write cache files, default to ~/cache-zeroshot/
    :return: result of fun()
    """
    if cache_path is None:
        cache_path = data_root / "cache"
    cache_file = cache_path / (cache_name + ".csv.zip")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists() and not ignore_cache:
        print(f"Loading cache {cache_file}")
        return pd.read_csv(cache_file)
    else:
        print(
            f"Cache {cache_file} not found or ignore_cache set to True, regenerating the file"
        )
        with Timeblock("Evaluate function."):
            df = fun()
            assert isinstance(df, pd.DataFrame)
            df.to_csv(cache_file, index=False)
            return pd.read_csv(cache_file)


if __name__ == "__main__":
    download_all()
