from pathlib import Path
import pandas as pd
from huggingface_hub import snapshot_download

from llmjudgeeval.utils import data_root


def load_m_arenahard(local_path, language: str | None = None):
    snapshot_download(
        repo_id="CohereLabs/m-ArenaHard",
        repo_type="dataset",
        allow_patterns=f"*",
        local_dir=local_path / "m-ArenaHard",
        force_download=False,
    )

    df_union = []
    m_arena_root = Path(local_path / "m-ArenaHard")
    eu_languages = [
        "cs",
        "de",
        "el",
        "en",
        "es",
        "fr",
        "it",
        "nl",
        "pl",
        "pt",
        "ro",
        "uk",
    ]
    for path in Path(m_arena_root).rglob("*.parquet"):
        lg = path.parent.name
        if language == "EU" and lg in eu_languages:
            df = pd.read_parquet(path)
            df["lang"] = lg
            df_union.append(df)
        elif language is None or language == lg:
            df = pd.read_parquet(path)
            df["lang"] = lg
            df_union.append(df)

    df_instructions = pd.concat(df_union, ignore_index=True)

    df_instructions.rename(
        {
            "question_id": "instruction_index",
            "prompt": "instruction",
        },
        axis=1,
        inplace=True,
    )
    df_instructions = df_instructions.set_index("instruction_index").sort_index()
    return df_instructions.loc[:, "instruction"]


if __name__ == "__main__":
    load_m_arenahard(local_path=data_root, language="EU")
