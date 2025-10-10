from pathlib import Path
import pandas as pd
from huggingface_hub import snapshot_download

from openjury.utils import data_root


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

    assert len(df_union) > 0, f"Invalid language passed {language}"
    return pd.concat(df_union, ignore_index=True)


if __name__ == "__main__":
    load_m_arenahard(local_path=data_root, language="EU")
