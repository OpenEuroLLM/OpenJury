from llmjudgeeval.instruction_dataset.m_arenahard import load_m_arenahard
from llmjudgeeval.utils import data_root, download_hf, read_df


def load_instructions(dataset: str):
    if "m-arena-hard" in dataset:
        if dataset == "m-arena-hard":
            language = None
        else:
            # read the suffix part "m-arena-hard-EU" -> "EU"
            language = dataset.split("-")[-1]
            assert language in [
                None,
                "ar",
                "cs",
                "de",
                "el",
                "en",
                "es",
                "fa",
                "fr",
                "he",
                "hi",
                "id",
                "it",
                "ja",
                "ko",
                "nl",
                "pl",
                "pt",
                "ro",
                "ru",
                "tr",
                "uk",
                "vi",
                "zh",
                "EU",
            ]
        print(f"Loading m-arena-hard with language specification set to {language}")
        df_instructions = load_m_arenahard(local_path=data_root, language=language)
        df_instructions.rename(
            {
                "question_id": "instruction_index",
                "prompt": "instruction",
            },
            axis=1,
            inplace=True,
        )

    else:
        assert dataset in ["alpaca-eval", "arena-hard"]
        local_path_tables = data_root / "tables"
        download_hf(name=dataset, local_path=local_path_tables)
        df_instructions = read_df(local_path_tables / "instructions" / f"{dataset}.csv")

    df_instructions = df_instructions.set_index("instruction_index").sort_index()
    return df_instructions.loc[:, "instruction"]
